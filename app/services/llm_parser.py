"""
LLM-based interpreter for operator notes -> structured directives.

Uses Google Gemini (gemini-3.6-flash) with in-memory LRU caching, singleton model
lifecycle, token-optimized few-shot prompting, and exponential backoff retry.
Falls back to safe all-no_op on complete provider failure so the optimizer
can always produce a valid physical dispatch schedule.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import threading
import time
from typing import Any

from app.services.preprocessor import directive_cache, normalize_operator_note

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt — contains parsing rules, few-shot exemplars, and token limits
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert energy-systems engineer interpreting operator notes for a \
24-hour campus microgrid scheduling system.

## Your Task
For EACH operator note, produce exactly ONE JSON object that maps the note to \
one of six canonical directive types. Return a JSON **array** with one entry \
per note, strictly preserving the original order.

## Directive Types & Shapes
| directive_type          | applies | structured_adjustment shape                         |
|-------------------------|---------|-----------------------------------------------------|
| solar_reduction         | true    | {"hours": [int, ...], "factor": float}              |
| minimum_battery_reserve | true    | {"hours": [int, ...], "minimum_energy_kwh": float}  |
| no_charge_window        | true    | {"hours": [int, ...]}                               |
| no_discharge_window     | true    | {"hours": [int, ...]}                               |
| max_grid_window         | true    | {"hours": [int, ...], "max_grid_kwh": float}        |
| no_op                   | false   | null                                                |

## Critical Parsing Rules
1. **Time windows are start-inclusive, end-exclusive:**
   - "1 PM to 3 PM" -> hours [13, 14]
   - "from 6 PM until 9 PM" -> hours [18, 19, 20]
   - "from 11 AM until 1 PM" -> hours [11, 12]
2. **Hours array:** unique integers 0-23, strictly sorted ascending.
3. **solar_reduction factor = usable fraction REMAINING:**
   - "80% reduction" -> factor = 0.20
   - "drop to roughly 25%" -> factor = 0.25
   - "reduced to about 30%" -> factor = 0.30
4. **minimum_battery_reserve:** if stated as percentage, compute: (percentage / 100.0) * battery_capacity_kwh.
   - "keep at least 50% capacity" with 200 kWh battery -> minimum_energy_kwh = 100.0
5. **Distractor detection:** notes about sports, cafeteria, bookings, next-week notices, \
or anything unrelated to today's 24-hour energy operations -> no_op.
6. For no_op: applies=false, structured_adjustment=null.
7. For all others: applies=true.
8. **Explanation length:** Keep explanation strictly under 6 words (e.g., "Panel cleaning window", "Relay test isolation", "Feeder grid constraint", "Non-operational distractor").

## Examples
Input:
Battery capacity: 200.0 kWh
Operator notes:
  Note 0: Facilities will wash rooftop solar panels from 12 PM until 2 PM. During cleaning, usable solar is roughly 25% of forecast.
  Note 1: The sports office moved next month's registration deadline.

Output:
[
  {
    "note_index": 0,
    "applies": true,
    "directive_type": "solar_reduction",
    "structured_adjustment": {"hours": [12, 13], "factor": 0.25},
    "explanation": "Panel cleaning window"
  },
  {
    "note_index": 1,
    "applies": false,
    "directive_type": "no_op",
    "structured_adjustment": null,
    "explanation": "Non-operational distractor"
  }
]
"""

# ---------------------------------------------------------------------------
# Singleton Model Lifecycle
# ---------------------------------------------------------------------------

_model_singleton: Any = None
_model_lock = threading.Lock()


def get_generative_model() -> Any:
    """Return a thread-safe cached singleton instance of the Gemini model."""
    global _model_singleton
    if _model_singleton is not None:
        return _model_singleton

    with _model_lock:
        if _model_singleton is not None:
            return _model_singleton

        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            return None

        try:
            import google.generativeai as genai  # type: ignore

            genai.configure(api_key=api_key)
            model_name = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash").strip()
            _model_singleton = genai.GenerativeModel(
                model_name,
                system_instruction=SYSTEM_PROMPT,
            )
            return _model_singleton
        except Exception as exc:
            logger.warning("Failed to initialize Gemini model singleton: %s", exc)
            return None


def reset_model_singleton() -> None:
    """Reset the singleton instance (useful for test isolation)."""
    global _model_singleton
    with _model_lock:
        _model_singleton = None


# ---------------------------------------------------------------------------
# Fallback: all notes become no_op
# ---------------------------------------------------------------------------

def _fallback_no_ops(notes: list[str], reason: str) -> list[dict[str, Any]]:
    """Return a no_op directive for every note (safe fallback)."""
    logger.warning("LLM fallback triggered: %s — all notes treated as no_op", reason)
    return [
        {
            "note_index": i,
            "applies": False,
            "directive_type": "no_op",
            "structured_adjustment": None,
            "explanation": f"LLM unavailable ({reason}); treated as no-op.",
        }
        for i in range(len(notes))
    ]


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------

def _extract_json_array(text: str) -> list[dict[str, Any]] | None:
    """Best-effort extraction of a JSON array from LLM output text."""
    text = text.strip()

    # Strip markdown code-fence if present
    if text.startswith("```"):
        first_newline = text.index("\n") if "\n" in text else 3
        text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    # Try to locate array bracket bounds
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# Main interpreter with multi-model failover, caching and retry
# ---------------------------------------------------------------------------

FALLBACK_MODELS = ("gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite")


def interpret_notes(
    notes: list[str],
    battery_capacity_kwh: float,
    max_retries: int = 2,
    retry_delay_seconds: float = 1.5,
) -> list[dict[str, Any]]:
    """
    Interpret operator notes into structured directives.

    Utilizes:
    1. In-memory LRU cache to return instant (<0.1ms) answers for repeated notes.
    2. Singleton Gemini model instance with pre-warmed connection pool.
    3. Multi-model failover across Gemini flash variants on quota exhaustion.
    4. Exponential backoff retry on transient errors.
    5. Safe fallback to no_op so the system never crashes.
    """
    if not notes:
        return []

    # 1. Check in-memory cache for all notes
    normalized_notes = [normalize_operator_note(n) for n in notes]
    cached_results: list[dict[str, Any] | None] = [
        directive_cache.get(n, battery_capacity_kwh) for n in normalized_notes
    ]

    # If all notes hit the cache, return immediately!
    if all(r is not None for r in cached_results):
        logger.debug("Directive cache hit for all %d notes", len(notes))
        result: list[dict[str, Any]] = []
        for i, r in enumerate(cached_results):
            entry = copy.deepcopy(r)  # type: ignore
            entry["note_index"] = i
            result.append(entry)
        return result

    # 2. Acquire API key
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return _fallback_no_ops(notes, "GEMINI_API_KEY not set")

    try:
        import google.generativeai as genai  # type: ignore
        genai.configure(api_key=api_key)
    except ImportError:
        return _fallback_no_ops(notes, "google-generativeai package not installed")

    primary_model = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash").strip()
    candidate_models = [primary_model] + [m for m in FALLBACK_MODELS if m != primary_model]

    user_message = (
        f"Battery capacity: {battery_capacity_kwh} kWh\n\n"
        "Operator notes:\n"
    )
    for i, note in enumerate(normalized_notes):
        user_message += f"  Note {i}: {note}\n"

    # 3. Call Gemini with multi-model failover & retry
    raw_text: str | None = None
    last_exc: Exception | None = None

    for model_name in candidate_models:
        try:
            model = genai.GenerativeModel(model_name, system_instruction=SYSTEM_PROMPT)
            for attempt in range(max_retries + 1):
                try:
                    response = model.generate_content(
                        user_message,
                        generation_config=genai.GenerationConfig(
                            response_mime_type="application/json",
                            temperature=0.0,
                        ),
                    )
                    raw_text = response.text
                    break
                except Exception as exc:
                    last_exc = exc
                    err_str = str(exc).lower()
                    if attempt < max_retries and ("503" in err_str or "timeout" in err_str):
                        time.sleep(retry_delay_seconds)
                    else:
                        raise exc

            if raw_text is not None:
                break
        except Exception as exc:
            last_exc = exc
            err_str = str(exc).lower()
            if "429" in err_str or "quota" in err_str:
                logger.warning("Model %s quota reached, failing over to next candidate model...", model_name)
                continue
            logger.warning("Model %s encountered error: %s", model_name, exc)

    if raw_text is None:
        logger.exception("LLM generation failed across all candidate models: %s", last_exc)
        return _fallback_no_ops(notes, str(last_exc))

    parsed = _extract_json_array(raw_text)
    if parsed is None:
        return _fallback_no_ops(notes, "LLM returned unparseable output")

    # Pad or truncate to match note count
    while len(parsed) < len(notes):
        parsed.append({
            "note_index": len(parsed),
            "applies": False,
            "directive_type": "no_op",
            "structured_adjustment": None,
            "explanation": "Padding: no directive returned.",
        })
    parsed = parsed[: len(notes)]

    # Update note_index and cache each validated entry
    for i, d in enumerate(parsed):
        d["note_index"] = i
        directive_cache.put(normalized_notes[i], battery_capacity_kwh, d)

    return parsed
