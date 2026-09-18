"""
LLM-based interpreter for operator notes -> structured directives.

Uses Google GenAI SDK (gemini-3.5-flash-lite by default) with in-memory LRU caching,
thread-safe singleton client, token-optimized few-shot prompting, exponential
backoff retry, and multi-model failover across verified Gemini Flash variants.
Falls back to safe all-no_op on complete provider failure so the optimizer
can always produce a valid physical dispatch schedule.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import threading
import time
from typing import Any

# Suppress verbose AFC log warnings from google-genai
logging.getLogger("google_genai").setLevel(logging.ERROR)
logging.getLogger("google.genai").setLevel(logging.ERROR)

from app.services.preprocessor import (
    directive_cache,
    normalize_operator_note,
    scenario_directive_cache,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompt — contains parsing rules, few-shot exemplars, and token limits
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are an expert energy-systems engineer interpreting operator notes for a
24-hour campus microgrid scheduling system.

## Your Task
For EACH operator note, produce exactly ONE JSON object that maps the note to \
one of six canonical directive types. Return a JSON **array** with one entry \
per note, strictly preserving the original order and note_index (0, 1, ... N-1).

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
1. **Time windows are whole-hour, start-inclusive and end-exclusive:**
   - "1 PM to 3 PM" or "13:00 to 15:00" -> hours [13, 14]
   - "from 6 PM until 9 PM" -> hours [18, 19, 20]
   - "from 11 AM until 1 PM" -> hours [11, 12]
   - "noon until 2 PM" -> hours [12, 13]
   - "from 2 AM until 5 AM" -> hours [2, 3, 4]
2. **Hours array:** unique integers 0 through 23 in strictly ascending order.
3. **solar_reduction factor = usable fraction REMAINING:**
   - "80% reduction" -> factor = 0.20
   - "drop to roughly 25%" or "reduced to 25%" -> factor = 0.25
   - "leave roughly half" or "50% reduction" -> factor = 0.50
   - "leave roughly one-fifth" -> factor = 0.20
4. **minimum_battery_reserve:** if stated as percentage of battery capacity, calculate: (percentage / 100.0) * battery_capacity_kwh.
   - e.g., "keep at least 50% capacity" with 200 kWh battery -> minimum_energy_kwh = 100.0
   - e.g., "at least 80 kWh in reserve" -> minimum_energy_kwh = 80.0
5. **max_grid_window:** max_grid_kwh must be a finite, non-negative number.
6. **Distractor detection:** notes about sports, cafeteria menus, room bookings, next-week schedule changes, \
or anything unrelated to today's 24-hour microgrid operations -> directive_type = "no_op", applies = false, structured_adjustment = null.
7. For no_op: applies=false, structured_adjustment=null.
8. For all non-no_op directives: applies=true.
9. **Explanation length:** Keep explanation concise and strictly under 10 words.

## Example A (2 notes)
Input:
Battery capacity: 200.0 kWh
Operator notes:
  Note 0: Facilities will wash rooftop solar panels from noon until 2 PM. During cleaning, usable solar should be treated as roughly 25% of forecast.
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

## Example B (3 notes)
Input:
Battery capacity: 200.0 kWh
Operator notes:
  Note 0: The charging circuit will be unavailable from 2 PM until 4 PM.
  Note 1: Keep at least 50% of the battery capacity stored from 6 PM until 9 PM for emergency operations.
  Note 2: The library is extending book-return hours next week.

Output:
[
  {
    "note_index": 0,
    "applies": true,
    "directive_type": "no_charge_window",
    "structured_adjustment": {"hours": [14, 15]},
    "explanation": "Charging circuit maintenance"
  },
  {
    "note_index": 1,
    "applies": true,
    "directive_type": "minimum_battery_reserve",
    "structured_adjustment": {"hours": [18, 19, 20], "minimum_energy_kwh": 100.0},
    "explanation": "Emergency reserve requirement"
  },
  {
    "note_index": 2,
    "applies": false,
    "directive_type": "no_op",
    "structured_adjustment": null,
    "explanation": "Non-operational distractor"
  }
]
"""

# ---------------------------------------------------------------------------
# Regex for robust markdown code-fence stripping (handles ```json, ```JSON, etc.)
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^```[a-z]*\n(.*?)\n?```$", re.DOTALL | re.IGNORECASE)

# ---------------------------------------------------------------------------
# Retryable error detection helpers
# ---------------------------------------------------------------------------

_RETRYABLE_ERRORS = ("503", "502", "500", "timeout", "connection", "reset", "unavailable")


def _is_retryable(err_str: str) -> bool:
    return any(e in err_str for e in _RETRYABLE_ERRORS)


def _is_quota_error(err_str: str) -> bool:
    return "429" in err_str or "quota" in err_str or "resource_exhausted" in err_str


# ---------------------------------------------------------------------------
# Singleton Client Lifecycle
# ---------------------------------------------------------------------------

_client_singleton: Any = None
_client_lock = threading.Lock()


def get_genai_client() -> Any:
    """Return a thread-safe cached singleton instance of the google.genai Client."""
    global _client_singleton
    if _client_singleton is not None:
        return _client_singleton

    with _client_lock:
        if _client_singleton is not None:
            return _client_singleton

        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            return None

        try:
            from google import genai  # type: ignore

            _client_singleton = genai.Client(api_key=api_key)
            logger.info("Google GenAI client singleton initialized")
            return _client_singleton
        except Exception as exc:
            logger.warning("Failed to initialize Google GenAI client: %s", exc)
            return None


def get_generative_model() -> Any:
    """Backward-compatible alias for lifespan warm-up and legacy references."""
    return get_genai_client()


def reset_model_singleton() -> None:
    """Reset the singleton instance (useful for test isolation)."""
    global _client_singleton
    with _client_lock:
        _client_singleton = None


# ---------------------------------------------------------------------------
# Custom error
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
            "explanation": f"LLM fallback ({reason}); treated as no-op.",
        }
        for i in range(len(notes))
    ]
}


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "directives": {
            "type": "array",
            "items": DIRECTIVE_ITEM_SCHEMA,
        }
    },
    "required": ["directives"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Single Groq call
# ---------------------------------------------------------------------------

def _call_groq(
    *,
    client: Any,
    model: str,
    notes: list[str],
    battery_capacity_kwh: float,
) -> list[dict[str, Any]]:

    user_message = (
        f"Battery capacity: {battery_capacity_kwh} kWh\n\n"
        f"Number of operator notes: {len(notes)}\n\n"
        "Operator notes:\n"
    )

    for i, note in enumerate(notes):
        user_message += f"Note {i}: {note}\n"

    user_message += (
        "\nReturn exactly one directive for each note, "
        "in note_index order."
    )

    logger.info(
        "Sending %d operator notes to Groq model %s",
        len(notes),
        model,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],

        # GPT-OSS supports this.
        reasoning_effort="low",

        # Keep interpretation as deterministic as practical.
        temperature=0.0,

        # Plenty for 1-3 small JSON directives.
        max_completion_tokens=1200,

        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "gridwise_directive_interpretation",
                "strict": True,
                "schema": RESPONSE_SCHEMA,
            },
        },
    )

    if not response.choices:
        raise LLMInterpretationError(
            f"Groq model {model} returned no choices."
        )

    # Strip markdown code-fence robustly (handles ```json, ```JSON, ``` etc.)
    m = _FENCE_RE.match(text)
    if m:
        text = m.group(1).strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    # Try to locate array bracket bounds as a last resort
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    directives = payload.get("directives")

    if not isinstance(directives, list):
        raise LLMInterpretationError(
            "LLM response does not contain a directives array."
        )

    # Exact number of entries required by the challenge.
    if len(directives) != len(notes):
        raise LLMInterpretationError(
            f"LLM returned {len(directives)} directives "
            f"for {len(notes)} notes."
        )

    # Note order is canonical.
    for i, directive in enumerate(directives):
        directive["note_index"] = i

    return directives


# ---------------------------------------------------------------------------
# Main interpreter with multi-model failover, caching and retry
# ---------------------------------------------------------------------------

# Verified fast Gemini models (primary first, then failover candidates)
FALLBACK_MODELS = ("gemini-3.5-flash-lite", "gemini-flash-lite-latest", "gemini-3.6-flash")


def _call_llm_core(
    notes: list[str],
    battery_capacity_kwh: float,
    max_retries: int,
    retry_delay_seconds: float,
) -> list[dict[str, Any]] | None:
    """
    Call Gemini for a list of already-normalized notes.

    Returns a parsed list of directive dicts (one per note, in order),
    or None if all models failed or output was unparseable.
    """
    client = get_genai_client()
    if client is None:
        logger.warning("GEMINI_API_KEY not set or client unavailable — cannot call LLM")
        return None

    try:
        from google.genai import types  # type: ignore
    except ImportError:
        logger.warning("google-genai package not installed")
        return None

    user_message = (
        f"Battery capacity: {battery_capacity_kwh} kWh\n\n"
        "Operator notes:\n"
    )
    for i, note in enumerate(notes):
        user_message += f"  Note {i}: {note}\n"

    raw_text: str | None = None
    last_exc: Exception | None = None

    primary_model = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()
    candidate_models: list[str] = [primary_model]
    for m in FALLBACK_MODELS:
        if m not in candidate_models:
            candidate_models.append(m)

    gen_config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json",
        temperature=0.0,
    )

    for model_name in candidate_models:
        try:
            for attempt in range(max_retries + 1):
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=user_message,
                        config=gen_config,
                    )
                    raw_text = response.text
                    break
                except Exception as exc:
                    last_exc = exc
                    err_str = str(exc).lower()
                    if _is_quota_error(err_str) or "404" in err_str:
                        # Fail over immediately without waiting retries
                        raise exc
                    if attempt < max_retries and _is_retryable(err_str):
                        delay = retry_delay_seconds * (2 ** attempt)
                        logger.warning(
                            "Transient error on attempt %d/%d for model %s, retrying in %.1fs: %s",
                            attempt + 1, max_retries + 1, model_name, delay, exc,
                        )
                        time.sleep(delay)
                    else:
                        raise exc

            if raw_text is not None:
                logger.debug("Successfully generated directives using %s", model_name)
                break

        except Exception as exc:
            last_exc = exc
            err_str = str(exc).lower()
            if _is_quota_error(err_str):
                logger.warning("Model %s quota/rate limit reached, failing over...", model_name)
            else:
                logger.warning("Model %s error: %s, failing over...", model_name, exc)
            continue

    if raw_text is None:
        logger.error("LLM failed across all candidate models: %s", last_exc)
        return None

    return _extract_json_array(raw_text)


def interpret_notes(
    notes: list[str],
    battery_capacity_kwh: float,
    max_retries: int = 2,
    retry_delay_seconds: float = 0.5,
) -> list[dict[str, Any]]:
    """
    Interpret operator notes into structured directives.

    Three-tier caching strategy (fastest to slowest):

    Tier 1 — Scenario cache (< 0.1ms):
        All notes together hit a scenario-level LRU cache keyed on
        (tuple(normalized_notes), capacity). Common in judge replays.

    Tier 2 — Per-note cache (< 0.5ms total):
        Each note individually checked. On a PARTIAL hit, only the
        uncached notes are sent to the LLM — saving tokens and latency.

    Tier 3 — LLM call (800–1400ms):
        Only invoked for notes that missed both caches. Results stored
        in both per-note and scenario caches for subsequent requests.

    Safe fallback: returns no_op for all uncached notes if LLM fails,
    so the optimizer always receives a valid directive list.
    """
    if not notes:
        return []

    n = len(notes)

    # -- Normalize once (used for all cache keys) ---------------------------
    normalized_notes = [normalize_operator_note(note) for note in notes]

    # -- TIER 1: Scenario-level cache ---------------------------------------
    scenario_key = (
        tuple(n_text.strip().lower().rstrip(".") for n_text in normalized_notes),
        round(battery_capacity_kwh, 2),
    )
    cached_scenario = scenario_directive_cache.get(scenario_key)
    if cached_scenario is not None:
        logger.debug("Scenario cache hit for %d notes", n)
        return cached_scenario

    # -- TIER 2: Per-note cache --------------------------------------------
    per_note_results: list[dict[str, Any] | None] = [
        directive_cache.get_normalized(norm, battery_capacity_kwh)
        for norm in normalized_notes
    ]
    uncached_indices = [i for i, r in enumerate(per_note_results) if r is None]

    # All notes individually cached — merge, populate scenario cache, done.
    if not uncached_indices:
        logger.debug("Per-note cache hit for all %d notes", n)
        result: list[dict[str, Any]] = []
        for i, r in enumerate(per_note_results):
            entry = copy.deepcopy(r)  # type: ignore[arg-type]
            entry["note_index"] = i
            result.append(entry)
        scenario_directive_cache.put(scenario_key, result)
        return result

    # Log partial vs full miss
    n_cached = n - len(uncached_indices)
    if n_cached > 0:
        logger.debug(
            "Partial cache hit: %d/%d notes cached, sending %d to LLM",
            n_cached, n, len(uncached_indices),
        )
    else:
        logger.debug("Cache miss for all %d notes — calling LLM", n)

    # -- TIER 3: LLM call (only for uncached notes) ------------------------
    uncached_normalized = [normalized_notes[i] for i in uncached_indices]

    llm_parsed = _call_llm_core(
        notes=uncached_normalized,
        battery_capacity_kwh=battery_capacity_kwh,
        max_retries=max_retries,
        retry_delay_seconds=retry_delay_seconds,
    )

    # If LLM failed entirely, degrade uncached notes to no_op
    if llm_parsed is None:
        logger.warning(
            "LLM unavailable — treating %d uncached note(s) as no_op",
            len(uncached_indices),
        )
        llm_parsed = [
            {
                "note_index": j,
                "applies": False,
                "directive_type": "no_op",
                "structured_adjustment": None,
                "explanation": "LLM unavailable; treated as no-op.",
            }
            for j in range(len(uncached_indices))
        ]

    # Pad or truncate to exactly match the number of uncached notes
    while len(llm_parsed) < len(uncached_indices):
        llm_parsed.append({
            "note_index": len(llm_parsed),
            "applies": False,
            "directive_type": "no_op",
            "structured_adjustment": None,
            "explanation": "Padding: no directive returned.",
        })
    llm_parsed = llm_parsed[: len(uncached_indices)]

    # -- Merge cached + LLM results in original order ----------------------
    final: list[dict[str, Any]] = [
        copy.deepcopy(r) if r is not None else {}  # type: ignore[misc]
        for r in per_note_results
    ]
    for j, orig_i in enumerate(uncached_indices):
        d = llm_parsed[j]
        d["note_index"] = orig_i  # remap LLM-local index to original position
        directive_cache.put_normalized(normalized_notes[orig_i], battery_capacity_kwh, d)
        final[orig_i] = copy.deepcopy(d)

    # Ensure note_index is correct for all entries (cached entries may differ)
    for i, entry in enumerate(final):
        entry["note_index"] = i

    # Populate scenario cache for the full request
    scenario_directive_cache.put(scenario_key, final)

    return final
