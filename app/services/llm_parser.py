"""
LLM-based interpreter for operator notes → structured directives.

Uses Google Gemini (gemini-2.0-flash) by default.  Falls back to all-no_op
on any API failure so the optimizer can still produce a valid schedule.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt — contains all parsing rules from the problem statement
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert energy-systems engineer interpreting operator notes for a \
24-hour campus microgrid scheduling system.

## Your Task
For EACH operator note, produce exactly ONE JSON object that maps the note to \
one of six canonical directive types.  Return a JSON **array** with one entry \
per note, in the same order as the notes.

## Directive Types

| directive_type            | applies | structured_adjustment shape                           |
|---------------------------|---------|-------------------------------------------------------|
| solar_reduction           | true    | {"hours": [int, ...], "factor": float}                |
| minimum_battery_reserve   | true    | {"hours": [int, ...], "minimum_energy_kwh": float}    |
| no_charge_window          | true    | {"hours": [int, ...]}                                 |
| no_discharge_window       | true    | {"hours": [int, ...]}                                 |
| max_grid_window           | true    | {"hours": [int, ...], "max_grid_kwh": float}          |
| no_op                     | false   | null                                                  |

## Critical Parsing Rules
1. **Time windows are start-inclusive, end-exclusive.**
   - "1 PM to 3 PM" → hours [13, 14]
   - "from 6 PM until 9 PM" → hours [18, 19, 20]
   - "from 11 AM until 1 PM" → hours [11, 12]
2. **Hours array**: unique integers 0–23, sorted ascending.
3. **solar_reduction factor** = usable fraction REMAINING.
   - "80% reduction" → factor = 0.20
   - "drop to roughly 25%" → factor = 0.25
   - "reduced to about 30%" → factor = 0.30
4. **minimum_battery_reserve**: if stated as percentage, compute from battery capacity.
   - "keep at least 50% capacity" with capacity 200 kWh → minimum_energy_kwh = 100
5. **Distractor detection**: notes about sports events, cafeteria, next-week plans, \
or anything unrelated to TODAY's 24-hour energy operations → no_op.
6. For no_op: applies=false, structured_adjustment=null.
7. For all others: applies=true.

## Output Schema
Return ONLY a JSON array (no markdown, no commentary):
[
  {
    "note_index": 0,
    "applies": true,
    "directive_type": "solar_reduction",
    "structured_adjustment": {"hours": [12, 13], "factor": 0.25},
    "explanation": "Brief reason"
  }
]
"""


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
        # Remove opening fence (with optional language tag)
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

    # Try to find array substring
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
# Main interpreter
# ---------------------------------------------------------------------------

def interpret_notes(
    notes: list[str],
    battery_capacity_kwh: float,
) -> list[dict[str, Any]]:
    """
    Use Gemini to interpret operator notes into structured directives.

    Parameters
    ----------
    notes : list[str]
        1–3 operator notes from the request.
    battery_capacity_kwh : float
        Battery capacity, included in the prompt so the LLM can convert
        percentage-based reserves to absolute values.

    Returns
    -------
    list[dict]
        One directive dict per note (may be unvalidated; guardrails come next).
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return _fallback_no_ops(notes, "GEMINI_API_KEY not set")

    try:
        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=api_key)
        model_name = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
        model = genai.GenerativeModel(
            model_name,
            system_instruction=SYSTEM_PROMPT,
        )

        user_message = (
            f"Battery capacity: {battery_capacity_kwh} kWh\n\n"
            "Operator notes:\n"
        )
        for i, note in enumerate(notes):
            user_message += f"  Note {i}: {note}\n"

        response = model.generate_content(
            user_message,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.0,
            ),
        )

        raw_text = response.text
        logger.debug("LLM raw response: %s", raw_text)

        parsed = _extract_json_array(raw_text)
        if parsed is None:
            return _fallback_no_ops(notes, "LLM returned unparseable output")

        # Ensure we have the right count
        if len(parsed) != len(notes):
            logger.warning(
                "LLM returned %d directives for %d notes; padding/truncating",
                len(parsed),
                len(notes),
            )
            # Pad with no_ops or truncate
            while len(parsed) < len(notes):
                parsed.append({
                    "note_index": len(parsed),
                    "applies": False,
                    "directive_type": "no_op",
                    "structured_adjustment": None,
                    "explanation": "Padding: LLM did not return a directive for this note.",
                })
            parsed = parsed[: len(notes)]

        # Ensure note_index is set correctly
        for i, d in enumerate(parsed):
            d["note_index"] = i

        return parsed

    except ImportError:
        return _fallback_no_ops(notes, "google-generativeai package not installed")
    except Exception as exc:
        logger.exception("LLM interpretation failed")
        return _fallback_no_ops(notes, str(exc))
