from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are an expert energy-systems engineer interpreting operator notes for a
24-hour campus microgrid scheduling system.

Your job is ONLY to interpret the operator notes.

For EACH operator note, produce exactly ONE structured directive in the same
order as the notes.

Supported directive types:

1. solar_reduction
   structured_adjustment:
   {
     "hours": [integer, ...],
     "factor": number
   }

2. minimum_battery_reserve
   structured_adjustment:
   {
     "hours": [integer, ...],
     "minimum_energy_kwh": number
   }

3. no_charge_window
   structured_adjustment:
   {
     "hours": [integer, ...]
   }

4. no_discharge_window
   structured_adjustment:
   {
     "hours": [integer, ...]
   }

5. max_grid_window
   structured_adjustment:
   {
     "hours": [integer, ...],
     "max_grid_kwh": number
   }

6. no_op
   structured_adjustment: null


CRITICAL RULES

1. Time windows are START-INCLUSIVE and END-EXCLUSIVE.

Examples:

"1 PM to 3 PM"
-> hours [13, 14]

"from 6 PM until 9 PM"
-> hours [18, 19, 20]

"from 11 AM until 1 PM"
-> hours [11, 12]


2. Hours must represent whole-hour intervals using integers 0 through 23.

3. solar_reduction factor means the usable fraction REMAINING.

Examples:

"80% reduction"
-> factor 0.20

"drop to 25%"
-> factor 0.25

"reduced to about 30%"
-> factor 0.30


4. minimum_battery_reserve:

If the reserve is expressed as a percentage of battery capacity, convert the
percentage into kWh using the battery capacity supplied in the user message.

Example:

Battery capacity = 200 kWh
"keep at least 50% capacity"
-> minimum_energy_kwh = 100


5. Distractors:

Notes unrelated to the CURRENT 24-hour energy schedule must be classified as
no_op.

Examples include:
- cafeteria menu changes
- sports registration changes
- administrative announcements
- unrelated future events


6. no_op rules:

applies = false
directive_type = "no_op"
structured_adjustment = null


7. Every other directive:

applies = true


8. Never invent:
- demand
- solar values
- tariff values
- battery limits
- unsupported directive types


Return only data matching the required structured output schema.
"""


# ---------------------------------------------------------------------------
# Custom error
# ---------------------------------------------------------------------------

class LLMInterpretationError(RuntimeError):
    """Raised when no LLM provider/model can interpret the notes."""


# ---------------------------------------------------------------------------
# JSON schemas
# ---------------------------------------------------------------------------

HOURS_SCHEMA = {
    "type": "array",
    "items": {
        "type": "integer",
        "minimum": 0,
        "maximum": 23,
    },
}


def _base_properties(
    directive_type: str,
    applies: bool,
    structured_adjustment: dict[str, Any],
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "note_index": {
                "type": "integer",
                "minimum": 0,
            },
            "applies": {
                "type": "boolean",
                "enum": [applies],
            },
            "directive_type": {
                "type": "string",
                "enum": [directive_type],
            },
            "structured_adjustment": structured_adjustment,
            "explanation": {
                "type": "string",
            },
        },
        "required": [
            "note_index",
            "applies",
            "directive_type",
            "structured_adjustment",
            "explanation",
        ],
        "additionalProperties": False,
    }


SOLAR_REDUCTION_SCHEMA = _base_properties(
    "solar_reduction",
    True,
    {
        "type": "object",
        "properties": {
            "hours": HOURS_SCHEMA,
            "factor": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
        },
        "required": ["hours", "factor"],
        "additionalProperties": False,
    },
)


MINIMUM_BATTERY_RESERVE_SCHEMA = _base_properties(
    "minimum_battery_reserve",
    True,
    {
        "type": "object",
        "properties": {
            "hours": HOURS_SCHEMA,
            "minimum_energy_kwh": {
                "type": "number",
                "minimum": 0,
            },
        },
        "required": [
            "hours",
            "minimum_energy_kwh",
        ],
        "additionalProperties": False,
    },
)


NO_CHARGE_SCHEMA = _base_properties(
    "no_charge_window",
    True,
    {
        "type": "object",
        "properties": {
            "hours": HOURS_SCHEMA,
        },
        "required": ["hours"],
        "additionalProperties": False,
    },
)


NO_DISCHARGE_SCHEMA = _base_properties(
    "no_discharge_window",
    True,
    {
        "type": "object",
        "properties": {
            "hours": HOURS_SCHEMA,
        },
        "required": ["hours"],
        "additionalProperties": False,
    },
)


MAX_GRID_SCHEMA = _base_properties(
    "max_grid_window",
    True,
    {
        "type": "object",
        "properties": {
            "hours": HOURS_SCHEMA,
            "max_grid_kwh": {
                "type": "number",
                "minimum": 0,
            },
        },
        "required": [
            "hours",
            "max_grid_kwh",
        ],
        "additionalProperties": False,
    },
)


NO_OP_SCHEMA = _base_properties(
    "no_op",
    False,
    {
        "type": "null",
    },
)


DIRECTIVE_ITEM_SCHEMA = {
    "anyOf": [
        SOLAR_REDUCTION_SCHEMA,
        MINIMUM_BATTERY_RESERVE_SCHEMA,
        NO_CHARGE_SCHEMA,
        NO_DISCHARGE_SCHEMA,
        MAX_GRID_SCHEMA,
        NO_OP_SCHEMA,
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

    raw_text = response.choices[0].message.content

    if not raw_text:
        raise LLMInterpretationError(
            f"Groq model {model} returned an empty response."
        )

    logger.debug(
        "Groq raw response from %s: %s",
        model,
        raw_text,
    )

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise LLMInterpretationError(
            f"Groq model {model} returned invalid JSON."
        ) from exc

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
# Main interpreter
# ---------------------------------------------------------------------------

def interpret_notes(
    notes: list[str],
    battery_capacity_kwh: float,
) -> list[dict[str, Any]]:
    """
    Interpret GridWise operator notes using Groq.

    Parameters
    ----------
    notes:
        1-3 natural-language operator notes.

    battery_capacity_kwh:
        Battery capacity used for percentage-based reserve calculations.

    Returns
    -------
    list[dict]:
        Exactly one structured directive per note.

    Raises
    ------
    LLMInterpretationError:
        If the LLM cannot produce a usable interpretation.
    """

    api_key = os.environ.get("GROQ_API_KEY", "").strip()

    if not api_key:
        raise LLMInterpretationError(
            "GROQ_API_KEY is not configured."
        )

    primary_model = os.environ.get(
        "GROQ_MODEL",
        "openai/gpt-oss-20b",
    ).strip()

    # Optional backup model.
    backup_model = os.environ.get(
        "GROQ_BACKUP_MODEL",
        "",
    ).strip()

    models = [primary_model]

    if backup_model and backup_model != primary_model:
        models.append(backup_model)

    try:
        from groq import Groq

    except ImportError as exc:
        raise LLMInterpretationError(
            "groq package is not installed. Run: pip install groq"
        ) from exc

    # Groq already retries transient connection/429/5xx errors.
    client = Groq(
        api_key=api_key,
        timeout=20.0,
        max_retries=2,
    )

    last_error: Exception | None = None

    for model in models:
        try:
            return _call_groq(
                client=client,
                model=model,
                notes=notes,
                battery_capacity_kwh=battery_capacity_kwh,
            )

        except Exception as exc:
            last_error = exc

            logger.exception(
                "Groq interpretation failed using model %s",
                model,
            )

            if len(models) > 1:
                logger.warning(
                    "Trying next Groq model after failure of %s",
                    model,
                )

    # IMPORTANT:
    # Do NOT silently convert all notes into no_op.
    # A 500/error is better than returning a logically false schedule.
    raise LLMInterpretationError(
        "LLM directive interpretation failed."
    ) from last_error