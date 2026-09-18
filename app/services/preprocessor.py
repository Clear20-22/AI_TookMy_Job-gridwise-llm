"""
Input preprocessing and in-memory caching for operator notes.

Normalizes colloquial time expressions, typography artifacts, and manages
an in-memory LRU cache to eliminate redundant LLM calls and minimize p95 latency.
"""

from __future__ import annotations

import copy
import re
import threading
from collections import OrderedDict
from typing import Any

# ---------------------------------------------------------------------------
# Text Normalization
# ---------------------------------------------------------------------------

_RE_CURLY_DOUBLE_QUOTES = re.compile(r"[\u201c\u201d\u201e\u201f\u00ab\u00bb]")
_RE_CURLY_SINGLE_QUOTES = re.compile(r"[\u2018\u2019\u201a\u201b]")
_RE_DASHES = re.compile(r"[\u2013\u2014\u2015]")
_RE_MULTIPLE_SPACES = re.compile(r"\s+")

# Colloquial time normalization
_RE_NOON = re.compile(r"\bnoon\b", re.IGNORECASE)
_RE_MIDNIGHT = re.compile(r"\bmidnight\b", re.IGNORECASE)
_RE_TIME_FORMAT = re.compile(r"\b(\d{1,2})(?::00)?\s*(am|pm)\b", re.IGNORECASE)


def normalize_operator_note(note: str) -> str:
    """
    Normalize operator note text for consistent parsing and caching.

    - Replaces typographic quotes with standard ASCII quotes.
    - Normalizes dashes (en-dash, em-dash) to hyphens.
    - Normalizes colloquial times ('noon' -> '12 PM', 'midnight' -> '12 AM').
    - Standardizes time representations ('12:00 PM' -> '12 PM', '2pm' -> '2 PM').
    - Collapses multiple whitespace characters.
    """
    if not note:
        return ""

    text = note.strip()
    text = _RE_CURLY_DOUBLE_QUOTES.sub('"', text)
    text = _RE_CURLY_SINGLE_QUOTES.sub("'", text)
    text = _RE_DASHES.sub("-", text)
    text = _RE_NOON.sub("12 PM", text)
    text = _RE_MIDNIGHT.sub("12 AM", text)

    # Standardize time strings like '2pm' or '02:00 PM' -> '2 PM'
    def _standardize_time(match: re.Match[str]) -> str:
        hour = int(match.group(1))
        meridiem = match.group(2).upper()
        return f"{hour} {meridiem}"

    text = _RE_TIME_FORMAT.sub(_standardize_time, text)
    text = _RE_MULTIPLE_SPACES.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Thread-safe LRU Directive Cache
# ---------------------------------------------------------------------------

class DirectiveLRUCache:
    """Thread-safe bounded LRU cache for parsed operator note directives."""

    def __init__(self, maxsize: int = 512) -> None:
        self.maxsize = maxsize
        self._cache: OrderedDict[tuple[str, float], dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, note: str, battery_capacity_kwh: float) -> dict[str, Any] | None:
        """Lookup cached directive for a normalized note and battery capacity."""
        key = (normalize_operator_note(note), round(float(battery_capacity_kwh), 2))
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self.hits += 1
                return copy.deepcopy(self._cache[key])
            self.misses += 1
            return None

    def put(self, note: str, battery_capacity_kwh: float, directive: dict[str, Any]) -> None:
        """Store a parsed directive in the cache."""
        key = (normalize_operator_note(note), round(float(battery_capacity_kwh), 2))
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = copy.deepcopy(directive)
            if len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        """Clear all cached entries and reset statistics."""
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0

    def stats(self) -> dict[str, int]:
        """Return cache performance statistics."""
        with self._lock:
            return {
                "size": len(self._cache),
                "maxsize": self.maxsize,
                "hits": self.hits,
                "misses": self.misses,
            }


# Global singleton instance
directive_cache = DirectiveLRUCache(maxsize=512)
