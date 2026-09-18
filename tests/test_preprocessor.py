"""
Unit tests for input preprocessing and LRU directive cache.
"""

from __future__ import annotations

import copy
import pytest

from app.services.preprocessor import (
    DirectiveLRUCache,
    directive_cache,
    normalize_operator_note,
)
from app.services.llm_parser import interpret_notes


class TestTextNormalization:
    """Test text canonicalization and normalization."""

    def test_normalize_empty_and_whitespace(self):
        assert normalize_operator_note("") == ""
        assert normalize_operator_note("   ") == ""

    def test_normalize_quotes(self):
        raw = "Facilities said “wash panels” and ‘inspect battery’."
        expected = "Facilities said \"wash panels\" and 'inspect battery'."
        assert normalize_operator_note(raw) == expected

    def test_normalize_dashes(self):
        raw = "Solar inspection – inverter repair — feeder limit."
        expected = "Solar inspection - inverter repair - feeder limit."
        assert normalize_operator_note(raw) == expected

    def test_normalize_noon_and_midnight(self):
        raw = "Washing from noon until 2 PM, inspection at midnight."
        expected = "Washing from 12 PM until 2 PM, inspection at 12 AM."
        assert normalize_operator_note(raw) == expected

    def test_normalize_time_formatting(self):
        raw = "From 2pm until 04:00 PM maintenance window."
        expected = "From 2 PM until 4 PM maintenance window."
        assert normalize_operator_note(raw) == expected


class TestDirectiveLRUCache:
    """Test thread-safe LRU cache operations."""

    def test_cache_put_and_get(self):
        cache = DirectiveLRUCache(maxsize=10)
        note = "Wash solar panels from 12 PM until 2 PM."
        directive = {
            "note_index": 0,
            "applies": True,
            "directive_type": "solar_reduction",
            "structured_adjustment": {"hours": [12, 13], "factor": 0.25},
            "explanation": "Panel cleaning",
        }

        cache.put(note, 200.0, directive)
        retrieved = cache.get(note, 200.0)

        assert retrieved is not None
        assert retrieved["directive_type"] == "solar_reduction"
        assert retrieved["structured_adjustment"]["factor"] == 0.25

    def test_cache_deepcopy_isolation(self):
        """Mutating the retrieved object must not mutate the cached state."""
        cache = DirectiveLRUCache(maxsize=10)
        note = "Maintenance window."
        directive = {"directive_type": "no_charge_window", "hours": [2, 3]}

        cache.put(note, 200.0, directive)
        retrieved = cache.get(note, 200.0)
        assert retrieved is not None
        retrieved["hours"].append(99)

        fresh = cache.get(note, 200.0)
        assert fresh is not None
        assert 99 not in fresh["hours"]

    def test_cache_lru_eviction(self):
        """Cache should evict least recently used entry when maxsize is exceeded."""
        cache = DirectiveLRUCache(maxsize=2)
        cache.put("note 1", 200.0, {"val": 1})
        cache.put("note 2", 200.0, {"val": 2})

        # Access note 1 to make note 2 the LRU
        _ = cache.get("note 1", 200.0)

        # Insert note 3 -> note 2 should be evicted
        cache.put("note 3", 200.0, {"val": 3})

        assert cache.get("note 1", 200.0) is not None
        assert cache.get("note 2", 200.0) is None
        assert cache.get("note 3", 200.0) is not None

    def test_cache_stats_and_clear(self):
        cache = DirectiveLRUCache(maxsize=10)
        cache.put("test note", 100.0, {"val": 1})

        _ = cache.get("test note", 100.0)  # hit
        _ = cache.get("unknown", 100.0)    # miss

        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1

        cache.clear()
        assert cache.stats()["size"] == 0
        assert cache.stats()["hits"] == 0


class TestCacheIntegrationWithParser:
    """Test that interpret_notes returns cached results without invoking the LLM."""

    def test_interpret_notes_cache_hit(self):
        # Pre-populate global cache
        note = "Cached test note"
        capacity = 250.0
        cached_entry = {
            "note_index": 0,
            "applies": True,
            "directive_type": "no_discharge_window",
            "structured_adjustment": {"hours": [18, 19]},
            "explanation": "Protection test",
        }
        directive_cache.put(note, capacity, cached_entry)

        # Calling interpret_notes with this note should hit cache and succeed
        # even without making any network calls
        results = interpret_notes([note], battery_capacity_kwh=capacity)
        assert len(results) == 1
        assert results[0]["directive_type"] == "no_discharge_window"
        assert results[0]["structured_adjustment"]["hours"] == [18, 19]
