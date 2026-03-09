"""Unit tests for task2_ner.py."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

import task2_ner  # noqa: E402
from task2_ner import extract_entities  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entity(
    text: str,
    label: str,
    start: int,
    end: int,
    kb_ents: list | None = None,
) -> MagicMock:
    """Build a mock spaCy entity span."""
    ent = MagicMock()
    ent.text = text
    ent.label_ = label
    ent.start_char = start
    ent.end_char = end
    # kb_ents is accessed via ent._.kb_ents
    ent._.kb_ents = kb_ents if kb_ents is not None else []
    return ent


def _make_nlp(entities: list[MagicMock], max_length: int = 1_000_000) -> MagicMock:
    """Build a mock spaCy nlp callable that returns a doc with given entities."""
    mock_doc = MagicMock()
    mock_doc.ents = entities

    mock_nlp = MagicMock()
    mock_nlp.max_length = max_length
    mock_nlp.return_value = mock_doc
    return mock_nlp


# ---------------------------------------------------------------------------
# Short-circuit cases (no model load)
# ---------------------------------------------------------------------------


class TestExtractEntitiesShortCircuit:
    def test_empty_string_returns_empty_list(self) -> None:
        result = extract_entities("")
        assert result == []

    def test_string_shorter_than_ten_chars_returns_empty_list(self) -> None:
        result = extract_entities("hi")
        assert result == []

    def test_string_of_exactly_nine_chars_returns_empty_list(self) -> None:
        result = extract_entities("123456789")
        assert result == []

    def test_string_of_exactly_ten_chars_calls_model(self) -> None:
        mock_nlp = _make_nlp([])
        with patch("task2_ner._get_nlp", return_value=mock_nlp):
            result = extract_entities("1234567890")
        assert result == []
        mock_nlp.assert_called_once()


# ---------------------------------------------------------------------------
# Entity dict shape
# ---------------------------------------------------------------------------


class TestEntityShape:
    def test_entity_dict_has_required_keys(self) -> None:
        ent = _make_entity("COVID-19", "DISEASE", 5, 13, kb_ents=[("C5203670", 1.0)])
        mock_nlp = _make_nlp([ent])

        with patch("task2_ner._get_nlp", return_value=mock_nlp):
            results = extract_entities("About COVID-19 treatment options here")

        assert len(results) == 1
        entity = results[0]
        assert set(entity.keys()) == {"text", "label", "kb_id", "umls_cui", "start_char", "end_char"}

    def test_entity_values_correct(self) -> None:
        ent = _make_entity("COVID-19", "DISEASE", 6, 14, kb_ents=[("C5203670", 0.99)])
        mock_nlp = _make_nlp([ent])

        with patch("task2_ner._get_nlp", return_value=mock_nlp):
            results = extract_entities("About COVID-19 and its treatment")

        entity = results[0]
        assert entity["text"] == "COVID-19"
        assert entity["label"] == "DISEASE"
        assert entity["umls_cui"] == "C5203670"
        assert entity["start_char"] == 6
        assert entity["end_char"] == 14

    def test_results_sorted_by_start_char(self) -> None:
        ents = [
            _make_entity("aspirin", "CHEMICAL", 20, 27),
            _make_entity("COVID-19", "DISEASE", 0, 8),
        ]
        mock_nlp = _make_nlp(ents)

        with patch("task2_ner._get_nlp", return_value=mock_nlp):
            results = extract_entities("COVID-19 patients received aspirin therapy")

        assert results[0]["start_char"] < results[1]["start_char"]


# ---------------------------------------------------------------------------
# UMLS CUI handling
# ---------------------------------------------------------------------------


class TestUmlsCui:
    def test_none_when_kb_ents_empty(self) -> None:
        ent = _make_entity("aspirin", "CHEMICAL", 0, 7, kb_ents=[])
        mock_nlp = _make_nlp([ent])

        with patch("task2_ner._get_nlp", return_value=mock_nlp):
            results = extract_entities("aspirin therapy is common treatment")

        assert results[0]["umls_cui"] is None

    def test_first_kb_ent_used_as_cui(self) -> None:
        kb_ents = [("C0004057", 0.95), ("C0001647", 0.80)]
        ent = _make_entity("aspirin", "CHEMICAL", 0, 7, kb_ents=kb_ents)
        mock_nlp = _make_nlp([ent])

        with patch("task2_ner._get_nlp", return_value=mock_nlp):
            results = extract_entities("aspirin therapy is common treatment")

        assert results[0]["umls_cui"] == "C0004057"

    def test_graceful_when_kb_ents_attribute_raises(self) -> None:
        ent = MagicMock()
        ent.text = "aspirin"
        ent.label_ = "CHEMICAL"
        ent.start_char = 0
        ent.end_char = 7
        # Accessing _.kb_ents raises an AttributeError.
        type(ent._).kb_ents = property(
            lambda self: (_ for _ in ()).throw(AttributeError("no kb_ents"))
        )

        mock_nlp = _make_nlp([ent])

        # Should not raise; umls_cui should be None.
        with patch("task2_ner._get_nlp", return_value=mock_nlp):
            results = extract_entities("aspirin therapy is common treatment")

        assert results[0]["umls_cui"] is None


# ---------------------------------------------------------------------------
# Chunking behaviour
# ---------------------------------------------------------------------------


class TestChunking:
    def test_text_within_max_length_not_chunked(self) -> None:
        """When text fits in max_length, _get_nlp called once (no chunking loop)."""
        long_text = "This is a sentence about COVID-19 therapy. " * 5
        ent = _make_entity("COVID-19", "DISEASE", 10, 18)
        mock_nlp = _make_nlp([ent], max_length=100_000)

        with patch("task2_ner._get_nlp", return_value=mock_nlp):
            results = extract_entities(long_text)

        # nlp called once (single-chunk path)
        mock_nlp.assert_called_once_with(long_text)

    def test_text_exceeding_max_length_is_chunked(self) -> None:
        """When text > max_length the chunking loop splits into multiple nlp calls."""
        # max_length = 50, text = 120 chars → multiple chunks.
        max_length = 50
        overlap = 200  # overlap > chunk for tiny test; we just want >1 nlp call

        # Build text of 120 chars.
        base = "ABCDE " * 20  # 120 chars
        text = base[:120]

        ent = _make_entity("ABCDE", "ENTITY", 0, 5)
        mock_nlp = _make_nlp([ent], max_length=max_length)

        with patch("task2_ner._get_nlp", return_value=mock_nlp):
            results = extract_entities(text)

        # nlp must have been called more than once
        assert mock_nlp.call_count > 1

    def test_duplicate_entities_deduplicated(self) -> None:
        """
        When overlapping chunks produce the same (text, start_char) entity twice,
        the result list must contain it only once.
        """
        max_length = 60
        # Craft text so first chunk ends mid-sentence and second chunk overlaps,
        # producing the same entity from both chunks.
        text = "The drug aspirin is effective. " * 4  # 124 chars

        # Entity sitting in the overlap zone: start_char=10 (inside first chunk).
        ent = _make_entity("aspirin", "CHEMICAL", 10, 17)
        mock_nlp = _make_nlp([ent], max_length=max_length)

        with patch("task2_ner._get_nlp", return_value=mock_nlp):
            results = extract_entities(text)

        # Deduplicate check: no two entries share the same (text, start_char).
        keys = [(r["text"], r["start_char"]) for r in results]
        assert len(keys) == len(set(keys))

    def test_char_offsets_adjusted_per_chunk(self) -> None:
        """
        Entities in the second chunk must have their char positions adjusted
        by the chunk's start offset.
        """
        max_length = 50
        text = "A" * 50 + " disease is studied extensively here"  # >50 chars

        # Entity at position 2 within the second chunk → absolute = 50 + overlap_adj + 2.
        # We will verify the entity's start_char equals chunk_start + ent.start_char.
        chunk2_start = max_length - 200  # overlap=200, but clamped by while loop
        # Simpler: patch _process_chunk directly to inspect call args.

        call_offsets: list[int] = []

        original_process_chunk = task2_ner._process_chunk

        def recording_process_chunk(nlp, chunk, char_offset):
            call_offsets.append(char_offset)
            return []

        with (
            patch("task2_ner._get_nlp", return_value=MagicMock(max_length=max_length)),
            patch("task2_ner._process_chunk", side_effect=recording_process_chunk),
        ):
            extract_entities(text)

        # First chunk always starts at offset 0.
        assert call_offsets[0] == 0
        # Subsequent chunks must have increasing offsets.
        assert all(
            call_offsets[i] < call_offsets[i + 1]
            for i in range(len(call_offsets) - 1)
        )
