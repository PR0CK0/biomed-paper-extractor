"""
Integration tests for the PMC paper analysis pipeline.

NETWORK: These tests make REAL HTTP calls to pmc.ncbi.nlm.nih.gov.
         An active internet connection is required.

MODELS:  Qwen2-VL (task1_figures.analyze_figures) and scispaCy
         (task2_ner.extract_entities) are fully mocked. No GPU,
         no model weights, and no HuggingFace downloads are needed.

Run only these tests:
    pytest -m integration tests/integration/

Skip them in offline / CI environments:
    pytest -m "not integration"
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from unittest.mock import patch

import pytest
import requests

from fetch_paper import fetch_paper

# ---------------------------------------------------------------------------
# Fixtures / constants
# ---------------------------------------------------------------------------

_PMC_ID = "PMC7614754"

_MOCK_FIGURES = [
    {
        "figure_id": "fig1",
        "figure_type": "bar_chart",
        "title": "test",
        "x_axis": {},
        "y_axis": {},
        "legend": [],
        "data_points": [],
        "notes": "",
    }
]

_MOCK_ENTITIES = [
    {
        "text": "imatinib",
        "label": "DRUG",
        "umls_cui": "C0935989",
        "start_char": 0,
        "end_char": 8,
    }
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_fetch_paper_returns_text_and_figures():
    """
    Real HTTP fetch of PMC7614754.

    Asserts that:
    - text is a non-empty string of at least 500 characters
    - figures is a list (empty is acceptable if PMC changes layout)
    - no exception is raised
    """
    text, figures = fetch_paper(_PMC_ID)

    assert isinstance(text, str), "text must be a string"
    assert len(text) > 500, f"text too short ({len(text)} chars); expected > 500"

    assert isinstance(figures, list), "figures must be a list"


@pytest.mark.integration
def test_fetch_paper_invalid_id_raises():
    """
    A PMC ID that does not match the PMC\\d+ pattern must raise ValueError
    before any network call is made.
    """
    with pytest.raises(ValueError, match="Invalid PMC ID"):
        fetch_paper("INVALID123")


@pytest.mark.integration
def test_fetch_paper_nonexistent_pmc_raises():
    """
    A syntactically valid but non-existent PMC ID must raise
    requests.HTTPError when the server returns a non-2xx response.
    """
    with pytest.raises(requests.HTTPError):
        fetch_paper("PMC9999999999")


@pytest.mark.integration
def test_full_pipeline_with_mocked_models():
    """
    Real fetch of PMC7614754 followed by mocked model inference.

    - analyze_figures is patched to return _MOCK_FIGURES (no GPU required)
    - extract_entities is patched to return _MOCK_ENTITIES (no scispaCy model)
    - Asserts both return the expected mocked shapes
    - Asserts JSON serialization succeeds for both results
    """
    text, figures = fetch_paper(_PMC_ID)

    with patch("task1_figures.analyze_figures", return_value=_MOCK_FIGURES) as mock_af, \
         patch("task2_ner.extract_entities", return_value=_MOCK_ENTITIES) as mock_ner:

        from task1_figures import analyze_figures
        from task2_ner import extract_entities

        figure_results = analyze_figures(figures)
        entity_results = extract_entities(text)

    # Shape assertions
    assert figure_results == _MOCK_FIGURES
    assert len(figure_results) == 1
    assert figure_results[0]["figure_id"] == "fig1"
    assert figure_results[0]["figure_type"] == "bar_chart"

    assert entity_results == _MOCK_ENTITIES
    assert len(entity_results) == 1
    assert entity_results[0]["text"] == "imatinib"
    assert entity_results[0]["label"] == "DRUG"
    assert entity_results[0]["umls_cui"] == "C0935989"

    # JSON serialization
    figures_json = json.dumps(figure_results)
    entities_json = json.dumps(entity_results)

    assert isinstance(figures_json, str)
    assert isinstance(entities_json, str)

    # Round-trip: parsed back must equal original
    assert json.loads(figures_json) == _MOCK_FIGURES
    assert json.loads(entities_json) == _MOCK_ENTITIES

    mock_af.assert_called_once_with(figures)
    mock_ner.assert_called_once_with(text)


@pytest.mark.integration
def test_fetch_paper_text_quality():
    """
    Real fetch of PMC7614754.

    Asserts that the extracted text:
    - exceeds 1000 characters (substantive article body was captured)
    - contains no raw HTML noise tags such as <script> or <nav> (the
      text extractor must have stripped all markup before returning)
    """
    text, _ = fetch_paper(_PMC_ID)

    assert len(text) > 1000, f"text too short ({len(text)} chars); expected > 1000"

    for forbidden_tag in ("<script", "<nav", "<header", "<footer", "<style"):
        assert forbidden_tag not in text.lower(), (
            f"Raw HTML tag '{forbidden_tag}' found in extracted text; "
            "markup stripping is not working correctly."
        )
