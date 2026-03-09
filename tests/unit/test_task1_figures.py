"""Unit tests for task1_figures.py."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

# ---------------------------------------------------------------------------
# Stub out heavy optional deps before the module is imported.
# torch is a real dep but we stub cuda/mps on each test as needed.
# ---------------------------------------------------------------------------

import task1_figures  # noqa: E402
from task1_figures import _get_device, _parse_json, analyze_figures  # noqa: E402


# ---------------------------------------------------------------------------
# _get_device
# ---------------------------------------------------------------------------


class TestGetDevice:
    def test_returns_cuda_when_cuda_available(self) -> None:
        with (
            patch("task1_figures.torch.cuda.is_available", return_value=True),
            patch("task1_figures.torch.backends.mps.is_available", return_value=False),
        ):
            assert _get_device() == "cuda"

    def test_returns_mps_when_only_mps_available(self) -> None:
        with (
            patch("task1_figures.torch.cuda.is_available", return_value=False),
            patch("task1_figures.torch.backends.mps.is_available", return_value=True),
        ):
            assert _get_device() == "mps"

    def test_returns_cpu_when_no_accelerator(self) -> None:
        with (
            patch("task1_figures.torch.cuda.is_available", return_value=False),
            patch("task1_figures.torch.backends.mps.is_available", return_value=False),
        ):
            assert _get_device() == "cpu"

    def test_cuda_takes_priority_over_mps(self) -> None:
        with (
            patch("task1_figures.torch.cuda.is_available", return_value=True),
            patch("task1_figures.torch.backends.mps.is_available", return_value=True),
        ):
            assert _get_device() == "cuda"


# ---------------------------------------------------------------------------
# _parse_json
# ---------------------------------------------------------------------------


class TestParseJson:
    def test_valid_json_string(self) -> None:
        raw = '{"figure_type": "bar", "title": "Test"}'
        result = _parse_json(raw)
        assert result == {"figure_type": "bar", "title": "Test"}

    def test_json_in_markdown_fences(self) -> None:
        raw = '```json\n{"figure_type": "line", "title": "Trend"}\n```'
        result = _parse_json(raw)
        assert result["figure_type"] == "line"
        assert result["title"] == "Trend"

    def test_json_with_surrounding_text(self) -> None:
        raw = 'Here is the analysis:\n{"figure_type": "scatter", "notes": "none"}\nEnd.'
        result = _parse_json(raw)
        assert result["figure_type"] == "scatter"
        assert result["notes"] == "none"

    def test_completely_invalid_returns_error_dict(self) -> None:
        raw = "This is not JSON at all, just prose."
        result = _parse_json(raw)
        assert "error" in result
        assert "raw" in result
        assert result["raw"] == raw

    def test_nested_json_preserved(self) -> None:
        raw = '{"x_axis": {"label": "Time", "units": "s", "ticks": [0, 1, 2]}}'
        result = _parse_json(raw)
        assert result["x_axis"]["label"] == "Time"
        assert result["x_axis"]["ticks"] == [0, 1, 2]

    def test_markdown_fence_without_language_tag(self) -> None:
        raw = '```\n{"figure_type": "table"}\n```'
        result = _parse_json(raw)
        assert result["figure_type"] == "table"


# ---------------------------------------------------------------------------
# analyze_figures
# ---------------------------------------------------------------------------


class TestAnalyzeFigures:
    def test_empty_list_returns_empty_list(self) -> None:
        # Even without mocking _load_model, an empty input list must short-circuit
        # after model load.  We patch _load_model to avoid real weight downloads.
        with patch("task1_figures._load_model") as mock_load:
            mock_load.return_value = (MagicMock(), MagicMock())
            with patch("task1_figures._get_device", return_value="cpu"):
                result = analyze_figures([])
        assert result == []

    def test_figure_id_injected(self) -> None:
        from PIL import Image as PILImage

        fake_image = PILImage.new("RGB", (100, 100))
        mock_parsed = {"figure_type": "bar", "title": "Chart"}

        with (
            patch("task1_figures._load_model") as mock_load,
            patch("task1_figures._get_device", return_value="cpu"),
            patch("task1_figures._analyze_single", return_value=mock_parsed),
        ):
            mock_load.return_value = (MagicMock(), MagicMock())
            results = analyze_figures([fake_image])

        assert len(results) == 1
        assert results[0]["figure_id"] == "fig1"
        assert results[0]["figure_type"] == "bar"

    def test_figure_ids_are_sequential(self) -> None:
        from PIL import Image as PILImage

        images = [PILImage.new("RGB", (100, 100)) for _ in range(3)]
        mock_parsed = {"figure_type": "other"}

        with (
            patch("task1_figures._load_model") as mock_load,
            patch("task1_figures._get_device", return_value="cpu"),
            patch("task1_figures._analyze_single", return_value=mock_parsed),
        ):
            mock_load.return_value = (MagicMock(), MagicMock())
            results = analyze_figures(images)

        ids = [r["figure_id"] for r in results]
        assert ids == ["fig1", "fig2", "fig3"]

    def test_per_figure_exception_yields_error_dict(self) -> None:
        from PIL import Image as PILImage

        images = [PILImage.new("RGB", (100, 100)) for _ in range(3)]

        call_count = 0

        def _side_effect(img, model, processor, device):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Inference failed on figure 2")
            return {"figure_type": "bar"}

        with (
            patch("task1_figures._load_model") as mock_load,
            patch("task1_figures._get_device", return_value="cpu"),
            patch("task1_figures._analyze_single", side_effect=_side_effect),
        ):
            mock_load.return_value = (MagicMock(), MagicMock())
            results = analyze_figures(images)

        assert len(results) == 3

        # Figure 1 and 3 succeed.
        assert results[0]["figure_type"] == "bar"
        assert results[0]["figure_id"] == "fig1"

        # Figure 2 fails but does NOT abort the loop.
        assert "error" in results[1]
        assert results[1]["figure_id"] == "fig2"
        assert "Inference failed" in results[1]["error"]

        assert results[2]["figure_type"] == "bar"
        assert results[2]["figure_id"] == "fig3"

    def test_error_dict_contains_figure_id(self) -> None:
        from PIL import Image as PILImage

        fake_image = PILImage.new("RGB", (100, 100))

        with (
            patch("task1_figures._load_model") as mock_load,
            patch("task1_figures._get_device", return_value="cpu"),
            patch("task1_figures._analyze_single", side_effect=ValueError("bad input")),
        ):
            mock_load.return_value = (MagicMock(), MagicMock())
            results = analyze_figures([fake_image])

        assert results[0]["figure_id"] == "fig1"
        assert "error" in results[0]
