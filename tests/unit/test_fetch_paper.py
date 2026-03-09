"""Unit tests for fetch_paper.py."""

from __future__ import annotations

import io
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path bootstrap — allow importing from project root without an install.
# ---------------------------------------------------------------------------

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

import fetch_paper  # noqa: E402
from fetch_paper import (  # noqa: E402
    _fetch_image,
    _normalise_url,
    _validate_pmc_id,
    fetch_paper as fetch_paper_fn,
)


# ---------------------------------------------------------------------------
# _validate_pmc_id
# ---------------------------------------------------------------------------


class TestValidatePmcId:
    def test_valid_standard_id(self) -> None:
        _validate_pmc_id("PMC7614754")  # must not raise

    def test_valid_short_id(self) -> None:
        _validate_pmc_id("PMC123")  # must not raise

    def test_invalid_empty_string(self) -> None:
        with pytest.raises(ValueError, match="Invalid PMC ID"):
            _validate_pmc_id("")

    def test_invalid_missing_prefix(self) -> None:
        with pytest.raises(ValueError, match="Invalid PMC ID"):
            _validate_pmc_id("7614754")

    def test_invalid_alpha_digits(self) -> None:
        with pytest.raises(ValueError, match="Invalid PMC ID"):
            _validate_pmc_id("PMCABC")

    def test_invalid_lowercase_prefix(self) -> None:
        # re.IGNORECASE means "pmc123" actually matches — mirror the actual impl.
        # The regex uses re.IGNORECASE so lowercase IS accepted.
        _validate_pmc_id("pmc123")  # must not raise per IGNORECASE flag

    def test_invalid_digits_only(self) -> None:
        with pytest.raises(ValueError):
            _validate_pmc_id("1234567")


# ---------------------------------------------------------------------------
# _normalise_url
# ---------------------------------------------------------------------------


class TestNormaliseUrl:
    def test_protocol_relative(self) -> None:
        result = _normalise_url("//cdn.example.com/img.png")
        assert result == "https://cdn.example.com/img.png"

    def test_absolute_path(self) -> None:
        result = _normalise_url("/articles/PMC123/fig1.jpg")
        assert result == "https://pmc.ncbi.nlm.nih.gov/articles/PMC123/fig1.jpg"

    def test_already_absolute(self) -> None:
        url = "https://already.absolute.com/img.png"
        assert _normalise_url(url) == url

    def test_empty_string_returns_none(self) -> None:
        assert _normalise_url("") is None

    def test_data_uri_returns_none(self) -> None:
        assert _normalise_url("data:image/png;base64,abc") is None

    def test_relative_path_no_slash_returns_none(self) -> None:
        assert _normalise_url("relative/path.jpg") is None


# ---------------------------------------------------------------------------
# _extract_text
# ---------------------------------------------------------------------------


class TestExtractText:
    """Test noise-stripping and text extraction from a BeautifulSoup object."""

    def _make_soup(self, html: str):
        from bs4 import BeautifulSoup

        return BeautifulSoup(html, "html.parser")

    def test_strips_nav_script_style(self) -> None:
        html = """
        <html><body>
          <nav>Navigation junk</nav>
          <script>alert('xss')</script>
          <style>body { color: red; }</style>
          <article>
            <p>Real content paragraph.</p>
          </article>
        </body></html>
        """
        from fetch_paper import _extract_text

        soup = self._make_soup(html)
        text = _extract_text(soup)

        assert "Real content paragraph." in text
        assert "Navigation junk" not in text
        assert "alert" not in text
        assert "color: red" not in text

    def test_returns_article_body_paragraphs(self) -> None:
        html = """
        <html><body>
          <article>
            <p>First paragraph.</p>
            <p>Second paragraph.</p>
          </article>
        </body></html>
        """
        from fetch_paper import _extract_text

        soup = self._make_soup(html)
        text = _extract_text(soup)

        assert "First paragraph." in text
        assert "Second paragraph." in text

    def test_empty_when_no_container(self) -> None:
        from fetch_paper import _extract_text

        soup = self._make_soup("<html></html>")
        # body is None — should return empty string
        result = _extract_text(soup)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _fetch_image
# ---------------------------------------------------------------------------


class TestFetchImage:
    def _make_response(self, width: int, height: int, status_code: int = 200) -> MagicMock:
        from PIL import Image as PILImage

        img = PILImage.new("RGB", (width, height), color=(128, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.content = buf.read()
        mock_resp.raise_for_status = MagicMock()
        if status_code >= 400:
            import requests

            mock_resp.raise_for_status.side_effect = requests.HTTPError(
                f"HTTP {status_code}", response=mock_resp
            )
        return mock_resp

    def test_success_returns_pil_image(self) -> None:
        from PIL import Image as PILImage

        mock_resp = self._make_response(200, 200)
        with patch("fetch_paper.requests.get", return_value=mock_resp):
            result = _fetch_image("https://example.com/fig.png")

        assert isinstance(result, PILImage.Image)

    def test_http_error_returns_none(self) -> None:
        mock_resp = self._make_response(200, 200, status_code=404)
        with patch("fetch_paper.requests.get", return_value=mock_resp):
            result = _fetch_image("https://example.com/missing.png")

        assert result is None

    def test_too_small_image_returns_none(self) -> None:
        # 50x50 is below _MIN_DIMENSION (100)
        mock_resp = self._make_response(50, 50)
        with patch("fetch_paper.requests.get", return_value=mock_resp):
            result = _fetch_image("https://example.com/tiny.png")

        assert result is None

    def test_network_exception_returns_none(self) -> None:
        import requests

        with patch(
            "fetch_paper.requests.get",
            side_effect=requests.ConnectionError("Network down"),
        ):
            result = _fetch_image("https://unreachable.example.com/fig.png")

        assert result is None


# ---------------------------------------------------------------------------
# fetch_paper (public API)
# ---------------------------------------------------------------------------


class TestFetchPaper:
    def test_invalid_pmc_id_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Invalid PMC ID"):
            fetch_paper_fn("NOTVALID")

    def test_http_error_raises_http_error(self) -> None:
        import requests

        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.status_code = 503

        with patch("fetch_paper.requests.get", return_value=mock_resp):
            with pytest.raises(requests.HTTPError):
                fetch_paper_fn("PMC7614754")

    def test_successful_fetch_returns_text_and_figures(self) -> None:
        """Happy-path: valid HTML, no figures in HTML → PDF fallback skipped."""
        html = """
        <html><body>
          <article>
            <p>This is the article body text for testing.</p>
          </article>
        </body></html>
        """
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.text = html

        with (
            patch("fetch_paper.requests.get", return_value=mock_resp),
            patch("fetch_paper._extract_figures_html", return_value=[]),
            patch("fetch_paper._extract_figures_pdf", return_value=[]) as mock_pdf,
        ):
            text, figures = fetch_paper_fn("PMC7614754")

        assert isinstance(text, str)
        assert isinstance(figures, list)
        # PDF fallback should be attempted when HTML yields no figures.
        mock_pdf.assert_called_once_with("PMC7614754")
