"""
fetch_paper.py

PMC (PubMed Central) paper fetcher for the pdf-extractor Gradio app.
Compatible with Apple Silicon (MPS) and HuggingFace Spaces.

Public API
----------
fetch_paper(pmc_id: str, use_jats: bool = False) -> tuple[str, list[PIL.Image.Image], list[str], dict, bytes | None, str]
fetch_url(url: str, use_jats: bool = False) -> tuple[str, list[PIL.Image.Image], list[str], dict, bytes | None, str]
render_pdf_pages(pdf_bytes: bytes, max_pages: int = 20) -> list[PIL.Image.Image]
"""

from __future__ import annotations

import base64
import io
import re
import sys
import xml.etree.ElementTree as ET
from typing import Optional
from urllib.parse import urljoin, urlparse

import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from PIL import Image

import requests
from curl_cffi.requests import Session as _CurlSession

_BASE_URL = "https://pmc.ncbi.nlm.nih.gov"

# curl_cffi impersonates Chrome's TLS fingerprint, bypassing NCBI's bot
# detection that blocks Python's default urllib3/OpenSSL stack on Windows.
# Works identically on Mac/Linux/HF where NCBI doesn't block plain requests.
_session = _CurlSession(impersonate="chrome120")


def _get(url: str, timeout: int) -> requests.Response:
    """GET via curl_cffi Chrome-impersonation session."""
    return _session.get(url, timeout=timeout)
_PAGE_TIMEOUT = 30
_IMG_TIMEOUT = 15
_MIN_DIMENSION = 100


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_pmc_id(pmc_id: str) -> None:
    """Raise ValueError if pmc_id is not in the expected PMC\\d+ format."""
    if not re.fullmatch(r"PMC\d+", pmc_id, re.IGNORECASE):
        raise ValueError(
            f"Invalid PMC ID '{pmc_id}'. "
            "Must start with 'PMC' followed by one or more digits (e.g. PMC7614754)."
        )


# ---------------------------------------------------------------------------
# URL normalisation
# ---------------------------------------------------------------------------


def _normalise_url(src: str) -> Optional[str]:
    """
    Resolve a raw src attribute into a fully-qualified HTTPS URL.

    Handles:
    - Protocol-relative  //example.com/img.png  →  https://example.com/img.png
    - Absolute path      /articles/…/fig1.jpg   →  https://pmc.ncbi.nlm.nih.gov/…
    - Already absolute   https://…              →  unchanged
    - Data URIs / empty  →  None (caller should skip)
    """
    if not src or src.startswith("data:"):
        return None
    if src.startswith("//"):
        return "https:" + src
    if src.startswith("/"):
        return _BASE_URL + src
    if src.startswith("http"):
        return src
    # Relative URL without leading slash — skip; too ambiguous without a base path.
    return None


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def _fetch_image(url: str) -> Optional[Image.Image]:
    """
    Download a single image from *url* and return it as an RGB PIL.Image.

    Returns None if the download fails or the image is too small to be a
    real figure (below _MIN_DIMENSION in either dimension).
    """
    try:
        resp = _get(url, timeout=_IMG_TIMEOUT)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        if img.width < _MIN_DIMENSION or img.height < _MIN_DIMENSION:
            return None
        return img
    except Exception:
        # Single-figure failures are non-fatal; caller decides what to do.
        return None


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------


def _extract_text(soup: BeautifulSoup) -> str:
    """
    Pull clean article body text from a PMC HTML page.

    Strategy:
    1. Try the canonical article body container (#mc_articles_content or
       .article-body / .full-report — PMC has used several class names).
    2. Fall back to <main> or <article> tags.
    3. Strip all <nav>, <header>, <footer>, <aside>, <script>, <style>
       nodes before collecting paragraph text.
    """
    # Remove noise nodes in-place.
    for tag in soup.find_all(["nav", "header", "footer", "aside", "script", "style"]):
        tag.decompose()

    # Candidate containers in preference order.
    container = (
        soup.find(id="mc_articles_content")
        or soup.find(class_=re.compile(r"article[_-]?(body|content|text)", re.I))
        or soup.find("article")
        or soup.find("main")
        or soup.body
    )

    if container is None:
        return ""

    paragraphs: list[str] = []
    for element in container.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"]):
        text = element.get_text(separator=" ", strip=True)
        if text:
            paragraphs.append(text)

    return "\n\n".join(paragraphs)


# ---------------------------------------------------------------------------
# Figure extraction — primary (HTML)
# ---------------------------------------------------------------------------


def _extract_caption_from_figure(figure_tag) -> str:
    """
    Extract a display caption from a <figure> (or JATS <fig>) element.

    Strategy (in preference order):
    1. Combine <label> text (e.g. "Figure 1.") with <fig-caption>, <caption>,
       or <figcaption> text — covers both JATS XML and standard HTML.
    2. Fall back to any remaining text inside the figure tag after stripping
       the image element itself.

    Returns an empty string if nothing useful is found.
    """
    # --- label (e.g. "Figure 1.", "Fig. 2") ---
    label_tag = figure_tag.find("label")
    label_text = label_tag.get_text(separator=" ", strip=True) if label_tag else ""

    # --- caption body ---
    caption_tag = (
        figure_tag.find("fig-caption")
        or figure_tag.find("caption")
        or figure_tag.find("figcaption")
    )
    caption_text = caption_tag.get_text(separator=" ", strip=True) if caption_tag else ""

    # Combine: "Figure 1. Some description of the figure."
    if label_text and caption_text:
        # Avoid duplicating the label if it's already the start of caption_text
        if caption_text.startswith(label_text):
            combined = caption_text
        else:
            sep = " " if label_text.endswith(".") else ". "
            combined = label_text + sep + caption_text
    elif label_text:
        combined = label_text
    else:
        combined = caption_text

    return combined.strip()


def _extract_figures_html(soup: BeautifulSoup, pmc_id: str) -> tuple[list[Image.Image], list[str]]:
    """
    Locate <figure> tags in the parsed HTML, download each <img src>, and
    return the successful downloads as RGB PIL.Image objects alongside their
    captions.

    Returns
    -------
    (images, captions) where both lists are the same length and each caption
    corresponds to the figure at the same index (empty string if not found).
    """
    images: list[Image.Image] = []
    captions: list[str] = []

    for figure_tag in soup.find_all("figure"):
        img_tag = figure_tag.find("img")
        if img_tag is None:
            continue

        src = img_tag.get("src") or img_tag.get("data-src") or ""
        url = _normalise_url(src)
        if url is None:
            continue

        img = _fetch_image(url)
        if img is not None:
            images.append(img)
            captions.append(_extract_caption_from_figure(figure_tag))

    return images, captions


# ---------------------------------------------------------------------------
# Figure extraction — fallback (PDF via PyMuPDF)
# ---------------------------------------------------------------------------


def _download_pdf(url: str) -> Optional[bytes]:
    """Download a URL and return bytes only if it's a real PDF."""
    try:
        resp = _get(url, timeout=_PAGE_TIMEOUT)
        resp.raise_for_status()
        content = resp.content
        content_type = resp.headers.get("Content-Type", "")
        if "pdf" in content_type.lower() or b"%PDF-" in content[:1024]:
            print(f"[fetch_paper] PDF accepted from {url}: {len(content):,} bytes")
            return content
        print(f"[fetch_paper] Not a PDF at {url} (content-type={content_type!r})")
        return None
    except Exception as exc:
        print(f"[fetch_paper] Download failed {url}: {exc}")
        return None


def _fetch_pdf_bytes(pmc_id: str, doi: Optional[str] = None) -> Optional[bytes]:
    """Try to obtain the PMC PDF via OA API, then Unpaywall, then give up.

    NCBI's direct PDF URLs are gated behind a JS Proof-of-Work challenge
    that requests cannot solve. Instead we use:
      1. PMC OA API  — works for Open Access papers
      2. Unpaywall   — finds any legal OA version by DOI
    """
    # 1. PMC OA API
    try:
        oa_resp = _get(
            f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmc_id}",
            timeout=15,
        )
        if oa_resp.ok and "error" not in oa_resp.text:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(oa_resp.text)
            for link in root.findall(".//link"):
                if link.get("format") == "pdf":
                    pdf_url = link.get("href", "")
                    print(f"[fetch_paper] OA API PDF: {pdf_url}")
                    result = _download_pdf(pdf_url)
                    if result:
                        return result
    except Exception as exc:
        print(f"[fetch_paper] OA API error: {exc}")

    # 2. Unpaywall (requires DOI)
    if doi:
        try:
            uw_resp = _get(
                f"https://api.unpaywall.org/v2/{doi}?email=pmc-extractor@example.com",
                timeout=15,
            )
            if uw_resp.ok:
                data = uw_resp.json()
                loc = data.get("best_oa_location") or {}
                pdf_url = loc.get("url_for_pdf")
                if pdf_url:
                    print(f"[fetch_paper] Unpaywall PDF: {pdf_url}")
                    result = _download_pdf(pdf_url)
                    if result:
                        return result
        except Exception as exc:
            print(f"[fetch_paper] Unpaywall error: {exc}")

    print(f"[fetch_paper] No PDF available for {pmc_id} (PoW-gated or not OA)")
    return None


def _extract_figures_pdf(pdf_bytes: bytes) -> tuple[list[Image.Image], list[str]]:
    """Extract embedded raster images from PDF bytes via PyMuPDF.

    Caption extraction from PDFs is not attempted — captions are returned as
    empty strings for every extracted image.

    Returns
    -------
    (images, captions) where captions is a list of empty strings of equal length.
    """
    images: list[Image.Image] = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            for image_info in page.get_images(full=True):
                xref = image_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    img = Image.open(io.BytesIO(base_image["image"])).convert("RGB")
                    if img.width >= _MIN_DIMENSION and img.height >= _MIN_DIMENSION:
                        images.append(img)
                except Exception:
                    continue
        doc.close()
    except Exception:
        pass
    captions: list[str] = [""] * len(images)
    return images, captions


# ---------------------------------------------------------------------------
# Public PDF page renderer
# ---------------------------------------------------------------------------


def render_pdf_pages(pdf_bytes: bytes, max_pages: int = 20) -> list[Image.Image]:
    """Render PDF pages as PIL Images using PyMuPDF at 2x zoom.

    Parameters
    ----------
    pdf_bytes:
        Raw PDF file content.
    max_pages:
        Maximum number of pages to render. Defaults to 20.

    Returns
    -------
    list of RGB PIL.Image objects, one per rendered page (up to max_pages).
    """
    images: list[Image.Image] = []
    # Strip any leading garbage before the %PDF- magic bytes (BOM, whitespace, etc.)
    pdf_start = pdf_bytes.find(b"%PDF-")
    if pdf_start > 0:
        print(f"[fetch_paper] Stripping {pdf_start} leading bytes before %PDF-")
        pdf_bytes = pdf_bytes[pdf_start:]
    print(f"[fetch_paper] render_pdf_pages: input={len(pdf_bytes):,}B first_bytes={pdf_bytes[:8]!r}")
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page_count = len(doc)
        print(f"[fetch_paper] PDF opened: {page_count} pages, rendering up to {max_pages}")
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            mat = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        doc.close()
        print(f"[fetch_paper] Rendered {len(images)} page(s)")
    except Exception as exc:
        print(f"[fetch_paper] PDF render failed: {exc}")
    return images


# ---------------------------------------------------------------------------
# JATS XML extraction
# ---------------------------------------------------------------------------


def _fetch_jats_text(pmc_id: str) -> Optional[str]:
    """
    Fetch and parse JATS XML from NCBI E-utilities, returning structured text.

    Extracts section text from <sec> elements. Falls back to all <p> elements
    in <body> if no <sec> elements are found. Excludes <ref-list>.

    Returns None on network failure or parse error (caller falls back to HTML).
    """
    numeric_id = re.sub(r"(?i)^pmc", "", pmc_id)
    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        f"?db=pmc&id={numeric_id}&rettype=xml&retmode=xml"
    )
    try:
        resp = _get(url, timeout=_PAGE_TIMEOUT)
        resp.raise_for_status()
        xml_text = resp.text
    except Exception as exc:
        print(f"[fetch_paper] JATS fetch failed: {exc}")
        return None

    # Strip namespace declarations so ElementTree tag lookups stay simple.
    xml_clean = re.sub(r'\s+xmlns(?::[^=]*)?\s*=\s*"[^"]*"', "", xml_text)
    try:
        root = ET.fromstring(xml_clean)
    except ET.ParseError as exc:
        print(f"[fetch_paper] JATS XML parse failed: {exc}")
        return None

    sections: list[str] = []

    # Locate <body> — may be nested under <article>.
    body = root.find(".//body")
    if body is None:
        return None

    # Helper: collect all text from an element and its descendants.
    def _element_text(el: ET.Element) -> str:
        return " ".join(t.strip() for t in el.itertext() if t.strip())

    # Try structured <sec> extraction first.
    sec_elements = body.findall(".//sec")
    # Filter out any sec that is itself inside a ref-list.
    ref_list = body.find(".//ref-list")

    def _inside_ref_list(el: ET.Element) -> bool:
        """Return True if el is the ref-list element or a descendant of it."""
        if ref_list is None:
            return False
        # Walk the ref-list subtree to check membership.
        return el in list(ref_list.iter())

    if sec_elements:
        for sec in sec_elements:
            if _inside_ref_list(sec):
                continue
            title_el = sec.find("title")
            section_title = _element_text(title_el).strip() if title_el is not None else ""
            paragraphs: list[str] = []
            for p in sec.findall("p"):
                text = _element_text(p).strip()
                if text:
                    paragraphs.append(text)
            if not paragraphs and not section_title:
                continue
            block = f"## {section_title}\n\n" + "\n\n".join(paragraphs) if section_title else "\n\n".join(paragraphs)
            sections.append(block)
    else:
        # Fallback: all <p> directly in body, excluding ref-list.
        for p in body.findall(".//p"):
            if _inside_ref_list(p):
                continue
            text = _element_text(p).strip()
            if text:
                sections.append(text)

    return "\n\n".join(sections) if sections else None


def _fetch_jats_metadata(pmc_id: str) -> Optional[dict]:
    """
    Fetch JATS XML and extract article metadata.

    Returns a metadata dict or None on failure.
    """
    numeric_id = re.sub(r"(?i)^pmc", "", pmc_id)
    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        f"?db=pmc&id={numeric_id}&rettype=xml&retmode=xml"
    )
    try:
        resp = _get(url, timeout=_PAGE_TIMEOUT)
        resp.raise_for_status()
        xml_text = resp.text
    except Exception as exc:
        print(f"[fetch_paper] JATS metadata fetch failed: {exc}")
        return None

    xml_clean = re.sub(r'\s+xmlns(?::[^=]*)?\s*=\s*"[^"]*"', "", xml_text)
    try:
        root = ET.fromstring(xml_clean)
    except ET.ParseError as exc:
        print(f"[fetch_paper] JATS XML metadata parse failed: {exc}")
        return None

    def _find_text(path: str) -> Optional[str]:
        el = root.find(path)
        if el is None:
            return None
        return " ".join(t.strip() for t in el.itertext() if t.strip()) or None

    # Authors: <contrib contrib-type="author"> → <surname> + <given-names>
    authors: list[str] = []
    for contrib in root.findall(".//contrib[@contrib-type='author']"):
        surname = contrib.findtext("surname") or contrib.findtext("name/surname") or ""
        given = contrib.findtext("given-names") or contrib.findtext("name/given-names") or ""
        full = f"{given} {surname}".strip()
        if full:
            authors.append(full)

    return {
        "title": _find_text(".//article-meta/title-group/article-title"),
        "authors": authors,
        "journal": _find_text(".//journal-title"),
        "year": _find_text(".//article-meta/pub-date/year"),
        "doi": _find_text(".//article-meta/article-id[@pub-id-type='doi']"),
        "pmid": _find_text(".//article-meta/article-id[@pub-id-type='pmid']"),
    }


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------


def _extract_metadata(soup: BeautifulSoup) -> dict:
    """
    Extract paper metadata from PMC citation <meta> tags.

    PMC embeds Google Scholar / Highwire citation tags:
    citation_title, citation_author, citation_journal_title,
    citation_publication_date, citation_doi, citation_pmid.
    """
    def _meta(name: str) -> Optional[str]:
        tag = soup.find("meta", attrs={"name": name})
        return tag["content"].strip() if tag and tag.get("content") else None

    def _meta_all(name: str) -> list[str]:
        return [
            t["content"].strip()
            for t in soup.find_all("meta", attrs={"name": name})
            if t.get("content")
        ]

    raw_date = _meta("citation_publication_date") or _meta("citation_date") or ""
    year = raw_date.split("/")[0] if raw_date else None

    return {
        "title": _meta("citation_title"),
        "authors": _meta_all("citation_author"),
        "journal": _meta("citation_journal_title"),
        "year": year,
        "doi": _meta("citation_doi"),
        "pmid": _meta("citation_pmid"),
    }


# ---------------------------------------------------------------------------
# Article HTML extraction
# ---------------------------------------------------------------------------


def _extract_article_html(soup: BeautifulSoup, pmc_id: str) -> str:
    """Extract and clean the article body HTML for in-app rendering.

    Rewrites relative image/link URLs to absolute PMC URLs so assets load
    correctly inside the Gradio gr.HTML component.
    """
    base = f"{_BASE_URL}/articles/{pmc_id}/"

    container = (
        soup.find(id="mc_articles_content")
        or soup.find(class_=re.compile(r"article[_-]?(body|content|text)", re.I))
        or soup.find("article")
        or soup.find("main")
    )
    if container is None:
        return "<p style='color:gray;padding:16px;'>Article HTML not available.</p>"

    # Rewrite src/href attributes to absolute URLs.
    for tag in container.find_all(True):
        for attr in ("src", "href", "data-src"):
            val = tag.get(attr, "")
            if not val or val.startswith("data:") or val.startswith("http") or val.startswith("//"):
                continue
            if val.startswith("/"):
                tag[attr] = _BASE_URL + val
            else:
                tag[attr] = base + val

    # Remove nav/sidebar/download-button noise.
    for noise in container.find_all(class_=re.compile(r"sidebar|nav|download|toc|ref-list", re.I)):
        noise.decompose()

    inner = container.decode_contents()
    return (
        f'<div style="max-width:860px;margin:0 auto;padding:24px 32px;'
        f'font-family:Georgia,serif;font-size:15px;line-height:1.7;color:#222;">'
        f'{inner}'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_paper(
    pmc_id: str,
    use_jats: bool = False,
) -> tuple[str, list[Image.Image], list[str], dict, Optional[bytes], str]:
    """
    Fetch a PubMed Central article and return its text, figures, captions, metadata, and PDF.

    Parameters
    ----------
    pmc_id:
        A valid PMC identifier, e.g. ``"PMC7614754"``.
    use_jats:
        When True, fetch full-text via NCBI JATS XML (E-utilities efetch) for
        richer structured text and metadata. Falls back to the HTML path if the
        JATS request fails. Figures are still sourced from the HTML page (or
        the PDF as a fallback). Default is False.

    Returns
    -------
    text:
        Full article body text, sections joined by blank lines.
    figures:
        List of RGB PIL.Image objects, one per figure found.
    figure_captions:
        List of caption strings, one per figure (empty string if not found).
        Sourced from <label> + <fig-caption>/<caption>/<figcaption> tags for
        HTML/JATS paths; empty strings for PDF-extracted figures.
    metadata:
        Dict with keys: title, authors, journal, year, doi, pmid.
    pdf_bytes:
        Raw PDF bytes, or None if no PDF is available.

    Raises
    ------
    ValueError
        If *pmc_id* does not match the expected ``PMC\\d+`` format.
    requests.HTTPError
        If the PMC article page returns a non-2xx status.
    """
    _validate_pmc_id(pmc_id)

    # Always fetch the HTML page — needed for figure extraction and HTML-path
    # text/metadata when JATS is unavailable or disabled.
    article_url = f"{_BASE_URL}/articles/{pmc_id}/"
    response = _get(article_url, timeout=_PAGE_TIMEOUT)
    if not response.ok:
        raise requests.HTTPError(
            f"Failed to fetch {article_url} — HTTP {response.status_code}",
            response=response,
        )

    soup = BeautifulSoup(response.text, "html.parser")

    # Text and metadata: JATS path or HTML fallback.
    text: Optional[str] = None
    metadata: Optional[dict] = None

    if use_jats:
        jats_text = _fetch_jats_text(pmc_id)
        if jats_text:
            text = jats_text
        else:
            print("[fetch_paper] JATS text unavailable, falling back to HTML extraction.")

        jats_metadata = _fetch_jats_metadata(pmc_id)
        if jats_metadata:
            metadata = jats_metadata
        else:
            print("[fetch_paper] JATS metadata unavailable, falling back to HTML extraction.")

    if text is None:
        text = _extract_text(soup)
    if metadata is None:
        metadata = _extract_metadata(soup)

    # Figures: HTML-first, PDF fallback.
    figures, figure_captions = _extract_figures_html(soup, pmc_id)
    doi = metadata.get("doi") if metadata else None
    pdf_bytes = None
    if not figures:
        pdf_bytes = _fetch_pdf_bytes(pmc_id, doi)
        if pdf_bytes:
            figures, figure_captions = _extract_figures_pdf(pdf_bytes)
    else:
        pdf_bytes = _fetch_pdf_bytes(pmc_id, doi)

    article_html = _extract_article_html(soup, pmc_id)

    return text, figures, figure_captions, metadata, pdf_bytes, article_html


# ---------------------------------------------------------------------------
# Generic URL helpers
# ---------------------------------------------------------------------------


def _is_pmc_url(url: str) -> Optional[str]:
    """Return the PMC ID if *url* is a PMC article URL or bare PMC ID, else None.

    Accepts:
    - "PMC7614754"  (bare ID, case-insensitive)
    - "https://pmc.ncbi.nlm.nih.gov/articles/PMC7614754/..."
    - "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7614754/..."
    """
    # Bare ID like "PMC7614754" or "pmc7614754".
    bare = re.fullmatch(r"(?i)(PMC\d+)", url.strip())
    if bare:
        return bare.group(1).upper()

    # URL containing /articles/PMC<digits>
    m = re.search(r"/articles/(PMC\d+)", url, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    return None


def _extract_text_pdf(pdf_bytes: bytes) -> str:
    """Extract plain text from PDF bytes via PyMuPDF.

    Concatenates text from every page in reading order.
    """
    parts: list[str] = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            parts.append(page.get_text())
        doc.close()
    except Exception as exc:
        print(f"[fetch_paper] PDF text extraction failed: {exc}")
    return "\n\n".join(parts)


def _build_pdf_html(pdf_bytes: bytes, max_pages: int = 20) -> str:
    """Render PDF pages as base64 PNG images wrapped in an HTML string."""
    pages = render_pdf_pages(pdf_bytes, max_pages=max_pages)
    if not pages:
        return "<p style='color:gray;padding:16px;'>Could not render PDF pages.</p>"

    items: list[str] = []
    for i, img in enumerate(pages):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        items.append(
            f'<div style="margin:16px 0;text-align:center;">'
            f'<img src="data:image/png;base64,{b64}" '
            f'style="max-width:100%;border:1px solid #ddd;border-radius:4px;" '
            f'alt="Page {i + 1}">'
            f'<p style="font-size:12px;color:#666;margin-top:4px;">Page {i + 1}</p>'
            f'</div>'
        )
    body = "".join(items)
    return (
        f'<div style="max-width:860px;margin:0 auto;padding:24px 32px;">'
        f'{body}'
        f'</div>'
    )


def _find_pdf_link(soup: BeautifulSoup, base_url: str) -> Optional[str]:
    """Heuristically find a PDF download link in an HTML page.

    Looks for <a> tags whose href ends in .pdf or whose text/title suggests
    a PDF download. Returns an absolute URL or None.
    """
    for a in soup.find_all("a", href=True):
        href: str = a["href"]
        text = (a.get_text() or "").lower()
        title = (a.get("title") or "").lower()
        if (
            href.lower().endswith(".pdf")
            or "pdf" in href.lower()
            or "pdf" in text
            or "pdf" in title
        ):
            return urljoin(base_url, href)
    return None


def _extract_metadata_generic(soup: BeautifulSoup, url: str) -> dict:
    """Extract metadata from arbitrary HTML via Open Graph, DC, and citation tags.

    Falls back to <title> for the title when richer tags are absent.
    """

    def _meta_prop(prop: str) -> Optional[str]:
        tag = soup.find("meta", attrs={"property": prop})
        return tag["content"].strip() if tag and tag.get("content") else None

    def _meta_name(name: str) -> Optional[str]:
        tag = soup.find("meta", attrs={"name": name})
        return tag["content"].strip() if tag and tag.get("content") else None

    def _meta_name_all(name: str) -> list[str]:
        return [
            t["content"].strip()
            for t in soup.find_all("meta", attrs={"name": name})
            if t.get("content")
        ]

    # Title: Open Graph > citation > DC > <title>
    title = (
        _meta_prop("og:title")
        or _meta_name("citation_title")
        or _meta_name("DC.title")
        or _meta_name("dc.title")
        or (soup.title.string.strip() if soup.title and soup.title.string else None)
    )

    # Authors
    authors = (
        _meta_name_all("citation_author")
        or _meta_name_all("DC.creator")
        or _meta_name_all("dc.creator")
    )

    # Journal / site name
    journal = (
        _meta_name("citation_journal_title")
        or _meta_prop("og:site_name")
        or _meta_name("DC.publisher")
    )

    # Year
    raw_date = (
        _meta_name("citation_publication_date")
        or _meta_name("citation_date")
        or _meta_name("DC.date")
        or _meta_prop("article:published_time")
        or ""
    )
    year: Optional[str] = None
    if raw_date:
        m = re.search(r"\d{4}", raw_date)
        year = m.group(0) if m else None

    # DOI
    doi = _meta_name("citation_doi") or _meta_name("DC.identifier")

    return {
        "title": title,
        "authors": authors,
        "journal": journal,
        "year": year,
        "doi": doi,
        "pmid": None,
        "source_url": url,
    }


def _extract_article_html_generic(soup: BeautifulSoup, page_url: str) -> str:
    """Extract and clean article body HTML for an arbitrary page URL.

    Rewrites relative URLs to absolute using *page_url* as the base.
    """
    container = (
        soup.find(class_=re.compile(r"article[_-]?(body|content|text)", re.I))
        or soup.find("article")
        or soup.find("main")
        or soup.find(id=re.compile(r"content|main|body", re.I))
        or soup.body
    )
    if container is None:
        return "<p style='color:gray;padding:16px;'>Article HTML not available.</p>"

    # Rewrite src/href attributes to absolute URLs.
    for tag in container.find_all(True):
        for attr in ("src", "href", "data-src"):
            val = tag.get(attr, "")
            if not val or val.startswith("data:"):
                continue
            if val.startswith("http") or val.startswith("//"):
                continue
            tag[attr] = urljoin(page_url, val)

    # Remove nav/sidebar noise.
    for noise in container.find_all(["nav", "header", "footer", "aside"]):
        noise.decompose()
    for noise in container.find_all(class_=re.compile(r"sidebar|nav|download|toc|ref-list|cookie|banner|ad", re.I)):
        noise.decompose()

    inner = container.decode_contents()
    return (
        f'<div style="max-width:860px;margin:0 auto;padding:24px 32px;'
        f'font-family:Georgia,serif;font-size:15px;line-height:1.7;color:#222;">'
        f'{inner}'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Public fetch_url
# ---------------------------------------------------------------------------


def fetch_url(
    url: str,
    use_jats: bool = False,
) -> tuple[str, list[Image.Image], list[str], dict, Optional[bytes], str]:
    """Fetch content from any URL and return the same tuple as fetch_paper.

    Routing
    -------
    - PMC URLs / bare PMC IDs  →  delegate to ``fetch_paper()``
    - Generic URLs             →  fetch as PDF or HTML and extract content

    Parameters
    ----------
    url:
        A PMC identifier (e.g. ``"PMC7614754"``), a PMC article URL, or any
        HTTP/HTTPS URL pointing to a paper or PDF.
    use_jats:
        Passed through to ``fetch_paper()`` for PMC inputs; ignored otherwise.

    Returns
    -------
    Identical to ``fetch_paper``:
    ``(text, figures, figure_captions, metadata, pdf_bytes, article_html)``
    """
    # 1. PMC shortcut.
    pmc_id = _is_pmc_url(url)
    if pmc_id:
        print(f"[fetch_url] Detected PMC ID {pmc_id}, delegating to fetch_paper()")
        return fetch_paper(pmc_id, use_jats=use_jats)

    print(f"[fetch_url] Generic URL: {url}")

    # 2. Fetch the URL.
    try:
        resp = _get(url, timeout=_PAGE_TIMEOUT)
        resp.raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch {url}: {exc}") from exc

    content_type = resp.headers.get("Content-Type", "").lower()
    raw = resp.content

    # 3a. Direct PDF response.
    is_pdf = "application/pdf" in content_type or raw[:5] == b"%PDF-"
    if is_pdf:
        print(f"[fetch_url] Response is PDF ({len(raw):,} bytes)")
        text = _extract_text_pdf(raw)
        figures, figure_captions = _extract_figures_pdf(raw)
        metadata: dict = {"title": None, "authors": [], "journal": None, "year": None, "doi": None, "pmid": None, "source_url": url}
        article_html = _build_pdf_html(raw)
        return text, figures, figure_captions, metadata, raw, article_html

    # 3b. HTML response.
    print(f"[fetch_url] Response is HTML, parsing...")
    soup = BeautifulSoup(raw, "html.parser")
    text = _extract_text(soup)
    figures, figure_captions = _extract_figures_html(soup, "")
    metadata = _extract_metadata_generic(soup, url)
    article_html = _extract_article_html_generic(soup, url)

    # Try to find and fetch a linked PDF.
    pdf_bytes: Optional[bytes] = None
    pdf_link = _find_pdf_link(soup, url)
    if pdf_link:
        print(f"[fetch_url] Found PDF link: {pdf_link}")
        pdf_bytes = _download_pdf(pdf_link)
        if pdf_bytes and not figures:
            figures, figure_captions = _extract_figures_pdf(pdf_bytes)

    return text, figures, figure_captions, metadata, pdf_bytes, article_html


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fetch_paper.py <PMC_ID>")
        print("Example: python fetch_paper.py PMC7614754")
        sys.exit(1)

    pmc_arg = sys.argv[1].strip()
    print(f"Fetching {pmc_arg} ...")

    paper_text, paper_figures, paper_captions, paper_meta, paper_pdf, _ = fetch_paper(pmc_arg)

    print(f"Text length  : {len(paper_text):,} characters")
    print(f"Figure count : {len(paper_figures)}")
    print(f"PDF available: {paper_pdf is not None}")
    print(f"Title        : {paper_meta.get('title')}")
    for i, cap in enumerate(paper_captions, 1):
        if cap:
            print(f"Caption {i}    : {cap[:120]}")
