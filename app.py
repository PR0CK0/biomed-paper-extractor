from __future__ import annotations

import base64
import io
import json
import random
import re
import threading
import time
from datetime import datetime
from pathlib import Path


import gradio as gr
import requests

from fetch_paper import fetch_paper, fetch_url
from task1_figures import (
    load_vlm,
    analyze_figures,
    check_ollama,
    toggle_deplot,
    get_last_prompt,
    set_ollama_timeout,
    set_ollama_retries,
    _HF_VLM_REGISTRY,
    _API_OPENAI_CHOICES,
    _API_ANTHROPIC_CHOICES,
    _API_GOOGLE_CHOICES,
)
from task2_ner import (
    load_ner,
    extract_entities,
    NER_MODEL_OPTIONS,
    NER_MODEL_INFO,
    GLINER_DEFAULT_ENTITY_TYPES,
)
from eval_suite import (
    run_eval,
    generate_md_report,
    save_eval_run,
    DEFAULT_PMC_IDS,
    get_vlm_summary_md,
    get_ner_summary_md,
)
from eval_charts import get_all_charts, chart_ner_type_distribution

try:
    from ner_viz import build_cooccurrence_graph, build_word_cloud, COOCCURRENCE_AVAILABLE, WORDCLOUD_AVAILABLE
except ImportError:
    build_cooccurrence_graph = None  # type: ignore[assignment]
    build_word_cloud = None  # type: ignore[assignment]
    COOCCURRENCE_AVAILABLE = False
    WORDCLOUD_AVAILABLE = False

_HF_VLM_CHOICES: list[tuple[str, str]] = _HF_VLM_REGISTRY

_API_PROVIDER_CHOICES = ["OpenAI", "Anthropic", "Google"]
_API_CHOICES_MAP: dict[str, list[tuple[str, str]]] = {
    "OpenAI": _API_OPENAI_CHOICES,
    "Anthropic": _API_ANTHROPIC_CHOICES,
    "Google": _API_GOOGLE_CHOICES,
}


def _ollama_status_html(running: bool) -> str:
    color, label = ("#22c55e", "Ollama running") if running else ("#ef4444", "Ollama not running")
    return (
        f'<span style="display:inline-flex;align-items:center;gap:6px;'
        f'font-size:13px;color:{color};font-weight:500;">'
        f'<span style="width:10px;height:10px;border-radius:50%;background:{color};'
        f'display:inline-block;flex-shrink:0;"></span>{label}</span>'
    )

def _get_ollama_vision_models() -> list[tuple[str, str]]:
    """Query Ollama for downloaded models that explicitly advertise vision capability.

    Uses /api/show per model to check the capabilities array. This is authoritative
    and works for all model families without pattern matching.
    """
    try:
        tags_resp = requests.get("http://localhost:11434/api/tags", timeout=2)
        if not tags_resp.ok:
            return []
        entries = []
        for m in tags_resp.json().get("models", []):
            name = m["name"]
            try:
                show = requests.post(
                    "http://localhost:11434/api/show",
                    json={"name": name, "verbose": False},
                    timeout=3,
                )
                if show.ok and "vision" in show.json().get("capabilities", []):
                    details = m.get("details", {})
                    params = details.get("parameter_size", "")
                    family = details.get("family", "")
                    size_bytes = m.get("size", 0)
                    gb = size_bytes / 1e9
                    parts = []
                    if params:
                        parts.append(f"{params} params")
                    parts.append(f"{gb:.1f} GB")
                    label = f"{name}  ({' · '.join(parts)})"
                    entries.append((label, f"ollama/{name}", family, size_bytes))
            except Exception:
                continue
        # Sort: family descending, then size descending within family
        entries.sort(key=lambda e: (e[2], e[3]), reverse=True)
        return [(label, value) for label, value, *_ in entries]
    except Exception:
        return []


def _update_vlm_choices(provider: str):
    """Returns: vlm_dropdown, ollama_indicator, api_provider_row, api_key_input, ollama_settings_row"""
    running = check_ollama()
    is_api = provider == "API (Cloud)"
    is_ollama = provider == "Ollama (local)"

    if is_ollama:
        choices = _get_ollama_vision_models() if running else []
        if not choices:
            choices = [("No vision models found — pull one with `ollama pull llava`", "")]
        return (
            gr.update(choices=choices, value=choices[0][1] if choices else None),
            _ollama_status_html(running),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
        )
    elif is_api:
        default_choices = _API_OPENAI_CHOICES
        return (
            gr.update(choices=default_choices, value=default_choices[0][1]),
            _ollama_status_html(running),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=False),
        )
    else:  # HuggingFace
        return (
            gr.update(choices=_HF_VLM_CHOICES, value=_HF_VLM_CHOICES[0][1]),
            _ollama_status_html(running),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )


def _update_api_model_choices(api_provider: str):
    choices = _API_CHOICES_MAP.get(api_provider, _API_OPENAI_CHOICES)
    return gr.update(choices=choices, value=choices[0][1])


_ollama_available = check_ollama()
_provider_choices = (
    ["HuggingFace", "Ollama (local)", "API (Cloud)"]
    if _ollama_available
    else ["HuggingFace", "API (Cloud)"]
)

NER_MODELS = NER_MODEL_OPTIONS


_NER_BASE_LABELS: dict[str, str] = {
    "scispacy_umls":        "MedMentions",
    "scispacy_triple":      "BC5CDR + CRAFT + JNLPBA",
    "d4data":               "MACCROBAT2018",
    "gliner":               "DeBERTa-v3 / biomedical distil",
    "pubmedbert":           "NCBI Disease + CRAFT",
    "scispacy_bionlp13cg":  "BioNLP13CG cancer genetics",
    "hunflair2":            "31 annotated corpora",
    "species_ner":          "Species-800",
}

_NER_SHORT_NAMES: dict[str, str] = {
    "scispacy_umls":        "scispaCy + UMLS",
    "scispacy_triple":      "Triple scispaCy Stack",
    "d4data":               "d4data DistilBERT",
    "gliner":               "GLiNER zero-shot",
    "pubmedbert":           "PubMedBERT Suite",
    "scispacy_bionlp13cg":  "BioNLP13CG Stack",
    "hunflair2":            "HunFlair2",
    "species_ner":          "Species NER",
}


def _ner_dropdown_choices() -> list[tuple[str, str]]:
    """Build NER dropdown labels showing base/dataset and output type count."""
    choices = []
    for key in NER_MODEL_OPTIONS:
        info = NER_MODEL_INFO.get(key, {})
        base = _NER_BASE_LABELS.get(key, "")
        if key == "gliner":
            n_types = len(GLINER_DEFAULT_ENTITY_TYPES)
            types_str = f"{n_types} default types"
        else:
            n_types = len(info.get("labels", []))
            types_str = f"{n_types} types"
        name = _NER_SHORT_NAMES.get(key, key)
        suffix = f"  ({base}  ·  {types_str})" if base else f"  ({types_str})"
        choices.append((name + suffix, key))
    return choices


_NER_CHOICES = _ner_dropdown_choices()


def _status_html(label: str, text: str) -> str:
    if text.startswith("Ready"):
        color = "#22c55e"
    elif text.startswith("Loading") or text.startswith("Waiting"):
        color = "#f59e0b"
    elif text.startswith("Failed"):
        color = "#ef4444"
    else:
        color = "#9ca3af"
    return (
        f'<p style="margin:4px 0;font-size:13px;">'
        f'<span style="color:#6b7280;">{label}: </span>'
        f'<span style="color:{color};font-weight:500;">{text}</span>'
        f'</p>'
    )


def _readiness_row(label: str, text: str) -> str:
    if text.startswith("Ready"):
        icon, color = "✓", "#22c55e"
    elif "Loading" in text or "Waiting" in text:
        icon, color = "◌", "#f59e0b"
    elif "Failed" in text:
        icon, color = "✗", "#ef4444"
    else:
        icon, color = "●", "#ef4444"
    return (
        f'<div style="color:{color};font-size:12px;margin:0;padding:0;line-height:1.3;">'
        f'{icon} {label}: {text}'
        f'</div>'
    )


def _pipeline_html(
    fetch: str = "idle", vlm: str = "idle", ner: str = "idle", complete: str = "idle",
    fetch_info: str = "", vlm_info: str = "", ner_info: str = "",
    total_info: str = "",
) -> str:
    _COLORS: dict[str, tuple[str, str]] = {
        "idle":    ("#4b5563", "#111827"),
        "running": ("#f59e0b", "#1c1200"),
        "done":    ("#22c55e", "#052e16"),
        "error":   ("#ef4444", "#1c0505"),
    }
    _ICONS = {"idle": "○", "running": "↻", "done": "✓", "error": "✗"}

    def node(label: str, state: str, info: str) -> str:
        color, bg = _COLORS.get(state, _COLORS["idle"])
        icon = _ICONS.get(state, "○")
        opacity = "1" if state != "idle" else "0.35"
        spin = "animation:pl-spin 1s linear infinite;display:inline-block;" if state == "running" else ""
        info_html = (
            f'<div style="font-size:11px;color:{color};opacity:0.85;margin-top:4px;'
            f'text-align:center;line-height:1.4;">{info}</div>'
            if info else '<div style="height:19px;"></div>'
        )
        return (
            f'<div style="flex:1;display:flex;flex-direction:column;align-items:center;'
            f'min-width:0;opacity:{opacity};">'
            f'<div style="width:48px;height:48px;border-radius:50%;border:2px solid {color};'
            f'background:{bg};display:flex;align-items:center;justify-content:center;flex-shrink:0;">'
            f'<span style="font-size:20px;color:{color};{spin}">{icon}</span>'
            f'</div>'
            f'<div style="margin-top:7px;font-size:12px;font-weight:600;color:{color};'
            f'text-align:center;white-space:nowrap;">{label}</div>'
            + info_html
            + '</div>'
        )

    def connector(from_state: str) -> str:
        color = (
            "#22c55e" if from_state == "done"
            else "#f59e0b" if from_state == "running"
            else "#374151"
        )
        return (
            f'<div style="flex:1;padding-top:24px;">'
            f'<div style="height:2px;background:{color};width:100%;"></div>'
            f'</div>'
        )

    return (
        '<style>@keyframes pl-spin{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}</style>'
        '<div style="width:100%;box-sizing:border-box;padding:20px 8px 12px;">'
        '<div style="display:flex;align-items:flex-start;width:100%;gap:0;">'
        + node("Fetch", fetch, fetch_info)
        + connector(fetch)
        + node("Image Analysis", vlm, vlm_info)
        + connector(vlm)
        + node("Entity Extraction", ner, ner_info)
        + connector(ner)
        + node("Complete", complete, total_info)
        + '</div></div>'
    )


_PIPELINE_IDLE = _pipeline_html()


_vlm_load_token: list[int] = [0]  # mutable so inner thread can read latest value


def _render_readiness(vlm_text: str, ner_text: str) -> str:
    return (
        '<div style="display:flex;flex-direction:column;gap:2px;margin-top:2px;">'
        + _readiness_row("VLM", vlm_text)
        + _readiness_row("NER", ner_text)
        + '</div>'
    )


def load_vlm_model(vlm_model_id: str, api_key: str):
    _vlm_load_token[0] += 1
    my_token = _vlm_load_token[0]

    # outputs: vlm_status, _vlm_loaded, run_btn, _vlm_ready_text, vlm_load_btn
    yield _status_html("VLM", "Loading..."), False, gr.update(interactive=False), "Loading...", gr.update(interactive=False)

    _error: list = [None]

    def _do():
        try:
            load_vlm(vlm_model_id, api_key=api_key.strip() if api_key else None)
        except Exception as exc:
            _error[0] = exc

    t = threading.Thread(target=_do, daemon=True)
    t.start()
    while t.is_alive():
        yield gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        t.join(timeout=0.1)
        if my_token != _vlm_load_token[0]:
            return  # superseded by a newer load request

    if my_token != _vlm_load_token[0]:
        return  # superseded after thread finished

    if _error[0]:
        text = f"Failed: {_error[0]}"
        yield _status_html("VLM", text), False, gr.update(), text, gr.update(interactive=True)
    else:
        text = f"Ready: {vlm_model_id.split('/')[-1]}"
        yield _status_html("VLM", text), True, gr.update(), text, gr.update(interactive=True)


def load_ner_model(ner_model_key: str):
    # outputs: ner_status, _ner_loaded, run_btn, _ner_ready_text, ner_load_btn
    yield _status_html("NER", "Loading..."), False, gr.update(interactive=False), "Loading...", gr.update(interactive=False)
    try:
        load_ner(ner_model_key)
        text = f"Ready: {NER_MODEL_OPTIONS.get(ner_model_key, ner_model_key)}"
        yield _status_html("NER", text), True, gr.update(), text, gr.update(interactive=True)
    except Exception as exc:
        text = f"Failed: {exc}"
        yield _status_html("NER", text), False, gr.update(), text, gr.update(interactive=True)


_RUN_HINT_PENDING = (
    '<div style="font-size:11px;color:#6b7280;margin-bottom:3px;line-height:1.4;">'
    'Load a <b>VLM</b> and <b>NER model</b> below before analyzing'
    '</div>'
)


def _update_run_btn(vlm_loaded: bool, ner_loaded: bool):
    ready = vlm_loaded and ner_loaded
    hint = "" if ready else _RUN_HINT_PENDING
    return gr.update(interactive=ready), hint


_ESEARCH_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_ESEARCH_PARAMS = {"db": "pmc", "term": "open access[filter]", "retmode": "json"}


def _esearch_json(params: dict) -> dict:
    """Call esearch and return parsed JSON, stripping any control characters first."""
    resp = requests.get(_ESEARCH_BASE, params={**_ESEARCH_PARAMS, **params}, timeout=15)
    resp.raise_for_status()
    # NCBI occasionally embeds stray control characters — strip all except tab/LF/CR.
    clean = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", resp.text)
    return json.loads(clean)


def fetch_random_pmc_id() -> str:
    """Return a random open-access PMC ID via NCBI E-utilities esearch.

    NCBI esearch hard-caps retstart at 9999 regardless of the total result
    count.  Passing an offset drawn from the full corpus size (millions of
    records) returns an empty idlist, causing a RuntimeError.  The fix keeps
    the offset inside [0, 9900] so there is always room for a full page, then
    picks one ID at random from the returned page of 100.
    """
    _NCBI_RETSTART_MAX = 9900  # leave headroom for retmax=100
    _PAGE_SIZE = 100
    offset = random.randint(0, _NCBI_RETSTART_MAX)
    ids = _esearch_json({"retmax": _PAGE_SIZE, "retstart": offset})["esearchresult"]["idlist"]
    if not ids:
        raise RuntimeError("No PMC IDs returned from esearch.")
    return f"PMC{random.choice(ids)}"


def _format_metadata_html(pmc_id: str, meta: dict) -> str:
    title = meta.get("title") or "Unknown title"
    authors = meta.get("authors") or []
    journal = meta.get("journal") or "Unknown journal"
    year = meta.get("year") or "Unknown year"
    doi = meta.get("doi")
    author_str = ", ".join(authors[:5]) + (" et al." if len(authors) > 5 else "")
    doi_link = (
        f' &middot; <a href="https://doi.org/{doi}" target="_blank" '
        f'style="color:#2563eb;text-decoration:none;">DOI</a>'
    ) if doi else ""
    return (
        f'<div style="font-family:Georgia,serif;max-width:900px;margin:0 auto;'
        f'padding:20px 24px 14px;border-bottom:1px solid rgba(255,255,255,0.15);">'
        f'<h1 style="font-size:1.3em;font-weight:600;margin:0 0 8px;line-height:1.4;color:#f0f0f0;">'
        f'{title}</h1>'
        f'<p style="margin:0 0 4px;font-size:0.9em;color:#b0b0b0;">{author_str}</p>'
        f'<p style="margin:0;font-size:0.85em;color:#888;">'
        f'<em>{journal}</em>, {year} &middot; {pmc_id}{doi_link}</p>'
        f'</div>'
    )


def normalize_pmc_id(raw: str) -> tuple[str | None, str | None]:
    pmc_id = raw.strip().upper()
    if not pmc_id.startswith("PMC"):
        pmc_id = f"PMC{pmc_id}"
    if not re.fullmatch(r"PMC\d+", pmc_id):
        return None, pmc_id
    return pmc_id, None


def _figures_html(figures: list, captions: list[str] | None = None) -> str:
    if not figures:
        return "<p style='color:gray;padding:16px;'>No figures extracted from this paper.</p>"

    uid = f"lb{id(figures) & 0xFFFFFF}"
    items = []
    for i, img in enumerate(figures):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        src = f"data:image/png;base64,{b64}"
        caption = (captions[i] if captions and i < len(captions) else "") or ""
        caption_html = (
            f'<p style="font-size:11px;color:#9ca3af;margin-top:4px;line-height:1.5;'
            f'max-width:100%;text-align:left;">{caption}</p>'
            if caption else ""
        )
        items.append(
            f'<div style="flex:1 1 45%;min-width:280px;text-align:center;margin:8px;">'
            f'<img src="{src}" '
            f'style="max-width:100%;border:1px solid #374151;border-radius:4px;cursor:zoom-in;transition:opacity 0.15s;" '
            f'onmouseover="this.style.opacity=\'0.85\'" onmouseout="this.style.opacity=\'1\'" '
            f'onclick="document.getElementById(\'{uid}-overlay\').style.display=\'flex\';'
            f'document.getElementById(\'{uid}-img\').src=this.src;" />'
            f'<p style="font-size:12px;color:#6b7280;margin-top:4px;">Figure {i + 1}</p>'
            f'{caption_html}'
            f'</div>'
        )

    grid = "".join(items)
    overlay = (
        f'<div id="{uid}-overlay" '
        f'style="display:none;position:fixed;inset:0;z-index:9999;'
        f'background:rgba(0,0,0,0.82);align-items:center;justify-content:center;cursor:zoom-out;" '
        f'onclick="this.style.display=\'none\';">'
        f'<img id="{uid}-img" src="" '
        f'style="max-width:90vw;max-height:90vh;border-radius:6px;box-shadow:0 8px 40px rgba(0,0,0,0.7);'
        f'object-fit:contain;pointer-events:none;" />'
        f'</div>'
    )
    return (
        overlay
        + f'<div style="display:flex;flex-wrap:wrap;gap:8px;padding:16px;">{grid}</div>'
    )



def _entities_to_highlighted(text: str, entities: list[dict]) -> list[tuple[str, str | None]]:
    """Convert entity list to gr.HighlightedText format."""
    if not entities or not text:
        return [(text, None)] if text else []
    result = []
    prev = 0
    for ent in sorted(entities, key=lambda e: e["start_char"]):
        start, end = ent["start_char"], ent["end_char"]
        if start < prev:  # skip overlapping
            continue
        if start > prev:
            result.append((text[prev:start], None))
        result.append((text[start:end], ent["label"]))
        prev = end
    if prev < len(text):
        result.append((text[prev:], None))
    return result


def _entities_table_html(entities: list[dict]) -> str:
    """Render entity list as HTML table with clickable UMLS links and label-pill toggle filters."""
    if not entities:
        return ""
    import uuid
    label_colors = {
        "Disease": "#ef4444", "Drug": "#3b82f6", "Chemical": "#3b82f6",
        "Gene": "#22c55e", "Protein": "#22c55e", "Species": "#f59e0b",
        "Organism": "#f59e0b", "CellLine": "#8b5cf6", "CellType": "#a78bfa",
        "Anatomy": "#ec4899", "Organ": "#ec4899", "Tissue": "#f472b6",
        "Cell": "#fb7185", "BiologicalProcess": "#14b8a6", "MolecularFunction": "#06b6d4",
        "DNA": "#84cc16", "RNA": "#a3e635", "Mutation": "#f97316",
    }
    # Deduplicate by text (case-insensitive), keeping first CUI seen
    seen: dict[str, dict] = {}
    for e in entities:
        key = e.get("text", "").strip().lower()
        if key and key not in seen:
            seen[key] = e

    # Collect ordered unique labels for the filter bar
    label_order: list[str] = []
    label_set: set[str] = set()
    for e in seen.values():
        lbl = e.get("label", "")
        if lbl and lbl not in label_set:
            label_order.append(lbl)
            label_set.add(lbl)

    tid = "et_" + uuid.uuid4().hex[:8]

    # Filter pill bar
    pills_html = ""
    for lbl in label_order:
        color = label_colors.get(lbl, "#94a3b8")
        pills_html += (
            f'<button id="{tid}_pill_{lbl}" data-label="{lbl}" data-active="0" '
            f'onclick="etToggle(\'{tid}\',\'{lbl}\',this)" '
            f'style="background:{color}22;color:{color};border:1px solid {color}55;'
            f'border-radius:4px;padding:2px 8px;font-size:11px;cursor:pointer;'
            f'margin:2px 3px;opacity:0.5;transition:opacity 0.15s;">'
            f'{lbl}</button>'
        )

    rows_html = ""
    for e in seen.values():
        text = e.get("text", "")
        label = e.get("label", "")
        cui = e.get("umls_cui") or e.get("kb_id") or ""
        color = label_colors.get(label, "#94a3b8")
        label_cell = (
            f'<span style="background:{color}22;color:{color};border:1px solid {color}55;'
            f'border-radius:3px;padding:1px 6px;font-size:11px;">{label}</span>'
        )
        if cui:
            cui_cell = (
                f'<a href="https://uts.nlm.nih.gov/uts/umls/concept/{cui}" '
                f'target="_blank" rel="noopener noreferrer" '
                f'style="color:#60a5fa;font-family:monospace;font-size:11px;'
                f'text-decoration:underline;">{cui}</a>'
            )
        else:
            cui_cell = '<span style="color:#4b5563;font-size:11px;">—</span>'
        rows_html += (
            f'<tr class="et-row" data-label="{label}">'
            f'<td style="padding:3px 8px;font-size:13px;">{text}</td>'
            f'<td style="padding:3px 8px;">{label_cell}</td>'
            f'<td style="padding:3px 8px;">{cui_cell}</td>'
            f"</tr>"
        )

    js = f"""
<script>
(function(){{
  function etToggle(tid, label, btn) {{
    var active = btn.dataset.active === '1';
    btn.dataset.active = active ? '0' : '1';
    btn.style.opacity = active ? '0.5' : '1';
    btn.style.boxShadow = active ? '' : '0 0 0 2px ' + btn.style.color;
    var filters = [];
    document.querySelectorAll('[id^="' + tid + '_pill_"][data-active="1"]').forEach(function(b) {{
      filters.push(b.dataset.label);
    }});
    document.querySelectorAll('#{tid} .et-row').forEach(function(row) {{
      row.style.display = (filters.length === 0 || filters.indexOf(row.dataset.label) !== -1) ? '' : 'none';
    }});
  }}
  window.etToggle = window.etToggle || etToggle;
  // expose per-table so multiple tables coexist
  window['etToggle'] = etToggle;
}})();
</script>
"""

    count = len(seen)
    return (
        js
        + f'<div id="{tid}" style="border:1px solid #374151;border-radius:6px;">'
        + f'<div style="padding:6px 8px 4px;background:#111827;border-bottom:1px solid #374151;'
        + f'border-radius:6px 6px 0 0;">'
        + f'<span style="font-size:11px;color:#6b7280;margin-right:6px;">Filter by type (click to toggle):</span>'
        + pills_html
        + f'<span style="float:right;font-size:11px;color:#6b7280;line-height:24px;">{count} unique entities</span>'
        + '</div>'
        + '<div style="max-height:380px;overflow-y:auto;">'
        + '<table style="width:100%;border-collapse:collapse;">'
        + '<thead><tr style="background:#1f2937;position:sticky;top:0;z-index:1;">'
        + '<th style="padding:6px 8px;text-align:left;font-size:12px;color:#9ca3af;">Entity</th>'
        + '<th style="padding:6px 8px;text-align:left;font-size:12px;color:#9ca3af;">Type</th>'
        + '<th style="padding:6px 8px;text-align:left;font-size:12px;color:#9ca3af;">UMLS CUI</th>'
        + '</tr></thead>'
        + f'<tbody>{rows_html}</tbody>'
        + '</table>'
        + '</div>'
        + '</div>'
    )


def _build_viz(entities_json_str: str | None, run_state: dict | None):
    """Build co-occurrence graph and word cloud from stored entities + text."""
    import plotly.graph_objects as go
    from PIL import Image as PILImage

    if not entities_json_str or not run_state:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="No data — run an extraction first", template="plotly_dark")
        return empty_fig, None

    try:
        entities = json.loads(entities_json_str)
        if not isinstance(entities, list):
            entities = []
    except Exception:
        entities = []

    text = run_state.get("text", "") if run_state else ""

    graph_fig = None
    if COOCCURRENCE_AVAILABLE and build_cooccurrence_graph is not None:
        try:
            graph_fig = build_cooccurrence_graph(entities, text)
        except Exception as exc:
            print(f"[viz] co-occurrence graph failed: {exc}")

    if graph_fig is None:
        graph_fig = go.Figure()
        msg = "networkx not installed — pip install networkx" if not COOCCURRENCE_AVAILABLE else "No entities for graph"
        graph_fig.update_layout(title=msg, template="plotly_dark")

    wc_img = None
    if WORDCLOUD_AVAILABLE and build_word_cloud is not None:
        try:
            wc_img = build_word_cloud(entities)
        except Exception as exc:
            print(f"[viz] word cloud failed: {exc}")

    return graph_fig, wc_img


def _save_run(
    run_dir: Path,
    pmc_id: str,
    figures: list,
    metadata: dict,
    figures_json: str,
    entities_json: str,
    run_info: dict,
    run_summary: dict | None = None,
    prompt_txt: str | None = None,
) -> None:
    figs_dir = run_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    for i, img in enumerate(figures):
        img.save(figs_dir / f"fig{i + 1}.png")

    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (run_dir / "task1_figures.json").write_text(figures_json)
    (run_dir / "task2_ner.json").write_text(entities_json)
    (run_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))

    if run_summary is not None:
        (run_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2))
    if prompt_txt:
        (run_dir / "prompt.txt").write_text(prompt_txt, encoding="utf-8")

    print(f"[output] Saved run to {run_dir}")


def _format_run_info(info: dict, save_error: str | None = None) -> str:
    lat = info.get("latency", {})
    stats = info.get("stats", {})

    meta_rows = [
        ("PMC ID", info.get("pmc_id", "")),
        ("Timestamp", info.get("timestamp", "").replace("_", " ")),
        ("Ingest Method", info.get("ingest_method", "")),
        ("VLM Model", info.get("vlm_model", "").split("/")[-1]),
        ("NER Model", info.get("ner_model_name", "")),
    ]
    if info.get("saved_to"):
        meta_rows.append(("Output", info["saved_to"]))
    meta_table = "\n".join(f"| **{k}** | {v} |" for k, v in meta_rows if v)

    lat_rows = []
    if "fetch_s" in lat:
        lat_rows.append(f"| Fetch | {lat['fetch_s']:.2f}s |")
    if "vlm_s" in lat:
        lat_rows.append(f"| Figure Analysis (VLM) | {lat['vlm_s']:.2f}s |")
    if "ner_s" in lat:
        lat_rows.append(f"| NER | {lat['ner_s']:.2f}s |")
    if "total_s" in lat:
        lat_rows.append(f"| **Total** | **{lat['total_s']:.2f}s** |")

    stat_items = []
    if "text_chars" in stats:
        stat_items.append(f"- Text: {stats['text_chars']:,} characters")
    if "figure_count" in stats:
        stat_items.append(f"- Figures: {stats['figure_count']}")
    if "entity_count" in stats:
        stat_items.append(f"- Entities: {stats['entity_count']}")
    stat_items.append(f"- PDF available: {'Yes' if stats.get('pdf_available') else 'No'}")
    stat_items.append(f"- HTML article available: {'Yes' if stats.get('html_available') else 'No'}")

    md = "### Run Info\n\n"
    if meta_table:
        md += "| Field | Value |\n|---|---|\n" + meta_table + "\n\n"
    if lat_rows:
        md += "#### Latency\n\n| Stage | Time |\n|---|---|\n" + "\n".join(lat_rows) + "\n\n"
    if stat_items:
        md += "#### Stats\n\n" + "\n".join(stat_items)
    if save_error:
        md += f"\n\n> ⚠️ **Save failed:** {save_error}"

    return md


_OUTPUT_ROOT = Path(__file__).parent / "output" / "extraction"
_EVAL_OUTPUT_ROOT = Path(__file__).parent / "output" / "evaluation"


# ---------------------------------------------------------------------------
# Run summary helpers (one-off extraction comparison)
# ---------------------------------------------------------------------------


def _compute_run_summary(
    figures_json_str: str,
    entities_json_str: str,
    run_info: dict,
    deplot_used: bool = False,
) -> dict:
    """Compute lightweight summary stats from a completed extraction run."""
    try:
        figs = json.loads(figures_json_str) if figures_json_str else []
        if not isinstance(figs, list):
            figs = []
    except Exception:
        figs = []

    n_figs = len(figs)
    n_compound = 0
    n_with_title = 0
    n_with_data = 0
    n_with_axes = 0
    total_panels = 0
    total_data_pts = 0
    total_chars = 0

    for fig in figs:
        if not isinstance(fig, dict):
            continue
        if fig.get("is_compound"):
            n_compound += 1
        panels = fig.get("panels") or []
        total_panels += len(panels)
        for p in panels:
            if not isinstance(p, dict):
                continue
            title = (p.get("title") or "").strip()
            if title and title.lower() not in ("", "none", "n/a", "unknown"):
                n_with_title += 1
            dp = p.get("data_points") or []
            if dp:
                n_with_data += 1
                total_data_pts += len(dp)
            x_label = (p.get("x_axis") or {}).get("label", "")
            y_label = (p.get("y_axis") or {}).get("label", "")
            if x_label or y_label:
                n_with_axes += 1
            for v in p.values():
                if isinstance(v, str):
                    total_chars += len(v)

    try:
        ents = json.loads(entities_json_str) if entities_json_str else []
        if not isinstance(ents, list):
            ents = []
    except Exception:
        ents = []

    n_entities = len(ents)
    unique_texts: set[str] = set()
    type_counts: dict[str, int] = {}
    for e in ents:
        if not isinstance(e, dict):
            continue
        t = e.get("text", "")
        if t:
            unique_texts.add(t.lower())
        lbl = e.get("label", "UNKNOWN")
        type_counts[lbl] = type_counts.get(lbl, 0) + 1

    text_chars = run_info.get("stats", {}).get("text_chars", 0)
    density = round(n_entities / text_chars * 1000, 2) if text_chars else 0.0

    return {
        "pmc_id": run_info.get("pmc_id", ""),
        "timestamp": run_info.get("timestamp", ""),
        "vlm_model": run_info.get("vlm_model", ""),
        "ner_model_name": run_info.get("ner_model_name", ""),
        "deplot_used": deplot_used,
        "latency": run_info.get("latency", {}),
        "vlm": {
            "n_figures": n_figs,
            "n_compound": n_compound,
            "n_with_title": n_with_title,
            "n_with_data_points": n_with_data,
            "n_with_axes": n_with_axes,
            "total_panels": total_panels,
            "total_data_points": total_data_pts,
            "avg_chars_per_fig": round(total_chars / n_figs, 1) if n_figs else 0.0,
        },
        "ner": {
            "n_entities": n_entities,
            "n_unique": len(unique_texts),
            "type_counts": type_counts,
            "density_per_1k": density,
            # Store normalized entity text set for cross-run diff (max 500 for memory)
            "entity_texts": sorted(unique_texts)[:500],
        },
    }


def _format_run_summary_md(summary: dict | None, slot_label: str = "Current Run") -> str:
    if not summary:
        return "_No run data available._"

    vlm = summary.get("vlm", {})
    ner = summary.get("ner", {})
    lat = summary.get("latency", {})
    deplot_tag = "  ·  **DePlot ✓**" if summary.get("deplot_used") else ""
    model_short = summary.get("vlm_model", "").split("/")[-1] or "—"

    n_figs = vlm.get("n_figures", 0)
    n_panels = vlm.get("total_panels", 0)

    vlm_rows = "\n".join([
        f"| Figures analyzed | {n_figs} |",
        f"| Compound figures | {vlm.get('n_compound', 0)} |",
        f"| Total panels | {n_panels} |",
        f"| Panels with title | {vlm.get('n_with_title', 0)}/{n_panels} |",
        f"| Panels with data points | {vlm.get('n_with_data_points', 0)}/{n_panels} |",
        f"| Panels with axis labels | {vlm.get('n_with_axes', 0)}/{n_panels} |",
        f"| Total data points extracted | {vlm.get('total_data_points', 0):,} |",
        f"| Avg chars per figure | {int(vlm.get('avg_chars_per_fig', 0)):,} |",
        f"| VLM time | {lat.get('vlm_s', 0):.2f}s |",
    ])

    type_counts = ner.get("type_counts") or {}
    top_types = sorted(type_counts.items(), key=lambda x: -x[1])[:10]
    ner_rows = "\n".join([
        f"| Total entities | {ner.get('n_entities', 0):,} |",
        f"| Unique entities | {ner.get('n_unique', 0):,} |",
        f"| Entity density | {ner.get('density_per_1k', 0):.2f} / 1k chars |",
        f"| NER time | {lat.get('ner_s', 0):.2f}s |",
    ])
    top_types_md = "  ·  ".join(f"`{lbl}` {cnt}" for lbl, cnt in top_types)

    return (
        f"#### {slot_label}\n"
        f"**PMC:** `{summary.get('pmc_id', '—')}`  ·  "
        f"**VLM:** `{model_short}`{deplot_tag}  ·  "
        f"**NER:** {summary.get('ner_model_name', '—')}\n\n"
        f"**VLM Output**\n\n| Metric | Value |\n|---|---|\n{vlm_rows}\n\n"
        f"**NER Output**\n\n| Metric | Value |\n|---|---|\n{ner_rows}"
        + (f"\n\n**Top entity types:** {top_types_md}" if top_types_md else "")
    )


def _format_comparison_md(a: dict | None, b: dict | None) -> str:
    if not a or not b:
        return ""

    def _short(s: dict) -> str:
        m = s.get("vlm_model", "").split("/")[-1] or "?"
        return f"{m} {'+ DePlot' if s.get('deplot_used') else '(base)'}"

    tag_a, tag_b = _short(a), _short(b)
    vlm_a, vlm_b = a.get("vlm", {}), b.get("vlm", {})
    ner_a, ner_b = a.get("ner", {}), b.get("ner", {})
    lat_a, lat_b = a.get("latency", {}), b.get("latency", {})

    def _row(label: str, va, vb, fmt: str = ".1f", higher_better: bool = True) -> str:
        try:
            fa, fb = float(va), float(vb)
            diff = fb - fa
            sign = "+" if diff >= 0 else ""
            arrow = ""
            if diff > 0:
                arrow = " ↑" if higher_better else " ↓"
            elif diff < 0:
                arrow = " ↓" if higher_better else " ↑"
            if fmt == "d":
                sa = f"{int(fa):,}"
                sb = f"{int(fb):,}"
                sd = f"{sign}{int(diff):,}{arrow}"
            else:
                sa = f"{fa:{fmt}}"
                sb = f"{fb:{fmt}}"
                sd = f"{sign}{diff:{fmt}}{arrow}"
        except Exception:
            sa, sb, sd = str(va), str(vb), ""
        return f"| {label} | {sa} | {sb} | {sd} |"

    rows = "\n".join([
        _row("Figures analyzed", vlm_a.get("n_figures", 0), vlm_b.get("n_figures", 0), "d"),
        _row("Compound figures", vlm_a.get("n_compound", 0), vlm_b.get("n_compound", 0), "d"),
        _row("Total panels", vlm_a.get("total_panels", 0), vlm_b.get("total_panels", 0), "d"),
        _row("Panels with title", vlm_a.get("n_with_title", 0), vlm_b.get("n_with_title", 0), "d"),
        _row("Panels with data points", vlm_a.get("n_with_data_points", 0), vlm_b.get("n_with_data_points", 0), "d"),
        _row("Total data points", vlm_a.get("total_data_points", 0), vlm_b.get("total_data_points", 0), "d"),
        _row("Avg chars / fig", vlm_a.get("avg_chars_per_fig", 0), vlm_b.get("avg_chars_per_fig", 0), ".1f"),
        _row("VLM time (s)", lat_a.get("vlm_s", 0), lat_b.get("vlm_s", 0), ".2f", False),
        _row("Total entities", ner_a.get("n_entities", 0), ner_b.get("n_entities", 0), "d"),
        _row("Unique entities", ner_a.get("n_unique", 0), ner_b.get("n_unique", 0), "d"),
        _row("Entity density / 1k", ner_a.get("density_per_1k", 0), ner_b.get("density_per_1k", 0), ".2f"),
        _row("NER time (s)", lat_a.get("ner_s", 0), lat_b.get("ner_s", 0), ".2f", False),
        _row("Total time (s)", lat_a.get("total_s", 0), lat_b.get("total_s", 0), ".2f", False),
    ])

    # Entity diff
    texts_a = set(ner_a.get("entity_texts") or [])
    texts_b = set(ner_b.get("entity_texts") or [])
    only_in_a = sorted(texts_a - texts_b)[:30]
    only_in_b = sorted(texts_b - texts_a)[:30]
    diff_md = ""
    if only_in_a or only_in_b:
        diff_md = "\n\n**Entity Diff**\n\n"
        if only_in_a:
            diff_md += f"Only in A ({len(only_in_a)} shown): " + "  ".join(f"`{t}`" for t in only_in_a) + "\n\n"
        if only_in_b:
            diff_md += f"Only in B ({len(only_in_b)} shown): " + "  ".join(f"`{t}`" for t in only_in_b)

    return (
        f"### A vs B Comparison\n\n"
        f"**A:** {tag_a}  ·  **B:** {tag_b}  ·  PMC: `{a.get('pmc_id', '')}`\n\n"
        f"| Metric | A | B | Δ (B − A) |\n|---|---|---|---|\n"
        + rows
        + diff_md
    )


def process_fetch(
    url_input: str,
    ingest_method: str,
    vlm_model_id: str,
    ner_model_key: str,
    gliner_types_str: str,
):
    """Phase 1: fetch paper from PMC ID, PMC URL, or any paper URL.

    Outputs: pdf_out (metadata header + article HTML), gallery_out, _run_state
    """
    raw = url_input.strip()

    if not raw:
        yield gr.update(), gr.update(), gr.update(), _pipeline_html(fetch="running")
        try:
            raw = fetch_random_pmc_id()
            gr.Info(f"Selected {raw}")
        except Exception as exc:
            yield f"<p style='color:#ef4444;padding:16px;'>Failed to fetch random PMC ID: {exc}</p>", "", None, _pipeline_html(fetch="error")
            return

    _fetch_result: list = [None]
    _fetch_error: list = [None]
    _use_jats = ingest_method == "JATS XML (full text)"

    def _do_fetch():
        try:
            _fetch_result[0] = fetch_url(raw, use_jats=_use_jats)
        except Exception as exc:
            _fetch_error[0] = exc

    t0 = time.perf_counter()
    _fetch_thread = threading.Thread(target=_do_fetch, daemon=True)
    _fetch_thread.start()
    while _fetch_thread.is_alive():
        yield gr.update(), gr.update(), gr.update(), _pipeline_html(fetch="running", fetch_info=f"{time.perf_counter() - t0:.1f}s")
        _fetch_thread.join(timeout=0.1)

    fetch_s = round(time.perf_counter() - t0, 2)
    if _fetch_error[0]:
        yield f"<pre style='color:#ef4444;padding:16px;white-space:pre-wrap;'>{_fetch_error[0]}</pre>", "", None, _pipeline_html(fetch="error", fetch_info=f"{fetch_s:.1f}s")
        return
    text, figures, figure_captions, metadata, pdf_bytes, article_html = _fetch_result[0]

    # Derive a slug for the run directory from the input
    pmc_id = metadata.get("pmid") or re.sub(r"[^\w]", "_", raw)[:40]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = _OUTPUT_ROOT / f"{pmc_id}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if pdf_bytes:
        pdf_path = run_dir / "paper.pdf"
        pdf_path.write_bytes(pdf_bytes)
        print(f"[app] PDF saved: {pdf_path} ({pdf_path.stat().st_size:,}B)")
    else:
        html_path = run_dir / "article.html"
        html_path.write_text(article_html, encoding="utf-8")
        print(f"[app] Article HTML saved: {html_path}")

    article_out_html = _format_metadata_html(raw, metadata) + article_html
    figs_html = _figures_html(figures, captions=figure_captions)

    state = {
        "pmc_id": pmc_id,
        "timestamp": timestamp,
        "text": text,
        "figures": figures,
        "figure_captions": figure_captions,
        "metadata": metadata,
        "run_dir": str(run_dir),
        "ingest_method": ingest_method,
        "vlm_model": vlm_model_id,
        "ner_model": ner_model_key,
        "ner_model_name": NER_MODEL_OPTIONS.get(ner_model_key, ner_model_key),
        "latency": {"fetch_s": fetch_s},
        "stats": {
            "text_chars": len(text),
            "figure_count": len(figures),
            "pdf_available": pdf_bytes is not None,
            "html_available": article_html is not None and len(article_html) > 0,
        },
    }

    yield article_out_html, figs_html, state, _pipeline_html(fetch="done", fetch_info=f"{fetch_s:.1f}s")


def process_models(
    state: dict | None,
    ner_model_key: str,
    gliner_types_str: str,
    vlm_model_id: str,
    deplot_enabled: bool = False,
):
    """Phase 2: run VLM and NER on the fetched data.

    Outputs: figures_out, entities_out, ner_highlighted, run_info_out,
             task1_tab, task2_tab, pipeline_status, run_summary_out, run_summary_state,
             entity_table_out, _entities_state
    gallery_out / pdf_out are NOT outputs here so they show without loading overlay.
    """
    _LABEL_T1 = "Task 1 - Image Information"
    _LABEL_T2 = "Task 2 — Entities"

    if state is None:
        yield "", "", [], "", gr.update(label=_LABEL_T1), gr.update(label=_LABEL_T2), gr.update(), gr.update(), None, gr.update(), None
        return

    text = state["text"]
    figures = state["figures"]
    figure_captions = state.get("figure_captions", [])
    metadata = state["metadata"]
    pmc_id = state["pmc_id"]
    run_dir = Path(state["run_dir"])
    latency = state["latency"]

    n_figs = len(figures)
    t_run = time.perf_counter()

    def _total() -> str:
        return f"{time.perf_counter() - t_run:.1f}s"

    yield "", "", [], "", gr.update(label=f"↻ {_LABEL_T1}"), gr.update(label=_LABEL_T2), _pipeline_html(
        fetch="done", vlm="running", vlm_info=f"0/{n_figs}", total_info=_total(),
    ), gr.update(), None, gr.update(), None

    t0 = time.perf_counter()
    figures_result: list[dict] = []
    for i, img in enumerate(figures):
        _partial: list = [None]
        _vlm_err: list = [None]

        def _do_vlm(_img=img, _i=i):
            try:
                _partial[0] = analyze_figures([_img], fig_offset=_i, total_figs=n_figs)
            except Exception as exc:
                _vlm_err[0] = exc

        _vlm_thread = threading.Thread(target=_do_vlm, daemon=True)
        _vlm_thread.start()
        while _vlm_thread.is_alive():
            yield gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), _pipeline_html(
                fetch="done", vlm="running",
                vlm_info=f"fig {i + 1}/{n_figs} · {time.perf_counter() - t0:.1f}s",
                total_info=_total(),
            ), gr.update(), None, gr.update(), None
            _vlm_thread.join(timeout=0.1)

        if _vlm_err[0]:
            figures_result.append({"figure_id": f"fig{i + 1}", "error": str(_vlm_err[0])})
        else:
            for entry in _partial[0]:
                entry["figure_id"] = f"fig{i + 1}"
            figures_result.extend(_partial[0])

    figures_json = json.dumps(figures_result, indent=2)
    latency["vlm_s"] = round(time.perf_counter() - t0, 2)
    vlm_s = latency["vlm_s"]

    yield figures_json, "", [], "", gr.update(label=_LABEL_T1), gr.update(label=f"↻ {_LABEL_T2}"), _pipeline_html(
        fetch="done", vlm="done", ner="running",
        vlm_info=f"{n_figs} fig{'s' if n_figs != 1 else ''} · {vlm_s:.1f}s",
        total_info=_total(),
    ), gr.update(), None, gr.update(), None

    gliner_types = (
        [t.strip() for t in gliner_types_str.split(",") if t.strip()]
        if ner_model_key == "gliner"
        else None
    )
    _ner_result: list = [None]
    _ner_error: list = [None]

    def _do_ner():
        try:
            _ner_result[0] = extract_entities(text, gliner_entity_types=gliner_types)
        except Exception as exc:
            _ner_error[0] = exc

    t_ner = time.perf_counter()
    _ner_thread = threading.Thread(target=_do_ner, daemon=True)
    _ner_thread.start()
    while _ner_thread.is_alive():
        yield gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), _pipeline_html(
            fetch="done", vlm="done", ner="running",
            vlm_info=f"{n_figs} fig{'s' if n_figs != 1 else ''} · {vlm_s:.1f}s",
            ner_info=f"{time.perf_counter() - t_ner:.1f}s",
            total_info=_total(),
        ), gr.update(), None, gr.update(), None
        _ner_thread.join(timeout=0.1)

    ner_s = round(time.perf_counter() - t_ner, 2)
    latency["ner_s"] = ner_s
    latency["total_s"] = round(latency["fetch_s"] + vlm_s + ner_s, 2)

    if _ner_error[0]:
        entities_result: list = []
        entities_json = json.dumps({"error": str(_ner_error[0]), "stage": "task2_ner"}, indent=2)
    else:
        entities_result = _ner_result[0]
        entities_json = json.dumps(entities_result, indent=2)

    state["stats"]["entity_count"] = len(entities_result)
    highlighted = _entities_to_highlighted(text, entities_result)

    run_info = {
        "pmc_id": pmc_id,
        "timestamp": state["timestamp"],
        "ingest_method": state["ingest_method"],
        "vlm_model": vlm_model_id,
        "ner_model": ner_model_key,
        "ner_model_name": NER_MODEL_OPTIONS.get(ner_model_key, ner_model_key),
        "latency": latency,
        "stats": state["stats"],
    }

    run_summary = _compute_run_summary(figures_json, entities_json, run_info, deplot_used=deplot_enabled)
    prompt_txt = get_last_prompt()

    save_error: str | None = None
    try:
        _save_run(run_dir, pmc_id, figures, metadata, figures_json, entities_json, run_info, run_summary, prompt_txt)
        run_info["saved_to"] = str(run_dir.resolve())
    except Exception as exc:
        save_error = str(exc)
        print(f"[output] Failed to save run: {exc}")

    yield (
        figures_json,
        entities_json,
        highlighted,
        _format_run_info(run_info, save_error),
        gr.update(label=_LABEL_T1),
        gr.update(label=_LABEL_T2),
        _pipeline_html(
            fetch="done", vlm="done", ner="done", complete="done",
            vlm_info=f"{n_figs} fig{'s' if n_figs != 1 else ''} · {vlm_s:.1f}s",
            ner_info=f"{ner_s:.1f}s",
            total_info=f"total {latency['total_s']:.1f}s",
        ),
        _format_run_summary_md(run_summary),
        run_summary,
        _entities_table_html(entities_result),
        entities_json,
    )


_SLOW_NER_MODELS = {"scispacy_umls", "scispacy_triple"}
_SLOW_NER_WARNING = (
    "\n\n> ⚠️ **Slow load:** this model loads the UMLS entity linker (~3 GB RAM) "
    "on first use. Cold start takes 30–120 s on hosted instances. "
    "Subsequent runs in the same session are fast."
)


def _update_ner_info(model_key: str) -> tuple[str, dict]:
    info = NER_MODEL_INFO.get(model_key, {})
    desc = info.get("description", "")
    labels = info.get("labels", [])
    size = info.get("size", "")
    trained_on = info.get("trained_on", "")
    labels_str = ", ".join(f"`{l}`" for l in labels)
    warning = _SLOW_NER_WARNING if model_key in _SLOW_NER_MODELS else ""
    md = (
        f"**{NER_MODEL_OPTIONS.get(model_key, model_key)}**\n\n"
        f"{desc}\n\n"
        f"**Labels:** {labels_str}\n\n"
        f"**Size:** {size} | **Trained on:** {trained_on}"
        f"{warning}"
    )
    is_gliner = model_key == "gliner"
    return md, gr.update(visible=is_gliner)


_ner_keys = list(NER_MODEL_OPTIONS.keys())

# Build eval VLM choices: HF + all API models + Ollama if running
_eval_vlm_choices = list(_HF_VLM_CHOICES)
_eval_vlm_choices += _API_OPENAI_CHOICES + _API_ANTHROPIC_CHOICES + _API_GOOGLE_CHOICES
if _ollama_available:
    _eval_vlm_choices += _get_ollama_vision_models()


# ---------------------------------------------------------------------------
# Eval runner (for Evaluation tab)
# ---------------------------------------------------------------------------


def run_eval_suite(
    papers_text: str,
    vlm_models: list,
    ner_models: list,
    ingest_method: str,
):
    """Run full evaluation suite and yield progressive status + final results."""
    _N_OUTPUTS = 14

    def _idle():
        return tuple(gr.update() for _ in range(_N_OUTPUTS))

    pmc_ids = [p.strip() for p in papers_text.strip().splitlines() if p.strip()]
    if not pmc_ids:
        yield (
            gr.update(value="<p style='color:#ef4444;'>No paper IDs provided.</p>"),
            *[gr.update() for _ in range(_N_OUTPUTS - 1)],
        )
        return
    if not vlm_models and not ner_models:
        yield (
            gr.update(value="<p style='color:#ef4444;'>Select at least one VLM or NER model.</p>"),
            *[gr.update() for _ in range(_N_OUTPUTS - 1)],
        )
        return

    use_jats = ingest_method == "JATS XML (full text)"

    _eval_result: list = [None]
    _eval_error: list = [None]
    _eval_progress: list = ["Starting evaluation..."]

    def _do_eval():
        def _cb(msg):
            _eval_progress[0] = msg
        try:
            _eval_result[0] = run_eval(
                pmc_ids,
                vlm_models or [],
                ner_models or [],
                use_jats=use_jats,
                progress_callback=_cb,
            )
        except Exception as exc:
            _eval_error[0] = exc

    t = threading.Thread(target=_do_eval, daemon=True)
    t.start()
    while t.is_alive():
        msg = _eval_progress[0]
        yield (
            gr.update(value=f'<p style="font-size:13px;color:#f59e0b;">↻ {msg}</p>'),
            *[gr.update() for _ in range(_N_OUTPUTS - 1)],
        )
        t.join(timeout=0.1)

    if _eval_error[0]:
        yield (
            gr.update(value=f'<p style="color:#ef4444;">Evaluation failed: {_eval_error[0]}</p>'),
            *[gr.update() for _ in range(_N_OUTPUTS - 1)],
        )
        return

    result = _eval_result[0]
    report_md = generate_md_report(result)

    _EVAL_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    run_path = save_eval_run(result, report_md, _EVAL_OUTPUT_ROOT)
    report_file_path = str(run_path / "report.md")

    charts = get_all_charts(result)

    # NER type distribution: show first NER model by default
    ner_dist_fig = None
    ner_keys_run = result.get("ner_model_keys", [])
    ner_dist_choices = [(NER_MODEL_OPTIONS.get(k, k), k) for k in ner_keys_run]
    ner_dist_first = ner_keys_run[0] if ner_keys_run else None
    if ner_dist_first:
        ner_dist_fig = chart_ner_type_distribution(result, ner_dist_first)

    result_json = json.dumps(result, indent=2, default=str)

    yield (
        gr.update(value=f'<p style="color:#22c55e;">Evaluation complete — saved to {run_path}</p>'),
        gr.update(value=get_vlm_summary_md(result)),
        gr.update(value=get_ner_summary_md(result)),
        charts.get("vlm_latency"),
        charts.get("vlm_richness_radar"),
        charts.get("vlm_panels"),
        charts.get("ner_latency"),
        charts.get("ner_entity_counts"),
        charts.get("vlm_latency_vs_richness"),
        ner_dist_fig,
        gr.update(choices=ner_dist_choices, value=ner_dist_first, visible=bool(ner_dist_choices)),
        gr.update(value=report_md),
        gr.update(value=report_file_path, visible=True),
        gr.update(value=result_json),
    )


def _eval_update_ner_dist(eval_json_str: str | None, model_key: str | None):
    if not eval_json_str or not model_key:
        return gr.update()
    try:
        result = json.loads(eval_json_str)
    except Exception:
        return gr.update()
    return chart_ner_type_distribution(result, model_key)


with gr.Blocks(title="BioMed Paper Information Extractor") as demo:
    gr.Markdown("## 🔬 BioMed Paper Information Extractor\nEnd-to-end pipeline for automated biomedical literature analysis — figure digitization via VLM and named entity recognition via configurable NER models.")
    with gr.Tabs():
        # ===================================================================
        # Tab 1: Extraction
        # ===================================================================
        with gr.Tab("Extraction"):
            gr.Markdown("### Single Paper Analysis\nEnter a PMC ID, PMC URL, or any paper URL. Leave blank for a random open-access paper.")

            with gr.Row():
                with gr.Column(scale=4, min_width=0):
                    pmc_input = gr.Textbox(
                        label="PMC ID / URL",
                        placeholder="PMC7614754  or  PMC URL  or  any PDF/article URL",
                        info="Leave blank for a random open-access paper.",
                    )
                    with gr.Row(equal_height=True):
                        gr.HTML('<div style="display:flex;align-items:center;height:100%;font-size:12px;color:#6b7280;white-space:nowrap;padding-right:4px;">Example papers</div>')
                        ex_btn_1 = gr.Button("PMC7614754", size="sm", variant="secondary", min_width=90)
                        ex_btn_2 = gr.Button("PMC6267067", size="sm", variant="secondary", min_width=90)
                        ex_btn_3 = gr.Button("PMC8563518", size="sm", variant="secondary", min_width=90)
                ingest_radio = gr.Radio(
                    choices=["HTML", "JATS XML (full text)"],
                    value="HTML",
                    label="Text Method",
                    info="PMC only — HTML: scrape  |  JATS: structured XML",
                    scale=3,
                )
                with gr.Column(scale=2, min_width=160):
                    run_hint = gr.HTML(value=_RUN_HINT_PENDING)
                    run_btn = gr.Button("Analyze", variant="primary", interactive=False)
                    readiness_html = gr.HTML(value=_render_readiness("Not loaded", "Not loaded"))

            gr.HTML('<hr style="border:none;border-top:1px solid #374151;margin:4px 0 8px;">')

            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Row():
                        vlm_provider = gr.Radio(
                            choices=_provider_choices,
                            value="HuggingFace",
                            label="VLM Provider",
                            scale=3,
                        )
                        ollama_indicator = gr.HTML(
                            value=_ollama_status_html(_ollama_available),
                        )
                    with gr.Row(visible=False) as ollama_settings_row:
                        ollama_timeout_input = gr.Number(
                            label="Timeout (s)",
                            value=180,
                            minimum=10,
                            maximum=600,
                            step=10,
                            scale=1,
                            info="Per-figure request timeout",
                        )
                        ollama_retries_input = gr.Number(
                            label="Retries on timeout",
                            value=1,
                            minimum=0,
                            maximum=5,
                            step=1,
                            scale=1,
                            info="Extra attempts after first timeout",
                        )
                    with gr.Row(visible=False) as api_provider_row:
                        api_provider_dropdown = gr.Dropdown(
                            choices=_API_PROVIDER_CHOICES,
                            value="OpenAI",
                            label="API Provider",
                            scale=2,
                        )
                    vlm_dropdown = gr.Dropdown(
                        choices=_HF_VLM_CHOICES,
                        value=_HF_VLM_CHOICES[0][1],
                        label="Vision Model",
                    )
                    vlm_reload_warn = gr.HTML(value="", visible=False)
                    api_key_input = gr.Textbox(
                        label="API Key",
                        placeholder="Paste your API key here...",
                        type="password",
                        visible=False,
                    )
                    with gr.Row():
                        vlm_load_btn = gr.Button("Load / Reload VLM", variant="secondary", scale=3)
                        deplot_toggle = gr.Checkbox(
                            label="DePlot context",
                            value=False,
                            info="Inject chart-extracted table into VLM prompt",
                            scale=2,
                        )
                    vlm_status = gr.HTML(value=_status_html("VLM", "Not loaded"))
                with gr.Column(scale=2):
                    ner_dropdown = gr.Dropdown(
                        choices=_NER_CHOICES,
                        value=_ner_keys[0],
                        label="NER Model",
                    )
                    ner_reload_warn = gr.HTML(value="", visible=False)
                    ner_status = gr.HTML(value=_status_html("NER", "Not loaded"))
                    ner_load_btn = gr.Button("Load / Reload NER", variant="secondary")
                    ner_info_md = gr.Markdown(value="")
                    gliner_types_input = gr.Textbox(
                        label="GLiNER Entity Types (comma-separated)",
                        value=", ".join(GLINER_DEFAULT_ENTITY_TYPES),
                        visible=False,
                        lines=2,
                    )

            gr.HTML('<hr style="border:none;border-top:1px solid #374151;margin:8px 0 4px;">')

            _run_state = gr.State(None)
            _vlm_loaded = gr.State(False)
            _ner_loaded = gr.State(False)
            _run_summary_state = gr.State(None)   # current run summary dict
            _run_a_state = gr.State(None)          # saved slot A
            _run_b_state = gr.State(None)          # saved slot B
            _entities_state = gr.State(None)       # latest entities JSON string for viz
            _vlm_ready_text = gr.State("Not loaded")  # text shown in combined readiness row
            _ner_ready_text = gr.State("Not loaded")

            pipeline_status = gr.HTML(value=_PIPELINE_IDLE)

            with gr.Tabs():
                with gr.Tab("Article"):
                    pdf_out = gr.HTML()
                with gr.Tab("Figures"):
                    gallery_out = gr.HTML()
                with gr.Tab("Task 1 - Image Information") as task1_tab:
                    figures_out = gr.Code(language="json", label="Figure JSON")
                with gr.Tab("Task 2 — Entities") as task2_tab:
                    with gr.Row():
                        ner_highlighted = gr.HighlightedText(
                            label="Entity Annotations",
                            combine_adjacent=False,
                            show_legend=True,
                            scale=3,
                        )
                        entities_out = gr.Code(language="json", label="Entity JSON", scale=2)
                    entity_table_out = gr.HTML(value="")
                with gr.Tab("Run Summary"):
                    gr.Markdown(
                        "Stats for the current run. Use **Set as A / Set as B** to compare two runs "
                        "(e.g. with and without DePlot). Results are also saved to `output/extraction/<run>/run_summary.json`.",
                        elem_classes=[],
                    )
                    run_summary_out = gr.Markdown(value="_Run an extraction to see stats._")
                    with gr.Row():
                        save_as_a_btn = gr.Button("Set as Run A", variant="secondary", scale=1)
                        save_as_b_btn = gr.Button("Set as Run B", variant="secondary", scale=1)
                        clear_cmp_btn = gr.Button("Clear Comparison", variant="stop", scale=1)
                    with gr.Row(visible=False) as comparison_row:
                        with gr.Column(scale=1):
                            comparison_col_a = gr.Markdown(value="")
                        with gr.Column(scale=1):
                            comparison_col_b = gr.Markdown(value="")
                    comparison_table_out = gr.Markdown(value="")
                with gr.Tab("Visualization"):
                    gr.Markdown(
                        "Entity co-occurrence network and word cloud for the current run. "
                        "Click **Build Visualizations** after an extraction completes."
                    )
                    viz_build_btn = gr.Button("Build Visualizations", variant="secondary")
                    viz_graph_out = gr.Plot(label="Entity Co-occurrence Network")
                    viz_cloud_out = gr.Image(label="Entity Word Cloud", type="pil")
                with gr.Tab("Run Info"):
                    run_info_out = gr.Markdown(value="")

            vlm_provider.change(
                _update_vlm_choices,
                inputs=[vlm_provider],
                outputs=[vlm_dropdown, ollama_indicator, api_provider_row, api_key_input, ollama_settings_row],
            )
            ollama_timeout_input.change(set_ollama_timeout, inputs=[ollama_timeout_input], outputs=[])
            ollama_retries_input.change(set_ollama_retries, inputs=[ollama_retries_input], outputs=[])
            api_provider_dropdown.change(
                _update_api_model_choices,
                inputs=[api_provider_dropdown],
                outputs=[vlm_dropdown],
            )
            deplot_toggle.change(toggle_deplot, inputs=[deplot_toggle], outputs=[])

            ner_dropdown.change(_update_ner_info, inputs=[ner_dropdown], outputs=[ner_info_md, gliner_types_input])
            demo.load(_update_ner_info, inputs=[ner_dropdown], outputs=[ner_info_md, gliner_types_input])

            _reload_warn_html = (
                '<span style="color:#f87171;font-size:11px;font-weight:500;">'
                '⚠ Selection changed — click Load / Reload to apply'
                '</span>'
            )
            vlm_dropdown.change(lambda: gr.update(value=_reload_warn_html, visible=True), outputs=[vlm_reload_warn])
            ner_dropdown.change(lambda: gr.update(value=_reload_warn_html, visible=True), outputs=[ner_reload_warn])

            vlm_load_btn.click(
                load_vlm_model,
                inputs=[vlm_dropdown, api_key_input],
                outputs=[vlm_status, _vlm_loaded, run_btn, _vlm_ready_text, vlm_load_btn],
            ).then(_update_run_btn, inputs=[_vlm_loaded, _ner_loaded], outputs=[run_btn, run_hint]
            ).then(_render_readiness, inputs=[_vlm_ready_text, _ner_ready_text], outputs=[readiness_html]
            ).then(lambda: gr.update(value="", visible=False), outputs=[vlm_reload_warn])
            ner_load_btn.click(
                load_ner_model,
                inputs=[ner_dropdown],
                outputs=[ner_status, _ner_loaded, run_btn, _ner_ready_text, ner_load_btn],
            ).then(_update_run_btn, inputs=[_vlm_loaded, _ner_loaded], outputs=[run_btn, run_hint]
            ).then(_render_readiness, inputs=[_vlm_ready_text, _ner_ready_text], outputs=[readiness_html]
            ).then(lambda: gr.update(value="", visible=False), outputs=[ner_reload_warn])

            _fetch_inputs = [pmc_input, ingest_radio, vlm_dropdown, ner_dropdown, gliner_types_input]
            _fetch_outputs = [pdf_out, gallery_out, _run_state, pipeline_status]

            _models_inputs = [_run_state, ner_dropdown, gliner_types_input, vlm_dropdown, deplot_toggle]
            _models_outputs = [
                figures_out, entities_out, ner_highlighted, run_info_out,
                task1_tab, task2_tab, pipeline_status,
                run_summary_out, _run_summary_state,
                entity_table_out, _entities_state,
            ]

            def _set_slot_a(summary, run_b):
                a_md = _format_run_summary_md(summary, "**Run A**")
                b_md = _format_run_summary_md(run_b, "**Run B**") if run_b else gr.update()
                table_md = _format_comparison_md(summary, run_b) if run_b else ""
                row_visible = gr.update(visible=True)
                return summary, row_visible, a_md, b_md, table_md

            def _set_slot_b(summary, run_a):
                b_md = _format_run_summary_md(summary, "**Run B**")
                a_md = _format_run_summary_md(run_a, "**Run A**") if run_a else gr.update()
                table_md = _format_comparison_md(run_a, summary) if run_a else ""
                row_visible = gr.update(visible=True)
                return summary, row_visible, a_md, b_md, table_md

            _slot_outputs = [_run_a_state, comparison_row, comparison_col_a, comparison_col_b, comparison_table_out]

            save_as_a_btn.click(
                _set_slot_a,
                inputs=[_run_summary_state, _run_b_state],
                outputs=_slot_outputs,
            )
            save_as_b_btn.click(
                _set_slot_b,
                inputs=[_run_summary_state, _run_a_state],
                outputs=[_run_b_state, comparison_row, comparison_col_a, comparison_col_b, comparison_table_out],
            )
            clear_cmp_btn.click(
                lambda: (None, None, gr.update(visible=False), "", "", ""),
                outputs=[_run_a_state, _run_b_state, comparison_row, comparison_col_a, comparison_col_b, comparison_table_out],
            )

            (
                run_btn.click(process_fetch, inputs=_fetch_inputs, outputs=_fetch_outputs, show_progress="hidden")
                .then(process_models, inputs=_models_inputs, outputs=_models_outputs, show_progress="minimal")
            )
            (
                pmc_input.submit(process_fetch, inputs=_fetch_inputs, outputs=_fetch_outputs, show_progress="hidden")
                .then(process_models, inputs=_models_inputs, outputs=_models_outputs, show_progress="minimal")
            )

            ex_btn_1.click(lambda: "PMC7614754", outputs=[pmc_input])
            ex_btn_2.click(lambda: "PMC6267067", outputs=[pmc_input])
            ex_btn_3.click(lambda: "PMC8563518", outputs=[pmc_input])

            viz_build_btn.click(
                _build_viz,
                inputs=[_entities_state, _run_state],
                outputs=[viz_graph_out, viz_cloud_out],
            )

        # ===================================================================
        # Tab 2: Evaluation
        # ===================================================================
        with gr.Tab("Evaluation"):
            gr.Markdown("### Evaluation Suite\nBenchmark VLM and NER models across multiple papers. Results are saved to `output/evaluation/`.")

            eval_papers_input = gr.Textbox(
                label="Paper IDs (one per line — PMC ID or URL)",
                value="\n".join(DEFAULT_PMC_IDS),
                lines=4,
            )

            with gr.Row():
                with gr.Column(scale=1):
                    eval_vlm_checkbox = gr.CheckboxGroup(
                        label="VLM Models",
                        choices=_eval_vlm_choices,
                        value=[_eval_vlm_choices[0][1]] if _eval_vlm_choices else [],
                    )
                with gr.Column(scale=1):
                    eval_ner_checkbox = gr.CheckboxGroup(
                        label="NER Models",
                        choices=_NER_CHOICES,
                        value=[_ner_keys[0]],
                    )

            eval_ingest_radio = gr.Radio(
                choices=["HTML", "JATS XML (full text)"],
                value="HTML",
                label="Text Method",
                info="PMC only — HTML: scrape  |  JATS: structured XML",
            )

            eval_run_btn = gr.Button("Run Evaluation", variant="primary")
            eval_pipeline = gr.HTML(value="")

            with gr.Tabs():
                with gr.Tab("Summary"):
                    eval_vlm_summary = gr.Markdown(value="")
                    eval_ner_summary = gr.Markdown(value="")

                with gr.Tab("Charts"):
                    with gr.Row():
                        with gr.Column():
                            eval_chart_vlm_latency = gr.Plot(label="VLM Latency per Figure")
                        with gr.Column():
                            eval_chart_vlm_richness = gr.Plot(label="VLM Output Richness (radar)")
                    with gr.Row():
                        with gr.Column():
                            eval_chart_vlm_panels = gr.Plot(label="Panel Detection")
                        with gr.Column():
                            eval_chart_vlm_scatter = gr.Plot(label="Latency vs Richness")
                    with gr.Row():
                        with gr.Column():
                            eval_chart_ner_latency = gr.Plot(label="NER Latency")
                        with gr.Column():
                            eval_chart_ner_counts = gr.Plot(label="Entity Counts")
                    eval_ner_dist_model = gr.Dropdown(
                        label="NER Type Distribution — select model",
                        choices=[],
                        visible=False,
                    )
                    eval_chart_ner_dist = gr.Plot(label="NER Type Distribution")

                with gr.Tab("Report"):
                    eval_report_md = gr.Markdown(value="")
                    eval_report_file = gr.File(label="Download Report", visible=False)

                with gr.Tab("Raw JSON"):
                    eval_raw_json = gr.Code(language="json", label="Evaluation JSON")

            _eval_outputs = [
                eval_pipeline,
                eval_vlm_summary,
                eval_ner_summary,
                eval_chart_vlm_latency,
                eval_chart_vlm_richness,
                eval_chart_vlm_panels,
                eval_chart_ner_latency,
                eval_chart_ner_counts,
                eval_chart_vlm_scatter,
                eval_chart_ner_dist,
                eval_ner_dist_model,
                eval_report_md,
                eval_report_file,
                eval_raw_json,
            ]

            eval_run_btn.click(
                run_eval_suite,
                inputs=[eval_papers_input, eval_vlm_checkbox, eval_ner_checkbox, eval_ingest_radio],
                outputs=_eval_outputs,
                show_progress="minimal",
            )

            eval_ner_dist_model.change(
                _eval_update_ner_dist,
                inputs=[eval_raw_json, eval_ner_dist_model],
                outputs=[eval_chart_ner_dist],
            )

        # ===================================================================
        # Tab 3: About
        # ===================================================================
        with gr.Tab("About"):
            _readme_path = Path(__file__).with_name("README.md")
            _readme_raw = _readme_path.read_text(encoding="utf-8") if _readme_path.exists() else ""
            # Strip YAML frontmatter (--- ... ---)
            if _readme_raw.startswith("---"):
                _fm_end = _readme_raw.find("\n---", 3)
                _readme_md = _readme_raw[_fm_end + 4:].lstrip("\n") if _fm_end != -1 else _readme_raw
            else:
                _readme_md = _readme_raw
            gr.Markdown(_readme_md)


if __name__ == "__main__":
    demo.launch()
