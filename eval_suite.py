"""
eval_suite.py — Evaluation runner for BioMed Paper Information Extractor.

Benchmarks VLM and NER models across multiple papers, computing per-model
latency, output quality, and entity metrics. No Gradio imports.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from fetch_paper import fetch_url
from task1_figures import load_vlm, analyze_figures
from task2_ner import load_ner, extract_entities, NER_MODEL_OPTIONS


DEFAULT_PMC_IDS: list[str] = [
    "PMC7614754",
    "PMC6267067",
    "PMC8563518",
]

_EXPECTED_PANEL_FIELDS = ["title", "figure_type", "legend", "data_points"]
_EXPECTED_AXIS_SUBFIELDS = ["label"]


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def compute_vlm_metrics(figures_result: list[dict], elapsed_s: float, n_figs: int) -> dict:
    """Compute VLM quality metrics from figures_result list."""
    compound_count = 0
    total_panels = 0
    total_panel_entries = 0
    total_field_score = 0.0
    total_data_points = 0
    output_chars = len(json.dumps(figures_result))

    for fig in figures_result:
        if fig.get("error"):
            continue
        if fig.get("is_compound"):
            compound_count += 1
        panels = fig.get("panels", [])
        for panel in panels:
            total_panel_entries += 1
            # Field completeness: check expected fields
            score = 0
            n_fields = 0
            for f in _EXPECTED_PANEL_FIELDS:
                n_fields += 1
                val = panel.get(f)
                if val and val != "" and val != [] and val != {}:
                    score += 1
            # Axis subfields
            for axis in ("x_axis", "y_axis"):
                ax = panel.get(axis, {})
                if isinstance(ax, dict):
                    for sf in _EXPECTED_AXIS_SUBFIELDS:
                        n_fields += 1
                        if ax.get(sf):
                            score += 1
            total_field_score += score / n_fields if n_fields > 0 else 0
            # Data points
            dp = panel.get("data_points", [])
            total_data_points += len(dp) if isinstance(dp, list) else 0
        total_panels += len(panels)

    valid_figs = sum(1 for f in figures_result if not f.get("error"))
    avg_panel_count = total_panels / valid_figs if valid_figs > 0 else 0.0
    field_completeness = total_field_score / total_panel_entries if total_panel_entries > 0 else 0.0
    compound_rate = compound_count / valid_figs if valid_figs > 0 else 0.0

    return {
        "latency_s": round(elapsed_s, 3),
        "latency_per_fig_s": round(elapsed_s / n_figs, 3) if n_figs > 0 else None,
        "json_valid": True,
        "valid_figure_count": valid_figs,
        "compound_count": compound_count,
        "compound_rate": round(compound_rate, 3),
        "avg_panel_count": round(avg_panel_count, 2),
        "field_completeness": round(field_completeness, 3),
        "total_data_points": total_data_points,
        "output_chars": output_chars,
    }


def compute_ner_metrics(entities: list[dict], text: str, elapsed_s: float) -> dict:
    """Compute NER quality metrics from entity list."""
    entity_count = len(entities)
    word_count = len(text.split()) if text else 1

    unique_spans = set()
    type_distribution: dict[str, int] = {}
    total_span_words = 0

    for ent in entities:
        span_text = ent.get("text", "")
        unique_spans.add(span_text.lower())
        label = ent.get("label", "UNKNOWN")
        type_distribution[label] = type_distribution.get(label, 0) + 1
        total_span_words += len(span_text.split())

    latency_per_entity_ms = (elapsed_s * 1000 / entity_count) if entity_count > 0 else None
    entity_density = entity_count / (word_count / 1000) if word_count > 0 else 0.0
    avg_span_length = total_span_words / entity_count if entity_count > 0 else 0.0

    return {
        "latency_s": round(elapsed_s, 3),
        "entity_count": entity_count,
        "unique_entity_count": len(unique_spans),
        "latency_per_entity_ms": round(latency_per_entity_ms, 3) if latency_per_entity_ms is not None else None,
        "entity_density_per_1k": round(entity_density, 2),
        "type_distribution": type_distribution,
        "avg_span_length_words": round(avg_span_length, 2),
    }


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


def run_eval(
    pmc_ids: list[str],
    vlm_model_ids: list[str],
    ner_model_keys: list[str],
    use_jats: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> dict:
    """Run evaluation across papers and models. Returns structured result dict."""

    def _progress(msg: str) -> None:
        print(f"[eval] {msg}")
        if progress_callback:
            progress_callback(msg)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    papers_results = []

    for pmc_id in pmc_ids:
        _progress(f"Fetching {pmc_id}...")
        t_fetch = time.perf_counter()
        try:
            text, figures, metadata, pdf_bytes, article_html = fetch_url(pmc_id, use_jats=use_jats)
            fetch_s = round(time.perf_counter() - t_fetch, 3)
            _progress(f"Fetched {pmc_id}: {len(figures)} figures, {len(text):,} chars in {fetch_s:.1f}s")
        except Exception as exc:
            fetch_s = round(time.perf_counter() - t_fetch, 3)
            _progress(f"Failed to fetch {pmc_id}: {exc}")
            papers_results.append({
                "pmc_id": pmc_id,
                "error": str(exc),
                "fetch_s": fetch_s,
            })
            continue

        paper_entry: dict = {
            "pmc_id": pmc_id,
            "fetch_s": fetch_s,
            "n_figures": len(figures),
            "text_chars": len(text),
            "title": metadata.get("title", ""),
            "vlm_results": {},
            "ner_results": {},
        }

        # --- VLM evaluation ---
        for vlm_id in vlm_model_ids:
            _progress(f"  VLM {vlm_id} on {pmc_id} ({len(figures)} figs)...")
            try:
                load_vlm(vlm_id)
                t_vlm = time.perf_counter()
                figures_result: list[dict] = []
                for i, img in enumerate(figures):
                    _progress(f"    fig {i + 1}/{len(figures)}...")
                    partial = analyze_figures([img])
                    for entry in partial:
                        entry["figure_id"] = f"fig{i + 1}"
                    figures_result.extend(partial)
                vlm_s = time.perf_counter() - t_vlm
                metrics = compute_vlm_metrics(figures_result, vlm_s, len(figures))
                metrics["raw"] = figures_result
                paper_entry["vlm_results"][vlm_id] = metrics
                _progress(f"    Done: {metrics['valid_figure_count']} figs in {vlm_s:.1f}s")
            except Exception as exc:
                _progress(f"    VLM error: {exc}")
                paper_entry["vlm_results"][vlm_id] = {"error": str(exc)}

        # --- NER evaluation ---
        for ner_key in ner_model_keys:
            model_name = NER_MODEL_OPTIONS.get(ner_key, ner_key)
            _progress(f"  NER {model_name} on {pmc_id}...")
            try:
                load_ner(ner_key)
                t_ner = time.perf_counter()
                entities = extract_entities(text)
                ner_s = time.perf_counter() - t_ner
                metrics = compute_ner_metrics(entities, text, ner_s)
                paper_entry["ner_results"][ner_key] = metrics
                _progress(f"    Done: {metrics['entity_count']} entities in {ner_s:.1f}s")
            except Exception as exc:
                _progress(f"    NER error: {exc}")
                paper_entry["ner_results"][ner_key] = {"error": str(exc)}

        papers_results.append(paper_entry)

    return {
        "timestamp": timestamp,
        "pmc_ids": pmc_ids,
        "vlm_model_ids": vlm_model_ids,
        "ner_model_keys": ner_model_keys,
        "papers": papers_results,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _vlm_summary_table(eval_result: dict) -> str:
    vlm_ids = eval_result.get("vlm_model_ids", [])
    papers = [p for p in eval_result["papers"] if "error" not in p]
    if not vlm_ids or not papers:
        return "_No VLM results._\n"

    header = "| Model | Lat/fig (s) | Compound % | Avg panels | Field compl. | Data pts | Output chars |"
    sep =    "|---|---|---|---|---|---|---|"
    rows = []
    for vlm_id in vlm_ids:
        # Average across papers
        lats, comp_rates, panel_counts, completeness, data_pts, chars = [], [], [], [], [], []
        for paper in papers:
            m = paper["vlm_results"].get(vlm_id, {})
            if m.get("error"):
                continue
            if m.get("latency_per_fig_s") is not None:
                lats.append(m["latency_per_fig_s"])
            comp_rates.append(m.get("compound_rate", 0))
            panel_counts.append(m.get("avg_panel_count", 0))
            completeness.append(m.get("field_completeness", 0))
            data_pts.append(m.get("total_data_points", 0))
            chars.append(m.get("output_chars", 0))

        def avg(lst): return sum(lst) / len(lst) if lst else None
        short_name = vlm_id.split("/")[-1]
        lat_str = f"{avg(lats):.2f}" if lats else "—"
        comp_str = f"{avg(comp_rates) * 100:.0f}%" if comp_rates else "—"
        panel_str = f"{avg(panel_counts):.1f}" if panel_counts else "—"
        compl_str = f"{avg(completeness) * 100:.0f}%" if completeness else "—"
        dp_str = str(int(sum(data_pts))) if data_pts else "—"
        chars_str = str(int(avg(chars))) if chars else "—"
        rows.append(f"| {short_name} | {lat_str} | {comp_str} | {panel_str} | {compl_str} | {dp_str} | {chars_str} |")

    return header + "\n" + sep + "\n" + "\n".join(rows) + "\n"


def _ner_summary_table(eval_result: dict) -> str:
    ner_keys = eval_result.get("ner_model_keys", [])
    papers = [p for p in eval_result["papers"] if "error" not in p]
    if not ner_keys or not papers:
        return "_No NER results._\n"

    header = "| Model | Latency (s) | Entities | Unique | Lat/ent (ms) | Density/1k | Avg span |"
    sep =    "|---|---|---|---|---|---|---|"
    rows = []
    for ner_key in ner_keys:
        lats, ents, unique, lat_per, density, spans = [], [], [], [], [], []
        for paper in papers:
            m = paper["ner_results"].get(ner_key, {})
            if m.get("error"):
                continue
            lats.append(m.get("latency_s", 0))
            ents.append(m.get("entity_count", 0))
            unique.append(m.get("unique_entity_count", 0))
            if m.get("latency_per_entity_ms") is not None:
                lat_per.append(m["latency_per_entity_ms"])
            density.append(m.get("entity_density_per_1k", 0))
            spans.append(m.get("avg_span_length_words", 0))

        def avg(lst): return sum(lst) / len(lst) if lst else None
        model_name = NER_MODEL_OPTIONS.get(ner_key, ner_key)
        lat_str = f"{sum(lats):.2f}" if lats else "—"
        ent_str = str(int(sum(ents))) if ents else "—"
        uniq_str = str(int(avg(unique))) if unique else "—"
        lpe_str = f"{avg(lat_per):.2f}" if lat_per else "—"
        dens_str = f"{avg(density):.1f}" if density else "—"
        span_str = f"{avg(spans):.1f}" if spans else "—"
        rows.append(f"| {model_name} | {lat_str} | {ent_str} | {uniq_str} | {lpe_str} | {dens_str} | {span_str} |")

    return header + "\n" + sep + "\n" + "\n".join(rows) + "\n"


def generate_md_report(eval_result: dict) -> str:
    timestamp = eval_result.get("timestamp", "unknown")
    pmc_ids = eval_result.get("pmc_ids", [])
    vlm_ids = eval_result.get("vlm_model_ids", [])
    ner_keys = eval_result.get("ner_model_keys", [])
    papers = eval_result.get("papers", [])

    lines = [
        f"# BioMed Extractor Evaluation Report",
        f"",
        f"**Timestamp:** {timestamp}  ",
        f"**Papers:** {', '.join(pmc_ids)}  ",
        f"**VLM Models:** {', '.join(vlm_ids) or 'none'}  ",
        f"**NER Models:** {', '.join(NER_MODEL_OPTIONS.get(k, k) for k in ner_keys) or 'none'}  ",
        f"",
        f"---",
        f"",
    ]

    for paper in papers:
        pmc = paper["pmc_id"]
        if paper.get("error"):
            lines += [f"## {pmc}", f"", f"> **Error:** {paper['error']}", ""]
            continue
        title = paper.get("title", "")
        lines += [
            f"## {pmc}",
            f"",
            f"**Title:** {title}  " if title else "",
            f"**Fetch:** {paper['fetch_s']:.2f}s | **Figures:** {paper['n_figures']} | **Text:** {paper['text_chars']:,} chars  ",
            f"",
        ]

        if vlm_ids:
            lines += ["### VLM Results", ""]
            lines += ["| Model | Lat/fig (s) | Compound % | Avg panels | Field compl. | Data pts |"]
            lines += ["|---|---|---|---|---|---|"]
            for vlm_id in vlm_ids:
                m = paper["vlm_results"].get(vlm_id, {})
                short = vlm_id.split("/")[-1]
                if m.get("error"):
                    lines.append(f"| {short} | error | — | — | — | — |")
                else:
                    lpf = f"{m['latency_per_fig_s']:.2f}" if m.get("latency_per_fig_s") else "—"
                    comp = f"{m.get('compound_rate', 0) * 100:.0f}%"
                    panels = f"{m.get('avg_panel_count', 0):.1f}"
                    compl = f"{m.get('field_completeness', 0) * 100:.0f}%"
                    dp = str(m.get("total_data_points", 0))
                    lines.append(f"| {short} | {lpf} | {comp} | {panels} | {compl} | {dp} |")
            lines.append("")

        if ner_keys:
            lines += ["### NER Results", ""]
            lines += ["| Model | Latency (s) | Entities | Unique | Lat/ent (ms) | Density/1k |"]
            lines += ["|---|---|---|---|---|---|"]
            for ner_key in ner_keys:
                m = paper["ner_results"].get(ner_key, {})
                model_name = NER_MODEL_OPTIONS.get(ner_key, ner_key)
                if m.get("error"):
                    lines.append(f"| {model_name} | error | — | — | — | — |")
                else:
                    lat = f"{m.get('latency_s', 0):.2f}"
                    ent = str(m.get("entity_count", 0))
                    uniq = str(m.get("unique_entity_count", 0))
                    lpe = f"{m['latency_per_entity_ms']:.2f}" if m.get("latency_per_entity_ms") else "—"
                    dens = f"{m.get('entity_density_per_1k', 0):.1f}"
                    lines.append(f"| {model_name} | {lat} | {ent} | {uniq} | {lpe} | {dens} |")
            lines.append("")

            # Type distribution per NER model
            for ner_key in ner_keys:
                m = paper["ner_results"].get(ner_key, {})
                if m.get("error") or not m.get("type_distribution"):
                    continue
                model_name = NER_MODEL_OPTIONS.get(ner_key, ner_key)
                dist = m["type_distribution"]
                sorted_types = sorted(dist.items(), key=lambda x: x[1], reverse=True)
                lines += [f"**{model_name} type distribution:**"]
                type_cells = " | ".join(f"{t}: {c}" for t, c in sorted_types[:10])
                lines.append(type_cells)
                lines.append("")

    lines += [
        "---",
        "",
        "## Summary",
        "",
        "### VLM Comparison (averaged across all papers)",
        "",
        _vlm_summary_table(eval_result),
        "",
        "### NER Comparison (aggregated across all papers)",
        "",
        _ner_summary_table(eval_result),
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_eval_run(eval_result: dict, report_md: str, output_root: Path) -> Path:
    """Save evaluation results to output_root / eval_{timestamp}/. Returns run dir."""
    timestamp = eval_result.get("timestamp", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    run_dir = output_root / f"eval_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "eval_result.json").write_text(
        json.dumps(eval_result, indent=2, default=str), encoding="utf-8"
    )
    (run_dir / "report.md").write_text(report_md, encoding="utf-8")

    print(f"[eval] Saved to {run_dir}")
    return run_dir


# ---------------------------------------------------------------------------
# Summary table helpers (exported for UI use)
# ---------------------------------------------------------------------------


def get_vlm_summary_md(eval_result: dict) -> str:
    return "### VLM Summary\n\n" + _vlm_summary_table(eval_result)


def get_ner_summary_md(eval_result: dict) -> str:
    return "### NER Summary\n\n" + _ner_summary_table(eval_result)
