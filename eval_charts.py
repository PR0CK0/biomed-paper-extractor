"""
eval_charts.py — Plotly chart generation for BioMed Extractor evaluation results.
"""
from __future__ import annotations

import plotly.graph_objects as go

try:
    from task2_ner import NER_MODEL_OPTIONS
except ImportError:
    NER_MODEL_OPTIONS: dict = {}

_TEMPLATE = "plotly_dark"
_COLORS = [
    "#60a5fa", "#34d399", "#f472b6", "#fbbf24", "#a78bfa",
    "#fb7185", "#38bdf8", "#4ade80", "#fb923c", "#c084fc",
]


def _short_vlm(model_id: str) -> str:
    return model_id.split("/")[-1]


def _short_ner(model_key: str) -> str:
    name = NER_MODEL_OPTIONS.get(model_key, model_key)
    # Truncate long names
    return name[:20] + "…" if len(name) > 20 else name


def _valid_papers(eval_result: dict) -> list[dict]:
    return [p for p in eval_result.get("papers", []) if "error" not in p]


# ---------------------------------------------------------------------------
# 1. VLM Latency per Figure — grouped bar by paper
# ---------------------------------------------------------------------------


def chart_vlm_latency(eval_result: dict) -> go.Figure:
    vlm_ids = eval_result.get("vlm_model_ids", [])
    papers = _valid_papers(eval_result)

    fig = go.Figure()
    for i, vlm_id in enumerate(vlm_ids):
        x, y = [], []
        for paper in papers:
            m = paper["vlm_results"].get(vlm_id, {})
            if not m.get("error") and m.get("latency_per_fig_s") is not None:
                x.append(paper["pmc_id"])
                y.append(m["latency_per_fig_s"])
        if x:
            fig.add_trace(go.Bar(
                name=_short_vlm(vlm_id),
                x=x, y=y,
                marker_color=_COLORS[i % len(_COLORS)],
            ))

    fig.update_layout(
        title="VLM Latency per Figure (seconds)",
        xaxis_title="Paper",
        yaxis_title="Seconds / figure",
        barmode="group",
        template=_TEMPLATE,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# ---------------------------------------------------------------------------
# 2. VLM Field Completeness Radar
# ---------------------------------------------------------------------------


def chart_vlm_field_completeness(eval_result: dict) -> go.Figure:
    vlm_ids = eval_result.get("vlm_model_ids", [])
    papers = _valid_papers(eval_result)

    categories = ["Field Completeness", "Avg Panels", "Data Points (norm)", "Output Chars (norm)"]
    categories_closed = categories + [categories[0]]

    # Compute raw averages per model
    raw: dict[str, list[float]] = {}
    for vlm_id in vlm_ids:
        vals = {"field_completeness": [], "avg_panel_count": [], "total_data_points": [], "output_chars": []}
        for paper in papers:
            m = paper["vlm_results"].get(vlm_id, {})
            if m.get("error"):
                continue
            vals["field_completeness"].append(m.get("field_completeness", 0))
            vals["avg_panel_count"].append(m.get("avg_panel_count", 0))
            vals["total_data_points"].append(m.get("total_data_points", 0))
            vals["output_chars"].append(m.get("output_chars", 0))
        avg = lambda lst: sum(lst) / len(lst) if lst else 0.0
        raw[vlm_id] = [
            avg(vals["field_completeness"]),
            avg(vals["avg_panel_count"]),
            avg(vals["total_data_points"]),
            avg(vals["output_chars"]),
        ]

    if not raw:
        return go.Figure(layout=go.Layout(title="VLM Output Richness (no data)", template=_TEMPLATE))

    # Normalize each axis to [0, 1] across models
    n_axes = len(categories)
    maxima = [max(raw[v][i] for v in raw) or 1.0 for i in range(n_axes)]

    fig = go.Figure()
    for idx, vlm_id in enumerate(vlm_ids):
        if vlm_id not in raw:
            continue
        vals_norm = [raw[vlm_id][i] / maxima[i] for i in range(n_axes)]
        vals_closed = vals_norm + [vals_norm[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals_closed,
            theta=categories_closed,
            fill="toself",
            name=_short_vlm(vlm_id),
            line_color=_COLORS[idx % len(_COLORS)],
            opacity=0.7,
        ))

    fig.update_layout(
        title="VLM Output Richness (normalized radar)",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        template=_TEMPLATE,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# ---------------------------------------------------------------------------
# 3. VLM Panel Detection — avg panel count bar, color by compound_rate
# ---------------------------------------------------------------------------


def chart_vlm_panel_detection(eval_result: dict) -> go.Figure:
    vlm_ids = eval_result.get("vlm_model_ids", [])
    papers = _valid_papers(eval_result)

    names, panel_counts, compound_rates = [], [], []
    for vlm_id in vlm_ids:
        panels, comp = [], []
        for paper in papers:
            m = paper["vlm_results"].get(vlm_id, {})
            if m.get("error"):
                continue
            panels.append(m.get("avg_panel_count", 0))
            comp.append(m.get("compound_rate", 0))
        if panels:
            names.append(_short_vlm(vlm_id))
            panel_counts.append(sum(panels) / len(panels))
            compound_rates.append(sum(comp) / len(comp))

    fig = go.Figure(go.Bar(
        x=names,
        y=panel_counts,
        marker=dict(
            color=compound_rates,
            colorscale="Viridis",
            colorbar=dict(title="Compound rate"),
            showscale=True,
        ),
        text=[f"{p:.1f}" for p in panel_counts],
        textposition="outside",
    ))
    fig.update_layout(
        title="Average Panel Count per Figure",
        xaxis_title="VLM Model",
        yaxis_title="Avg panels",
        template=_TEMPLATE,
    )
    return fig


# ---------------------------------------------------------------------------
# 4. NER Latency — grouped bar by paper
# ---------------------------------------------------------------------------


def chart_ner_latency(eval_result: dict) -> go.Figure:
    ner_keys = eval_result.get("ner_model_keys", [])
    papers = _valid_papers(eval_result)

    fig = go.Figure()
    for i, ner_key in enumerate(ner_keys):
        x, y = [], []
        for paper in papers:
            m = paper["ner_results"].get(ner_key, {})
            if not m.get("error"):
                x.append(paper["pmc_id"])
                y.append(m.get("latency_s", 0))
        if x:
            fig.add_trace(go.Bar(
                name=_short_ner(ner_key),
                x=x, y=y,
                marker_color=_COLORS[i % len(_COLORS)],
            ))

    fig.update_layout(
        title="NER Latency (seconds)",
        xaxis_title="Paper",
        yaxis_title="Seconds",
        barmode="group",
        template=_TEMPLATE,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# ---------------------------------------------------------------------------
# 5. NER Entity Counts — total vs unique, grouped
# ---------------------------------------------------------------------------


def chart_ner_entity_counts(eval_result: dict) -> go.Figure:
    ner_keys = eval_result.get("ner_model_keys", [])
    papers = _valid_papers(eval_result)

    model_names = [_short_ner(k) for k in ner_keys]
    total_counts = []
    unique_counts = []
    for ner_key in ner_keys:
        totals, uniques = [], []
        for paper in papers:
            m = paper["ner_results"].get(ner_key, {})
            if not m.get("error"):
                totals.append(m.get("entity_count", 0))
                uniques.append(m.get("unique_entity_count", 0))
        total_counts.append(sum(totals))
        unique_counts.append(sum(uniques))

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Total entities", x=model_names, y=total_counts, marker_color=_COLORS[0]))
    fig.add_trace(go.Bar(name="Unique entities", x=model_names, y=unique_counts, marker_color=_COLORS[1]))
    fig.update_layout(
        title="Entity Counts (total vs unique, all papers)",
        xaxis_title="NER Model",
        yaxis_title="Entity count",
        barmode="group",
        template=_TEMPLATE,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# ---------------------------------------------------------------------------
# 6. NER Type Distribution — stacked bar per paper
# ---------------------------------------------------------------------------


def chart_ner_type_distribution(eval_result: dict, model_key: str) -> go.Figure:
    papers = _valid_papers(eval_result)

    all_types: set[str] = set()
    for paper in papers:
        m = paper["ner_results"].get(model_key, {})
        if not m.get("error"):
            all_types.update(m.get("type_distribution", {}).keys())

    sorted_types = sorted(all_types)
    paper_ids = [p["pmc_id"] for p in papers]

    fig = go.Figure()
    for i, entity_type in enumerate(sorted_types):
        counts = []
        for paper in papers:
            m = paper["ner_results"].get(model_key, {})
            if m.get("error"):
                counts.append(0)
            else:
                counts.append(m.get("type_distribution", {}).get(entity_type, 0))
        fig.add_trace(go.Bar(
            name=entity_type,
            x=paper_ids,
            y=counts,
            marker_color=_COLORS[i % len(_COLORS)],
        ))

    model_name = NER_MODEL_OPTIONS.get(model_key, model_key)
    fig.update_layout(
        title=f"Entity Type Distribution — {model_name}",
        xaxis_title="Paper",
        yaxis_title="Entity count",
        barmode="stack",
        template=_TEMPLATE,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# ---------------------------------------------------------------------------
# 7. VLM Latency vs Richness Scatter
# ---------------------------------------------------------------------------


def chart_vlm_latency_vs_richness(eval_result: dict) -> go.Figure:
    vlm_ids = eval_result.get("vlm_model_ids", [])
    papers = _valid_papers(eval_result)

    fig = go.Figure()
    for i, vlm_id in enumerate(vlm_ids):
        xs, ys, texts = [], [], []
        for paper in papers:
            m = paper["vlm_results"].get(vlm_id, {})
            if m.get("error") or m.get("latency_per_fig_s") is None:
                continue
            xs.append(m["latency_per_fig_s"])
            ys.append(m.get("field_completeness", 0) * 100)
            texts.append(f"{_short_vlm(vlm_id)}<br>{paper['pmc_id']}")
        if xs:
            fig.add_trace(go.Scatter(
                x=xs, y=ys,
                mode="markers+text",
                name=_short_vlm(vlm_id),
                text=[_short_vlm(vlm_id)] * len(xs),
                textposition="top center",
                marker=dict(size=12, color=_COLORS[i % len(_COLORS)]),
                hovertext=texts,
                hoverinfo="text",
            ))

    fig.update_layout(
        title="VLM: Latency vs Output Richness",
        xaxis_title="Latency per figure (s)",
        yaxis_title="Field completeness (%)",
        template=_TEMPLATE,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# ---------------------------------------------------------------------------
# Convenience: all charts at once
# ---------------------------------------------------------------------------


def get_all_charts(eval_result: dict) -> dict[str, go.Figure]:
    return {
        "vlm_latency": chart_vlm_latency(eval_result),
        "vlm_richness_radar": chart_vlm_field_completeness(eval_result),
        "vlm_panels": chart_vlm_panel_detection(eval_result),
        "ner_latency": chart_ner_latency(eval_result),
        "ner_entity_counts": chart_ner_entity_counts(eval_result),
        "vlm_latency_vs_richness": chart_vlm_latency_vs_richness(eval_result),
    }
