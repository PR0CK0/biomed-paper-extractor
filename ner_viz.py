"""
ner_viz.py

Visualization utilities for biomedical NER entity outputs.

Provides:
    build_cooccurrence_graph  - Plotly co-occurrence network figure
    build_word_cloud          - PIL word-cloud image
    COOCCURRENCE_AVAILABLE    - bool flag (networkx present)
    WORDCLOUD_AVAILABLE       - bool flag (wordcloud present)
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict
from typing import Any

import plotly.graph_objects as go
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy dependencies
# ---------------------------------------------------------------------------

try:
    import networkx as nx  # type: ignore[import]
    COOCCURRENCE_AVAILABLE = True
except ImportError:
    nx = None  # type: ignore[assignment]
    COOCCURRENCE_AVAILABLE = False
    logger.warning(
        "networkx is not installed. build_cooccurrence_graph will raise RuntimeError. "
        "Install with: pip install networkx"
    )

try:
    from wordcloud import WordCloud  # type: ignore[import]
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WordCloud = None  # type: ignore[assignment]
    WORDCLOUD_AVAILABLE = False
    logger.warning(
        "wordcloud is not installed. build_word_cloud will return None. "
        "Install with: pip install wordcloud"
    )

# ---------------------------------------------------------------------------
# Entity type color map
# ---------------------------------------------------------------------------

_TYPE_COLORS: dict[str, str] = {
    "Disease": "#ef4444",
    "Drug": "#3b82f6",
    "Chemical": "#3b82f6",
    "Gene": "#22c55e",
    "Protein": "#22c55e",
    "Species": "#f59e0b",
    "Organism": "#f59e0b",
    "CellLine": "#8b5cf6",
    "CellType": "#a78bfa",
    "Anatomy": "#ec4899",
    "Organ": "#ec4899",
    "Tissue": "#f472b6",
    "Cell": "#fb7185",
    "BiologicalProcess": "#14b8a6",
    "MolecularFunction": "#06b6d4",
    "DNA": "#84cc16",
    "RNA": "#a3e635",
    "Mutation": "#f97316",
    "ENTITY": "#6b7280",
}
_DEFAULT_COLOR = "#94a3b8"


def _entity_color(label: str) -> str:
    return _TYPE_COLORS.get(label, _DEFAULT_COLOR)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _deduplicate_entities(
    entities: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """
    Collapse entities by normalized (lowercase) text.

    Returns a mapping of normalized_text -> {text, label, frequency}.
    The label kept is the most common one observed for that text.
    """
    freq_counter: Counter[str] = Counter()
    label_votes: dict[str, Counter[str]] = defaultdict(Counter)
    canonical_text: dict[str, str] = {}

    for ent in entities:
        raw = ent.get("text", "")
        norm = raw.strip().lower()
        if not norm:
            continue
        freq_counter[norm] += 1
        label_votes[norm][ent.get("label", "ENTITY")] += 1
        if norm not in canonical_text:
            canonical_text[norm] = raw.strip()

    deduped: dict[str, dict[str, Any]] = {}
    for norm, freq in freq_counter.items():
        best_label = label_votes[norm].most_common(1)[0][0]
        deduped[norm] = {
            "text": canonical_text[norm],
            "label": best_label,
            "frequency": freq,
        }
    return deduped


def _build_sentence_offsets(text: str) -> list[tuple[int, int]]:
    """
    Split text into sentences by '. ' or newline and return
    (start_char, end_char) ranges for each sentence.
    """
    offsets: list[tuple[int, int]] = []
    # Split on sentence-ending '. ' sequences or newlines, keeping delimiters
    pattern = re.compile(r"(?<=\.)\s+|\n+")
    prev = 0
    for m in pattern.finditer(text):
        end = m.start() + len(m.group())
        offsets.append((prev, end))
        prev = end
    if prev < len(text):
        offsets.append((prev, len(text)))
    return offsets


def _entities_in_sentence(
    entities: list[dict[str, Any]],
    sent_start: int,
    sent_end: int,
) -> list[str]:
    """Return normalized texts of entities whose char range falls in the sentence."""
    result: list[str] = []
    for ent in entities:
        s = ent.get("start_char", -1)
        e = ent.get("end_char", -1)
        if s >= sent_start and e <= sent_end:
            norm = ent.get("text", "").strip().lower()
            if norm:
                result.append(norm)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_cooccurrence_graph(
    entities: list[dict[str, Any]],
    text: str,
    min_cooccurrence: int = 2,
    max_nodes: int = 60,
) -> go.Figure:
    """
    Build a Plotly co-occurrence network figure from NER entities.

    Parameters
    ----------
    entities:
        List of entity dicts with keys: text, label, start_char, end_char.
    text:
        Full paper text used for sentence-level co-occurrence detection.
    min_cooccurrence:
        Minimum edge weight (co-occurrence count) required to include an edge.
    max_nodes:
        Maximum number of nodes to render (kept by total weighted degree).

    Returns
    -------
    go.Figure
    """
    if not COOCCURRENCE_AVAILABLE:
        raise RuntimeError(
            "networkx is required for build_cooccurrence_graph. "
            "Install with: pip install networkx"
        )

    if not entities:
        fig = go.Figure()
        fig.update_layout(
            title="Entity Co-occurrence Network (0 entities, 0 connections)",
            template="plotly_dark",
        )
        return fig

    # 1. Deduplicate
    deduped = _deduplicate_entities(entities)

    # 2. Build sentence-level co-occurrence
    sentence_offsets = _build_sentence_offsets(text)
    edge_weights: Counter[tuple[str, str]] = Counter()

    for sent_start, sent_end in sentence_offsets:
        present = list(set(_entities_in_sentence(entities, sent_start, sent_end)))
        present = [p for p in present if p in deduped]
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                a, b = sorted([present[i], present[j]])
                edge_weights[(a, b)] += 1

    # 3. Build graph with filtered edges
    G = nx.Graph()
    for norm, info in deduped.items():
        G.add_node(norm, **info)

    for (a, b), weight in edge_weights.items():
        if weight >= min_cooccurrence:
            G.add_edge(a, b, weight=weight)

    # Remove isolated nodes (no edges after filtering)
    isolated = [n for n in list(G.nodes()) if G.degree(n) == 0]
    G.remove_nodes_from(isolated)

    # Keep top max_nodes by total weighted degree
    if len(G.nodes()) > max_nodes:
        weighted_degrees = {
            n: sum(d.get("weight", 1) for _, _, d in G.edges(n, data=True))
            for n in G.nodes()
        }
        top_nodes = sorted(
            weighted_degrees, key=weighted_degrees.get, reverse=True  # type: ignore[arg-type]
        )[:max_nodes]
        G = G.subgraph(top_nodes).copy()

    if len(G.nodes()) == 0:
        fig = go.Figure()
        fig.update_layout(
            title="Entity Co-occurrence Network (0 entities, 0 connections)",
            template="plotly_dark",
        )
        return fig

    # 4. Layout
    n = len(G.nodes())
    k_val = 1.5 / math.sqrt(n) if n > 1 else 1.5
    pos: dict[str, tuple[float, float]] = nx.spring_layout(
        G, seed=42, k=k_val
    )

    # Compute per-node degree for sizing
    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 1
    min_degree = min(degrees.values()) if degrees else 1
    degree_range = max(max_degree - min_degree, 1)

    def _node_size(deg: int) -> float:
        return 8 + (deg - min_degree) / degree_range * (28 - 8)

    # 5a. Edge traces
    all_weights = [d.get("weight", 1) for _, _, d in G.edges(data=True)]
    max_w = max(all_weights) if all_weights else 1
    min_w = min(all_weights) if all_weights else 1
    w_range = max(max_w - min_w, 1)

    edge_traces: list[go.Scatter] = []
    for u, v, data in G.edges(data=True):
        w = data.get("weight", 1)
        opacity = 0.1 + (w - min_w) / w_range * (0.7 - 0.1)
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_traces.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(color="#374151", width=1),
                opacity=opacity,
                hoverinfo="none",
                showlegend=False,
            )
        )

    # 5b. Node traces — one per entity type for legend grouping
    type_to_nodes: dict[str, list[str]] = defaultdict(list)
    for node in G.nodes():
        label = G.nodes[node].get("label", "ENTITY")
        type_to_nodes[label].append(node)

    node_traces: list[go.Scatter] = []
    for entity_type, nodes in sorted(type_to_nodes.items()):
        color = _entity_color(entity_type)
        xs: list[float] = []
        ys: list[float] = []
        sizes: list[float] = []
        texts: list[str] = []
        hover_texts: list[str] = []

        for node in nodes:
            x, y = pos[node]
            xs.append(x)
            ys.append(y)
            sizes.append(_node_size(degrees[node]))
            display = G.nodes[node].get("text", node)
            texts.append(display)
            freq = G.nodes[node].get("frequency", 1)
            deg = degrees[node]
            hover_texts.append(
                f"<b>{display}</b><br>"
                f"Type: {entity_type}<br>"
                f"Frequency: {freq}<br>"
                f"Connections: {deg}"
            )

        node_traces.append(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text",
                name=entity_type,
                marker=dict(
                    size=sizes,
                    color=color,
                    line=dict(width=1, color="#1f2937"),
                ),
                text=texts,
                textposition="top center",
                textfont=dict(size=9),
                hovertext=hover_texts,
                hoverinfo="text",
            )
        )

    # 6. Assemble figure
    fig = go.Figure(data=edge_traces + node_traces)

    # 7. Title
    title = (
        f"Entity Co-occurrence Network "
        f"({len(G.nodes())} entities, {len(G.edges())} connections)"
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        template="plotly_dark",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=10),
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="closest",
    )

    return fig


def build_word_cloud(
    entities: list[dict[str, Any]],
    max_words: int = 120,
) -> Image.Image | None:
    """
    Build a word cloud PIL image from NER entities colored by entity type.

    Parameters
    ----------
    entities:
        List of entity dicts with keys: text, label, start_char, end_char.
    max_words:
        Maximum number of words to include in the cloud.

    Returns
    -------
    PIL.Image.Image or None
        Returns None if wordcloud is not installed or entities list is empty.
    """
    if not WORDCLOUD_AVAILABLE:
        return None

    if not entities:
        return None

    # 1. Build frequency dict, exclude entities shorter than 2 chars
    freq_counter: Counter[str] = Counter()
    norm_to_label: dict[str, str] = {}

    for ent in entities:
        raw = ent.get("text", "").strip()
        norm = raw.lower()
        if len(norm) < 2:
            continue
        freq_counter[norm] += 1
        if norm not in norm_to_label:
            norm_to_label[norm] = ent.get("label", "ENTITY")

    if not freq_counter:
        return None

    freq_dict = dict(freq_counter)

    # 2. Color function keyed on normalized word
    def _color_func(
        word: str,
        font_size: int,
        position: tuple[int, int],
        orientation: Any,
        random_state: Any = None,
        **kwargs: Any,
    ) -> str:
        label = norm_to_label.get(word.lower(), "ENTITY")
        return _entity_color(label)

    # 3 & 4. Generate word cloud
    wc = WordCloud(
        width=900,
        height=450,
        background_color="#111827",
        max_words=max_words,
        colormap=None,
        color_func=_color_func,
        prefer_horizontal=0.8,
        min_font_size=9,
    )
    wc.generate_from_frequencies(freq_dict)

    # 5. Return PIL image
    return wc.to_image()
