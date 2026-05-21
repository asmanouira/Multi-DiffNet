# multi_diffnet/networks.py
# -*- coding: utf-8 -*-

from __future__ import annotations


import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patheffects as pe

from .data import block_slices_from_dims

CMAP = plt.cm.YlOrRd

EDGE_COLORS = {
    "common": "#7f7f7f",
    "low_specific": "#4C78A8",
    "high_specific": "#D62728",
}

MIN_EDGE_WIDTH = 0.25
MAX_EDGE_WIDTH = 1.8
EDGE_ALPHA = 0.42
SPRING_SEED = 7


def matrix_to_edge_table(M, feature_names=None, threshold=1e-6):
    """
    Convert a square symmetric matrix into an edge table.

    Parameters
    ----------
    M : ndarray of shape (p, p)
        Precision or differential precision matrix.

    feature_names : list[str], optional
        Feature names. If None, generic names are used.

    threshold : float, default=1e-6
        Minimum absolute edge weight to retain.

    Returns
    -------
    pandas.DataFrame
        Table containing source, target, weight, and absolute weight.
    """
    M = np.asarray(M)
    p = M.shape[0]

    if M.shape[0] != M.shape[1]:
        raise ValueError("matrix_to_edge_table expects a square matrix.")

    if feature_names is None:
        feature_names = [f"V{i}" for i in range(p)]

    edges = []

    for i in range(p):
        for j in range(i + 1, p):
            w = float(M[i, j])
            if abs(w) > threshold:
                edges.append(
                    {
                        "source": feature_names[i],
                        "target": feature_names[j],
                        "weight": w,
                        "abs_weight": abs(w),
                    }
                )

    return pd.DataFrame(edges)


def rectangular_matrix_to_edge_table(
    M,
    row_names,
    col_names,
    threshold=1e-6,
):
    """
    Convert a rectangular matrix block into a bipartite edge table.

    Parameters
    ----------
    M : ndarray of shape (p_row, p_col)
        Rectangular cross-block matrix.

    row_names : list[str]
        Names of row features.

    col_names : list[str]
        Names of column features.

    threshold : float, default=1e-6
        Minimum absolute edge weight to retain.

    Returns
    -------
    pandas.DataFrame
        Table containing source, target, weight, and absolute weight.
    """
    M = np.asarray(M)

    edges = []

    for i, source in enumerate(row_names):
        for j, target in enumerate(col_names):
            w = float(M[i, j])
            if abs(w) > threshold:
                edges.append(
                    {
                        "source": source,
                        "target": target,
                        "weight": w,
                        "abs_weight": abs(w),
                    }
                )

    return pd.DataFrame(edges)


def extract_all_blocks(M, data):
    """
    Extract all diagonal and off-diagonal modality blocks from a matrix.

    Parameters
    ----------
    M : ndarray of shape (p, p)
        Precision or differential precision matrix.

    data : MultiModalData
        Multi-modal data object.

    Returns
    -------
    dict
        Dictionary mapping block names to block matrices.
    """
    idx = block_slices_from_dims(data.dims)
    names = list(data.dims.keys())

    blocks = {}

    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if j < i:
                continue

            I = idx[a]
            J = idx[b]

            block_name = f"{a}__{b}"
            blocks[block_name] = M[np.ix_(I, J)]

    return blocks


def matrix_to_edge_dict_square(M, names, threshold=1e-6):
    """
    Convert a square matrix into an undirected edge dictionary.

    Parameters
    ----------
    M : ndarray of shape (p, p)
        Square matrix.

    names : list[str]
        Node names.

    threshold : float
        Minimum absolute edge weight.

    Returns
    -------
    dict
        Dictionary mapping edge tuples to weights.
    """
    M = np.asarray(M)
    d = M.shape[0]

    edges = {}

    for i in range(d):
        for j in range(i + 1, d):
            w = float(M[i, j])
            if abs(w) > threshold:
                edges[(names[i], names[j])] = w

    return edges


def matrix_to_edge_dict_rect(
    M,
    row_names,
    col_names,
    threshold=1e-6,
):
    """
    Convert a rectangular matrix into a bipartite edge dictionary.

    Parameters
    ----------
    M : ndarray of shape (p_row, p_col)
        Rectangular block matrix.

    row_names : list[str]
        Row node names.

    col_names : list[str]
        Column node names.

    threshold : float
        Minimum absolute edge weight.

    Returns
    -------
    dict
        Dictionary mapping bipartite edge tuples to weights.
    """
    M = np.asarray(M)

    edges = {}

    for i, r in enumerate(row_names):
        for j, c in enumerate(col_names):
            w = float(M[i, j])
            if abs(w) > threshold:
                edges[(f"L::{r}", f"R::{c}")] = w

    return edges


def classify_low_high_edges(low_edges, high_edges):
    """
    Classify edges into common, low-specific and high-specific sets.

    Parameters
    ----------
    low_edges : dict
        Edge dictionary for the low group.

    high_edges : dict
        Edge dictionary for the high group.

    Returns
    -------
    tuple
        common_edges, low_specific_edges, high_specific_edges.
    """
    low_set = set(low_edges.keys())
    high_set = set(high_edges.keys())

    common = []
    low_specific = []
    high_specific = []

    for e in sorted(low_set & high_set):
        w = 0.5 * (low_edges[e] + high_edges[e])
        common.append((e[0], e[1], float(w)))

    for e in sorted(low_set - high_set):
        low_specific.append((e[0], e[1], float(low_edges[e])))

    for e in sorted(high_set - low_set):
        high_specific.append((e[0], e[1], float(high_edges[e])))

    return common, low_specific, high_specific


def compute_strength(edges, nodes):
    """
    Compute node strength as the sum of absolute edge weights.

    Parameters
    ----------
    edges : list[tuple]
        Weighted edge list.

    nodes : list[str]
        Node names.

    Returns
    -------
    dict
        Node strength dictionary.
    """
    strength = {n: 0.0 for n in nodes}

    for u, v, w in edges:
        aw = abs(float(w))
        strength[u] = strength.get(u, 0.0) + aw
        strength[v] = strength.get(v, 0.0) + aw

    return strength


def merge_edges(*edge_lists):
    """
    Merge several weighted edge lists.

    Parameters
    ----------
    edge_lists : list
        Weighted edge lists.

    Returns
    -------
    list
        Merged weighted edge list.
    """
    out = {}

    for edges in edge_lists:
        for u, v, w in edges:
            key = tuple(sorted((u, v)))
            out[key] = out.get(key, 0.0) + abs(float(w))

    return [(u, v, w) for (u, v), w in out.items()]


def rescale(values, out_min, out_max):
    """
    Rescale numeric values to a target interval.

    Parameters
    ----------
    values : array-like
        Values to rescale.

    out_min : float
        Minimum output value.

    out_max : float
        Maximum output value.

    Returns
    -------
    ndarray
        Rescaled values.
    """
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return values

    vmin = values.min()
    vmax = values.max()

    if vmin == vmax:
        return np.full_like(values, 0.5 * (out_min + out_max))

    return out_min + (values - vmin) / (vmax - vmin) * (out_max - out_min)


def select_label_nodes(
    strength,
    quantile=0.95,
    max_labels=10,
):
    """
    Select high-strength nodes to label.

    Parameters
    ----------
    strength : dict
        Node strength dictionary.

    quantile : float
        Quantile threshold.

    max_labels : int
        Maximum number of labels.

    Returns
    -------
    list[str]
        Selected node names.
    """
    positive = [(n, s) for n, s in strength.items() if s > 0]

    if not positive:
        return []

    vals = np.array([s for _, s in positive])
    thr = np.quantile(vals, quantile)

    selected = [(n, s) for n, s in positive if s >= thr]
    selected = sorted(selected, key=lambda x: x[1], reverse=True)

    return [n for n, _ in selected[:max_labels]]


def compute_shared_layout(all_nodes, union_edges):
    """
    Compute a shared graph layout for multiple network panels.

    Parameters
    ----------
    all_nodes : list[str]
        All node names.

    union_edges : list[tuple]
        Union edge list.

    Returns
    -------
    dict
        Node positions.
    """
    G = nx.Graph()
    G.add_nodes_from(all_nodes)

    for u, v, w in union_edges:
        G.add_edge(u, v, weight=abs(float(w)))

    non_isolates = [n for n in G.nodes if G.degree(n) > 0]

    if not non_isolates:
        return nx.circular_layout(all_nodes)

    G_noniso = G.subgraph(non_isolates).copy()

    try:
        pos = nx.kamada_kawai_layout(G_noniso, weight=None, scale=9.0)
    except Exception:
        pos = nx.spring_layout(G_noniso, seed=SPRING_SEED)

    missing = [n for n in all_nodes if n not in pos]

    if missing:
        ring = nx.circular_layout(missing, scale=11.0)
        pos.update(ring)

    return pos


def draw_panel(
    ax,
    edges,
    all_nodes,
    pos,
    title,
    edge_color,
    color_min,
    color_max,
    label_quantile=0.95,
    max_labels=10,
):
    """
    Draw one network panel.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis where the network is drawn.

    edges : list[tuple]
        Weighted edge list.

    all_nodes : list[str]
        All nodes to display.

    pos : dict
        Shared layout positions.

    title : str
        Panel title.

    edge_color : str
        Edge color.

    color_min : float
        Minimum value for node color scale.

    color_max : float
        Maximum value for node color scale.

    label_quantile : float
        Node strength quantile used for labeling.

    max_labels : int
        Maximum number of node labels.
    """
    G = nx.Graph()
    G.add_nodes_from(all_nodes)

    for u, v, w in edges:
        G.add_edge(u, v, weight=float(w))

    strength = compute_strength(edges, all_nodes)
    strength_values = np.array([strength[n] for n in all_nodes])

    node_sizes = np.full(len(all_nodes), 8.0)
    positive = strength_values > 0

    if positive.any():
        node_sizes[positive] = rescale(
            np.sqrt(strength_values[positive]),
            25,
            220,
        )

    if len(edges) > 0:
        abs_w = np.array([abs(w) for _, _, w in edges])
        edge_widths = rescale(abs_w, MIN_EDGE_WIDTH, MAX_EDGE_WIDTH)

        nx.draw_networkx_edges(
            G,
            pos=pos,
            ax=ax,
            edgelist=[(u, v) for u, v, _ in edges],
            width=edge_widths,
            edge_color=edge_color,
            alpha=EDGE_ALPHA,
        )

    nx.draw_networkx_nodes(
        G,
        pos=pos,
        ax=ax,
        nodelist=all_nodes,
        node_size=node_sizes,
        node_color=strength_values,
        cmap=CMAP,
        vmin=color_min,
        vmax=color_max,
        linewidths=0.25,
        edgecolors="black",
    )

    label_nodes = select_label_nodes(
        strength,
        quantile=label_quantile,
        max_labels=max_labels,
    )

    texts = []
    xs = []
    ys = []

    for n in label_nodes:
        x, y = pos[n]
        xs.append(x)
        ys.append(y)

        txt = ax.text(
            x,
            y + 0.08,
            str(n).split("::", 1)[-1],
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="center",
            color="black",
            zorder=20,
        )

        txt.set_path_effects(
            [
                pe.Stroke(linewidth=2.5, foreground="white"),
                pe.Normal(),
            ]
        )

        texts.append(txt)

    n_nodes = sum(1 for n in all_nodes if strength.get(n, 0.0) > 0)

    ax.set_title(
        f"{title}\n({n_nodes} nodes, {G.number_of_edges()} edges total)",
        pad=10,
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_network_from_matrix(
    M,
    feature_names=None,
    threshold=1e-6,
    top_edges=100,
    title="Network",
    node_size=300,
    figsize=(8, 6),
):
    """
    Plot a simple network from a square matrix.

    Parameters
    ----------
    M : ndarray of shape (p, p)
        Matrix to visualize.

    feature_names : list[str], optional
        Feature names.

    threshold : float
        Minimum absolute value to keep an edge.

    top_edges : int
        Maximum number of strongest edges to display.

    title : str
        Plot title.

    node_size : int
        Node size.

    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure or None
        Network figure, or None if no edges are found.
    """
    edges = matrix_to_edge_table(
        M,
        feature_names=feature_names,
        threshold=threshold,
    )

    if edges.empty:
        print("No edges found with the selected threshold.")
        return None

    edges = edges.sort_values("abs_weight", ascending=False).head(top_edges)

    G = nx.Graph()

    for _, row in edges.iterrows():
        G.add_edge(
            row["source"],
            row["target"],
            weight=row["weight"],
        )

    pos = nx.spring_layout(G, seed=42)

    weights = np.array([abs(G[u][v]["weight"]) for u, v in G.edges()])
    widths = 1 + 4 * weights / weights.max()

    fig, ax = plt.subplots(figsize=figsize)

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_size,
        ax=ax,
    )

    nx.draw_networkx_edges(
        G,
        pos,
        width=widths,
        alpha=0.7,
        ax=ax,
    )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=8,
        ax=ax,
    )

    ax.set_title(title)
    ax.axis("off")

    return fig


def plot_low_high_common_networks(
    low_matrix,
    high_matrix,
    row_names,
    col_names=None,
    block_name="block",
    threshold=1e-6,
    title=None,
    label_quantile=0.95,
    max_labels=10,
    figsize=(16, 6),
):
    """
    Plot common, low-specific and high-specific networks for one block.

    Parameters
    ----------
    low_matrix : ndarray
        Differential matrix for the low group.

    high_matrix : ndarray
        Differential matrix for the high group.

    row_names : list[str]
        Feature names for rows.

    col_names : list[str], optional
        Feature names for columns. If None, the block is assumed square.

    block_name : str
        Block name.

    threshold : float
        Minimum absolute edge weight.

    title : str, optional
        Figure title.

    label_quantile : float
        Node strength quantile used for labeling.

    max_labels : int
        Maximum number of node labels.

    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with common, low-specific and high-specific panels.
    """
    if col_names is None:
        all_nodes = list(row_names)

        low_edges = matrix_to_edge_dict_square(
            low_matrix,
            all_nodes,
            threshold=threshold,
        )

        high_edges = matrix_to_edge_dict_square(
            high_matrix,
            all_nodes,
            threshold=threshold,
        )

    else:
        left_nodes = [f"L::{x}" for x in row_names]
        right_nodes = [f"R::{x}" for x in col_names]
        all_nodes = left_nodes + right_nodes

        low_edges = matrix_to_edge_dict_rect(
            low_matrix,
            row_names,
            col_names,
            threshold=threshold,
        )

        high_edges = matrix_to_edge_dict_rect(
            high_matrix,
            row_names,
            col_names,
            threshold=threshold,
        )

    common_edges, low_specific_edges, high_specific_edges = classify_low_high_edges(
        low_edges,
        high_edges,
    )

    union_edges = merge_edges(
        common_edges,
        low_specific_edges,
        high_specific_edges,
    )

    pos = compute_shared_layout(all_nodes, union_edges)

    all_strengths = []

    for edges in [common_edges, low_specific_edges, high_specific_edges]:
        s = compute_strength(edges, all_nodes)
        all_strengths.extend(s.values())

    color_min = 0.0
    color_max = max(all_strengths) if all_strengths else 1.0

    if color_max == 0:
        color_max = 1.0

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        1,
        4,
        width_ratios=[1, 1, 1, 0.045],
        wspace=0.08,
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    cax = fig.add_subplot(gs[0, 3])

    draw_panel(
        ax1,
        common_edges,
        all_nodes,
        pos,
        title="Common",
        edge_color=EDGE_COLORS["common"],
        color_min=color_min,
        color_max=color_max,
        label_quantile=label_quantile,
        max_labels=max_labels,
    )

    draw_panel(
        ax2,
        low_specific_edges,
        all_nodes,
        pos,
        title="Low-specific",
        edge_color=EDGE_COLORS["low_specific"],
        color_min=color_min,
        color_max=color_max,
        label_quantile=label_quantile,
        max_labels=max_labels,
    )

    draw_panel(
        ax3,
        high_specific_edges,
        all_nodes,
        pos,
        title="High-specific",
        edge_color=EDGE_COLORS["high_specific"],
        color_min=color_min,
        color_max=color_max,
        label_quantile=label_quantile,
        max_labels=max_labels,
    )

    norm = mpl.colors.Normalize(vmin=color_min, vmax=color_max)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=CMAP)
    sm.set_array([])

    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Node strength")

    fig.suptitle(
        title or block_name,
        y=0.98,
        fontweight="bold",
    )

    return fig