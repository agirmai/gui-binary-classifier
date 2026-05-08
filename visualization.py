from __future__ import annotations

import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier, export_graphviz


@dataclass(frozen=True)
class BoundaryFigure:
    fig: plt.Figure
    ax: plt.Axes


def scatter_2d(
    X: np.ndarray,
    y: np.ndarray,
    *,
    feature_names_2d: list[str],
    target_names: list[str],
    title: str,
) -> BoundaryFigure:
    fig, ax = plt.subplots(figsize=(6, 4))
    classes = np.unique(y)
    for c in classes:
        mask = y == c
        ax.scatter(X[mask, 0], X[mask, 1], s=28, alpha=0.85, label=target_names[int(c)] if int(c) < len(target_names) else str(c))
    ax.set_title(title)
    ax.set_xlabel(feature_names_2d[0])
    ax.set_ylabel(feature_names_2d[1])
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return BoundaryFigure(fig=fig, ax=ax)


def decision_boundary_2d(
    model: DecisionTreeClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    feature_names_2d: list[str],
    target_names: list[str],
    title: str,
    animate_delay_s: float = 0.15,
    grid_step: float = 0.03,
) -> BoundaryFigure:
    # Simple “animation”: small delay so the redraw feels smoother during rapid widget changes.
    if animate_delay_s > 0:
        time.sleep(float(animate_delay_s))

    x_min, x_max = X_train[:, 0].min() - 1.0, X_train[:, 0].max() + 1.0
    y_min, y_max = X_train[:, 1].min() - 1.0, X_train[:, 1].max() + 1.0

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, grid_step),
        np.arange(y_min, y_max, grid_step),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.contourf(xx, yy, Z, alpha=0.28, levels=np.arange(Z.max() + 2) - 0.5)

    classes = np.unique(y_train)
    for c in classes:
        mask = y_train == c
        ax.scatter(
            X_train[mask, 0],
            X_train[mask, 1],
            s=22,
            alpha=0.9,
            edgecolor="k",
            linewidth=0.25,
            label=target_names[int(c)] if int(c) < len(target_names) else str(c),
        )

    ax.set_title(title)
    ax.set_xlabel(feature_names_2d[0])
    ax.set_ylabel(feature_names_2d[1])
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return BoundaryFigure(fig=fig, ax=ax)


def tree_graphviz_source(
    model: DecisionTreeClassifier,
    *,
    feature_names_2d: list[str],
    class_names: list[str],
    filled: bool = True,
    rounded: bool = True,
) -> str:
    dot = export_graphviz(
        model,
        out_file=None,
        feature_names=feature_names_2d,
        class_names=class_names,
        filled=filled,
        rounded=rounded,
        special_characters=True,
    )
    return str(dot)


@dataclass(frozen=True)
class TreeLayout:
    x: np.ndarray  # shape (n_nodes,)
    y: np.ndarray  # shape (n_nodes,)
    depth: np.ndarray  # shape (n_nodes,)


def _compute_tree_depths(model: DecisionTreeClassifier) -> np.ndarray:
    t = model.tree_
    n = int(t.node_count)
    depths = np.zeros(n, dtype=int)
    stack = [(0, 0)]
    while stack:
        node, d = stack.pop()
        depths[node] = d
        left = int(t.children_left[node])
        right = int(t.children_right[node])
        if left != -1:
            stack.append((left, d + 1))
        if right != -1:
            stack.append((right, d + 1))
    return depths


def compute_tree_layout(model: DecisionTreeClassifier) -> TreeLayout:
    """
    Deterministic 2D layout for a binary decision tree:
    - y is -depth (root on top)
    - x is leaf-order, with internal nodes centered over their children
    """
    t = model.tree_
    n = int(t.node_count)

    depths = _compute_tree_depths(model)
    x = np.zeros(n, dtype=float)
    y = -depths.astype(float)

    next_leaf_x = 0.0

    def dfs(node: int) -> None:
        nonlocal next_leaf_x
        left = int(t.children_left[node])
        right = int(t.children_right[node])
        is_leaf = left == -1 and right == -1
        if is_leaf:
            x[node] = next_leaf_x
            next_leaf_x += 1.0
            return
        if left != -1:
            dfs(left)
        if right != -1:
            dfs(right)
        # center internal node between children
        child_xs = []
        if left != -1:
            child_xs.append(x[left])
        if right != -1:
            child_xs.append(x[right])
        x[node] = float(np.mean(child_xs)) if child_xs else next_leaf_x

    dfs(0)
    return TreeLayout(x=x, y=y, depth=depths)


def interactive_tree_figure(
    model: DecisionTreeClassifier,
    *,
    feature_names_2d: list[str],
    class_names: list[str],
) -> go.Figure:
    """
    Plotly tree graph with clickable nodes (via streamlit-plotly-events in app).

    Marker size and figure height are scaled by leaf count and tree depth so
    larger trees (Titanic with one-hot features and ``max_depth=None``, etc.)
    don't get crammed into a fixed-size canvas.
    """
    t = model.tree_
    layout = compute_tree_layout(model)
    n = int(t.node_count)

    # Tree-shape stats used to size the figure.
    if n > 0:
        is_leaf_arr = np.fromiter(
            (
                (int(t.children_left[i]) == -1) and (int(t.children_right[i]) == -1)
                for i in range(n)
            ),
            dtype=bool,
            count=n,
        )
        n_leaves = int(is_leaf_arr.sum()) or 1
        max_depth = int(layout.depth.max())
    else:
        n_leaves = 1
        max_depth = 0

    # Marker / font sizing: shrink as the leaf count grows so neighbouring
    # nodes stop overlapping in a horizontally-fixed Streamlit container, but
    # never below 18 px (~touch-target minimum) so individual nodes stay
    # readable and clickable. For very wide trees (Titanic at full depth has
    # ~300 leaves) some horizontal overlap is unavoidable; lower max_depth
    # if you need a less crowded view.
    marker_size = int(max(18, min(28, 800 / n_leaves)))
    font_size = max(10, min(12, marker_size // 2 + 1))

    # Vertical sizing: each level of depth gets ~80 px of room. Capped so the
    # figure doesn't become unscrollable.
    fig_height = int(max(520, min(1400, 90 * (max_depth + 1) + 40)))

    # Build edge segments
    xs: list[float] = []
    ys: list[float] = []
    for node in range(n):
        for child in (int(t.children_left[node]), int(t.children_right[node])):
            if child == -1:
                continue
            xs += [layout.x[node], layout.x[child], None]
            ys += [layout.y[node], layout.y[child], None]

    edge_trace = go.Scatter(
        x=xs,
        y=ys,
        mode="lines",
        line=dict(color="rgba(120,120,120,0.7)", width=1),
        hoverinfo="skip",
        showlegend=False,
    )

    # Node labels
    labels: list[str] = []
    hover: list[str] = []
    custom: list[int] = []

    for node in range(n):
        feat = int(t.feature[node])
        thr = float(t.threshold[node])
        impurity = float(t.impurity[node])
        samples = int(t.n_node_samples[node])

        is_leaf = (int(t.children_left[node]) == -1) and (int(t.children_right[node]) == -1)
        if feat == -2 or is_leaf:
            label = f"{node}"
            split_txt = "leaf"
        else:
            fname = feature_names_2d[feat] if feat < len(feature_names_2d) else f"f{feat}"
            label = f"{node}"
            split_txt = f"{fname} ≤ {thr:.3g}"

        labels.append(label)
        hover.append(
            "<br>".join(
                [
                    f"<b>node</b>: {node}",
                    f"<b>split</b>: {split_txt}",
                    f"<b>impurity</b>: {impurity:.4f}",
                    f"<b>samples</b>: {samples}",
                    "<i>Click to inspect</i>",
                ]
            )
        )
        custom.append(int(node))

    node_trace = go.Scatter(
        x=layout.x.tolist(),
        y=layout.y.tolist(),
        mode="markers+text",
        text=labels,
        textposition="middle center",
        textfont=dict(size=font_size, color="white"),
        marker=dict(
            size=marker_size,
            color="rgba(31,119,180,0.9)",
            line=dict(color="white", width=1),
        ),
        hovertext=hover,
        hoverinfo="text",
        customdata=custom,
        showlegend=False,
    )

    fig = go.Figure(data=[edge_trace, node_trace])

    # Tiny axis padding so leaves at the edges aren't half-clipped by the plot
    # area, plus extra headroom at the top so the root marker has breathing space.
    if n > 0:
        x_min = float(layout.x.min())
        x_max = float(layout.x.max())
        y_min = float(layout.y.min())
        y_max = float(layout.y.max())
        x_pad = max(0.5, (x_max - x_min) * 0.02)
        y_pad = max(0.5, (y_max - y_min) * 0.05)
    else:
        x_min = x_max = y_min = y_max = 0.0
        x_pad = y_pad = 0.5

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False, range=[x_min - x_pad, x_max + x_pad]),
        yaxis=dict(visible=False, range=[y_min - y_pad, y_max + y_pad]),
        height=fig_height,
        plot_bgcolor="white",
    )
    return fig

