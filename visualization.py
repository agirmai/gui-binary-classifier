from __future__ import annotations

import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
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

