from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.tree import DecisionTreeClassifier


@dataclass(frozen=True)
class NodeInfo:
    node_id: int
    is_leaf: bool
    feature_index: int | None
    feature_name: str | None
    threshold: float | None
    impurity: float
    n_node_samples: int
    weighted_n_node_samples: float
    value: np.ndarray


def node_count(tree_model: DecisionTreeClassifier) -> int:
    return int(tree_model.tree_.node_count)


def get_node_info(
    tree_model: DecisionTreeClassifier,
    *,
    node_id: int,
    feature_names: list[str],
) -> NodeInfo:
    t = tree_model.tree_
    node_id = int(node_id)

    feature = int(t.feature[node_id])
    threshold = float(t.threshold[node_id])

    is_leaf = (t.children_left[node_id] == -1) and (t.children_right[node_id] == -1)
    if feature == -2:  # sklearn constant for TREE_UNDEFINED
        feature_index = None
        feature_name = None
        threshold_out = None
    else:
        feature_index = feature
        feature_name = feature_names[feature]
        threshold_out = threshold

    value = np.asarray(t.value[node_id])
    return NodeInfo(
        node_id=node_id,
        is_leaf=is_leaf,
        feature_index=feature_index,
        feature_name=feature_name,
        threshold=threshold_out,
        impurity=float(t.impurity[node_id]),
        n_node_samples=int(t.n_node_samples[node_id]),
        weighted_n_node_samples=float(t.weighted_n_node_samples[node_id]),
        value=value,
    )


def format_class_distribution(value: np.ndarray) -> list[dict[str, float]]:
    """
    Returns per-output class distributions.
    For classifiers, sklearn stores shape (n_outputs, n_classes) under tree_.value[node].
    """
    v = np.asarray(value, dtype=float)
    if v.ndim == 3:
        v = v[0]
    if v.ndim == 2 and v.shape[0] == 1:
        v = v[0]
    if v.ndim == 1:
        total = float(v.sum()) if float(v.sum()) > 0 else 1.0
        return [{"class_index": int(i), "count": float(c), "pct": float(c) / total} for i, c in enumerate(v)]

    rows: list[dict[str, float]] = []
    for i, c in enumerate(v.ravel().tolist()):
        rows.append({"class_index": int(i), "count": float(c), "pct": float("nan")})
    return rows

