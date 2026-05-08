from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import math

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier


@dataclass(frozen=True)
class ModelResult:
    model: DecisionTreeClassifier
    accuracy: float
    f1: float
    roc_auc: float  # NaN when AUC is undefined (e.g. test set has a single class)
    depth: int


def train_decision_tree(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    params: dict[str, Any] | None = None,
) -> ModelResult:
    params = params or {}
    clf = DecisionTreeClassifier(**params)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    try:
        y_proba = clf.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, y_proba))
    except (ValueError, IndexError):
        auc = math.nan
    depth = int(clf.get_depth())
    return ModelResult(model=clf, accuracy=acc, f1=f1, roc_auc=auc, depth=depth)


def default_params() -> dict[str, Any]:
    return {}


def current_params(
    *,
    max_depth: int | None,
    min_samples_split: int,
    min_samples_leaf: int,
    criterion: str,
    random_state: int,
) -> dict[str, Any]:
    return {
        "max_depth": max_depth,
        "min_samples_split": int(min_samples_split),
        "min_samples_leaf": int(min_samples_leaf),
        "criterion": criterion,
        "random_state": int(random_state),
    }

