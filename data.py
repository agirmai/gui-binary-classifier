from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DatasetName = Literal["Iris", "Wine", "Breast Cancer"]


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    target_names: list[str]


@dataclass(frozen=True)
class SplitBundle:
    name: str
    feature_names_2d: list[str]
    target_names: list[str]
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


def load_dataset(name: DatasetName) -> DatasetBundle:
    if name == "Iris":
        ds = load_iris()
    elif name == "Wine":
        ds = load_wine()
    elif name == "Breast Cancer":
        ds = load_breast_cancer()
    else:
        raise ValueError(f"Unknown dataset: {name}")

    X = np.asarray(ds.data, dtype=float)
    y = np.asarray(ds.target, dtype=int)
    feature_names = [str(f) for f in ds.feature_names]
    target_names = [str(t) for t in ds.target_names]
    return DatasetBundle(name=name, X=X, y=y, feature_names=feature_names, target_names=target_names)


def make_2d_split(
    bundle: DatasetBundle,
    feature_x: str,
    feature_y: str,
    *,
    test_size: float = 0.25,
    random_state: int = 42,
    stratify: bool = True,
    scale: bool = True,
) -> SplitBundle:
    fx_idx = bundle.feature_names.index(feature_x)
    fy_idx = bundle.feature_names.index(feature_y)

    X2 = bundle.X[:, [fx_idx, fy_idx]]
    y = bundle.y

    strat = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X2, y, test_size=test_size, random_state=random_state, stratify=strat
    )

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return SplitBundle(
        name=bundle.name,
        feature_names_2d=[feature_x, feature_y],
        target_names=bundle.target_names,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

