from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DatasetName = Literal[
    "Titanic",
    "Adult Income",
    "Pima Diabetes",
    "Breast Cancer",
    "Iris",
]


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
    target_names: list[str]  # exactly 2 entries: [negative_label, positive_label]
    positive_class: str  # the original class label treated as the positive (1) class
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray  # binary {0, 1}
    y_test: np.ndarray  # binary {0, 1}


# ---- Loaders ----------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def load_dataset(name: DatasetName) -> DatasetBundle:
    """Return a uniform DatasetBundle for any of the supported datasets.

    Network-backed datasets (Titanic, Adult, Pima) are fetched via
    :func:`sklearn.datasets.fetch_openml` and cached on disk under sklearn's
    standard data home (`~/scikit_learn_data` by default), so subsequent calls
    do not require internet access.
    """
    if name == "Iris":
        return _from_sklearn_bunch("Iris", load_iris())
    if name == "Breast Cancer":
        return _from_sklearn_bunch("Breast Cancer", load_breast_cancer())
    if name == "Titanic":
        return _load_titanic()
    if name == "Adult Income":
        return _load_adult()
    if name == "Pima Diabetes":
        return _load_pima()
    raise ValueError(f"Unknown dataset: {name}")


def _from_sklearn_bunch(name: str, ds) -> DatasetBundle:
    return DatasetBundle(
        name=name,
        X=np.asarray(ds.data, dtype=float),
        y=np.asarray(ds.target, dtype=int),
        feature_names=[str(f) for f in ds.feature_names],
        target_names=[str(t) for t in ds.target_names],
    )


def _fetch_openml_safe(name: str, **kwargs):
    try:
        return fetch_openml(name, as_frame=True, parser="auto", **kwargs)
    except Exception as e:  # network failures, OpenML outages, etc.
        raise RuntimeError(
            f"Could not download the '{name}' dataset from OpenML. "
            "An internet connection is required on the first run; results "
            "are cached on disk after that. "
            f"Underlying error: {e}"
        ) from e


def _load_titanic() -> DatasetBundle:
    ds = _fetch_openml_safe("titanic", version=1)
    df = ds.frame.copy()
    df = df.dropna(subset=["survived"])

    # A tractable feature set: a few numerics plus two obvious categoricals
    # one-hot encoded. This keeps the 2D feature picker honest.
    feat_df = df[["pclass", "age", "sibsp", "parch", "fare", "sex", "embarked"]].copy()
    feat_df = pd.get_dummies(feat_df, columns=["sex", "embarked"], drop_first=False)

    target_series = df["survived"]
    mask = feat_df.notna().all(axis=1) & target_series.notna()
    feat_df = feat_df.loc[mask]
    target_series = target_series.loc[mask]

    return DatasetBundle(
        name="Titanic",
        X=feat_df.to_numpy(dtype=float),
        y=target_series.astype(int).to_numpy(),
        feature_names=feat_df.columns.tolist(),
        target_names=["did_not_survive", "survived"],
    )


def _load_adult() -> DatasetBundle:
    ds = _fetch_openml_safe("adult", version=2)
    df = ds.frame.copy().dropna()

    target_col = ds.target.name if ds.target.name in df.columns else "class"
    target_str = df[target_col].astype(str).str.strip()

    # Numeric-only feature subset keeps the 2D scatter / boundary plots simple
    # (no need to one-hot 100+ levels of categorical features).
    numeric_cols = [
        "age",
        "fnlwgt",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    X = df[numeric_cols].astype(float).to_numpy()
    y = (target_str == ">50K").astype(int).to_numpy()

    return DatasetBundle(
        name="Adult Income",
        X=X,
        y=y,
        feature_names=numeric_cols,
        target_names=["<=50K", ">50K"],
    )


def _load_pima() -> DatasetBundle:
    ds = _fetch_openml_safe("diabetes", version=1)

    # OpenML's `diabetes` v1 uses cryptic short names; rename for clarity.
    rename_map = {
        "preg": "pregnancies",
        "plas": "glucose",
        "pres": "blood_pressure",
        "skin": "skin_thickness",
        "insu": "insulin",
        "mass": "bmi",
        "pedi": "diabetes_pedigree",
        "age": "age",
    }
    X_df = ds.data.rename(columns=rename_map).astype(float)
    feature_names = X_df.columns.tolist()

    y = (ds.target.astype(str) == "tested_positive").astype(int).to_numpy()

    return DatasetBundle(
        name="Pima Diabetes",
        X=X_df.to_numpy(),
        y=y,
        feature_names=feature_names,
        target_names=["not_diabetic", "diabetic"],
    )


# ---- Split + binarisation ---------------------------------------------------


def make_2d_split(
    bundle: DatasetBundle,
    feature_x: str,
    feature_y: str,
    *,
    positive_class: str,
    test_size: float = 0.25,
    random_state: int = 42,
    stratify: bool = True,
    scale: bool = True,
) -> SplitBundle:
    """Build a stratified train/test split on two features and binarise y.

    The class named ``positive_class`` becomes label 1 (positive); every other
    class collapses to 0 (one-vs-rest reduction for multiclass datasets, plain
    re-labelling for natively-binary ones).
    """
    if positive_class not in bundle.target_names:
        raise ValueError(
            f"positive_class={positive_class!r} not in dataset classes "
            f"{bundle.target_names!r}"
        )
    pos_idx = bundle.target_names.index(positive_class)

    fx_idx = bundle.feature_names.index(feature_x)
    fy_idx = bundle.feature_names.index(feature_y)

    X2 = bundle.X[:, [fx_idx, fy_idx]]
    y_binary = (bundle.y == pos_idx).astype(int)

    strat = y_binary if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X2, y_binary, test_size=test_size, random_state=random_state, stratify=strat
    )

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    if len(bundle.target_names) > 2:
        negative_label = "rest"
    else:
        negative_label = next(
            n for n in bundle.target_names if n != positive_class
        )
    target_names = [negative_label, positive_class]

    return SplitBundle(
        name=bundle.name,
        feature_names_2d=[feature_x, feature_y],
        target_names=target_names,
        positive_class=positive_class,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
