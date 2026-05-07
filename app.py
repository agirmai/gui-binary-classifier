from __future__ import annotations

import math

import streamlit as st

from data import DatasetName, load_dataset, make_2d_split
from inspector import format_class_distribution, get_node_info, node_count
from model import current_params, default_params, train_decision_tree
from visualization import decision_boundary_2d, scatter_2d, tree_graphviz_source


st.set_page_config(page_title="Decision Tree Lab", layout="wide")


def _depth_delta(baseline_depth: int, current_depth: int) -> str:
    d = current_depth - baseline_depth
    if d == 0:
        return "0"
    return f"{d:+d}"


def _acc_delta(baseline_acc: float, current_acc: float) -> str:
    d = current_acc - baseline_acc
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.3f}"


st.title("Decision Tree Lab")
st.caption("Interactively compare a baseline DecisionTreeClassifier vs a user-controlled one. All retraining happens automatically on widget changes.")

with st.sidebar:
    st.header("Controls")

    dataset_name: DatasetName = st.selectbox("Dataset", ["Iris", "Wine", "Breast Cancer"])
    bundle = load_dataset(dataset_name)

    st.subheader("2D feature projection")
    f1 = st.selectbox("Feature X", bundle.feature_names, index=0)
    default_y_idx = 1 if len(bundle.feature_names) > 1 else 0
    f2 = st.selectbox("Feature Y", bundle.feature_names, index=default_y_idx)
    if f1 == f2:
        st.warning("Pick two different features for a 2D decision boundary.")

    st.subheader("Random seed explorer")
    random_state = st.slider("random_state", min_value=0, max_value=10_000, value=42, step=1)

    st.subheader("Hyperparameters (current model)")
    criterion = st.selectbox("criterion", ["gini", "entropy", "log_loss"], index=0)
    max_depth_opt = st.selectbox("max_depth", ["None", "1", "2", "3", "4", "5", "8", "12", "20"], index=0)
    max_depth = None if max_depth_opt == "None" else int(max_depth_opt)

    min_samples_split = st.slider("min_samples_split", min_value=2, max_value=50, value=2, step=1)
    min_samples_leaf = st.slider("min_samples_leaf", min_value=1, max_value=50, value=1, step=1)

split = make_2d_split(bundle, f1, f2, random_state=random_state)

# Train baseline (defaults) and current (controlled).
baseline_result = train_decision_tree(
    split.X_train,
    split.y_train,
    split.X_test,
    split.y_test,
    params={**default_params(), "random_state": int(random_state)},
)

curr_params = current_params(
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    criterion=criterion,
    random_state=random_state,
)
current_result = train_decision_tree(
    split.X_train,
    split.y_train,
    split.X_test,
    split.y_test,
    params=curr_params,
)

st.divider()

colA, colB, colC, colD = st.columns(4)
colA.metric("Baseline accuracy", f"{baseline_result.accuracy:.3f}")
colB.metric("Current accuracy", f"{current_result.accuracy:.3f}", delta=_acc_delta(baseline_result.accuracy, current_result.accuracy))
colC.metric("Baseline depth", f"{baseline_result.depth:d}")
colD.metric("Current depth", f"{current_result.depth:d}", delta=_depth_delta(baseline_result.depth, current_result.depth))

st.divider()

st.subheader("Decision boundary comparison (train set)")
left, right = st.columns(2, gap="large")
with left:
    fig1 = decision_boundary_2d(
        baseline_result.model,
        split.X_train,
        split.y_train,
        feature_names_2d=split.feature_names_2d,
        target_names=split.target_names,
        title="Baseline (defaults)",
    )
    st.pyplot(fig1.fig, clear_figure=True, use_container_width=True)
with right:
    fig2 = decision_boundary_2d(
        current_result.model,
        split.X_train,
        split.y_train,
        feature_names_2d=split.feature_names_2d,
        target_names=split.target_names,
        title="Current (controlled)",
    )
    st.pyplot(fig2.fig, clear_figure=True, use_container_width=True)

st.divider()

st.subheader("Dataset visual explorer")
sc = scatter_2d(
    split.X_train,
    split.y_train,
    feature_names_2d=split.feature_names_2d,
    target_names=split.target_names,
    title=f"{split.name} (train split, standardized)",
)
st.pyplot(sc.fig, clear_figure=True, use_container_width=True)

st.divider()

st.subheader("Tree visualization + node inspector")

tree_col, inspect_col = st.columns([1.2, 0.8], gap="large")

with tree_col:
    st.caption("Current model tree (Graphviz). Node clicking isn’t supported in Streamlit’s Graphviz renderer, so use the node selector on the right.")
    dot_src = tree_graphviz_source(
        current_result.model,
        feature_names_2d=split.feature_names_2d,
        class_names=split.target_names,
    )
    st.graphviz_chart(dot_src, use_container_width=True)

with inspect_col:
    n_nodes = node_count(current_result.model)
    node_id = st.selectbox("Select node", options=list(range(n_nodes)), index=0)

    info = get_node_info(
        current_result.model,
        node_id=int(node_id),
        feature_names=split.feature_names_2d,
    )

    st.markdown("#### Node details")
    st.write(
        {
            "node_id": info.node_id,
            "is_leaf": info.is_leaf,
            "feature": info.feature_name,
            "threshold": None if info.threshold is None else float(info.threshold),
            "impurity": float(info.impurity),
            "samples": int(info.n_node_samples),
            "weighted_samples": float(info.weighted_n_node_samples),
        }
    )

    dist = format_class_distribution(info.value)
    st.markdown("#### Class distribution")
    pretty = []
    for row in dist:
        idx = int(row["class_index"])
        name = split.target_names[idx] if idx < len(split.target_names) else str(idx)
        cnt = float(row["count"])
        pct = row["pct"]
        pretty.append(
            {
                "class": name,
                "count": cnt,
                "pct": None if (pct is None or (isinstance(pct, float) and (math.isnan(pct)))) else float(pct),
            }
        )
    st.table(pretty)

