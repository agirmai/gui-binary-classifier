from __future__ import annotations

import json
import math

import streamlit as st
import streamlit.components.v1 as components
from streamlit_plotly_events import plotly_events

from data import DatasetName, load_dataset, make_2d_split
from inspector import format_class_distribution, get_node_info, node_count
from model import current_params, default_params, train_decision_tree
from visualization import decision_boundary_2d, interactive_tree_figure, scatter_2d, tree_graphviz_source


st.set_page_config(page_title="Binary Decision Tree Classifier", layout="wide")

APP_BUILD = "binary-classifier-train-button-2026-05-08"


@st.cache_data(show_spinner="Loading dataset...")
def _load_bundle(name: str):
    """Streamlit-cached wrapper around load_dataset so OpenML downloads only
    happen once per process and don't block subsequent reruns."""
    return load_dataset(name)


def _int_delta(baseline: int, current: int) -> str:
    d = current - baseline
    if d == 0:
        return "0"
    return f"{d:+d}"


def _float_delta(baseline: float, current: float) -> str:
    if math.isnan(baseline) or math.isnan(current):
        return "n/a"
    d = current - baseline
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.3f}"


def _fmt(x: float) -> str:
    return "n/a" if math.isnan(x) else f"{x:.3f}"


st.title("Binary Decision Tree Classifier")
st.caption(
    "Interactive GUI for training and exploring a binary "
    "`sklearn.tree.DecisionTreeClassifier`. Pick a dataset, define the binary "
    "task by choosing the positive class, tweak hyperparameters, and inspect "
    "the resulting tree. Retraining happens automatically on every change."
)
st.caption(f"Build: {APP_BUILD}")

DATASET_OPTIONS: list[DatasetName] = [
    "Titanic",
    "Adult Income",
    "Pima Diabetes",
    "Breast Cancer",
    "Iris",
]


# Per-dataset sensible defaults for the positive class (label = 1).
DEFAULT_POSITIVE_CLASS = {
    "Titanic": "survived",
    "Adult Income": ">50K",
    "Pima Diabetes": "diabetic",
    "Breast Cancer": "malignant",
    "Iris": "versicolor",
}


with st.sidebar:
    st.header("Controls")

    dataset_name: DatasetName = st.selectbox("Dataset", DATASET_OPTIONS)
    bundle = _load_bundle(dataset_name)

    st.subheader("Binary task")
    default_pos = DEFAULT_POSITIVE_CLASS.get(dataset_name, bundle.target_names[0])
    default_pos_idx = (
        bundle.target_names.index(default_pos)
        if default_pos in bundle.target_names
        else 0
    )
    positive_class = st.selectbox(
        "Positive class (label = 1)",
        bundle.target_names,
        index=default_pos_idx,
        key=f"positive_class::{dataset_name}",
        help=(
            "The selected class is treated as the positive (1) class. For "
            "multiclass datasets all other classes are collapsed into the "
            "negative (0) class (one-vs-rest)."
        ),
    )

    st.subheader("2D feature projection")
    f1 = st.selectbox(
        "Feature X",
        bundle.feature_names,
        index=0,
        key=f"feature_x::{dataset_name}",
    )
    default_y_idx = 1 if len(bundle.feature_names) > 1 else 0
    f2 = st.selectbox(
        "Feature Y",
        bundle.feature_names,
        index=default_y_idx,
        key=f"feature_y::{dataset_name}",
    )
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

    st.divider()
    train_clicked = st.button(
        "Train model",
        type="primary",
        width="stretch",
        help=(
            "Re-fits both the baseline and current models with the settings "
            "above. Settings changes do not auto-train; use this button."
        ),
    )

# Build a stable signature of the current sidebar config. Training results
# are cached in session_state and only refreshed when the user clicks Train
# (or on first load, so the page isn't empty).
current_cfg = json.dumps(
    {
        "dataset": dataset_name,
        "positive_class": positive_class,
        "f1": f1,
        "f2": f2,
        "random_state": int(random_state),
        "criterion": criterion,
        "max_depth": max_depth,
        "min_samples_split": int(min_samples_split),
        "min_samples_leaf": int(min_samples_leaf),
    },
    sort_keys=True,
)

needs_train = train_clicked or "trained_cfg" not in st.session_state

if needs_train:
    if f1 == f2:
        st.error("Pick two different features for a 2D decision boundary, then click 'Train model'.")
        st.stop()

    with st.spinner("Training..."):
        split = make_2d_split(
            bundle,
            f1,
            f2,
            positive_class=positive_class,
            random_state=random_state,
        )
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

    st.session_state["split"] = split
    st.session_state["baseline_result"] = baseline_result
    st.session_state["current_result"] = current_result
    st.session_state["trained_cfg"] = current_cfg
    # Clicking Train invalidates any node selected from a previous tree.
    st.session_state["selected_node_id"] = 0
    st.session_state["node_select"] = 0
    st.session_state.pop("_last_click_token", None)
else:
    split = st.session_state["split"]
    baseline_result = st.session_state["baseline_result"]
    current_result = st.session_state["current_result"]

if st.session_state.get("trained_cfg") != current_cfg:
    st.info(
        "Settings have changed since the last training run. "
        "Click **Train model** in the sidebar to refresh the model and plots."
    )

st.divider()
st.markdown(
    f"**Task:** `{split.name}` — predict **{split.target_names[1]}** "
    f"vs **{split.target_names[0]}** (binary classification)."
)

row1 = st.columns(4)
row1[0].metric("Baseline accuracy", _fmt(baseline_result.accuracy))
row1[1].metric(
    "Current accuracy",
    _fmt(current_result.accuracy),
    delta=_float_delta(baseline_result.accuracy, current_result.accuracy),
)
row1[2].metric("Baseline F1", _fmt(baseline_result.f1))
row1[3].metric(
    "Current F1",
    _fmt(current_result.f1),
    delta=_float_delta(baseline_result.f1, current_result.f1),
)

row2 = st.columns(4)
row2[0].metric("Baseline ROC AUC", _fmt(baseline_result.roc_auc))
row2[1].metric(
    "Current ROC AUC",
    _fmt(current_result.roc_auc),
    delta=_float_delta(baseline_result.roc_auc, current_result.roc_auc),
)
row2[2].metric("Baseline depth", f"{baseline_result.depth:d}")
row2[3].metric(
    "Current depth",
    f"{current_result.depth:d}",
    delta=_int_delta(baseline_result.depth, current_result.depth),
)

# Tree section moved to the top (right after metrics) so it appears before boundaries/scatter.
st.divider()
st.subheader("Tree visualization + node inspector")

tree_view_options = ["Interactive (click nodes)", "Graphviz"]
if "tree_view_mode" not in st.session_state:
    st.session_state["tree_view_mode"] = tree_view_options[0]

# Render mode toggle comes *below* so the tree visualization is always the first thing on top.
view_mode = st.session_state["tree_view_mode"]

n_nodes = node_count(current_result.model)
tree_ = current_result.model.tree_


def _node_label(node_idx: int) -> str:
    node_idx = int(node_idx)
    is_leaf = (int(tree_.children_left[node_idx]) == -1) and (int(tree_.children_right[node_idx]) == -1)
    samples = int(tree_.n_node_samples[node_idx])
    impurity = float(tree_.impurity[node_idx])

    feat = int(tree_.feature[node_idx])
    thr = float(tree_.threshold[node_idx])

    if is_leaf or feat == -2:
        split_txt = "leaf"
    else:
        fname = split.feature_names_2d[feat] if 0 <= feat < len(split.feature_names_2d) else f"f{feat}"
        split_txt = f"{fname} ≤ {thr:.3g}"

    return f"{node_idx} · {split_txt} · samples={samples} · impurity={impurity:.3f}"


label_map = {i: _node_label(i) for i in range(n_nodes)}

# Source of truth for the currently inspected node.
if "selected_node_id" not in st.session_state:
    st.session_state["selected_node_id"] = 0
# Tree shape can change (different dataset / hyperparameters). Clamp to a valid range.
if not (0 <= int(st.session_state["selected_node_id"]) < n_nodes):
    st.session_state["selected_node_id"] = 0

if view_mode == "Interactive (click nodes)":
    fig = interactive_tree_figure(
        current_result.model,
        feature_names_2d=split.feature_names_2d,
        class_names=split.target_names,
    )
    # Forward the figure's dynamic height to the click iframe — otherwise it
    # stays at streamlit-plotly-events' 450px default and the bottom of deeper
    # trees gets clipped.
    tree_fig_height = int(fig.layout.height) if fig.layout.height else 520
    # Bind the click component's identity to the tree shape so its internal click
    # cache resets whenever the tree is rebuilt (e.g. dataset / hyperparameters change).
    clicked = plotly_events(
        fig,
        click_event=True,
        hover_event=False,
        select_event=False,
        key=f"tree_click_events_{n_nodes}",
        override_height=tree_fig_height,
    )

    # streamlit-plotly-events v0.0.6 strips `customdata` from the event payload
    # before returning it to Streamlit (only x, y, curveNumber, pointNumber,
    # pointIndex make it through). The node trace is curveNumber == 1 (edges
    # are curveNumber == 0) and points are emitted in node-id order, so
    # pointNumber *is* the node id.
    #
    # The same payload is re-returned on every rerun until a new click
    # happens, so we dedupe by a stable token to keep cached clicks from
    # clobbering whatever the user picked via the dropdown below.
    if clicked:
        ev = clicked[0]
        curve = ev.get("curveNumber")
        pt = ev.get("pointNumber")
        if pt is None:
            pt = ev.get("pointIndex")
        if curve == 1 and pt is not None:
            new_node = int(pt)
            click_token = (curve, new_node)
            if (
                click_token != st.session_state.get("_last_click_token")
                and 0 <= new_node < n_nodes
            ):
                st.session_state["_last_click_token"] = click_token
                st.session_state["selected_node_id"] = new_node
                # Push into the selectbox's own session-state slot *before*
                # the widget is instantiated below so the dropdown picks up
                # the click on this same rerun.
                st.session_state["node_select"] = new_node
    st.caption("Tip: click a blue node to update the stats below.")
else:
    dot_src = tree_graphviz_source(
        current_result.model,
        feature_names_2d=split.feature_names_2d,
        class_names=split.target_names,
    )
    st.graphviz_chart(dot_src, width="stretch")

# Initialise / clamp the selectbox's stored value (keep it consistent with the
# source of truth, but don't blindly overwrite a fresh user choice on every rerun).
if "node_select" not in st.session_state:
    st.session_state["node_select"] = int(st.session_state["selected_node_id"])
if not (0 <= int(st.session_state["node_select"]) < n_nodes):
    st.session_state["node_select"] = int(st.session_state["selected_node_id"])

node_id = st.selectbox(
    "Select node",
    options=list(range(n_nodes)),
    format_func=lambda i: label_map[int(i)],
    key="node_select",
)
# After the widget renders, mirror its value back into the source of truth so a
# dropdown change is honoured on subsequent reruns.
st.session_state["selected_node_id"] = int(node_id)

# Streamlit's selectbox is always typeable; we don't want that for the node
# picker (the list is long, but you should pick by mouse / click-on-tree only).
# Inject JS that flips the underlying <input readonly>, scoped by the label so
# the sidebar selectboxes stay normal. A MutationObserver re-applies after
# every Streamlit rerender.
components.html(
    """
<script>
(function() {
  const w = window.parent;
  const NAV_KEYS = new Set([
    'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight',
    'Enter', 'Escape', 'Tab', 'Home', 'End', 'PageUp', 'PageDown',
    'Shift', 'Control', 'Alt', 'Meta'
  ]);

  const harden = (input) => {
    if (!input || input.dataset.nodeSelectHardened === '1') return;
    input.dataset.nodeSelectHardened = '1';

    // NOTE: don't set the `readonly` attribute on the input — BaseWeb (the
    // UI library Streamlit uses) styles `input[readonly]` with a transparent
    // text fill, which makes the selected value disappear visually. We block
    // typing purely through capture-phase event listeners instead.
    input.style.caretColor = 'transparent';
    input.style.cursor = 'pointer';

    // Block typing at the source: capture-phase listeners run before BaseWeb's.
    input.addEventListener('keydown', (e) => {
      if (!NAV_KEYS.has(e.key)) {
        e.preventDefault();
        e.stopPropagation();
      }
    }, { capture: true });
    input.addEventListener('keypress', (e) => {
      e.preventDefault();
      e.stopPropagation();
    }, { capture: true });
    input.addEventListener('beforeinput', (e) => {
      e.preventDefault();
      e.stopPropagation();
    }, { capture: true });
    input.addEventListener('paste', (e) => {
      e.preventDefault();
      e.stopPropagation();
    }, { capture: true });
    input.addEventListener('drop', (e) => {
      e.preventDefault();
      e.stopPropagation();
    }, { capture: true });
  };

  // Idempotent: harden() short-circuits on already-hardened inputs.
  const apply = () => {
    const boxes = w.document.querySelectorAll('[data-testid="stSelectbox"]');
    for (let i = 0; i < boxes.length; i++) {
      const box = boxes[i];
      const label = box.querySelector('label');
      const txt = label ? (label.innerText || label.textContent || '').trim() : '';
      if (txt !== 'Select node') continue;
      const input = box.querySelector('input');
      if (input) harden(input);
    }
  };

  apply();

  // Cheap 500ms poll. Avoids a MutationObserver on document.body, which would
  // fire on every Plotly / matplotlib DOM mutation and starve the main thread
  // (graphs would never finish rendering).
  if (w.__node_select_interval__) {
    try { clearInterval(w.__node_select_interval__); } catch (e) {}
  }
  w.__node_select_interval__ = setInterval(apply, 500);
})();
</script>
""",
    height=0,
)

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
            "pct": None
            if (pct is None or (isinstance(pct, float) and (math.isnan(pct))))
            else float(pct),
        }
    )
st.table(pretty)

with st.expander("Tree render mode (optional)"):
    new_view_mode = st.radio(
        "Tree view",
        options=tree_view_options,
        index=tree_view_options.index(view_mode),
        horizontal=True,
    )
    st.session_state["tree_view_mode"] = new_view_mode

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
    st.pyplot(fig1.fig, clear_figure=True, width="stretch")
with right:
    fig2 = decision_boundary_2d(
        current_result.model,
        split.X_train,
        split.y_train,
        feature_names_2d=split.feature_names_2d,
        target_names=split.target_names,
        title="Current (controlled)",
    )
    st.pyplot(fig2.fig, clear_figure=True, width="stretch")

st.divider()
st.subheader("Dataset visual explorer")
sc = scatter_2d(
    split.X_train,
    split.y_train,
    feature_names_2d=split.feature_names_2d,
    target_names=split.target_names,
    title=f"{split.name} (train split, standardized)",
)
st.pyplot(sc.fig, clear_figure=True, width="stretch")

