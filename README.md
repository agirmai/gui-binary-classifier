# Binary Decision Tree Classifier — Interactive GUI

A small Streamlit app that lets you train and explore a binary
`sklearn.tree.DecisionTreeClassifier` end-to-end:

- pick a dataset and define the binary task (positive class)
- choose two features to project onto for visualisation
- tune the model's hyperparameters (`criterion`, `max_depth`,
  `min_samples_split`, `min_samples_leaf`, `random_state`)
- compare a default-hyperparameter **baseline** model against your
  **current** model on accuracy, F1 score, ROC AUC, and tree depth
- click any node in the trained tree to inspect its split, impurity,
  sample count, and class distribution
- see the decision boundary the tree induces on the 2D projection

Retraining happens automatically on every widget change — there is no
explicit "train" button to push.

---

## Repository

`https://github.com/agirmai/gui-binary-classifier`

## Project layout

```
gui-binary-classifier/
├── app.py             # Streamlit entry point (UI + glue)
├── data.py            # dataset loading + binarised 2D train/test split
├── model.py           # DecisionTreeClassifier training + metrics
├── inspector.py       # tree node introspection helpers
├── visualization.py   # decision boundary, scatter, interactive tree, graphviz
├── requirements.txt
└── README.md
```

## Datasets and binary task

The app ships with three scikit-learn datasets:

| Dataset         | Native classes                              | Binary framing in this app                     |
| --------------- | ------------------------------------------- | ---------------------------------------------- |
| Iris            | `setosa`, `versicolor`, `virginica`         | one-vs-rest (pick the positive class)          |
| Wine            | `class_0`, `class_1`, `class_2`             | one-vs-rest (pick the positive class)          |
| Breast Cancer   | `malignant`, `benign`                       | natively binary (pick which class is positive) |

For multiclass datasets, the chosen positive class becomes label `1`
and every other class is collapsed into label `0`. The split is
stratified, the two selected features are standardised (z-score) using
statistics fit on the train set only.

---

## Setup

### Prerequisites

- Python 3.10 or newer
- `pip`
- (optional) the [Graphviz](https://graphviz.org/download/) system
  binary if you want to use the static **Graphviz** tree view. The
  default **Interactive (click nodes)** tree renders without it.

### Install

Clone the repo and create a virtual environment.

**Windows (PowerShell):**

```powershell
git clone https://github.com/agirmai/gui-binary-classifier.git
cd gui-binary-classifier
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux:**

```bash
git clone https://github.com/agirmai/gui-binary-classifier.git
cd gui-binary-classifier
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

From the project root, with the virtual environment activated:

```bash
streamlit run app.py
```

Streamlit will print a local URL (default
`http://localhost:8501`) — open it in a browser. The app is fully
local; no data leaves your machine.

To stop the server, press `Ctrl+C` in the terminal.

---

## Using the GUI

The sidebar groups every control:

1. **Dataset** — pick `Iris`, `Wine`, or `Breast Cancer`.
2. **Binary task → Positive class** — select the class that becomes
   label `1`. All other classes become `0`. This is what makes the
   problem binary.
3. **2D feature projection** — pick the two features used for both
   training and the decision-boundary plot. (The full feature space is
   *not* used so the boundary stays plottable in 2D.)
4. **Random seed explorer** — slide `random_state` to see how stable
   the model is across different train/test shuffles.
5. **Hyperparameters** — `criterion`, `max_depth`,
   `min_samples_split`, `min_samples_leaf`. These feed directly into
   `DecisionTreeClassifier(...)`.

The main panel shows, top to bottom:

- the current binary task, then four metrics for the **baseline**
  (default-hyperparameter) and **current** (your hyperparameters)
  models: accuracy, F1, ROC AUC, tree depth, with deltas;
- an interactive Plotly tree where every node is clickable — clicking
  a node updates the searchable dropdown and the inspector panel
  underneath;
- node details (feature, threshold, impurity, sample count) and the
  per-class distribution at the selected node;
- side-by-side decision-boundary plots for the baseline and current
  models on the 2D projection;
- a scatter plot of the (standardised) training data.

A collapsible "Tree render mode" expander lets you switch between the
interactive Plotly tree and a static Graphviz rendering.

---

## How it works (brief)

- `data.load_dataset` wraps the three sklearn loaders and returns a
  uniform `DatasetBundle`.
- `data.make_2d_split` selects the two requested features, builds a
  binary target (`y == positive_class_index`), stratified-splits it
  75/25, and standardises the features.
- `model.train_decision_tree` fits a `DecisionTreeClassifier`,
  predicts on the test split, and returns accuracy, F1, ROC AUC, and
  the realised tree depth.
- `visualization.interactive_tree_figure` lays the tree out
  deterministically (DFS, leaves on a 1D grid, internal nodes
  centred over their children) and emits a Plotly figure whose nodes
  carry `customdata=node_id` — `streamlit-plotly-events` returns the
  click events back into Streamlit.
- `inspector.get_node_info` and `format_class_distribution` pull
  per-node statistics straight off the underlying
  `sklearn.tree._tree.Tree` object.

## Troubleshooting

- **`pip install` fails on `graphviz`** — only the Python `graphviz`
  package is required (it is a thin wrapper). The system Graphviz
  binary is only needed for the optional Graphviz tree view; the
  default interactive view does not need it.
- **`Module not found: streamlit_plotly_events`** — make sure your
  virtual environment is activated and `pip install -r
  requirements.txt` finished successfully.
- **App launches but the tree is empty** — try a smaller
  `min_samples_leaf` or set `max_depth` to `None`. With overly
  restrictive constraints the tree can collapse to a single leaf.
- **"Same X and Y feature" warning** — pick two distinct features so
  the 2D decision boundary is meaningful.
