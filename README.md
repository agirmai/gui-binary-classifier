# Binary Decision Tree Classifier — Interactive GUI

A small Streamlit app that lets you train and explore a binary
`sklearn.tree.DecisionTreeClassifier` end-to-end:

- pick from five recognisable binary classification datasets (Titanic,
  Adult Income, Pima Diabetes, Breast Cancer, Iris)
- define the binary task by choosing the positive class
- choose two features to project onto for visualisation
- tune the model's hyperparameters (`criterion`, `max_depth`,
  `min_samples_split`, `min_samples_leaf`, `random_state`)
- click **Train model** to (re)fit. Sidebar changes don't auto-train;
  the app shows a "settings changed" notice until you re-train
- compare a default-hyperparameter **baseline** model against your
  **current** model on accuracy, F1 score, ROC AUC, and tree depth
- click any node in the trained tree to inspect its split, impurity,
  sample count, and class distribution
- see the decision boundary the tree induces on the 2D projection

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

The app ships with five datasets, all framed as binary classification:

| Dataset        | Source                       | Native classes                        | Binary framing                                       | First-load network? |
| -------------- | ---------------------------- | ------------------------------------- | ---------------------------------------------------- | ------------------- |
| Titanic        | OpenML (`titanic`, v1)       | `did_not_survive`, `survived`         | natively binary (pick the positive class)            | yes                 |
| Adult Income   | OpenML (`adult`, v2)         | `<=50K`, `>50K`                       | natively binary (pick the positive class)            | yes                 |
| Pima Diabetes  | OpenML (`diabetes`, v1)      | `not_diabetic`, `diabetic`            | natively binary (pick the positive class)            | yes                 |
| Breast Cancer  | scikit-learn (`load_*`)      | `malignant`, `benign`                 | natively binary (pick the positive class)            | no                  |
| Iris           | scikit-learn (`load_*`)      | `setosa`, `versicolor`, `virginica`   | one-vs-rest reduction (pick the positive class)      | no                  |

OpenML datasets are downloaded once and cached on disk under sklearn's
standard data home (`~/scikit_learn_data` by default), so subsequent
runs are offline.

For multiclass datasets (Iris) the chosen positive class becomes label
`1` and every other class is collapsed into label `0`. The split is
stratified, the two selected features are standardised (z-score) using
statistics fit on the train set only.

For the Adult and Titanic datasets only a tractable subset of features
is exposed — numeric columns for Adult, and a small numeric + one-hot
encoded categorical mix for Titanic — so the 2D feature picker stays
honest.

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

1. **Dataset** — pick `Titanic`, `Adult Income`, `Pima Diabetes`,
   `Breast Cancer`, or `Iris`.
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
6. **Train model** — re-fits both the baseline (default
   hyperparameters) and current (your hyperparameters) models with
   the selected dataset / features / seed. The app trains once
   automatically on first load, after that you have to click this
   button to refresh; while sidebar settings differ from the
   currently-displayed model an "settings have changed" banner
   appears at the top of the main panel.

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
