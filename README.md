# Decision Tree Lab

Local Streamlit app for interactive experimentation with `sklearn.tree.DecisionTreeClassifier`, including:
- side-by-side baseline vs current model comparison
- animated decision boundary updates
- dataset explorer (Iris/Wine/Breast Cancer)
- tree visualization (Graphviz) + node inspector
- random seed explorer

## Setup

Create and activate a virtual environment (recommended), then install dependencies:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Notes

- Decision boundary plots are based on **two selected features** (2D training + prediction).
- Baseline model uses **default** `DecisionTreeClassifier()` parameters, while the “current” model uses sidebar controls.