# Explainable AML Detection on the Elliptic Bitcoin Graph

Graph neural network for flagging illicit Bitcoin transactions — built to cut
false positives and explain every flag as a money-laundering typology.

## Result
- Non-graph baseline (XGBoost): flags **131 of 15,587 licit transactions as
  illicit — a 0.84% false-positive rate** on the temporal test set (steps
  35–49). Precision 0.86 / recall 0.74 / F1 0.79 on the illicit class.
- GNN (GraphSAGE/GCN): _pending — milestone 3._
- Key finding: _pending — milestone 3._

The baseline (`baseline.py`) uses node features only — no edges, no graph
structure — and is the comparison anchor for the GNN. Confusion matrix on the
test set:

|                  | pred licit | pred illicit |
|------------------|-----------:|-------------:|
| **true licit**   |     15,456 |          131 |
| **true illicit** |        287 |          796 |

Milestone 1 (EDA) is complete. The dataset is severely imbalanced — only
**2.23% of all transactions are labeled illicit** — and carries a known
mid-timeline distribution shift that will make the later test steps the hard
part of the problem. Details below.

## Why this framing
False positives, not accuracy, are the real cost in AML triage. Flags are mapped
to named typologies (structuring, layering, rapid movement of funds) and each
comes with an explainer subgraph — auditable, not a black box.

## Explainer examples
_Pending — milestone 4._

## Dataset
Elliptic Bitcoin Dataset — 203,769 transactions, 234,355 directed payment
edges, 165 features, 49 time steps. Loaded via
`torch_geometric.datasets.EllipticBitcoinDataset`.

### Class imbalance (from `eda.py`)
| Class    | Count   | Share  |
|----------|---------|--------|
| illicit  | 4,545   | 2.23%  |
| licit    | 42,019  | 20.62% |
| unknown  | 157,205 | 77.15% |

Only 22.85% of nodes carry a label at all; among labeled nodes the illicit
share is 9.76%. Any baseline must be judged on illicit-class precision/recall,
not accuracy.

![Class imbalance and illicit share over time](figures/class_imbalance.png)

### Temporal structure & distribution shift
The 49 time steps are confirmed present. The standard temporal split uses
steps 1–34 for training and 35–49 for testing. There is a well-documented
distribution shift around **time step ~43** — attributed to the sudden shutdown
of a dark marketplace — after which the illicit share collapses (steps 44–46
fall to 1.51% / 0.41% / 0.28% of labeled nodes). This is a known property of
the dataset, not noise: it is why temporal splits are used and why later test
steps are expected to be hard. We do **not** model the time axis explicitly
(no temporal GNN) — we only report it so later metrics are read in context.

## Run it
```bash
# macOS only: xgboost needs the OpenMP runtime — run `brew install libomp` first
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python eda.py        # milestone 1 — prints stats, writes figures/class_imbalance.png
python baseline.py   # milestone 2 — trains XGBoost baseline, prints test metrics
```

## Stack
PyTorch Geometric · XGBoost · GNNExplainer
