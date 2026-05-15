"""Milestone 3 — Graph neural network (GraphSAGE) for the Elliptic Bitcoin Graph.

Trains a 2-layer GraphSAGE on the full transaction graph. Unlike the XGBoost
baseline, message passing uses the graph structure: a node's representation is
built from its neighbours, including the ~77% unlabeled nodes (semi-supervised).
Loss is computed only on labeled training nodes, evaluation only on labeled
test nodes.

  - graph loaded via torch_geometric's EllipticBitcoinDataset
  - edges treated as undirected for message passing (standard for this dataset)
  - temporal split built EXPLICITLY from time steps (train 1-34, test 35-49),
    NOT the dataset's built-in masks — so the test nodes are provably identical
    to baseline.py's split
  - class imbalance handled with a weighted cross-entropy loss (the illicit
    class is up-weighted, comparable in spirit to the baseline's
    scale_pos_weight); no hyperparameter search, one reasonable config

Reports on the test set the same block as baseline.py: precision / recall / F1
on the illicit class, the false-positive rate, and a confusion matrix. Saves
the test-set predictions to outputs/ for milestone 4 (the explainer).

This is a graph model, so it imports torch / torch_geometric and deliberately
does NOT import xgboost — the two never share a process (OpenMP separation).
"""

import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch_geometric.datasets import EllipticBitcoinDataset
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import to_undirected

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
OUT_DIR = os.path.join(ROOT, "outputs")

# label encoding — EllipticBitcoinDataset uses 0=licit, 1=illicit, 2=unknown,
# which already matches the rest of the project.
LICIT, ILLICIT, UNKNOWN = 0, 1, 2
TRAIN_STEPS_MAX = 34  # train on 1-34, test on 35-49

HIDDEN_DIM = 64
EPOCHS = 200
LR = 0.01
WEIGHT_DECAY = 5e-4
SEED = 42


def set_seed(seed=SEED):
    """Seed python, numpy and torch so a run produces identical predictions.

    Milestone 4 (the explainer) joins this model's predictions against the
    baseline's — that disagreement analysis is only meaningful if the GNN run
    is reproducible. CPU GraphSAGE training is deterministic once these three
    generators are fixed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class GraphSAGE(torch.nn.Module):
    """2-layer GraphSAGE — one reasonable config, no architecture search."""

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def load():
    """Load the Elliptic graph and build the temporal split explicitly.

    The node ordering of EllipticBitcoinDataset is the row order of
    elliptic_txs_features.csv, so we read that raw CSV to recover each node's
    txId and time step — and build the train/test masks from the time step
    directly, exactly as baseline.py does.
    """
    dataset = EllipticBitcoinDataset(root=DATA_DIR)
    data = dataset[0]

    # edges -> undirected for message passing (standard for this dataset)
    data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)

    # recover txId + time step per node from the raw features CSV (same order)
    feat = pd.read_csv(os.path.join(RAW_DIR, "elliptic_txs_features.csv"), header=None)
    assert len(feat) == data.num_nodes, "feature CSV row count != graph node count"
    tx_id = torch.from_numpy(feat[0].values.copy())
    time_step = torch.from_numpy(feat[1].values.copy())

    labeled = data.y != UNKNOWN
    train_mask = labeled & (time_step <= TRAIN_STEPS_MAX)
    test_mask = labeled & (time_step > TRAIN_STEPS_MAX)

    print("=" * 60)
    print("TEMPORAL SPLIT (labeled nodes only, built from time steps)")
    print("=" * 60)
    print(f"  nodes in graph   : {data.num_nodes:,} ({data.x.size(1)} features each)")
    print(f"  edges (undirected): {data.edge_index.size(1):,}")
    for name, mask in (("train (steps 1-34)", train_mask), ("test  (steps 35-49)", test_mask)):
        n = int(mask.sum())
        illicit = int((data.y[mask] == ILLICIT).sum())
        print(f"  {name}: {n:>7,} nodes  |  illicit {illicit:>6,} ({100 * illicit / n:5.2f}%)")

    return data, train_mask, test_mask, tx_id


def train(data, train_mask):
    device = torch.device("cpu")
    data = data.to(device)
    train_mask = train_mask.to(device)

    model = GraphSAGE(data.x.size(1), HIDDEN_DIM, out_dim=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # weighted loss: up-weight the illicit class by licit/illicit ratio on the
    # training nodes — same spirit as the baseline's scale_pos_weight.
    y_train = data.y[train_mask]
    pos = int((y_train == ILLICIT).sum())
    neg = int((y_train == LICIT).sum())
    class_weight = torch.tensor([1.0, neg / pos], device=device)
    print()
    print(f"class weight (illicit) = {neg:,} licit / {pos:,} illicit = {neg / pos:.2f}")

    model.train()
    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[train_mask], data.y[train_mask], weight=class_weight)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 50 == 0:
            print(f"  epoch {epoch:>3} | train loss {loss.item():.4f}")

    return model


@torch.no_grad()
def evaluate(model, data, test_mask, tx_id):
    model.eval()
    logits = model(data.x, data.edge_index)
    probs = F.softmax(logits, dim=1)[:, ILLICIT]
    y_pred = logits.argmax(dim=1)

    y_true_test = data.y[test_mask].cpu().numpy()
    y_pred_test = y_pred[test_mask].cpu().numpy()

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_test, y_pred_test, labels=[ILLICIT], average=None, zero_division=0
    )
    precision, recall, f1 = precision[0], recall[0], f1[0]

    cm = confusion_matrix(y_true_test, y_pred_test, labels=[LICIT, ILLICIT])
    tn, fp, fn, tp = cm.ravel()
    fp_rate = fp / (fp + tn)  # licit transactions wrongly flagged illicit

    print()
    print("=" * 60)
    print("TEST-SET RESULTS — illicit class")
    print("=" * 60)
    print(f"  precision          : {precision:.4f}")
    print(f"  recall             : {recall:.4f}")
    print(f"  F1                 : {f1:.4f}")
    print(f"  false-positive rate: {fp_rate:.4f}  ({fp:,} of {fp + tn:,} licit flagged)")
    print()
    print("CONFUSION MATRIX (rows = true, cols = predicted)")
    print(f"               pred licit   pred illicit")
    print(f"  true licit   {tn:>10,}   {fp:>12,}")
    print(f"  true illicit {fn:>10,}   {tp:>12,}")

    # save test-set predictions for milestone 4 (the explainer)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "gnn_test_predictions.csv")
    pd.DataFrame(
        {
            "txId": tx_id[test_mask].cpu().numpy(),
            "true_label": y_true_test,
            "predicted_label": y_pred_test,
            "predicted_prob": probs[test_mask].cpu().numpy(),
        }
    ).to_csv(out_path, index=False)
    print()
    print(f"saved test-set predictions -> {os.path.relpath(out_path, ROOT)}")


def main():
    set_seed()
    data, train_mask, test_mask, tx_id = load()
    model = train(data, train_mask)
    evaluate(model, data, test_mask, tx_id)

    # save the trained model so explain.py (milestone 4) runs GNNExplainer on
    # the *identical* model rather than a re-trained copy. gitignored (*.pt) —
    # it is fully regenerable by re-running this script.
    os.makedirs(OUT_DIR, exist_ok=True)
    model_path = os.path.join(OUT_DIR, "gnn_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"saved trained model       -> {os.path.relpath(model_path, ROOT)}")


if __name__ == "__main__":
    main()
