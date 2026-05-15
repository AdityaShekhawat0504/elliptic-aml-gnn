"""Milestone 2 — Non-graph baseline (XGBoost) for the Elliptic Bitcoin Graph.

Trains a single XGBoost classifier on node features only — no edges, no graph
structure. This is the comparison anchor for the GNN in milestone 3.

  - labeled nodes only (illicit + licit; the ~77% unknown are dropped)
  - temporal split: train on time steps 1-34, test on 35-49 (not random)
  - class imbalance handled with scale_pos_weight (no hyperparameter search)

Reports on the test set: precision / recall / F1 on the illicit class, the
false-positive rate, and a confusion matrix.

Loads the raw Elliptic CSVs directly with pandas (downloaded by milestone 1's
EDA). This is a non-graph baseline, so it deliberately does NOT import torch /
torch_geometric — it only needs node features and labels as arrays.
"""

import os

import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from xgboost import XGBClassifier

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(ROOT, "data", "raw")
OUT_DIR = os.path.join(ROOT, "outputs")

# our label encoding (kept consistent with the rest of the project)
LICIT, ILLICIT = 0, 1
# raw elliptic_txs_classes.csv encoding -> ours
RAW_LABEL_MAP = {"1": ILLICIT, "2": LICIT}
TRAIN_STEPS_MAX = 34  # train on 1-34, test on 35-49


def load():
    """Load labeled nodes only from the raw CSVs: features, time step, label."""
    # features: no header — col 0 = txId, col 1 = time step, cols 2+ = features
    feat = pd.read_csv(os.path.join(RAW_DIR, "elliptic_txs_features.csv"), header=None)
    feat = feat.rename(columns={0: "txId", 1: "time_step"})

    # classes: header "txId,class" — raw class is "1"=illicit, "2"=licit, "unknown"
    classes = pd.read_csv(os.path.join(RAW_DIR, "elliptic_txs_classes.csv"))
    classes["label"] = classes["class"].map(RAW_LABEL_MAP)
    classes = classes.dropna(subset=["label"])  # drop the ~77% unknown
    classes["label"] = classes["label"].astype(int)

    df = feat.merge(classes[["txId", "label"]], on="txId", how="inner")
    feature_cols = [c for c in feat.columns if c not in ("txId", "time_step")]
    return df, feature_cols


def build_split(df, feature_cols):
    """Temporal split: steps 1-34 train, 35-49 test."""
    train = df[df["time_step"] <= TRAIN_STEPS_MAX]
    test = df[df["time_step"] > TRAIN_STEPS_MAX]

    X_train, y_train = train[feature_cols].values, train["label"].values
    X_test, y_test = test[feature_cols].values, test["label"].values
    tx_test = test["txId"].values

    print("=" * 60)
    print("TEMPORAL SPLIT (labeled nodes only)")
    print("=" * 60)
    print(f"  features per node: {len(feature_cols)}")
    for name, yy in (("train (steps 1-34)", y_train), ("test  (steps 35-49)", y_test)):
        n = len(yy)
        illicit = int((yy == ILLICIT).sum())
        print(f"  {name}: {n:>7,} nodes  |  illicit {illicit:>6,} ({100 * illicit / n:5.2f}%)")
    return X_train, y_train, X_test, y_test, tx_test


def train(X_train, y_train):
    pos = int((y_train == ILLICIT).sum())
    neg = int((y_train == LICIT).sum())
    scale_pos_weight = neg / pos
    print()
    print(f"scale_pos_weight = {neg:,} licit / {pos:,} illicit = {scale_pos_weight:.2f}")

    clf = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate(clf, X_test, y_test, tx_test):
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, ILLICIT]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=[ILLICIT], average=None, zero_division=0
    )
    precision, recall, f1 = precision[0], recall[0], f1[0]

    # confusion matrix with explicit label order [licit, illicit]
    cm = confusion_matrix(y_test, y_pred, labels=[LICIT, ILLICIT])
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

    # save test-set predictions so milestone 4 can compare the two models
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "baseline_test_predictions.csv")
    pd.DataFrame(
        {
            "txId": tx_test,
            "true_label": y_test,
            "predicted_label": y_pred,
            "predicted_prob": y_prob,
        }
    ).to_csv(out_path, index=False)
    print()
    print(f"saved test-set predictions -> {os.path.relpath(out_path, ROOT)}")
    return precision, recall, f1, fp_rate, (tn, fp, fn, tp)


def main():
    df, feature_cols = load()
    X_train, y_train, X_test, y_test, tx_test = build_split(df, feature_cols)
    clf = train(X_train, y_train)
    evaluate(clf, X_test, y_test, tx_test)


if __name__ == "__main__":
    main()
