import pandas as pd
import pytest

import explain

COLS = ["txId", "true_label", "predicted_label", "predicted_prob"]


def write_preds(tmp_path, monkeypatch, gnn_rows, base_rows):
    g, b = tmp_path / "gnn.csv", tmp_path / "base.csv"
    pd.DataFrame(gnn_rows, columns=COLS).to_csv(g, index=False)
    pd.DataFrame(base_rows, columns=COLS).to_csv(b, index=False)
    monkeypatch.setattr(explain, "GNN_PRED_PATH", str(g))
    monkeypatch.setattr(explain, "BASELINE_PRED_PATH", str(b))


def test_disagreement_sets_join_on_txid_not_row_order(tmp_path, monkeypatch):
    gnn_rows = [
        (1, 1, 1, 0.9),  # GNN caught, baseline missed -> key
        (2, 1, 0, 0.1),  # baseline caught, GNN missed -> reverse
        (3, 1, 1, 0.8),  # both caught -> overlap
        (4, 1, 0, 0.2),  # both missed
        (5, 0, 1, 0.7),  # licit: never in any illicit set
    ]
    base_rows = [  # deliberately shuffled
        (5, 0, 0, 0.1),
        (3, 1, 1, 0.9),
        (1, 1, 0, 0.3),
        (4, 1, 0, 0.1),
        (2, 1, 1, 0.8),
    ]
    write_preds(tmp_path, monkeypatch, gnn_rows, base_rows)
    key, reverse, overlap = explain.disagreement_analysis()
    assert key["txId"].tolist() == [1]
    assert reverse["txId"].tolist() == [2]
    assert overlap["txId"].tolist() == [3]


def test_disagreement_rejects_mismatched_txids(tmp_path, monkeypatch):
    write_preds(tmp_path, monkeypatch, [(1, 1, 1, 0.9)], [(2, 1, 1, 0.9)])
    with pytest.raises(AssertionError):
        explain.disagreement_analysis()


def test_select_cases_ranks_by_gnn_confidence_and_caps(monkeypatch):
    key = pd.DataFrame({
        "txId": range(8),
        "predicted_prob_gnn": [0.1, 0.9, 0.5, 0.99, 0.3, 0.7, 0.2, 0.6],
        "predicted_prob_base": [0.0] * 8,
    })
    chosen = explain.select_cases(key)
    assert len(chosen) == explain.MAX_CASES
    assert chosen["txId"].tolist() == [3, 1, 5, 7, 2]
    assert len(explain.select_cases(key.head(0))) == 0


def test_describe_structure_shape_hints():
    chain = [((0, 1), 1.0), ((1, 2), 0.5)]
    s = explain.describe_structure(0, chain, true_label=None, full_deg=1)
    assert s["shape_hint"].startswith("chain")
    assert s["target_deg_in_fig"] == 1 and s["n_nodes"] == 3

    hub = [((0, i), 1.0) for i in range(1, 6)]
    s = explain.describe_structure(0, hub, true_label=None, full_deg=5)
    assert s["shape_hint"].startswith("hub")
    assert s["leaves"] == 5


def test_load_model_without_checkpoint_says_run_gnn(tmp_path, monkeypatch):
    monkeypatch.setattr(explain, "MODEL_PATH", str(tmp_path / "missing.pt"))
    with pytest.raises(FileNotFoundError, match="gnn.py"):
        explain.load_model(4)
