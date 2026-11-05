import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

import baseline
import gnn


def fake_dataset_from_raw(raw_dir):
    """Build what EllipticBitcoinDataset would return, from the synthetic CSVs."""
    feat = pd.read_csv(f"{raw_dir}/elliptic_txs_features.csv", header=None)
    classes = pd.read_csv(f"{raw_dir}/elliptic_txs_classes.csv", dtype={"class": str})
    raw_to_y = {"1": gnn.ILLICIT, "2": gnn.LICIT, "unknown": gnn.UNKNOWN}
    y = torch.tensor(classes["class"].map(raw_to_y).values)
    x = torch.tensor(feat.iloc[:, 2:].values, dtype=torch.float)
    n = len(feat)
    # a directed chain 0->1->2->... ; gnn.load should make it undirected
    edge_index = torch.tensor([list(range(n - 1)), list(range(1, n))])
    return [Data(x=x, edge_index=edge_index, y=y)]


@pytest.fixture
def patched(monkeypatch, raw_dir, tmp_path):
    monkeypatch.setattr(gnn, "RAW_DIR", raw_dir)
    monkeypatch.setattr(gnn, "OUT_DIR", str(tmp_path / "outputs"))
    # no download: replace the PyG dataset class with an in-memory fake
    monkeypatch.setattr(gnn, "EllipticBitcoinDataset", lambda root: fake_dataset_from_raw(raw_dir))
    return tmp_path


def test_graphsage_output_shape():
    model = gnn.GraphSAGE(in_dim=4, hidden_dim=8, out_dim=2)
    x = torch.randn(5, 4)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    assert model(x, edge_index).shape == (5, 2)


def test_load_builds_temporal_masks_and_undirected_edges(patched):
    data, train_mask, test_mask, tx_id = gnn.load()
    assert not (train_mask & test_mask).any()
    assert (data.y[train_mask] != gnn.UNKNOWN).all()
    assert (data.y[test_mask] != gnn.UNKNOWN).all()
    assert int(train_mask.sum()) == 9 and int(test_mask.sum()) == 9
    # chain of n-1 directed edges -> 2(n-1) undirected
    assert data.edge_index.size(1) == 2 * (data.num_nodes - 1)


def test_gnn_and_baseline_share_identical_test_set(patched, monkeypatch, raw_dir):
    """The core claim of the project: both models score the same test nodes."""
    monkeypatch.setattr(baseline, "RAW_DIR", raw_dir)
    df, cols = baseline.load()
    *_, base_tx = baseline.build_split(df, cols)
    _, _, test_mask, tx_id = gnn.load()
    assert sorted(base_tx.tolist()) == sorted(tx_id[test_mask].tolist())


def test_train_and_evaluate_end_to_end(patched, monkeypatch):
    monkeypatch.setattr(gnn, "EPOCHS", 3)
    gnn.set_seed()
    data, train_mask, test_mask, tx_id = gnn.load()
    model = gnn.train(data, train_mask)
    gnn.evaluate(model, data, test_mask, tx_id)
    out = pd.read_csv(patched / "outputs" / "gnn_test_predictions.csv")
    assert len(out) == int(test_mask.sum())
    assert out["predicted_prob"].between(0, 1).all()


def test_set_seed_makes_training_reproducible(patched, monkeypatch):
    monkeypatch.setattr(gnn, "EPOCHS", 3)
    data, train_mask, _, _ = gnn.load()
    params = []
    for _ in range(2):
        gnn.set_seed()
        model = gnn.train(data, train_mask)
        params.append(torch.cat([p.detach().flatten() for p in model.parameters()]))
    assert torch.equal(params[0], params[1])
