import numpy as np
import pandas as pd
import pytest

import baseline


@pytest.fixture
def patched(monkeypatch, raw_dir, tmp_path):
    monkeypatch.setattr(baseline, "RAW_DIR", raw_dir)
    monkeypatch.setattr(baseline, "OUT_DIR", str(tmp_path / "outputs"))
    return tmp_path


def test_load_drops_unknown_and_maps_labels(patched):
    df, feature_cols = baseline.load()
    # 6 steps x 4 nodes, one in four is unknown -> 18 labeled
    assert len(df) == 18
    assert set(df["label"].unique()) == {baseline.LICIT, baseline.ILLICIT}
    assert (df["label"] == baseline.ILLICIT).sum() == 6
    assert "txId" not in feature_cols and "time_step" not in feature_cols
    assert len(feature_cols) == 4


def test_temporal_split_has_no_leakage(patched):
    df, feature_cols = baseline.load()
    X_train, y_train, X_test, y_test, tx_test = baseline.build_split(df, feature_cols)
    test_steps = df.set_index("txId").loc[tx_test, "time_step"]
    assert (test_steps > baseline.TRAIN_STEPS_MAX).all()
    assert len(y_train) == 9  # steps 1, 2, 34
    assert len(y_test) == 9   # steps 35, 36, 49
    assert X_train.shape[1] == X_test.shape[1] == 4


def test_missing_raw_data_gives_actionable_error(monkeypatch, tmp_path):
    monkeypatch.setattr(baseline, "RAW_DIR", str(tmp_path / "nope"))
    with pytest.raises(FileNotFoundError, match="eda.py"):
        baseline.load()


class StubClassifier:
    """Deterministic stand-in for XGBClassifier.predict / predict_proba."""

    def __init__(self, y_pred):
        self.y_pred = np.asarray(y_pred)

    def predict(self, X):
        return self.y_pred

    def predict_proba(self, X):
        p = self.y_pred.astype(float)
        return np.column_stack([1 - p, p])


def test_evaluate_metrics_and_prediction_csv(patched):
    y_test = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1, 1, 0])  # tn=3 fp=1 fn=1 tp=3
    tx = np.arange(8) + 100
    precision, recall, f1, fp_rate, cm = baseline.evaluate(
        StubClassifier(y_pred), np.zeros((8, 1)), y_test, tx
    )
    assert cm == (3, 1, 1, 3)
    assert precision == pytest.approx(0.75)
    assert recall == pytest.approx(0.75)
    assert f1 == pytest.approx(0.75)
    assert fp_rate == pytest.approx(0.25)

    out = pd.read_csv(patched / "outputs" / "baseline_test_predictions.csv")
    assert list(out.columns) == ["txId", "true_label", "predicted_label", "predicted_prob"]
    assert out["txId"].tolist() == tx.tolist()


def test_train_fits_on_tiny_data(patched):
    df, feature_cols = baseline.load()
    X_train, y_train, X_test, y_test, _ = baseline.build_split(df, feature_cols)
    clf = baseline.train(X_train, y_train)
    assert clf.predict(X_test).shape == y_test.shape
