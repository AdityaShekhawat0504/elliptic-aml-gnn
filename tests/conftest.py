"""Shared fixtures: a tiny synthetic Elliptic-shaped dataset.

No test touches the network or the real ~700 MB dataset. Everything is built
in a temp dir and the scripts' path constants are monkeypatched to point there.
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

N_FEATURES = 4


def make_raw_frames(n_per_step=4, steps=(1, 2, 34, 35, 36, 49)):
    """Build (features_df, classes_df) mimicking the raw Elliptic CSVs.

    features: no header — col 0 = txId, col 1 = time step, cols 2+ = features
    classes : header txId,class — "1"=illicit, "2"=licit, "unknown"
    """
    feat_rows, class_rows = [], []
    tx = 1000
    raw_cycle = ["1", "2", "2", "unknown"]
    for s in steps:
        for i in range(n_per_step):
            raw = raw_cycle[i % len(raw_cycle)]
            # make illicit nodes separable on feature 0 so tiny models can learn
            f0 = 5.0 if raw == "1" else 0.0
            feat_rows.append([tx, s, f0] + [float(tx % 7 + j) for j in range(N_FEATURES - 1)])
            class_rows.append({"txId": tx, "class": raw})
            tx += 1
    return pd.DataFrame(feat_rows), pd.DataFrame(class_rows)


@pytest.fixture
def raw_dir(tmp_path):
    feat, classes = make_raw_frames()
    d = tmp_path / "raw"
    d.mkdir()
    feat.to_csv(d / "elliptic_txs_features.csv", header=False, index=False)
    classes.to_csv(d / "elliptic_txs_classes.csv", index=False)
    return str(d)
