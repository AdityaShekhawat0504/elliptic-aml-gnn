"""Milestone 1 — Setup + EDA for the Elliptic Bitcoin Graph.

Loads the dataset via torch_geometric, then reports:
  - class imbalance (illicit / licit / unknown)
  - temporal structure (49 time steps)
  - a note on the known mid-timeline distribution shift
and saves a visualization of the class imbalance + illicit rate over time.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch_geometric.datasets import EllipticBitcoinDataset

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
FIG_DIR = os.path.join(ROOT, "figures")

# y encoding used by torch_geometric's EllipticBitcoinDataset
LABELS = {0: "licit", 1: "illicit", 2: "unknown"}


def load():
    dataset = EllipticBitcoinDataset(root=DATA_DIR)
    data = dataset[0]
    # The time_step column is dropped from data.x, so read it from the raw CSV.
    feat_df = pd.read_csv(dataset.raw_paths[0], header=None)
    time_step = torch.from_numpy(feat_df[1].values)
    return data, time_step


def report_class_imbalance(data):
    n = data.num_nodes
    counts = {LABELS[c]: int((data.y == c).sum()) for c in LABELS}
    print("=" * 60)
    print("CLASS IMBALANCE")
    print("=" * 60)
    print(f"total nodes (transactions): {n:,}")
    for name, c in counts.items():
        print(f"  {name:<8}: {c:>8,}  ({100 * c / n:5.2f}%)")
    labeled = counts["licit"] + counts["illicit"]
    print(f"  labeled : {labeled:>8,}  ({100 * labeled / n:5.2f}%)")
    print(
        f"  -> among labeled nodes, illicit share is "
        f"{100 * counts['illicit'] / labeled:.2f}% — severe imbalance"
    )
    return counts


def report_temporal_structure(data, time_step):
    steps = torch.unique(time_step)
    print()
    print("=" * 60)
    print("TEMPORAL STRUCTURE")
    print("=" * 60)
    print(f"number of time steps: {len(steps)} (expected 49)")
    print(f"time step range: {int(steps.min())} .. {int(steps.max())}")

    per_step = []
    for s in steps:
        mask = time_step == s
        y = data.y[mask]
        total = int(mask.sum())
        illicit = int((y == 1).sum())
        licit = int((y == 0).sum())
        labeled = illicit + licit
        rate = (100 * illicit / labeled) if labeled else 0.0
        per_step.append((int(s), total, illicit, licit, rate))

    print(f"{'step':>4} {'nodes':>8} {'illicit':>8} {'licit':>8} {'illicit%':>9}")
    for s, total, illicit, licit, rate in per_step:
        print(f"{s:>4} {total:>8,} {illicit:>8,} {licit:>8,} {rate:>8.2f}%")
    return per_step


def report_distribution_shift():
    print()
    print("=" * 60)
    print("NOTE — MID-TIMELINE DISTRIBUTION SHIFT")
    print("=" * 60)
    print(
        "The Elliptic dataset has a well-documented distribution shift around\n"
        "time step ~43, attributed to the sudden shutdown of a dark marketplace.\n"
        "Models trained on the earlier steps (the temporal split uses steps 1-34\n"
        "for training) tend to see a sharp drop in illicit-class recall on the\n"
        "later test steps. This is a known phenomenon, not a bug — it makes the\n"
        "dataset a realistic stress test and is why temporal, not random, splits\n"
        "are used. We do NOT model the time axis explicitly (no temporal GNN);\n"
        "we simply report it so the later metrics are read in context."
    )


def plot(counts, per_step):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    names = ["illicit", "licit", "unknown"]
    values = [counts[k] for k in names]
    colors = ["#d62728", "#2ca02c", "#7f7f7f"]
    bars = ax1.bar(names, values, color=colors)
    ax1.set_title("Class imbalance — Elliptic Bitcoin transactions")
    ax1.set_ylabel("number of transactions")
    total = sum(counts.values())
    for bar, v in zip(bars, values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            v,
            f"{v:,}\n({100 * v / total:.1f}%)",
            ha="center",
            va="bottom",
        )
    ax1.set_ylim(0, max(values) * 1.15)

    steps = [s for s, *_ in per_step]
    rates = [r for *_, r in per_step]
    ax2.plot(steps, rates, marker="o", color="#d62728")
    ax2.axvline(43, color="#7f7f7f", linestyle="--", linewidth=1)
    ax2.text(43, ax2.get_ylim()[1], " ~step 43: dark-market shutdown",
             color="#7f7f7f", va="top", fontsize=9)
    ax2.set_title("Illicit share among labeled nodes, per time step")
    ax2.set_xlabel("time step")
    ax2.set_ylabel("illicit % of labeled nodes")

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "class_imbalance.png")
    fig.savefig(out, dpi=150)
    print()
    print(f"saved plot -> {os.path.relpath(out, ROOT)}")


def main():
    data, time_step = load()
    print(f"loaded: {data}")
    print()
    counts = report_class_imbalance(data)
    per_step = report_temporal_structure(data, time_step)
    report_distribution_shift()
    plot(counts, per_step)


if __name__ == "__main__":
    main()
