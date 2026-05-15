"""Milestone 4 — Explainability: disagreement analysis + GNNExplainer subgraphs.

This is the centerpiece. It does NOT re-train anything — it loads the exact
GraphSAGE model saved by gnn.py and the test-set prediction CSVs written by
both models, then:

  1. Disagreement analysis. Joins the two prediction CSVs on txId (on the id,
     not row order) and isolates the KEY SET — illicit transactions the GNN
     caught that the XGBoost baseline missed. Also reports the reverse set and
     the overlap, for honest context.
  2. Case selection. Takes up to 5 cases from the key set, highest GNN
     confidence first. Fewer than 5 available -> explains all and says so.
  3. GNNExplainer. Runs torch_geometric.explain.GNNExplainer on each case,
     over that node's 2-hop neighbourhood (the exact computation graph of the
     2-layer GraphSAGE), and pulls the edge importance mask.
  4. Visualisation. Thresholds the edge mask to the most important edges so
     the figure stays legible, draws the subgraph, colours nodes by true class
     and outlines GNN-flagged-illicit nodes. Saves figures/explain_<txId>.png.
  5. Structural readout. Prints concrete structure per case (neighbour count,
     shape) so the README narratives can be written from real numbers, not a
     template.

Imports torch / torch_geometric (it reuses gnn.py's model + loader) and does
NOT import xgboost — same OpenMP separation as gnn.py.
"""

import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import degree, k_hop_subgraph

from gnn import (
    HIDDEN_DIM,
    ILLICIT,
    LICIT,
    OUT_DIR,
    ROOT,
    SEED,
    UNKNOWN,
    GraphSAGE,
    load,
    set_seed,
)

FIG_DIR = os.path.join(ROOT, "figures")
MODEL_PATH = os.path.join(OUT_DIR, "gnn_model.pt")
GNN_PRED_PATH = os.path.join(OUT_DIR, "gnn_test_predictions.csv")
BASELINE_PRED_PATH = os.path.join(OUT_DIR, "baseline_test_predictions.csv")

MAX_CASES = 5            # explain at most this many cases from the key set
EXPLAINER_EPOCHS = 200   # GNNExplainer mask-optimisation steps
TOP_EDGES = 15           # keep this many top edges per figure for legibility

CLASS_COLOR = {LICIT: "#2ca02c", ILLICIT: "#d62728", UNKNOWN: "#bbbbbb"}
CLASS_NAME = {LICIT: "licit", ILLICIT: "illicit", UNKNOWN: "unknown"}


def load_model(in_dim):
    """Reconstruct the exact GraphSAGE saved by gnn.py — no re-training."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"{os.path.relpath(MODEL_PATH, ROOT)} not found — run `python gnn.py` first."
        )
    model = GraphSAGE(in_dim, HIDDEN_DIM, out_dim=2)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    return model


def disagreement_analysis():
    """Join the two prediction CSVs on txId and isolate the disagreement sets."""
    gnn = pd.read_csv(GNN_PRED_PATH)
    base = pd.read_csv(BASELINE_PRED_PATH)

    # join on the id — row order of the two CSVs is NOT assumed to match
    merged = gnn.merge(base, on="txId", how="inner", suffixes=("_gnn", "_base"))
    assert len(merged) == len(gnn) == len(base), "prediction CSVs do not cover the same txIds"
    # both CSVs carry the ground-truth label; they must agree
    assert (merged["true_label_gnn"] == merged["true_label_base"]).all(), "true labels disagree across CSVs"
    merged = merged.rename(columns={"true_label_gnn": "true_label"})

    true_illicit = merged["true_label"] == ILLICIT
    gnn_illicit = merged["predicted_label_gnn"] == ILLICIT
    base_illicit = merged["predicted_label_base"] == ILLICIT

    key = merged[true_illicit & gnn_illicit & ~base_illicit]      # GNN caught, baseline missed
    reverse = merged[true_illicit & ~gnn_illicit & base_illicit]  # baseline caught, GNN missed
    overlap = merged[true_illicit & gnn_illicit & base_illicit]   # both caught
    neither = merged[true_illicit & ~gnn_illicit & ~base_illicit]  # both missed

    print("=" * 64)
    print("DISAGREEMENT ANALYSIS  (test set, steps 35-49, illicit class)")
    print("=" * 64)
    print(f"  test transactions joined on txId : {len(merged):,}")
    print(f"  truly illicit                    : {int(true_illicit.sum()):,}")
    print("  ----------------------------------------------------------")
    print(f"  GNN caught, baseline missed  (KEY): {len(key):,}")
    print(f"  baseline caught, GNN missed       : {len(reverse):,}")
    print(f"  both caught (overlap)             : {len(overlap):,}")
    print(f"  both missed                       : {len(neither):,}")
    print()

    return key, reverse, overlap


def select_cases(key):
    """Up to MAX_CASES from the key set, highest GNN confidence first."""
    ranked = key.sort_values("predicted_prob_gnn", ascending=False)
    chosen = ranked.head(MAX_CASES)
    if len(key) == 0:
        print("KEY SET IS EMPTY — the GNN caught no illicit transaction the baseline missed.")
        print("That is itself the finding; there is nothing to explain. Stopping.")
    elif len(key) < MAX_CASES:
        print(f"KEY SET HAS ONLY {len(key)} case(s) — explaining all of them.")
    else:
        print(f"Selected the top {MAX_CASES} key cases by GNN confidence:")
    for _, row in chosen.iterrows():
        print(
            f"  txId {int(row['txId'])}  |  GNN p(illicit)={row['predicted_prob_gnn']:.4f}"
            f"  baseline p(illicit)={row['predicted_prob_base']:.4f}"
        )
    print()
    return chosen


def build_explainer(model):
    return Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=EXPLAINER_EPOCHS),
        explanation_type="model",
        node_mask_type=None,
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="node",
            return_type="raw",
        ),
    )


def explain_case(explainer, model, data, node_idx, full_pred, case_seed):
    """Run GNNExplainer on one node's 2-hop neighbourhood; return everything
    needed to draw it and describe it.

    The 2-hop subgraph IS the computation graph of the 2-layer GraphSAGE, so
    the model's prediction on the subgraph is identical to its prediction on
    the full graph — we run the explainer on the subgraph purely for speed.
    """
    subset, sub_edge_index, mapping, _ = k_hop_subgraph(
        node_idx, num_hops=2, edge_index=data.edge_index,
        relabel_nodes=True, num_nodes=data.num_nodes,
    )
    sub_x = data.x[subset]
    target_local = int(mapping[0])

    # sanity: subgraph prediction must match the full-graph prediction
    with torch.no_grad():
        sub_logit = model(sub_x, sub_edge_index)[target_local]
    assert int(sub_logit.argmax()) == int(full_pred[node_idx]), "subgraph prediction drifted"

    set_seed(case_seed)  # GNNExplainer inits its mask randomly — pin it
    explanation = explainer(sub_x, sub_edge_index, index=target_local)
    edge_imp = explanation.edge_mask.detach().cpu().numpy()  # one score per directed edge

    # map sub-edges back to original node ids, collapse to undirected pairs
    src = subset[sub_edge_index[0]].cpu().numpy()
    dst = subset[sub_edge_index[1]].cpu().numpy()
    pair_imp = {}
    for u, v, w in zip(src, dst, edge_imp):
        key = (int(u), int(v)) if u < v else (int(v), int(u))
        pair_imp[key] = max(pair_imp.get(key, 0.0), float(w))

    # full 2-hop neighbourhood size — honest "how big was the computation graph"
    full_nbhd_nodes = len(subset)
    full_nbhd_edges = len(pair_imp)

    # keep the most important edges so the figure stays readable
    ranked = sorted(pair_imp.items(), key=lambda kv: kv[1], reverse=True)
    top = ranked[:TOP_EDGES]
    # guarantee the target node is shown: if no kept edge touches it, add its best one
    if not any(node_idx in pair for pair, _ in top):
        target_edges = [(p, w) for p, w in ranked if node_idx in p]
        if target_edges:
            top.append(target_edges[0])

    return {
        "top_edges": top,
        "full_nbhd_nodes": full_nbhd_nodes,
        "full_nbhd_edges": full_nbhd_edges,
    }


def describe_structure(node_idx, top_edges, true_label, full_deg):
    """Compute concrete structural facts about the thresholded subgraph."""
    G = nx.Graph()
    for (u, v), w in top_edges:
        G.add_edge(u, v, weight=w)
    nodes = list(G.nodes)
    degs = dict(G.degree())
    target_deg = degs.get(node_idx, 0)
    other_degs = [d for n, d in degs.items() if n != node_idx]
    max_other = max(other_degs) if other_degs else 0
    n_nodes, n_edges = G.number_of_nodes(), G.number_of_edges()
    density = nx.density(G) if n_nodes > 1 else 0.0
    leaves = sum(1 for d in other_degs if d == 1)

    # shape heuristic — only a hint; the README narrative is written from the figure
    if target_deg >= 4 and target_deg >= 2 * max_other:
        shape = "hub (target dominates — fan-in / fan-out)"
    elif max(degs.values(), default=0) <= 2 and n_edges <= n_nodes:
        shape = "chain / path (pass-through)"
    elif density >= 0.5:
        shape = "dense cluster"
    else:
        shape = "mixed / branching tree"

    return {
        "G": G,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "target_deg_in_fig": target_deg,
        "max_other_deg": max_other,
        "leaves": leaves,
        "density": density,
        "shape_hint": shape,
        "full_graph_degree": int(full_deg),
    }


def draw_case(node_idx, txId, struct, true_label, full_pred, gnn_conf, out_path):
    G = struct["G"]
    pos = nx.spring_layout(G, seed=SEED, k=0.9)

    node_fill, node_edge, node_size = [], [], []
    for n in G.nodes:
        node_fill.append(CLASS_COLOR[int(true_label[n])])
        # outline GNN-flagged-illicit nodes in black, others in white
        node_edge.append("black" if int(full_pred[n]) == ILLICIT else "white")
        node_size.append(900 if n == node_idx else 320)

    weights = np.array([G[u][v]["weight"] for u, v in G.edges])
    wmax = weights.max() if len(weights) else 1.0
    edge_widths = 0.6 + 4.0 * (weights / wmax)

    plt.figure(figsize=(8, 6))
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color="#555555", alpha=0.7)
    nx.draw_networkx_nodes(
        G, pos, node_color=node_fill, edgecolors=node_edge,
        linewidths=2.0, node_size=node_size,
    )
    # label only the target node with its txId
    nx.draw_networkx_labels(G, pos, labels={node_idx: str(txId)}, font_size=8, font_color="black")

    legend = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=CLASS_COLOR[ILLICIT],
                   markersize=11, label="true: illicit"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=CLASS_COLOR[LICIT],
                   markersize=11, label="true: licit"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=CLASS_COLOR[UNKNOWN],
                   markersize=11, label="true: unknown (unlabeled)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#ffffff",
                   markeredgecolor="black", markeredgewidth=2, markersize=11,
                   label="GNN flags illicit (black outline)"),
    ]
    plt.legend(handles=legend, loc="upper left", fontsize=8, frameon=True)
    plt.title(
        f"GNNExplainer subgraph — txId {txId}  (GNN p(illicit)={gnn_conf:.3f})\n"
        f"top {struct['n_edges']} edges by importance · "
        f"target degree in full graph: {struct['full_graph_degree']}",
        fontsize=10,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    set_seed()
    os.makedirs(FIG_DIR, exist_ok=True)

    data, _, _, tx_id = load()
    model = load_model(data.x.size(1))

    with torch.no_grad():
        full_pred = model(data.x, data.edge_index).argmax(dim=1)
    full_deg = degree(data.edge_index[0], num_nodes=data.num_nodes)
    txid_to_node = {int(t): i for i, t in enumerate(tx_id.tolist())}
    true_label = data.y.cpu().numpy()

    key, reverse, overlap = disagreement_analysis()
    cases = select_cases(key)
    if len(cases) == 0:
        return

    explainer = build_explainer(model)

    print("=" * 64)
    print("PER-CASE STRUCTURE  (write the README narratives from these numbers)")
    print("=" * 64)
    for i, (_, row) in enumerate(cases.iterrows()):
        txId = int(row["txId"])
        node_idx = txid_to_node[txId]
        gnn_conf = float(row["predicted_prob_gnn"])

        result = explain_case(explainer, model, data, node_idx, full_pred, case_seed=SEED + i)
        struct = describe_structure(node_idx, result["top_edges"], true_label, full_deg[node_idx])

        out_path = os.path.join(FIG_DIR, f"explain_{txId}.png")
        draw_case(node_idx, txId, struct, true_label, full_pred, gnn_conf, out_path)

        # how many figure neighbours are themselves illicit / unknown
        fig_nodes = [n for n in struct["G"].nodes if n != node_idx]
        n_illicit_nb = sum(1 for n in fig_nodes if int(true_label[n]) == ILLICIT)
        n_unknown_nb = sum(1 for n in fig_nodes if int(true_label[n]) == UNKNOWN)
        n_licit_nb = sum(1 for n in fig_nodes if int(true_label[n]) == LICIT)

        print(f"\n[case {i + 1}] txId {txId}")
        print(f"  GNN p(illicit)={gnn_conf:.4f}  baseline p(illicit)={row['predicted_prob_base']:.4f}")
        print(f"  degree in FULL graph (real neighbour count): {struct['full_graph_degree']}")
        print(f"  2-hop computation graph: {result['full_nbhd_nodes']:,} nodes / "
              f"{result['full_nbhd_edges']:,} edges")
        print(f"  figure = top {struct['n_edges']} edges -> {struct['n_nodes']} nodes")
        print(f"    target degree within figure : {struct['target_deg_in_fig']}")
        print(f"    max non-target degree       : {struct['max_other_deg']}")
        print(f"    degree-1 leaves             : {struct['leaves']}")
        print(f"    density                     : {struct['density']:.3f}")
        print(f"    figure neighbours by true class: "
              f"{n_illicit_nb} illicit / {n_licit_nb} licit / {n_unknown_nb} unknown")
        print(f"    shape hint                   : {struct['shape_hint']}")
        print(f"  saved -> {os.path.relpath(out_path, ROOT)}")

    print()
    print("Done. Figures in figures/explain_<txId>.png — write honest narratives from the above.")


if __name__ == "__main__":
    main()
