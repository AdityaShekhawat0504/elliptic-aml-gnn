# CLAUDE.md — Explainable AML on the Elliptic Bitcoin Graph

## What this project is
A graph neural network that flags illicit Bitcoin transactions on the Elliptic
dataset, benchmarked against a non-graph baseline, where **every flag ships with
an explainer subgraph that reads as a money-laundering typology**.

## Definition of done — the finish line, do not drift from it
The project is done when a stranger can open this repo and in 60 seconds understand:
1. A GNN flags illicit Bitcoin transactions.
2. It reduces false positives vs. a non-graph (XGBoost) baseline.
3. Each flag comes with an explainer subgraph mapped to a named typology (e.g. layering).

The README leads with a **false-positive number** and the word **"layering"** —
NOT an AUC score. Training the model is the *middle* of the project. The README
plus one conversation with an AML practitioner is the *end*.

## Stack — this is the whole dependency list, do not add to it
- **PyTorch Geometric** — use the built-in `EllipticBitcoinDataset` (no manual CSV wrangling)
- **XGBoost** — non-graph baseline
- **GNN: GraphSAGE or GCN** — pick one, do not build both
- **`torch_geometric.explain` (GNNExplainer)** — explainability

## Milestones — build and COMMIT one at a time
Do not start milestone N+1 until milestone N is committed AND the README is updated.
Target: ~2–3 focused weekends total. If it is ballooning past that, scope has crept.

1. **Setup + EDA.** Load `EllipticBitcoinDataset`. Report class imbalance (~2%
   illicit), temporal structure (49 time steps), and note the known mid-timeline
   distribution shift. Output: short EDA script/notebook + README section.
2. **Non-graph baseline.** XGBoost on node features only. Report precision/recall
   on the illicit class AND false-positive rate on a fixed test split. This is the
   comparison anchor — do not skip or rush it.
3. **GNN.** GraphSAGE or GCN on the same split, same metrics. Honest reporting: if
   it does not beat XGBoost on raw recall, that is fine and expected. The story is
   *what kind of pattern the graph catches that features alone miss.*
4. **Explainability — the centerpiece, not a bolt-on.** Run GNNExplainer on ~5
   illicit predictions. Visualize each surfaced subgraph. Write 2–3 sentences per
   case mapping it to a typology (chains of rapid pass-through = layering).
5. **README + one conversation.** Finalize a recruiter-skimmable README. Then show
   it to one person who works in AML. This step is NOT the agent's job — it is the
   human's. It is still the finish line.

## DO NOT — scope guardrails
The failure mode for this project is sprawl: lots of half-built code, no shipped
artifact. Hard nos:
- No EvolveGCN / temporal GNNs (the dataset is temporal — resist it).
- No Elliptic2 / subgraph-level work.
- No Streamlit dashboard, no web UI, no real-time pipeline.
- No hyperparameter grid-search marathon — one reasonable config per model.
- No federated learning, no synthetic-data engine, no format-preserving encryption.
  Those belong in a separate startup thesis, not this repo.
- No new dependencies beyond the stack above.
- Do not add a 6th milestone. If tempted, ship milestone 5 instead.

## Working rules for the agent
- After each milestone: commit with a clear message, then update the relevant
  README section.
- Keep the README metric-first and typology-first from milestone 1 onward — it is
  a living file, not a final write-up.
- Prefer the simplest thing that completes the milestone over the most
  sophisticated thing.
- If asked to do something on the DO NOT list, flag it and refer back to this file.

## README template — maintain from milestone 1, do not leave for the end
```markdown
# Explainable AML Detection on the Elliptic Bitcoin Graph

Graph neural network for flagging illicit Bitcoin transactions — built to cut
false positives and explain every flag as a money-laundering typology.

## Result
- Non-graph baseline (XGBoost): <precision> / <recall> on illicit class,
  <FP rate> false positives.
- GNN (GraphSAGE/GCN): <precision> / <recall>, <FP rate> false positives.
- Key finding: <one sentence — what the graph catches that features miss>.

## Why this framing
False positives, not accuracy, are the real cost in AML triage. Flags are mapped
to named typologies (structuring, layering, rapid movement of funds) and each
comes with an explainer subgraph — auditable, not a black box.

## Explainer examples
<2–3 subgraph visualizations, each with a short typology read>

## Dataset
Elliptic Bitcoin Dataset — ~200k transactions, 166 features, 49 time steps,
~2% labeled illicit. Loaded via `torch_geometric.datasets.EllipticBitcoinDataset`.

## Run it
<setup + commands>

## Stack
PyTorch Geometric · XGBoost · GNNExplainer
```