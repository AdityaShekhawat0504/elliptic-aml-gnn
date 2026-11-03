# CLAUDE.md

## Attribution rule (mandatory)
Never add `Co-Authored-By` trailers, "Generated with Claude Code" lines, session
links, or any other AI attribution to commits, PR titles/descriptions, issues,
comments, or files in this repo.

## Project
Explainable AML detection on the Elliptic Bitcoin graph: XGBoost baseline vs
GraphSAGE + GNNExplainer. See `README.md` and `docs/ARCHITECTURE.md`.

## Rules
- Never invent or change reported numbers. README results must match script output.
- Keep the temporal split (train steps 1-34, test 35-49) identical across models.
- `baseline.py` must not import torch; `gnn.py`/`explain.py` must not import xgboost.
- Tests must stay small, fast, offline (no dataset download, no network).

## Commands
- Install: `pip install -r requirements-dev.txt --extra-index-url https://download.pytorch.org/whl/cpu`
- Lint: `ruff check .`
- Test: `pytest`
