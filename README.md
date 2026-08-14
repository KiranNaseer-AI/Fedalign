# FedAlign

Pretraining-task domain alignment predicts whether federated PEFT helps or
harms the worst client.

Generated 2026-08-14.

- `fedalign/` - the package (partitioning, models, 11 federated algorithms,
  metrics, statistics, analysis)
- `alignment_table7.json` - a-priori alignment scores AND signed per-cell
  predictions, committed **before** any federated run
- `manifest.csv` - every run with status, priority and full config
- `deviations.md` - logged departures from the frozen protocol

Reproduce: open the notebook in Colab, run sections 0-3 once, then section 7.
## Scope of this release

`confirm.py`, the `fedavg_wscale` aggregation rule and the four additional backbones in `core.py` belong to a confirmatory replication in progress and support no claim in the manuscript. Every result reported in the paper comes from the runs marked `done` in `manifest.csv`. Rows flagged in the `provenance` column were reconstructed from the append-only result log; see `deviations.md`.
