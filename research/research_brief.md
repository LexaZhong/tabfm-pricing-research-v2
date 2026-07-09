# Research brief: Tabular foundation models for pricing

## What is TabPFN? (the 60-second version for the team)

TabPFN is a *tabular foundation model* (TFM): a transformer pre-trained on
millions of synthetic tabular datasets. To use it, you don't train anything —
you feed your labelled training rows and your test rows into one forward pass,
and it predicts via **in-context learning** (like an LLM answering from
examples in its prompt). No hyperparameter tuning, results in seconds.

- v2 published in Nature (2025); v2.5 (Nov 2025) scales to ~50k rows / 2k
  features; TabPFN-3 (May 2026) to ~1M rows / 200 features.
- Tops the TabArena benchmark, beating tuned GBDTs on small/medium data.
- Regression + classification; also uncertainty estimates and synthetic data.

## Why insurers should care — and be skeptical

Care: pricing niches with thin data (new products, small segments, new
territories) are exactly where TFMs claim an edge; zero tuning shortens
model development cycles.

Skeptical: pricing has structure TFMs don't natively speak — exposure
offsets, Poisson/Gamma/Tweedie objectives, monotonicity constraints,
regulatory explainability, calibration requirements. Deprez et al. (2026,
arXiv:2605.22892) found TabPFN did **not** consistently beat GLM/XGBoost on
MTPL data and was slow at inference and sensitive to context size.

## Hypotheses to test

- H1: TabPFN ≈ XGBoost > GLM on ranking (Gini) for small n; gap closes or
  reverses as n grows.
- H2: TabPFN is materially worse on total calibration (no Poisson objective)
  unless we post-calibrate (multiplicative rebalancing).
- H3: A TabPFN+GLM blend (GLM backbone, TabPFN on residuals or as ensemble
  member) beats both alone on small segments.
- H4: Inference cost/latency makes standalone TabPFN impractical for full
  book scoring without the paid API/GPU.

## Options landscape ("LQM" ⇒ tabular foundation models)

| Option | License | Notes |
|---|---|---|
| TabPFN v2 (OSS weights) | Apache 2.0 + attribution | commercial OK, smaller scale |
| TabPFN 2.5/2.6/3 weights | Non-commercial | internal benchmarking OK; production/procurement decisions need enterprise license |
| TabPFN API (Prior Labs) | Commercial | never send internal data without DPA/legal sign-off |
| TabICL / LimiX | open source | alternatives worth a smoke test |
| Build our own TFM | n/a | multi-month pretraining project; phase 3 at most. Claude Code useful for scaffolding experiments, not a shortcut to pretraining |

## Plan (4 weeks)

1. **Wk 1** — Replicate Deprez et al. on freMTPL2freq with this repo;
   sanity-check metrics against the paper.
2. **Wk 2** — Extensions: severity (freMTPL2sev), sample-size sweep,
   post-calibration, TabPFN+GLM ensemble.
3. **Wk 3** — (If legal OK) pilot on one internal line with thin data;
   robustness by segment.
4. **Wk 4** — Presentation + recommendation memo.
