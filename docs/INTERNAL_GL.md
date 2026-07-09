# How to run the benchmark on the internal General Liability extract

## 0. Before touching data

- Legal/compliance check: TabPFN 2.5+/3 weights are licensed for internal
  testing and benchmarking only; results must not drive production pricing
  or procurement decisions without a commercial license.
- Run everything **locally** (the pip `tabpfn` package, not `tabpfn-client`).
  Never send internal data to the hosted API.
- Extract must contain no PII: no insured names, addresses, policy numbers.

## 1. Build the extract (one CSV, one row per policy-year)

Scope per research/gl_scoping.md: premises/ops subline, single dominant
exposure base, mature years only, claim counts at a fixed maturity.

Required columns (names are yours; the config maps them):
- claim count at fixed maturity (e.g. `rpt_claim_cnt_30mo`)
- exposure in a single normalized unit (e.g. `sales_millions`)
- categorical features: class group, class code, state, new/renewal, ...
- numeric features: log(exposure), limit, deductible, years in business, ...
- optional filter columns (`subline`, `exposure_base`) and a
  `credibility_bucket_col` (class code) for the thin-class analysis

Sanity checks before modelling (30 minutes that save your Day 4):
- claim counts are per policy-year, not per claimant/feature record
- exposure > 0 everywhere; investigate the top/bottom 1%
- overall frequency (sum counts / sum exposure) matches what your
  actuaries quote for the segment

## 2. Configure and run

Copy `configs/gl_frequency_example.json`, point it at your column names,
place the CSV under `data/` (gitignored), then:

```bash
# Baselines first — fast, catches data problems early
python src/run_benchmark.py --internal-csv data/gl_extract.csv \
  --config configs/gl_frequency.json --models glm xgboost \
  --sample-sizes 50000 --seeds 1 2 3

# TabPFN with context sweep (GPU: ~minutes; CPU: start with 2000 5000)
python src/run_benchmark.py --internal-csv data/gl_extract.csv \
  --config configs/gl_frequency.json --models tabpfn \
  --sample-sizes 2000 10000 50000 --seeds 1 2 3
```

Outputs: `results/benchmark_*.csv` (overall) and `*_by_bucket.csv`
(thin/medium/thick class-code buckets — the chart most likely to show
TabPFN value, if any exists).

## 3. How to read the results

- **poisson_deviance** (lower better): distributional fit; the headline metric.
- **gini** (higher better): risk ranking — what drives rate segmentation.
- **total_calibration**: sum(pred)/sum(actual). GLM/XGB with Poisson
  objectives sit near 1.00; expect TabPFN to drift (no Poisson objective).
  If TabPFN ranks well but is miscalibrated, note that a one-line
  multiplicative rebalance fixes totals — report both raw and rebalanced.
- **fit_s / pred_s**: the operational-cost story.
- By-bucket file: does the GLM-vs-TabPFN gap shrink in "thin" classes?
  That's the data-scarcity hypothesis, and your most interesting slide
  either way.

## 4. Caveats to state on the slide (credibility with actuaries)

- Random train/test split, not out-of-time; counts at fixed maturity, not
  ultimate; single subline and exposure base; no severity, so this is a
  frequency comparison, not a rate adequacy study.
- Exposure enters TabPFN as a feature + target rate (no native offset) —
  a methodological choice, disclosed, consistent with the paper's spirit.
