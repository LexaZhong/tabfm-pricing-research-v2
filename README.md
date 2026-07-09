# TabPFN vs XGBoost vs GLM for Insurance Pricing

Research project evaluating tabular foundation models (TFMs) for non-life
insurance pricing, benchmarked against the industry-standard GLM and XGBoost.

## Research questions

1. **Accuracy**: Does TabPFN match/beat a Poisson GLM and tuned XGBoost on
   claim frequency (and later severity) out of the box?
2. **Data efficiency**: How does the gap change with training set size
   (1k / 5k / 20k / 100k policies)? TFMs are expected to shine when data is scarce.
3. **Practicality**: Inference time, GPU needs, calibration of the total
   predicted claim count (critical for pricing — we need the premium pool to balance),
   licensing constraints.
4. **Actuarial validity**: Lift, Lorenz/Gini, and calibration by segment —
   not just RMSE/deviance.

## Key prior work (read first)

- Deprez, Verbeke & Verdonck (2026), *Is TabPFN the Silver Bullet for Insurance
  Pricing?* arXiv:2605.22892 — benchmarks TabPFN v2.6/v3 vs GLM/XGBoost on
  freMTPL2 and beMTPL97. Our starting point to replicate & extend.
- Hollmann et al. (2025), *TabPFN v2*, Nature.
- Grinsztajn et al. (2026), *TabPFN-3 Technical Report*, arXiv:2605.13986.
- Brauer (2024), *Enhancing actuarial non-life pricing models via transformers*,
  European Actuarial Journal.

## Licensing note (important)

TabPFN **code** and **v2 weights**: Prior Labs License (Apache 2.0 + attribution,
commercial OK). TabPFN **2.5 / 2.6 / 3 weights**: non-commercial license —
internal testing/benchmarking allowed, but outputs may not drive commercial or
production decisions without an enterprise license. Confirm scope with legal
before this research influences vendor selection or pricing decisions.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/run_benchmark.py --sample-sizes 5000 20000 --seeds 42   # public data
```

- Replicate the Deprez et al. paper: see **docs/REPLICATION.md**
- Run on the internal GL extract: see **docs/INTERNAL_GL.md** (+ research/gl_scoping.md for scope)
Results land in `results/`.

## Layout

- `src/` — data loading, models, actuarial metrics, benchmark runner
- `research/` — research brief, literature notes, presentation outline
- `prompts/` — reusable prompts for the Claude Code research agent
- `CLAUDE.md` — instructions that turn Claude Code into your research assistant
