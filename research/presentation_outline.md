# Presentation outline (20–30 min, DS team, no TFM background)

1. **Hook (2 min)** — "What if you could get XGBoost-level accuracy with zero
   training and zero tuning?" One-slide claim + one-slide catch.
2. **What is a tabular foundation model? (5 min)**
   - Analogy: LLM few-shot prompting, but rows instead of words.
   - Pre-trained once on synthetic datasets → in-context learning at use time.
   - TabPFN timeline: v1 (2023 proof of concept) → v2 (Nature 2025) →
     v2.5/v3 (2025–26, up to 1M rows). One architecture diagram, keep it light.
3. **Why it might matter for pricing (3 min)** — thin segments, new products,
   speed of iteration; uncertainty quantification for free.
4. **Why it might not (3 min)** — no exposure offset / Poisson objective,
   explainability & regulatory fit, inference cost, licensing.
5. **Our benchmark (8 min)** — setup slide (freMTPL2, GLM/XGB/TabPFN, metrics:
   deviance, Gini, calibration, wall time), then 3 result charts:
   - metric vs training-set size (the data-efficiency story)
   - lift / Lorenz curves at one representative n
   - total calibration + runtime table
   - one slide comparing our numbers to Deprez et al. (2026).
6. **Live-ish demo (3 min, optional)** — notebook: fit TabPFN on 5k policies
   in seconds vs XGBoost tuning loop.
7. **Recommendation & roadmap (3 min)** — where TFMs fit (fast baseline,
   thin-data segments, ensemble member), licensing path, proposed pilot.
8. **Q&A (5 min)** — anticipate: "can we explain it to regulators?",
   "what about our data volumes?", "cost?".
