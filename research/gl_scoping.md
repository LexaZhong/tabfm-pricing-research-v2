# Scoping: testing TabPFN on the internal General Liability book

## 1. Why GL is hard to price (and why that matters for this test)

**Multiple, incompatible exposure bases.** GL premium is rated per ISO class
code, and different classes use different bases: gross sales, payroll, square
footage, admissions, units, total cost. A "claims per unit of exposure" model
only makes sense within one base — you cannot pool payroll-rated contractors
with sales-rated retailers in one frequency model without normalisation.

**Hundreds of thin classes.** The classic GLM weakness is multi-level factors
where many levels have little exposure, forcing credibility weighting /
blending with ISO loss costs. This low-credibility regime is *exactly* where
TabPFN claims an advantage — which makes GL a more interesting testbed than
motor, not less.

**Low frequency, heavy-tailed severity, long development.** Occurrence
claims report and develop over years (products/completed-ops especially).
Standard practice: use mature accident years, develop or fix the evaluation
maturity, cap/censor large losses at a basic limit, and handle limits via
ILFs and severity separately.

**Coverage structure.** Premises/ops vs products/completed-ops sublines,
occurrence vs claims-made forms, per-occurrence/aggregate limits and
deductibles — each shifts the loss distribution.

**Current practice** (what your audience does today): ISO loss costs × loss
cost multiplier + schedule rating for small commercial; in-house
frequency/severity GLMs per subline with capped losses and exposure offsets
for larger books; GBMs as challenger models; credibility for thin classes.

## 2. The chunk to test TabPFN on (5-day scope)

Pick the **cleanest possible Poisson problem** hiding inside GL:

- **Subline**: premises/operations only (shorter tail than products).
- **Exposure base**: the single base covering the largest share of your book
  (typically gross sales or payroll). Drop other bases for now.
- **Target**: reported claim count at a fixed maturity (e.g. 18 or 30
  months), capped at ~4 per policy-year; frequency per normalized exposure
  unit (e.g. per $1M sales).
- **Years**: mature accident/policy years only (e.g. AY ≤ current-3); hold
  out the most recent mature year as a time-based test set (more honest than
  random CV for a pricing use case).
- **Features (~15–30)**: class group / industry rollup, detailed class code,
  state/territory, log(exposure size), limit, deductible, new/renewal,
  years in business, subline indicators. Well under TabPFN-3's 200-feature
  cap; 50k–500k policy-years fits its 1M-row context.
- **Exclusions**: large-loss counts unchanged (frequency capping handles
  them); no severity modelling this week; no PII, and run TabPFN **locally**
  (never the hosted API) on internal data.

**The one chart that will land with your team**: performance (Poisson
deviance / Gini) **by class-code credibility bucket** (classes with <500,
500–5k, >5k policy-years). H0 from the paper: GLM wins overall. H1 worth
testing: TabPFN closes the gap or wins in the thin-class buckets, where its
in-context "prior" substitutes for credibility.

## 3. Other modeling potentials of TabPFN (one slide's worth)

- **Full predictive distributions**: TabPFN outputs a posterior predictive
  distribution, not a point estimate — quantiles for large-loss propensity,
  and uncertainty that could feed credibility-style blending with the GLM.
- **Instant challenger model**: zero-tuning benchmark to pressure-test every
  new GLM/GBM build (model validation use, low regulatory stakes).
- **Thin-data niches**: new products, new states, small sublines.
- **Data quality**: exposure misclassification / audit flagging as a
  classification task; missing-value imputation.
- **Synthetic tabular data** for thin classes or data sharing.
- **Beyond pricing** (per Deprez et al.): claims fraud detection, triage,
  and reserving on run-off triangles.

## 4. 5-day plan

- **Day 1**: run replication on freMTPL2freq (docs/REPLICATION.md); verify
  numbers against the paper. In parallel, request the GL extract.
- **Day 2**: prep GL extract to the schema in docs/INTERNAL_GL.md; smoke-run
  GLM + XGBoost baselines.
- **Day 3**: TabPFN runs (context sweep + credibility-bucket analysis).
- **Day 4**: charts + sanity review with an actuary; recalibration check
  (total predicted vs actual claims).
- **Day 5**: build the deck from research/presentation_outline.md.
