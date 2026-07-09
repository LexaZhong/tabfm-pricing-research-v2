# Reusable prompts for the Claude Code research agent

Run these from the repo root so CLAUDE.md context is loaded.

## 1. Literature scan (weekly)
> Search arXiv and the actuarial literature for papers from the last 3 months
> on tabular foundation models (TabPFN, TabICL, LimiX, in-context learning for
> tabular data) applied to insurance, credit risk, or claim modelling. Append
> summaries to research/lit_notes.md: citation, 2-sentence summary, and one
> line "so what for our pricing benchmark". Flag anything that changes our
> experimental design.

## 2. Replication run
> Run src/run_benchmark.py with sample sizes 1000 5000 20000 50000 and seeds
> 1..5. Aggregate mean±std per model/size into results/replication.csv and a
> matplotlib figure of Poisson deviance and Gini vs n_train. Compare
> qualitatively to Figure 1 of arXiv:2605.22892 and write 5 bullet findings.

## 3. Fair-comparison audit
> Review src/models.py for unfair advantages between models (encoding,
> exposure handling, tuning budget). Propose and implement: (a) tuned-XGBoost
> variant with a 30-min Optuna budget, (b) post-hoc multiplicative calibration
> for TabPFN, (c) frequency-as-classification TabPFN variant (P(claim>0) with
> exposure feature). Rerun and report.

## 4. Ensemble experiment
> Implement a GLM+TabPFN ensemble: GLM prediction as an extra feature for
> TabPFN, and a simple 50/50 log-space blend. Evaluate on 5k and 20k contexts.

## 5. Presentation assets
> From the latest results/ csvs, generate presentation-ready charts (PNG,
> large fonts, one message per chart) and update
> research/presentation_outline.md section 5 with the actual numbers.

## 6. Severity extension
> Add freMTPL2sev (OpenML 41215): Gamma GLM, XGBoost gamma objective, TabPFN
> regression on log severity. Same metric discipline; then combine with
> frequency models into pure-premium comparison (Tweedie deviance).
