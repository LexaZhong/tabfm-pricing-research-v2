# How to replicate Deprez et al. (2026), arXiv:2605.22892

Goal for Day 1: reproduce the paper's qualitative findings on public data so
your GL results are credible by contrast.

## Option A (recommended): this repo on freMTPL2freq

Setup on any machine (Python 3.10–3.12; GPU strongly recommended for TabPFN):

```bash
git clone <your-internal-git-url>/tabfm-pricing-research && cd tabfm-pricing-research
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

First TabPFN run downloads model weights from Hugging Face (needs internet
once; afterwards they're cached in ~/.cache — copy that folder to offline
machines if needed).

Run the paper's context-size sweep (5 seeds ≈ paper's 5 folds):

```bash
python src/run_benchmark.py \
  --sample-sizes 2000 5000 10000 50000 100000 \
  --seeds 1 2 3 4 5 --models glm xgboost tabpfn
```

No GPU available? Add `--models glm xgboost` first, then run TabPFN with
`--sample-sizes 2000 5000` only (CPU inference at 50k+ context is very slow —
that slowness is itself one of the paper's findings; report it).

## What "successful replication" looks like

Compare your results/benchmark_*.csv against the paper's Table 1
(freMTPL2 frequency): GLM Poisson deviance ≈ 0.296 (best), XGBoost ≈ 0.344,
TabPFN between ~0.34 and ~0.55 depending on context size. You will not match
numbers exactly (different preprocessing details, folds, and our added Gini/
calibration metrics), but you should reproduce all four qualitative findings:

1. GLM has the best deviance; TabPFN never wins outright.
2. TabPFN deviance is sensitive (non-monotonically) to context size.
3. TabPFN seed-to-seed variance exceeds GLM/XGBoost.
4. TabPFN inference time exceeds GLM/XGBoost total time even at small contexts.

## Option A-GPU: Google Colab (free GPU)

No local GPU, or hitting the Apple Silicon MPS out-of-memory error? Colab's
free tier gives a T4 GPU, which sidesteps both problems (see the caveat in
`src/models.py`'s `TabPFNFrequency.predict_batch_size` docstring — batching
still helps but matters much less with a discrete GPU's larger VRAM).

1. Open [colab.research.google.com](https://colab.research.google.com),
   `File -> Upload notebook`, and pick `notebooks/colab_quickstart.ipynb`
   from this repo (or open it directly from GitHub via
   `File -> Open notebook -> GitHub` and search this repo).
2. `Runtime -> Change runtime type -> T4 GPU`.
3. Run the cells top to bottom. If the repo is private, the token cell
   prompts for a GitHub personal access token (hidden input, never saved to
   the notebook); leave it blank for a public repo.
4. The quickstart cell runs the test command
   `python src/run_benchmark.py --models tabpfn --sample-sizes 2000 --seeds 42`;
   edit the `--sample-sizes`/`--seeds`/`--models` args for the full sweep.
5. Download `results/*.csv` via the Colab file browser, or copy to Drive
   (last cell) before the runtime recycles — Colab free tier does not
   persist the VM's disk between sessions.

## Option B: authors' exact code

For an exact replication: https://github.com/B-Deprez/tabpfn_insurance
(uses the CASdatasets R package for freMTPL2/beMTPL97). Only worth it if a
reviewer challenges Option A; it costs setup time you don't have this week.

## Presentation tip

One slide: "We replicated the published result" (your chart next to the
paper's Figure 1) buys you the room's trust before you show internal data.
