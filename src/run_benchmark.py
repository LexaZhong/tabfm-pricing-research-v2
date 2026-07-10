"""Run the GLM vs XGBoost vs TabPFN frequency benchmark.

Replication (public data, freMTPL2freq from OpenML):
    python src/run_benchmark.py --sample-sizes 2000 5000 10000 50000 100000 \
        --seeds 1 2 3 4 5
Internal data (e.g. General Liability extract):
    python src/run_benchmark.py --internal-csv data/gl_extract.csv \
        --config configs/gl_frequency_example.json --sample-sizes 50000
Skip TabPFN (no GPU):
    python src/run_benchmark.py --models glm xgboost

Outputs results/benchmark_<timestamp>.csv, one row per (model, n_train, seed):
Poisson deviance, Gini, total calibration, fit/predict wall time. If the
config defines credibility_bucket_col, also writes per-bucket metrics.
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

import schema
from data import load_fremtpl2freq, train_test_split_df
from models import MODELS
from metrics import evaluate


def run_one(name, train, test, tabpfn_context, tabpfn_predict_batch_size):
    kwargs = ({"max_context": tabpfn_context,
               "predict_batch_size": tabpfn_predict_batch_size}
              if name == "tabpfn" else {})
    model = MODELS[name](**kwargs)
    t0 = time.time(); model.fit(train); t_fit = time.time() - t0
    t0 = time.time(); pred = model.predict_counts(test); t_pred = time.time() - t0
    m = evaluate(test["ClaimNb"].values, pred, test["Exposure"].values)
    return pred, {**m, "fit_s": round(t_fit, 2), "pred_s": round(t_pred, 2)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample-sizes", type=int, nargs="+", default=[5000, 20000])
    p.add_argument("--models", nargs="+", default=["glm", "xgboost", "tabpfn"])
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--n-test", type=int, default=100_000)
    p.add_argument("--tabpfn-context", type=int, default=100_000,
                   help="max in-context training rows for TabPFN")
    p.add_argument("--tabpfn-predict-batch-size", type=int, default=1000,
                   help="test rows per TabPFN predict() call; lower this on "
                        "MPS/OOM (attention scales with batch size)")
    p.add_argument("--internal-csv", type=str, default=None)
    p.add_argument("--config", type=str, default=None)
    args = p.parse_args()
    t0_total = time.time()

    bucket_col = None
    if args.internal_csv:
        from internal_data import load_internal
        df, cfg = load_internal(args.internal_csv, args.config)
        bucket_col = cfg.get("credibility_bucket_col")
        print(f"Loaded internal data: {len(df):,} rows, "
              f"features={schema.features()}")
    else:
        print("Loading freMTPL2freq from OpenML...")
        df = load_fremtpl2freq()

    n_test = min(args.n_test, len(df) // 3)
    rows, bucket_rows = [], []
    for seed in args.seeds:
        for n in args.sample_sizes:
            n_eff = min(n, len(df) - n_test)
            train, test = train_test_split_df(df, n_train=n_eff, seed=seed,
                                              n_test=n_test)
            for name in args.models:
                print(f"[seed={seed} n_train={n_eff}] {name} ...", flush=True)
                pred, res = run_one(name, train, test, args.tabpfn_context,
                                     args.tabpfn_predict_batch_size)
                rows.append({"model": name, "n_train": n_eff, "seed": seed, **res})
                print(f"    dev={res['poisson_deviance']:.4f} "
                      f"gini={res['gini']:.3f} calib={res['total_calibration']:.3f}")
                if bucket_col and bucket_col in test.columns:
                    t = test.copy(); t["pred"] = pred
                    sizes = t.groupby(bucket_col)["Exposure"].transform("count")
                    t["bucket"] = pd.cut(sizes, [0, 500, 5000, float("inf")],
                                         labels=["thin", "medium", "thick"])
                    for b, g in t.groupby("bucket", observed=True):
                        if g["ClaimNb"].sum() == 0:
                            continue
                        bm = evaluate(g["ClaimNb"].values, g["pred"].values,
                                      g["Exposure"].values)
                        bucket_rows.append({"model": name, "n_train": n_eff,
                                            "seed": seed, "bucket": str(b),
                                            "n_policies": len(g), **bm})

    total_s = time.time() - t0_total

    out = Path(__file__).resolve().parents[1] / "results"
    out.mkdir(exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d_%H%M}"
    pd.DataFrame(rows).to_csv(out / f"benchmark_{stamp}.csv", index=False)
    print(f"\nSaved results/benchmark_{stamp}.csv")
    if bucket_rows:
        pd.DataFrame(bucket_rows).to_csv(
            out / f"benchmark_{stamp}_by_bucket.csv", index=False)
        print(f"Saved results/benchmark_{stamp}_by_bucket.csv")

    print(f"Total runtime: {total_s:.1f}s "
          f"(sum of fit_s={sum(r['fit_s'] for r in rows):.1f}s, "
          f"pred_s={sum(r['pred_s'] for r in rows):.1f}s)")
    with open(out / f"benchmark_{stamp}_meta.json", "w") as f:
        json.dump({"args": vars(args), "total_runtime_s": round(total_s, 2)},
                   f, indent=2)
    print(f"Saved results/benchmark_{stamp}_meta.json")


if __name__ == "__main__":
    main()
