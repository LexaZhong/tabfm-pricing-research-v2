"""Actuarial evaluation metrics for frequency models.

Beyond generic loss functions, pricing models must rank risks well (Gini/lift)
and be calibrated in total (premium pool balance).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_poisson_deviance

# numpy.trapezoid replaced the deprecated numpy.trapz in NumPy 2.0; support both.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def poisson_deviance(y_counts, pred_counts):
    pred = np.clip(pred_counts, 1e-9, None)
    return mean_poisson_deviance(y_counts, pred)


def total_calibration(y_counts, pred_counts):
    """Sum(predicted)/Sum(actual). 1.00 = portfolio-level balance."""
    return pred_counts.sum() / y_counts.sum()


def gini_index(y_counts, pred_counts, exposure):
    """Exposure-weighted Gini from the Lorenz curve, ordering policies by
    predicted frequency (pred/exposure). Higher = better risk ranking."""
    rate = pred_counts / exposure
    order = np.argsort(rate)
    exp_c = np.cumsum(exposure[order]) / exposure.sum()
    claims_c = np.cumsum(y_counts[order]) / y_counts.sum()
    # area between diagonal and Lorenz curve, trapezoid rule
    return 1 - 2 * _trapezoid(claims_c, exp_c)


def decile_lift(y_counts, pred_counts, exposure, n_bins=10) -> pd.DataFrame:
    """Actual vs predicted frequency by predicted-risk decile."""
    rate = pred_counts / exposure
    df = pd.DataFrame({"y": y_counts, "pred": pred_counts,
                       "exp": exposure, "rate": rate})
    df["bin"] = pd.qcut(df["rate"].rank(method="first"), n_bins, labels=False)
    g = df.groupby("bin").apply(
        lambda d: pd.Series({
            "actual_freq": d["y"].sum() / d["exp"].sum(),
            "pred_freq": d["pred"].sum() / d["exp"].sum(),
            "exposure": d["exp"].sum(),
        }), include_groups=False)
    return g


def evaluate(y_counts, pred_counts, exposure) -> dict:
    return {
        "poisson_deviance": poisson_deviance(y_counts, pred_counts),
        "gini": gini_index(y_counts, pred_counts, exposure),
        "total_calibration": total_calibration(y_counts, pred_counts),
    }
