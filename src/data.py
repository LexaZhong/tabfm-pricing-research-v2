"""Load and prepare the French MTPL frequency dataset (freMTPL2freq).

Standard benchmark for claim-frequency modelling (~678k policies).
Fetched from OpenML (dataset id 41214).
"""
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

import schema

CATEGORICAL = ["Area", "VehBrand", "VehGas", "Region"]
NUMERIC = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
FEATURES = CATEGORICAL + NUMERIC


def load_fremtpl2freq() -> pd.DataFrame:
    df = fetch_openml(data_id=41214, as_frame=True, parser="auto").frame
    # Standard cleaning used in the literature (e.g. Noll, Salzmann & Wüthrich 2020)
    df["ClaimNb"] = df["ClaimNb"].astype(float).clip(upper=4)
    df["Exposure"] = df["Exposure"].astype(float).clip(upper=1.0)
    df["Frequency"] = df["ClaimNb"] / df["Exposure"]
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    schema.set_schema(CATEGORICAL, NUMERIC)
    return df


def train_test_split_df(df: pd.DataFrame, n_train: int, seed: int = 42,
                        n_test: int = 100_000):
    """Subsample n_train policies for training; fixed held-out test set."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(df))
    test = df.iloc[idx[:n_test]].reset_index(drop=True)
    pool = idx[n_test:]
    train = df.iloc[pool[:n_train]].reset_index(drop=True)
    return train, test
