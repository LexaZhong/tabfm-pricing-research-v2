"""Frequency model wrappers: Poisson GLM, XGBoost, TabPFN.

All wrappers expose fit(train_df) and predict_counts(test_df) -> expected
claim counts for each policy's actual exposure, so metrics are comparable.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import schema


class GLMFrequency:
    """Poisson GLM with log link; exposure handled via sample weights on
    frequency (equivalent to an offset for the weighted deviance)."""

    def __init__(self, alpha: float = 1e-4):
        self.alpha = alpha

    def fit(self, train: pd.DataFrame):
        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), schema.CATEGORICAL),
            ("num", StandardScaler(), schema.NUMERIC),
        ])
        self.pipe = Pipeline([
            ("pre", pre),
            ("glm", PoissonRegressor(alpha=self.alpha, max_iter=1000)),
        ])
        self.pipe.fit(train[schema.features()], train["Frequency"],
                      glm__sample_weight=train["Exposure"])
        return self

    def predict_counts(self, test: pd.DataFrame) -> np.ndarray:
        return self.pipe.predict(test[schema.features()]) * test["Exposure"].values


class XGBFrequency:
    """XGBoost Poisson with log(exposure) offset via base_margin."""

    def __init__(self, **params):
        import xgboost as xgb
        self.xgb = xgb
        self.params = {
            "objective": "count:poisson", "tree_method": "hist",
            "learning_rate": 0.05, "max_depth": 5, "min_child_weight": 5,
            "subsample": 0.8, "colsample_bytree": 0.8, **params,
        }
        self.num_rounds = params.pop("num_rounds", 400)

    def _dmatrix(self, df, with_label=True):
        X = pd.get_dummies(df[schema.features()], columns=schema.CATEGORICAL)
        X = X.reindex(columns=self.columns_, fill_value=0) if hasattr(self, "columns_") else X
        d = self.xgb.DMatrix(X, label=df["ClaimNb"] if with_label else None)
        d.set_base_margin(np.log(df["Exposure"].values))
        return d, X

    def fit(self, train: pd.DataFrame):
        X = pd.get_dummies(train[schema.features()], columns=schema.CATEGORICAL)
        self.columns_ = X.columns
        d = self.xgb.DMatrix(X, label=train["ClaimNb"])
        d.set_base_margin(np.log(train["Exposure"].values))
        self.booster = self.xgb.train(self.params, d, num_boost_round=self.num_rounds)
        return self

    def predict_counts(self, test: pd.DataFrame) -> np.ndarray:
        d, _ = self._dmatrix(test, with_label=False)
        return self.booster.predict(d)  # base_margin already includes exposure


class TabPFNFrequency:
    """TabPFN regressor on claim frequency (rate), Exposure also as feature.

    CAVEAT: TabPFN has no native Poisson objective, offset, or sample
    weights. We regress on Frequency = ClaimNb/Exposure with Exposure as an
    additional feature, then multiply predictions by exposure. Document this
    choice in any writeup; it is one of the fair-comparison questions of the
    project (cf. Deprez et al. 2026, arXiv:2605.22892).
    """

    def __init__(self, max_context: int = 50_000, device: str = "auto"):
        self.max_context = max_context
        self.device = device

    def fit(self, train: pd.DataFrame):
        from tabpfn import TabPFNRegressor
        t = train
        if len(t) > self.max_context:
            t = t.sample(self.max_context, random_state=0)
        X = t[schema.features() + ["Exposure"]].copy()
        for c in schema.CATEGORICAL:  # ordinal-encode; TabPFN handles raw numerics
            X[c] = X[c].cat.codes
        self.model = TabPFNRegressor(device=self.device)
        self.model.fit(X.values, t["Frequency"].values)
        return self

    def predict_counts(self, test: pd.DataFrame) -> np.ndarray:
        X = test[schema.features() + ["Exposure"]].copy()
        for c in schema.CATEGORICAL:
            X[c] = X[c].cat.codes
        rate = np.clip(self.model.predict(X.values), a_min=0, a_max=None)
        return rate * test["Exposure"].values


MODELS = {"glm": GLMFrequency, "xgboost": XGBFrequency, "tabpfn": TabPFNFrequency}
