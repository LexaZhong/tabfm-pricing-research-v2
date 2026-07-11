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


class _TabPFNFrequencyBase:
    """TabPFN regressor on claim frequency (rate), Exposure also as feature.

    CAVEAT: TabPFN has no native Poisson objective, offset, or sample
    weights. We regress on Frequency = ClaimNb/Exposure with Exposure as an
    additional feature, then multiply predictions by exposure. Document this
    choice in any writeup; it is one of the fair-comparison questions of the
    project (cf. Deprez et al. 2026, arXiv:2605.22892).

    Subclasses supply the backend regressor (local OSS package vs hosted API).
    """

    def __init__(self, max_context: int = 50_000, predict_batch_size: int = 1000):
        self.max_context = max_context
        self.predict_batch_size = predict_batch_size

    def _make_regressor(self):  # local vs hosted backend
        raise NotImplementedError

    def _encode(self, df: pd.DataFrame) -> np.ndarray:
        X = df[schema.features() + ["Exposure"]].copy()
        for c in schema.CATEGORICAL:  # ordinal-encode; TabPFN handles raw numerics
            X[c] = X[c].cat.codes
        return X.values

    def fit(self, train: pd.DataFrame):
        t = train
        if len(t) > self.max_context:
            t = t.sample(self.max_context, random_state=0)
        self.model = self._make_regressor()
        self.model.fit(self._encode(t), t["Frequency"].values)
        return self

    def predict_counts(self, test: pd.DataFrame) -> np.ndarray:
        Xv = self._encode(test)
        # Batch predictions: TabPFN's attention scales with query batch size,
        # and local GPU/MPS memory (or the hosted API request size) can't hold
        # a large test set in one forward pass.
        chunks = [self.model.predict(Xv[i:i + self.predict_batch_size])
                  for i in range(0, len(Xv), self.predict_batch_size)]
        rate = np.clip(np.concatenate(chunks), a_min=0, a_max=None)
        return rate * test["Exposure"].values


class TabPFNFrequency(_TabPFNFrequencyBase):
    """Local TabPFN via the OSS `tabpfn` package. GPU recommended; on CPU it is
    capped at ~1000 rows unless TABPFN_ALLOW_CPU_LARGE_DATASET=1."""

    def __init__(self, max_context: int = 50_000, device: str = "auto",
                 predict_batch_size: int = 1000):
        super().__init__(max_context, predict_batch_size)
        self.device = device

    def _make_regressor(self):
        from tabpfn import TabPFNRegressor
        return TabPFNRegressor(device=self.device)


class TabPFNClientFrequency(_TabPFNFrequencyBase):
    """Hosted TabPFN via the `tabpfn-client` API: inference runs on Prior Labs'
    GPU servers, so no local GPU is needed (e.g. a CPU-only Colab runtime).

    DATA CAVEAT: every fit/predict uploads the feature rows to an external
    service. Public datasets only -- never point this at internal data (see the
    guardrails in CLAUDE.md and requirements.txt). run_benchmark.py refuses to
    combine this model with --internal-csv.
    """

    def _make_regressor(self):
        from tabpfn_client import TabPFNRegressor
        return TabPFNRegressor()


MODELS = {"glm": GLMFrequency, "xgboost": XGBFrequency,
          "tabpfn": TabPFNFrequency, "tabpfn-client": TabPFNClientFrequency}
