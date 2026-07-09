"""Adapter for internal datasets (e.g. General Liability policy-year data).

Reads a CSV plus a JSON config mapping your columns onto the standard schema
used by the benchmark: Exposure, ClaimNb, Frequency + feature lists.

Config format (see configs/gl_frequency_example.json):
{
  "claim_count_col": "rpt_claim_cnt_30mo",
  "exposure_col": "sales_millions",
  "categorical": ["class_group", "state", "new_renewal"],
  "numeric": ["log_sales", "limit_occ", "deductible", "years_in_business"],
  "claim_cap": 4,
  "min_exposure": 0.001,
  "filters": {"subline": "premises_ops", "exposure_base": "sales"},
  "credibility_bucket_col": "class_code"   // optional, for by-segment eval
}

Privacy: keep the extract free of PII (no names, addresses, policy numbers).
Run TabPFN locally on internal data — never through the hosted API.
"""
import json
import pandas as pd

import schema


def load_internal(csv_path: str, config_path: str) -> tuple[pd.DataFrame, dict]:
    with open(config_path) as f:
        cfg = json.load(f)
    df = pd.read_csv(csv_path)

    for col, val in cfg.get("filters", {}).items():
        df = df[df[col] == val]

    df = df.rename(columns={cfg["claim_count_col"]: "ClaimNb",
                            cfg["exposure_col"]: "Exposure"})
    df = df[df["Exposure"] >= cfg.get("min_exposure", 1e-6)].copy()
    df["ClaimNb"] = df["ClaimNb"].astype(float).clip(upper=cfg.get("claim_cap", 4))
    df["Frequency"] = df["ClaimNb"] / df["Exposure"]

    for c in cfg["categorical"]:
        df[c] = df[c].astype("category")
    schema.set_schema(cfg["categorical"], cfg["numeric"])

    keep = ["ClaimNb", "Exposure", "Frequency"] + schema.features()
    if cfg.get("credibility_bucket_col"):
        keep.append(cfg["credibility_bucket_col"])
    keep = list(dict.fromkeys(keep))  # dedupe, preserve order
    return df[keep].reset_index(drop=True), cfg
