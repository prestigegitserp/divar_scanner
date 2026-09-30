from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from .config import Config


def _robust_z(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    med = x.median()
    mad = (x - med).abs().median()
    if pd.isna(mad) or mad <= 1e-12:
        return pd.Series(np.zeros(len(x)), index=x.index, dtype=float)
    return (x - med) / (1.4826 * mad)


def _peer_robust_z(
    df: pd.DataFrame, value_col: str, peer_min_count: int, clip: float
) -> pd.Series:
    result = pd.Series(np.nan, index=df.index, dtype=float)
    levels = [
        ["neighborhood", "area_bucket", "rooms"],
        ["neighborhood", "area_bucket"],
        ["neighborhood"],
    ]
    for group_cols in levels:
        valid_cols = [c for c in group_cols if c in df.columns]
        if not valid_cols:
            continue
        for _, idx in df.groupby(valid_cols, dropna=False).groups.items():
            idx = pd.Index(idx)
            remaining = idx[result.loc[idx].isna()]
            if len(remaining) == 0:
                continue
            s = pd.to_numeric(df.loc[idx, value_col], errors="coerce")
            if s.notna().sum() < peer_min_count:
                continue
            z = _robust_z(s).abs().clip(0, clip) / clip
            result.loc[remaining] = z.loc[remaining]
    global_z = _robust_z(df[value_col]).abs().clip(0, clip) / clip
    return result.fillna(global_z).fillna(0.0)


def add_market_anomaly(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    out = df.copy()
    a = config.section("anomaly")
    clip = float(a.get("robust_z_clip", 8.0))
    peer_min_count = int(a.get("peer_min_count", 8))

    z_cols = []
    for value_col in ("deposit_per_m2", "rent_per_m2", "equivalent_deposit_per_m2"):
        if value_col in out:
            name = f"peer_anomaly_{value_col}"
            out[name] = _peer_robust_z(out, value_col, peer_min_count, clip)
            z_cols.append(name)
    out["robust_market_anomaly"] = out[z_cols].max(axis=1) if z_cols else 0.0

    numeric = pd.DataFrame(index=out.index)
    base_cols = [
        "area_m2",
        "rooms",
        "year_built_shamsi",
        "deposit_toman",
        "rent_monthly_toman",
        "equivalent_deposit_per_m2",
        "description_len",
    ]
    for col in base_cols:
        if col not in out:
            continue
        s = pd.to_numeric(out[col], errors="coerce")
        if col in {"deposit_toman", "rent_monthly_toman", "equivalent_deposit_per_m2"}:
            s = np.log1p(s.clip(lower=0))
        med = s.median()
        numeric[col] = s.fillna(0.0 if pd.isna(med) else med)

    if len(out) >= 12 and numeric.shape[1] >= 2:
        model = IsolationForest(
            n_estimators=250,
            contamination=a.get("isolation_contamination", "auto"),
            random_state=int(config.get("project.random_seed", 42)),
            n_jobs=-1,
        )
        model.fit(numeric)
        raw = -model.score_samples(numeric)
        lo, hi = np.quantile(raw, [0.05, 0.95])
        denom = max(float(hi - lo), 1e-9)
        out["isolation_anomaly_score"] = np.clip((raw - lo) / denom, 0, 1)
    else:
        out["isolation_anomaly_score"] = 0.0

    out["market_anomaly_score"] = np.clip(
        0.65 * out["robust_market_anomaly"] + 0.35 * out["isolation_anomaly_score"], 0, 1
    )
    return out


def add_duplicate_score(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    out = df.copy()
    if len(out) < 2:
        out["duplicate_similarity"] = 0.0
        out["near_duplicate_token"] = ""
        return out
    a = config.section("anomaly")
    max_features = int(a.get("duplicate_max_features", 10000))
    texts = (
        out["title"].fillna("").astype(str)
        + " "
        + out["description"].fillna("").astype(str).str.slice(0, 2500)
    )
    if texts.str.strip().eq("").all():
        out["duplicate_similarity"] = 0.0
        out["near_duplicate_token"] = ""
        return out
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        max_features=max_features,
        sublinear_tf=True,
    )
    X = vec.fit_transform(texts)
    nn = NearestNeighbors(n_neighbors=min(2, len(out)), metric="cosine", algorithm="brute")
    nn.fit(X)
    distances, indices = nn.kneighbors(X)
    sims = np.zeros(len(out), dtype=float)
    tokens = [""] * len(out)
    if distances.shape[1] >= 2:
        sims = 1.0 - distances[:, 1]
        for i, j in enumerate(indices[:, 1]):
            tokens[i] = str(out.iloc[int(j)].get("token", ""))
    out["duplicate_similarity"] = np.clip(sims, 0, 1)
    out["near_duplicate_token"] = tokens
    return out


def prefilter_score(df: pd.DataFrame) -> pd.Series:
    dup = pd.to_numeric(
        df.get("duplicate_similarity", pd.Series(0.0, index=df.index)), errors="coerce"
    ).fillna(0.0)
    bait = pd.to_numeric(
        df.get("duplicate_bait_score", pd.Series(0.0, index=df.index)), errors="coerce"
    ).fillna(0.0)
    market = pd.to_numeric(
        df.get("market_anomaly_score", pd.Series(0.0, index=df.index)), errors="coerce"
    ).fillna(0.0)
    quality = pd.to_numeric(
        df.get("data_quality_score", pd.Series(0.0, index=df.index)), errors="coerce"
    ).fillna(0.0)
    price_model = pd.to_numeric(
        df.get("price_model_anomaly_score", pd.Series(0.0, index=df.index)), errors="coerce"
    ).fillna(0.0)
    return np.clip(
        0.32 * market + 0.26 * quality + 0.18 * price_model + 0.16 * bait + 0.08 * dup,
        0, 1,
    )
