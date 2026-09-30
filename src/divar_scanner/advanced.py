from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .config import Config


def _rank01(values: np.ndarray) -> np.ndarray:
    if len(values) <= 1:
        return np.zeros(len(values), dtype=float)
    s = pd.Series(values)
    return s.rank(method="average", pct=True).to_numpy(dtype=float)


def _robust_abs_score(values: np.ndarray, clip: float = 6.0) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    finite = np.isfinite(x)
    out = np.zeros(len(x), dtype=float)
    if finite.sum() < 4:
        return out
    med = float(np.nanmedian(x[finite]))
    mad = float(np.nanmedian(np.abs(x[finite] - med)))
    if mad <= 1e-12:
        return out
    z = np.abs((x - med) / (1.4826 * mad))
    out[finite] = np.clip(z[finite] / clip, 0, 1)
    return out


def add_price_model_anomaly(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Out-of-fold price expectation model.

    This is deliberately OOF: each row is scored by a model that was not trained on
    that row, reducing the self-fit leakage common in unsupervised price residuals.
    """
    out = df.copy()
    n = len(out)
    out["price_model_expected_equiv_deposit"] = np.nan
    out["price_model_ratio"] = np.nan
    out["price_model_signed_log_residual"] = 0.0
    out["price_model_anomaly_score"] = 0.0
    if n < 40 or "equivalent_deposit_toman" not in out:
        return out

    target = pd.to_numeric(out["equivalent_deposit_toman"], errors="coerce")
    valid = target.notna() & (target > 0)
    if valid.sum() < 35:
        return out

    numeric_cols = [
        "area_m2", "rooms", "year_built_shamsi", "parking", "elevator", "storage",
        "description_len",
    ]
    X = pd.DataFrame(index=out.index)
    for col in numeric_cols:
        if col not in out:
            continue
        s = out[col]
        if s.dtype == bool:
            s = s.astype(float)
        s = pd.to_numeric(s, errors="coerce")
        med = s[valid].median()
        X[col] = s.fillna(0.0 if pd.isna(med) else med)

    # Low-cardinality location context is useful even without precise coordinates.
    if "neighborhood" in out:
        cat = out["neighborhood"].fillna("unknown").astype(str).str.strip().replace("", "unknown")
        dummies = pd.get_dummies(cat, prefix="nbh", dtype=float)
        # Bound dimensionality against noisy schema drift.
        if dummies.shape[1] <= 80:
            X = pd.concat([X, dummies], axis=1)

    idx = np.flatnonzero(valid.to_numpy())
    y = np.log1p(target.iloc[idx].to_numpy(dtype=float))
    Xv = X.iloc[idx].to_numpy(dtype=float)
    folds = min(5, max(3, len(idx) // 20))
    kf = KFold(n_splits=folds, shuffle=True, random_state=int(config.get("project.random_seed", 42)))
    pred = np.full(len(idx), np.nan, dtype=float)

    for train_i, test_i in kf.split(Xv):
        model = HistGradientBoostingRegressor(
            learning_rate=0.055,
            max_iter=220,
            max_leaf_nodes=15,
            min_samples_leaf=max(8, min(20, len(train_i) // 8)),
            l2_regularization=1.5,
            random_state=int(config.get("project.random_seed", 42)),
        )
        model.fit(Xv[train_i], y[train_i])
        pred[test_i] = model.predict(Xv[test_i])

    residual = y - pred
    score = _robust_abs_score(residual, clip=5.0)
    expected = np.expm1(pred)
    actual = target.iloc[idx].to_numpy(dtype=float)
    ratio = actual / np.maximum(expected, 1.0)

    out.loc[out.index[idx], "price_model_expected_equiv_deposit"] = expected
    out.loc[out.index[idx], "price_model_ratio"] = ratio
    out.loc[out.index[idx], "price_model_signed_log_residual"] = residual
    out.loc[out.index[idx], "price_model_anomaly_score"] = score
    return out


def add_density_anomaly(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    out = df.copy()
    out["lof_anomaly_score"] = 0.0
    if len(out) < 20:
        return out

    cols = [
        "area_m2",
        "rooms",
        "year_built_shamsi",
        "equivalent_deposit_per_m2",
        "contract_rent_share",
    ]
    X = pd.DataFrame(index=out.index)
    for col in cols:
        if col not in out:
            continue
        s = pd.to_numeric(out[col], errors="coerce")
        if col == "equivalent_deposit_per_m2":
            s = np.log1p(s.clip(lower=0))
        med = s.median()
        X[col] = s.fillna(0 if pd.isna(med) else med)
    if X.shape[1] < 2:
        return out

    Z = StandardScaler().fit_transform(X)
    k = min(25, max(8, len(out) // 12), len(out) - 1)
    lof = LocalOutlierFactor(n_neighbors=k, contamination="auto")
    lof.fit_predict(Z)
    raw = -lof.negative_outlier_factor_
    out["lof_anomaly_score"] = np.clip(_rank01(raw), 0, 1)
    return out


class _UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def add_duplicate_graph_signals(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Build a lightweight similarity graph and expose cluster inconsistency signals."""
    out = df.copy()
    for col, default in [
        ("duplicate_cluster_id", ""),
        ("duplicate_cluster_size", 1),
        ("duplicate_cluster_neighborhoods", 1),
        ("duplicate_cluster_price_span", 0.0),
        ("duplicate_bait_score", 0.0),
    ]:
        out[col] = default

    if len(out) < 3:
        return out

    texts = (
        out.get("title", pd.Series("", index=out.index)).fillna("").astype(str)
        + " "
        + out.get("description", pd.Series("", index=out.index)).fillna("").astype(str).str.slice(0, 3000)
    )
    if texts.str.strip().eq("").all():
        return out

    a = config.section("anomaly")
    threshold = float(a.get("duplicate_similarity_threshold", 0.88))
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        max_features=int(a.get("duplicate_max_features", 12000)),
        sublinear_tf=True,
    )
    X = vec.fit_transform(texts)
    k = min(7, len(out))
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute").fit(X)
    distances, neighbors = nn.kneighbors(X)

    uf = _UnionFind(len(out))
    for i in range(len(out)):
        for dist, j in zip(distances[i, 1:], neighbors[i, 1:]):
            sim = 1.0 - float(dist)
            if sim >= threshold:
                uf.union(i, int(j))

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(out)):
        groups[uf.find(i)].append(i)

    equiv = pd.to_numeric(out.get("equivalent_deposit_toman"), errors="coerce")
    nbh = out.get("neighborhood", pd.Series("", index=out.index)).fillna("").astype(str)

    for root, members in groups.items():
        if len(members) <= 1:
            continue
        vals = equiv.iloc[members].dropna()
        if len(vals) >= 2 and (vals > 0).all():
            span = float(np.log1p(vals.max()) - np.log1p(vals.min()))
            span_score = float(np.clip(span / np.log(2.5), 0, 1))
        else:
            span_score = 0.0
        neighborhoods = max(1, nbh.iloc[members].replace("", np.nan).nunique(dropna=True))
        nbh_score = float(np.clip((neighborhoods - 1) / 2.0, 0, 1))
        size_score = float(np.clip((len(members) - 1) / 4.0, 0, 1))
        bait = float(np.clip(0.45 * size_score + 0.30 * span_score + 0.25 * nbh_score, 0, 1))
        stable = hashlib.sha1("|".join(sorted(str(out.iloc[m].get("token", m)) for m in members)).encode()).hexdigest()[:10]
        for m in members:
            out.at[out.index[m], "duplicate_cluster_id"] = stable
            out.at[out.index[m], "duplicate_cluster_size"] = len(members)
            out.at[out.index[m], "duplicate_cluster_neighborhoods"] = neighborhoods
            out.at[out.index[m], "duplicate_cluster_price_span"] = span_score
            out.at[out.index[m], "duplicate_bait_score"] = bait

    return out


def fuse_advanced_market_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    robust = pd.to_numeric(out.get("robust_market_anomaly", 0), errors="coerce").fillna(0)
    iso = pd.to_numeric(out.get("isolation_anomaly_score", 0), errors="coerce").fillna(0)
    price = pd.to_numeric(out.get("price_model_anomaly_score", 0), errors="coerce").fillna(0)
    lof = pd.to_numeric(out.get("lof_anomaly_score", 0), errors="coerce").fillna(0)
    out["market_anomaly_score"] = np.clip(0.36 * robust + 0.22 * iso + 0.30 * price + 0.12 * lof, 0, 1)
    return out
