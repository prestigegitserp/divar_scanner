from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .advanced import (
    add_density_anomaly,
    add_duplicate_graph_signals,
    add_price_model_anomaly,
    fuse_advanced_market_signals,
)
from .anomaly import add_duplicate_score, add_market_anomaly, prefilter_score
from .config import Config, load_config
from .crawler import DivarCrawler
from .decision import apply_decisions
from .features import add_features
from .normalize import normalize_many
from .report import build_html_report


def _as_float(row: pd.Series, name: str) -> float:
    try:
        v = row.get(name, 0)
        if v is None or pd.isna(v):
            return 0.0
        return float(v)
    except Exception:
        return 0.0


def _reason(row: pd.Series, config: Config) -> str:
    reasons: list[str] = []
    if _as_float(row, "data_quality_score") >= 0.42:
        reasons.append("data/field inconsistency")
    if _as_float(row, "price_model_anomaly_score") >= 0.55:
        ratio = row.get("price_model_ratio")
        if pd.notna(ratio):
            reasons.append(f"OOF price residual (actual/expected={float(ratio):.2f})")
        else:
            reasons.append("OOF price residual")
    if _as_float(row, "lof_anomaly_score") >= 0.92:
        reasons.append("low-density numeric outlier")
    if _as_float(row, "market_anomaly_score") >= 0.55:
        reasons.append("unusual vs local market")
    if _as_float(row, "duplicate_bait_score") >= 0.50:
        reasons.append(
            f"duplicate cluster inconsistency (n={int(_as_float(row, 'duplicate_cluster_size'))})"
        )

    if bool(row.get("decision_evaluated", False)):
        bait = _as_float(row, "decision_bait_probability")
        data_error = _as_float(row, "decision_data_error_probability")
        review = _as_float(row, "decision_manual_review_probability")
        if bait >= 0.60:
            reasons.append(f"bounded decision: bait/misleading p={bait:.2f}")
        if data_error >= 0.65:
            reasons.append(f"bounded decision: data-error p={data_error:.2f}")
        if review >= 0.65:
            reasons.append(f"bounded decision: manual-review p={review:.2f}")
        disposition = str(row.get("decision_disposition", "") or "")
        if disposition and disposition not in {"plausible", "market_outlier"}:
            reasons.append(f"decision:{disposition}")

    if _as_float(row, "uncertainty_score") >= 0.32:
        reasons.append("detectors disagree → human review")
    return "; ".join(dict.fromkeys(reasons)) or "low combined review risk"


def _rowwise_weighted_mean(
    frame: pd.DataFrame,
    weighted_columns: dict[str, float],
) -> tuple[pd.Series, pd.Series]:
    numerator = pd.Series(0.0, index=frame.index)
    denominator = pd.Series(0.0, index=frame.index)
    for column, weight in weighted_columns.items():
        values = pd.to_numeric(frame[column], errors="coerce")
        mask = values.notna()
        numerator.loc[mask] += values.loc[mask] * float(weight)
        denominator.loc[mask] += float(weight)
    score = numerator / denominator.replace(0, np.nan)
    return score.fillna(0.0), denominator


def _available_std(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    values = frame[columns].apply(pd.to_numeric, errors="coerce")
    return values.std(axis=1, skipna=True).fillna(0.0)


def _finalize_scores(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    out = df.copy()
    s = config.section("scoring")
    threshold = float(config.get("anomaly.duplicate_similarity_threshold", 0.88))
    denom = max(1e-6, 1.0 - threshold)
    nearest_dup = np.clip(
        (pd.to_numeric(out["duplicate_similarity"], errors="coerce").fillna(0) - threshold)
        / denom,
        0,
        1,
    )
    bait_graph = pd.to_numeric(out.get("duplicate_bait_score", 0), errors="coerce").fillna(0)
    out["duplicate_risk"] = np.maximum(nearest_dup, bait_graph)

    # The bounded decision model contributes only when that row was actually evaluated.
    decision_bait = pd.to_numeric(
        out.get("decision_bait_probability", np.nan), errors="coerce"
    )
    decision_consistency = pd.to_numeric(
        out.get("decision_consistency_score", np.nan), errors="coerce"
    )
    decision_conflict = 1.0 - decision_consistency
    out["decision_risk_score"] = np.where(
        pd.to_numeric(out.get("decision_evaluated", False), errors="coerce").fillna(0).astype(bool),
        np.maximum(decision_bait, 0.35 * decision_conflict),
        np.nan,
    )

    weights = {
        "data_quality_score": float(s.get("data_quality_weight", 0.22)),
        "market_anomaly_score": float(s.get("market_anomaly_weight", 0.38)),
        "duplicate_risk": float(s.get("duplicate_weight", 0.16)),
        "decision_risk_score": float(s.get("decision_weight", 0.24)),
    }
    suspicion, weight_sum = _rowwise_weighted_mean(out, weights)
    out["suspicion_score"] = np.clip(suspicion, 0, 1)
    out["evidence_weight_sum"] = weight_sum

    component_cols = [
        "data_quality_score",
        "market_anomaly_score",
        "duplicate_risk",
        "decision_risk_score",
    ]
    out["uncertainty_score"] = np.clip(
        _available_std(out, component_cols) * 2.0,
        0,
        1,
    )

    review_prob = pd.to_numeric(
        out.get("decision_manual_review_probability", np.nan), errors="coerce"
    )
    review_component = review_prob.where(review_prob.notna(), 0.0)
    out["review_priority_score"] = np.clip(
        0.80 * out["suspicion_score"]
        + 0.12 * out["uncertainty_score"]
        + 0.08 * review_component,
        0,
        1,
    )

    review_t = float(s.get("review_threshold", 0.58))
    high_t = float(s.get("high_risk_threshold", 0.78))
    out["risk_band"] = np.select(
        [
            out["review_priority_score"] >= high_t,
            out["review_priority_score"] >= review_t,
        ],
        ["high", "review"],
        default="normal",
    )
    out["flag_reasons"] = [_reason(row, config) for _, row in out.iterrows()]
    return out.sort_values(
        ["review_priority_score", "suspicion_score", "market_anomaly_score"],
        ascending=False,
    ).reset_index(drop=True)


def _read_input(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    if p.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(p)
    raise ValueError(f"Unsupported input format: {p.suffix}")


def run_pipeline(
    config_path: str | Path = "config/fatemi.yaml",
    *,
    crawl: bool = True,
    input_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    config = load_config(config_path)
    meta: dict[str, Any] = {
        "project": config.get("project.name", "Divar Scanner"),
        "pipeline_version": "0.3.0",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
    }

    if crawl:
        crawler = DivarCrawler(config)
        raw, crawl_meta = crawler.crawl()
        meta["crawl"] = crawl_meta
        meta.update(crawl_meta)
        df = normalize_many(
            raw,
            redact_phones=bool(config.get("crawl.redact_phone_numbers", True)),
        )
    else:
        if input_path is None:
            raise ValueError("input_path is required when crawl=False")
        df = _read_input(input_path)
        meta["input_path"] = str(input_path)

    if df.empty:
        meta["counts"] = {"total": 0, "normal": 0, "review": 0, "high": 0}
        return df, meta

    # Independent evidence first; bounded decision layer last.
    df = add_features(df, config)
    df = add_market_anomaly(df, config)
    df = add_price_model_anomaly(df, config)
    df = add_density_anomaly(df, config)
    df = fuse_advanced_market_signals(df)
    df = add_duplicate_score(df, config)
    df = add_duplicate_graph_signals(df, config)
    df["prefilter_score"] = prefilter_score(df)
    df, decision_meta = apply_decisions(df, config)
    meta["decision"] = decision_meta
    df = _finalize_scores(df, config)

    out_dir = Path(config.get("output.directory", "outputs/fatemi"))
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    paths: dict[str, str] = {}

    if bool(config.get("output.write_csv", True)):
        csv_path = out_dir / f"listings_scored_{stamp}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        paths["csv"] = str(csv_path)
        review_path = out_dir / f"review_queue_{stamp}.csv"
        df[df["risk_band"].isin(["review", "high"])].to_csv(
            review_path, index=False, encoding="utf-8-sig"
        )
        paths["review_csv"] = str(review_path)

    if bool(config.get("output.write_parquet", True)):
        pq_path = out_dir / f"listings_scored_{stamp}.parquet"
        df.to_parquet(pq_path, index=False)
        paths["parquet"] = str(pq_path)

    if bool(config.get("output.write_html_report", True)):
        html_path = out_dir / f"report_{stamp}.html"
        build_html_report(df, meta, html_path)
        paths["html_report"] = str(html_path)

    meta["outputs"] = paths
    meta["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    meta["counts"] = {
        "total": int(len(df)),
        "normal": int((df["risk_band"] == "normal").sum()),
        "review": int((df["risk_band"] == "review").sum()),
        "high": int((df["risk_band"] == "high").sum()),
    }
    meta_path = out_dir / f"run_meta_{stamp}.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    meta["outputs"]["meta"] = str(meta_path)
    return df, meta
