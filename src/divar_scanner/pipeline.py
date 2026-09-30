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
from .features import add_features
from .normalize import normalize_many
from .report import build_html_report
from .semantic import apply_semantic


def _as_float(row: pd.Series, name: str) -> float:
    try:
        v = row.get(name, 0)
        if pd.isna(v):
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
    threshold = float(config.get("anomaly.duplicate_similarity_threshold", 0.88))
    if _as_float(row, "duplicate_similarity") >= threshold:
        reasons.append("near-duplicate text")
    if _as_float(row, "semantic_score") >= 0.55:
        provider = str(row.get("semantic_provider", "semantic"))
        reasons.append(f"semantic suspicion ({provider})")
    cls = str(row.get("semantic_classification", ""))
    if cls and cls not in {"plausible", "market_outlier", "not_evaluated", "unknown"}:
        reasons.append(f"class:{cls}")
    if _as_float(row, "uncertainty_score") >= 0.28:
        reasons.append("detectors disagree → human review")
    return "; ".join(dict.fromkeys(reasons)) or "low combined risk"


def _finalize_scores(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    out = df.copy()
    s = config.section("scoring")
    threshold = float(config.get("anomaly.duplicate_similarity_threshold", 0.88))
    denom = max(1e-6, 1.0 - threshold)
    nearest_dup = np.clip(
        (pd.to_numeric(out["duplicate_similarity"], errors="coerce").fillna(0) - threshold) / denom,
        0, 1,
    )
    bait = pd.to_numeric(out.get("duplicate_bait_score", 0), errors="coerce").fillna(0)
    out["duplicate_risk"] = np.maximum(nearest_dup, bait)

    confidence = pd.to_numeric(out.get("semantic_confidence", 0.35), errors="coerce").fillna(0.35)
    semantic = pd.to_numeric(out.get("semantic_score", 0), errors="coerce").fillna(0)
    # A weak/fallback semantic opinion must not dominate deterministic evidence.
    out["semantic_effective_score"] = semantic * (0.45 + 0.55 * confidence.clip(0, 1))

    channels = pd.DataFrame(
        {
            "quality": pd.to_numeric(out["data_quality_score"], errors="coerce").fillna(0),
            "market": pd.to_numeric(out["market_anomaly_score"], errors="coerce").fillna(0),
            "duplicate": out["duplicate_risk"].fillna(0),
            "semantic": out["semantic_effective_score"].fillna(0),
        },
        index=out.index,
    )
    out["uncertainty_score"] = channels.std(axis=1).clip(0, 0.5) * 2.0

    weights = {
        "data_quality_score": float(s.get("data_quality_weight", 0.22)),
        "market_anomaly_score": float(s.get("market_anomaly_weight", 0.38)),
        "duplicate_risk": float(s.get("duplicate_weight", 0.16)),
        "semantic_effective_score": float(s.get("semantic_weight", 0.24)),
    }
    total = max(sum(weights.values()), 1e-9)
    score = sum(pd.to_numeric(out[col], errors="coerce").fillna(0) * w for col, w in weights.items()) / total
    out["suspicion_score"] = np.clip(score, 0, 1)

    review_prob = pd.to_numeric(
        out.get("semantic_review_probability", 0), errors="coerce"
    ).fillna(0)
    out["review_priority_score"] = np.clip(
        0.78 * out["suspicion_score"]
        + 0.14 * out["uncertainty_score"]
        + 0.08 * review_prob,
        0, 1,
    )

    review_t = float(s.get("review_threshold", 0.58))
    high_t = float(s.get("high_risk_threshold", 0.78))
    out["risk_band"] = np.select(
        [out["review_priority_score"] >= high_t, out["review_priority_score"] >= review_t],
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
        "pipeline_version": "0.2.0",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
    }

    if crawl:
        crawler = DivarCrawler(config)
        raw, crawl_meta = crawler.crawl()
        meta["crawl"] = crawl_meta
        meta.update(crawl_meta)
        df = normalize_many(raw, redact_phones=bool(config.get("crawl.redact_phone_numbers", True)))
    else:
        if input_path is None:
            raise ValueError("input_path is required when crawl=False")
        df = _read_input(input_path)
        meta["input_path"] = str(input_path)

    if df.empty:
        meta["counts"] = {"total": 0, "normal": 0, "review": 0, "high": 0}
        return df, meta

    # Evidence pipeline: deterministic → distributional → learned OOF → graph → semantic.
    df = add_features(df, config)
    df = add_market_anomaly(df, config)
    df = add_price_model_anomaly(df, config)
    df = add_density_anomaly(df, config)
    df = fuse_advanced_market_signals(df)
    df = add_duplicate_score(df, config)
    df = add_duplicate_graph_signals(df, config)
    df["prefilter_score"] = prefilter_score(df)
    df, semantic_meta = apply_semantic(df, config)
    meta["semantic"] = semantic_meta
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
