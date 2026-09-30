from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .anomaly import add_duplicate_score, add_market_anomaly, prefilter_score
from .config import Config, load_config
from .crawler import DivarCrawler
from .features import add_features
from .jev import apply_semantic
from .normalize import normalize_many
from .report import build_html_report


def _reason(row: pd.Series, config: Config) -> str:
    reasons: list[str] = []
    if float(row.get("data_quality_score", 0) or 0) >= 0.45:
        reasons.append("data/field inconsistency")
    if float(row.get("market_anomaly_score", 0) or 0) >= 0.55:
        reasons.append("unusual vs local peers")
    threshold = float(config.get("anomaly.duplicate_similarity_threshold", 0.88))
    if float(row.get("duplicate_similarity", 0) or 0) >= threshold:
        reasons.append("near-duplicate listing")
    if float(row.get("semantic_score", 0) or 0) >= 0.55:
        reasons.append("semantic suspicion")
    cls = str(row.get("jev_classification", ""))
    if cls and cls not in {"not_evaluated", "plausible", "market_outlier", "unknown"}:
        reasons.append(f"Jev:{cls}")
    return "; ".join(dict.fromkeys(reasons)) or "low combined risk"


def _finalize_scores(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    out = df.copy()
    s = config.section("scoring")
    weights = {
        "data_quality_score": float(s.get("data_quality_weight", 0.24)),
        "market_anomaly_score": float(s.get("market_anomaly_weight", 0.36)),
        "duplicate_risk": float(s.get("duplicate_weight", 0.12)),
        "semantic_score": float(s.get("semantic_weight", 0.28)),
    }
    threshold = float(config.get("anomaly.duplicate_similarity_threshold", 0.88))
    denom = max(1e-6, 1.0 - threshold)
    out["duplicate_risk"] = np.clip((out["duplicate_similarity"].fillna(0) - threshold) / denom, 0, 1)
    total = max(sum(weights.values()), 1e-9)
    score = sum(out[col].fillna(0).astype(float) * w for col, w in weights.items()) / total
    out["suspicion_score"] = np.clip(score, 0, 1)

    review_t = float(s.get("review_threshold", 0.62))
    high_t = float(s.get("high_risk_threshold", 0.80))
    out["risk_band"] = np.select(
        [out["suspicion_score"] >= high_t, out["suspicion_score"] >= review_t],
        ["high", "review"],
        default="normal",
    )
    out["flag_reasons"] = [_reason(row, config) for _, row in out.iterrows()]
    return out.sort_values(["suspicion_score", "market_anomaly_score"], ascending=False).reset_index(drop=True)


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
        return df, meta

    df = add_features(df, config)
    df = add_market_anomaly(df, config)
    df = add_duplicate_score(df, config)
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
