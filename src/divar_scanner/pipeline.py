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
    if _as_float(row, "data_problem_score") >= 0.55:
        reasons.append(f"data-quality risk={_as_float(row, 'data_problem_score'):.2f}")
    if _as_float(row, "market_outlier_score") >= 0.55:
        ratio = row.get("price_model_ratio")
        if pd.notna(ratio):
            reasons.append(
                f"market outlier={_as_float(row, 'market_outlier_score'):.2f} "
                f"(actual/expected={float(ratio):.2f})"
            )
        else:
            reasons.append(f"market outlier={_as_float(row, 'market_outlier_score'):.2f}")
    if _as_float(row, "misleading_risk_score") >= 0.50:
        reasons.append(f"misleading/bait evidence={_as_float(row, 'misleading_risk_score'):.2f}")
    if _as_float(row, "duplicate_bait_score") >= 0.50:
        reasons.append(
            f"duplicate-cluster inconsistency (n={int(_as_float(row, 'duplicate_cluster_size'))})"
        )

    if bool(row.get("decision_evaluated", False)):
        disposition = str(row.get("decision_disposition", "") or "")
        if disposition and disposition not in {"plausible", "market_outlier"}:
            reasons.append(f"decision:{disposition}")
        if bool(row.get("decision_abstain", False)):
            why = str(row.get("decision_abstain_reasons", "") or "low decision reliability")
            reasons.append(f"System-One abstained:{why}")
        elif _as_float(row, "decision_effective_confidence") < 0.68:
            reasons.append(
                f"decision confidence={_as_float(row, 'decision_effective_confidence'):.2f}"
            )

    if _as_float(row, "uncertainty_score") >= 0.40:
        reasons.append("evidence families disagree / need human review")
    return "; ".join(dict.fromkeys(reasons)) or "low review priority"


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


def _numeric_series(
    frame: pd.DataFrame,
    column: str,
    default: float = np.nan,
) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").astype(float)


def _json_probability(frame: pd.DataFrame, column: str, key: str) -> pd.Series:
    def parse(value: Any) -> float:
        try:
            payload = json.loads(str(value or "{}"))
            return float(payload.get(key, np.nan))
        except Exception:
            return np.nan
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return frame[column].map(parse).astype(float)


def _finalize_scores(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Fuse independent evidence channels without equating anomaly with deception."""
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

    evaluated = _numeric_series(out, "decision_evaluated", 0.0).fillna(0).astype(bool)
    decision_conf = _numeric_series(
        out, "decision_effective_confidence", np.nan
    ).clip(0, 1)
    confidence_weight = decision_conf.where(evaluated, np.nan)

    # Channel 1 — data correctness. This is not fraud.
    data_decision = _numeric_series(
        out, "decision_data_error_probability", np.nan
    )
    integrity_error = (
        _json_probability(out, "decision_integrity_probs_json", "extraction_error").fillna(0)
        + _json_probability(out, "decision_integrity_probs_json", "listing_claim_conflict").fillna(0)
    ).clip(0, 1)
    data_model = (0.55 * data_decision + 0.45 * integrity_error).where(evaluated)
    out["data_problem_score"], _ = _rowwise_weighted_mean(
        pd.DataFrame(
            {
                "rule": pd.to_numeric(out["data_quality_score"], errors="coerce").fillna(0),
                "model": data_model * confidence_weight,
            },
            index=out.index,
        ),
        {"rule": float(s.get("data_rule_weight", 0.72)), "model": float(s.get("data_decision_weight", 0.28))},
    )
    out["data_problem_score"] = out["data_problem_score"].clip(0, 1)

    # Channel 2 — market outlier. Kept independent from deception.
    market_decision = (
        _json_probability(out, "decision_market_status_probs_json", "moderate_outlier").fillna(0)
        + _json_probability(out, "decision_market_status_probs_json", "extreme_outlier").fillna(0)
    ).clip(0, 1).where(evaluated)
    market_frame = pd.DataFrame(
        {
            "stats": pd.to_numeric(out["market_anomaly_score"], errors="coerce").fillna(0),
            "model": market_decision * confidence_weight,
        },
        index=out.index,
    )
    out["market_outlier_score"], _ = _rowwise_weighted_mean(
        market_frame,
        {"stats": float(s.get("market_stat_weight", 0.82)), "model": float(s.get("market_decision_weight", 0.18))},
    )
    out["market_outlier_score"] = out["market_outlier_score"].clip(0, 1)

    # Channel 3 — misleading/bait. Deliberately receives no single-listing market score.
    decision_bait = _numeric_series(
        out, "decision_bait_probability", np.nan
    ).where(evaluated)
    disposition_bait = _json_probability(
        out, "decision_disposition_probs_json", "misleading_or_bait"
    ).where(evaluated)
    duplicate_conflict = _json_probability(
        out, "decision_duplicate_pattern_probs_json", "cross_property_conflict"
    ).where(evaluated)
    bounded_bait = pd.concat(
        [decision_bait, disposition_bait, duplicate_conflict], axis=1
    ).max(axis=1, skipna=True).where(evaluated)
    misleading_frame = pd.DataFrame(
        {
            "duplicate": out["duplicate_risk"],
            "decision": bounded_bait * confidence_weight,
        },
        index=out.index,
    )
    out["misleading_risk_score"], _ = _rowwise_weighted_mean(
        misleading_frame,
        {
            "duplicate": float(s.get("misleading_duplicate_weight", 0.42)),
            "decision": float(s.get("misleading_decision_weight", 0.58)),
        },
    )
    out["misleading_risk_score"] = out["misleading_risk_score"].clip(0, 1)
    # Backwards-compatible name now has a precise meaning: deception/bait suspicion only.
    out["suspicion_score"] = out["misleading_risk_score"]

    channels = ["data_problem_score", "market_outlier_score", "misleading_risk_score"]
    out["channel_disagreement"] = np.clip(_available_std(out, channels) * 2.0, 0, 1)

    model_uncertainty = pd.Series(0.0, index=out.index)
    model_uncertainty.loc[evaluated] = (
        1.0 - decision_conf.loc[evaluated].fillna(0.0)
    ).clip(0, 1)
    coherence_series = _numeric_series(
        out, "decision_coherence_score", np.nan
    )
    coherence_uncertainty = pd.Series(0.0, index=out.index)
    coherence_uncertainty.loc[evaluated] = (
        1.0 - coherence_series.loc[evaluated].fillna(0.0)
    ).clip(0, 1)
    out["uncertainty_score"] = np.clip(
        0.45 * model_uncertainty
        + 0.25 * coherence_uncertainty
        + 0.30 * out["channel_disagreement"],
        0,
        1,
    )

    channel_values = out[channels].to_numpy(dtype=float)
    sorted_channels = np.sort(channel_values, axis=1)
    strongest = sorted_channels[:, -1]
    second = sorted_channels[:, -2] if len(channels) > 1 else strongest
    review_prob = _numeric_series(
        out, "decision_manual_review_probability", np.nan
    ).fillna(0.0)
    abstain = out.get("decision_abstain", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    abstain_uplift = abstain.astype(float) * float(s.get("abstain_review_uplift", 0.08))

    out["review_priority_score"] = np.clip(
        0.52 * strongest
        + 0.20 * second
        + 0.16 * out["uncertainty_score"]
        + 0.12 * review_prob
        + abstain_uplift,
        0,
        1,
    )
    out["overall_review_risk"] = out["review_priority_score"]

    review_t = float(s.get("review_threshold", 0.55))
    high_t = float(s.get("high_risk_threshold", 0.75))
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
        ["review_priority_score", "misleading_risk_score", "market_outlier_score"],
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
        "pipeline_version": "0.4.0",
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
        review_df = df[df["risk_band"].isin(["review", "high"])].copy()
        for label_col in (
            "human_disposition",
            "human_integrity_class",
            "human_duplicate_pattern",
            "human_market_status",
            "human_bait",
            "human_data_error",
            "human_manual_review",
            "human_consistency_level",
            "human_label_confidence",
            "human_notes",
        ):
            if label_col not in review_df:
                review_df[label_col] = ""
        review_df.to_csv(review_path, index=False, encoding="utf-8-sig")
        paths["review_csv"] = str(review_path)

        annotation_path = out_dir / f"decision_annotation_sample_{stamp}.csv"
        annotation_df = df[df["decision_evaluated"] == True].copy()  # noqa: E712
        for label_col in (
            "human_disposition",
            "human_integrity_class",
            "human_duplicate_pattern",
            "human_market_status",
            "human_bait",
            "human_data_error",
            "human_manual_review",
            "human_consistency_level",
            "human_label_confidence",
            "human_notes",
        ):
            if label_col not in annotation_df:
                annotation_df[label_col] = ""
        annotation_df.to_csv(annotation_path, index=False, encoding="utf-8-sig")
        paths["annotation_csv"] = str(annotation_path)

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
