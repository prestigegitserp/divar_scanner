from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import requests

from .config import Config


@dataclass
class JevRuntime:
    enabled: bool
    endpoint: str | None
    api_key: str | None
    model: str
    reason: str = ""


def resolve_jev(config: Config) -> JevRuntime:
    s = config.section("semantic")
    if not bool(s.get("enabled", True)):
        return JevRuntime(False, None, None, str(s.get("model", "jev-latest")), "disabled in config")
    hosted = os.getenv("JEV_API_KEY")
    official = os.getenv("TYPESAFE_API_KEY")
    endpoint = os.getenv("JEV_ENDPOINT")
    if hosted:
        return JevRuntime(
            True,
            endpoint or "https://jevtypesafeai.com/api/v1/decide",
            hosted,
            str(s.get("model", "jev-latest")),
        )
    if official:
        return JevRuntime(
            True,
            endpoint or "https://api.typesafe.ai/v1/systemone",
            official,
            str(s.get("model", "jev-latest")),
        )
    return JevRuntime(False, None, None, str(s.get("model", "jev-latest")), "no API key found")


def _safe(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, float) and np.isnan(v):
        return None
    if pd.isna(v) if not isinstance(v, (list, dict, str, bool)) else False:
        return None
    return v


def _state(row: pd.Series) -> dict[str, Any]:
    fields = [
        "title",
        "description",
        "neighborhood",
        "area_m2",
        "rooms",
        "year_built_shamsi",
        "deposit_toman",
        "rent_monthly_toman",
        "equivalent_deposit_per_m2",
        "peer_anomaly_equivalent_deposit_per_m2",
        "data_quality_score",
        "market_anomaly_score",
        "duplicate_similarity",
        "area_text_conflict",
        "rooms_text_conflict",
        "amenity_text_conflict",
    ]
    listing = {k: _safe(row.get(k)) for k in fields if k in row.index}
    listing["analysis_note"] = (
        "Scores are evidence signals, not proof of fraud. Distinguish parser/data errors, "
        "legitimate market outliers, misleading ads, and cases that require human review."
    )
    return listing


def _questions() -> dict[str, Any]:
    return {
        "classification": {
            "type": "choice",
            "instructions": "Classify this rental listing using only the supplied evidence.",
            "criteria": {
                "plausible": "internally coherent and plausibly a real listing",
                "data_error": "likely parser/structured-data error or accidental inconsistent data",
                "market_outlier": "unusual versus peers but not clearly misleading",
                "misleading_or_bait": "material claims look inconsistent, bait-like, or misleading",
                "needs_review": "evidence is ambiguous and should be manually reviewed",
            },
        },
        "semantic_suspicion": {
            "type": "noul",
            "instructions": (
                "Is there meaningful evidence that this ad may be misleading, bait-like, fabricated, "
                "or otherwise suspicious enough to merit scrutiny? Do not equate a low/high price alone with fraud."
            ),
        },
        "manual_review": {
            "type": "noul",
            "instructions": "Should a human analyst review this listing before trusting it?",
        },
        "consistency": {
            "type": "score",
            "instructions": "Rate internal consistency between structured fields and listing text.",
            "criteria": [
                "major contradictions",
                "multiple concerning inconsistencies",
                "some ambiguity or minor mismatches",
                "mostly consistent",
                "strongly consistent",
            ],
        },
    }


def decide(row: pd.Series, runtime: JevRuntime, timeout: int = 30, retries: int = 3) -> dict[str, Any]:
    if not runtime.enabled or not runtime.endpoint or not runtime.api_key:
        return {}
    payload = {"model": runtime.model, "state": _state(row), "questions": _questions()}
    last_exc: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            r = requests.post(
                runtime.endpoint,
                headers={
                    "Authorization": f"Bearer {runtime.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=timeout,
            )
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            last_exc = exc
            if attempt + 1 < retries:
                time.sleep(min(2 ** attempt, 8))
    if last_exc:
        raise last_exc
    return {}


def _heuristic_semantic(row: pd.Series) -> float:
    conflict = max(
        float(row.get("area_text_conflict", 0) or 0),
        float(row.get("rooms_text_conflict", 0) or 0),
        float(row.get("amenity_text_conflict", 0) or 0),
    )
    text = f"{row.get('title', '')} {row.get('description', '')}".lower()
    bait_terms = ["فوری", "زیر قیمت", "استثنایی", "باور نکردنی", "فقط امروز", "بیعانه", "رزرو"]
    bait = min(1.0, sum(term in text for term in bait_terms) / 3.0)
    return float(np.clip(0.70 * conflict + 0.30 * bait, 0, 1))


def apply_semantic(df: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    s = config.section("semantic")
    runtime = resolve_jev(config)
    out["semantic_score"] = [_heuristic_semantic(row) for _, row in out.iterrows()]
    out["jev_classification"] = "not_evaluated"
    out["jev_review_probability"] = np.nan
    out["jev_consistency_score"] = np.nan
    out["jev_raw_json"] = ""

    if not runtime.enabled or out.empty:
        return out, {"jev_enabled": False, "jev_reason": runtime.reason, "jev_evaluated": 0}

    top_k = min(int(s.get("top_k", 80)), len(out))
    min_prefilter = float(s.get("min_prefilter_score", 0.35))
    candidates = out[out["prefilter_score"] >= min_prefilter].nlargest(top_k, "prefilter_score")
    timeout = int(s.get("timeout_seconds", 30))
    retries = int(s.get("retries", 3))
    errors: list[str] = []
    evaluated = 0

    for idx, row in candidates.iterrows():
        try:
            response = decide(row, runtime, timeout=timeout, retries=retries)
            answers = response.get("answers", {}) if isinstance(response, dict) else {}
            classification = answers.get("classification", {})
            susp = answers.get("semantic_suspicion", {})
            review = answers.get("manual_review", {})
            consistency = answers.get("consistency", {})
            p_susp = float(susp.get("noul", 0.0) or 0.0)
            p_review = float(review.get("noul", 0.0) or 0.0)
            out.at[idx, "semantic_score"] = np.clip(0.65 * p_susp + 0.35 * p_review, 0, 1)
            out.at[idx, "jev_classification"] = str(classification.get("choice", "unknown"))
            out.at[idx, "jev_review_probability"] = p_review
            out.at[idx, "jev_consistency_score"] = consistency.get("score", np.nan)
            out.at[idx, "jev_raw_json"] = json.dumps(response, ensure_ascii=False)
            evaluated += 1
        except Exception as exc:  # keep the pipeline usable when the optional API is unavailable
            errors.append(f"{row.get('token', idx)}: {type(exc).__name__}: {exc}")

    return out, {
        "jev_enabled": True,
        "jev_endpoint": runtime.endpoint,
        "jev_model": runtime.model,
        "jev_evaluated": evaluated,
        "jev_errors": errors[:10],
    }
