from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import requests

from .config import Config


CLASSES = ["plausible", "data_error", "market_outlier", "misleading_or_bait", "needs_review"]


@dataclass(frozen=True)
class ProviderRuntime:
    name: str
    endpoint: str
    api_key: str
    model: str
    native_jev: bool = False


def _env(name: str) -> str | None:
    v = os.getenv(name)
    return v.strip() if v and v.strip() else None


def _provider_candidates(config: Config) -> list[ProviderRuntime]:
    s = config.section("semantic")
    models = s.get("models", {}) if isinstance(s.get("models"), dict) else {}
    candidates: list[ProviderRuntime] = []

    if _env("JEV_API_KEY"):
        candidates.append(ProviderRuntime(
            "jev",
            _env("JEV_ENDPOINT") or "https://jevtypesafeai.com/api/v1/decide",
            _env("JEV_API_KEY") or "",
            str(models.get("jev", s.get("model", "jev-1.13.0"))),
            True,
        ))
    if _env("TYPESAFE_API_KEY"):
        candidates.append(ProviderRuntime(
            "jev-official",
            _env("JEV_ENDPOINT") or "https://api.typesafe.ai/v1/systemone",
            _env("TYPESAFE_API_KEY") or "",
            str(models.get("jev", s.get("model", "jev-1.13.0"))),
            True,
        ))
    if _env("GROQ_API_KEY"):
        candidates.append(ProviderRuntime(
            "groq", "https://api.groq.com/openai/v1/chat/completions",
            _env("GROQ_API_KEY") or "", str(models.get("groq", "openai/gpt-oss-20b"))
        ))
    if _env("CEREBRAS_API_KEY"):
        candidates.append(ProviderRuntime(
            "cerebras", "https://api.cerebras.ai/v1/chat/completions",
            _env("CEREBRAS_API_KEY") or "", str(models.get("cerebras", "gpt-oss-120b"))
        ))
    if _env("GEMINI_API_KEY"):
        candidates.append(ProviderRuntime(
            "gemini", "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            _env("GEMINI_API_KEY") or "", str(models.get("gemini", "gemini-3.8-flash"))
        ))
    if _env("COHERE_API_KEY"):
        candidates.append(ProviderRuntime(
            "cohere", "https://api.cohere.ai/compatibility/v1/chat/completions",
            _env("COHERE_API_KEY") or "", str(models.get("cohere", "c4ai-aya-expanse-32b"))
        ))
    if _env("OPENROUTER_API_KEY"):
        candidates.append(ProviderRuntime(
            "openrouter", "https://openrouter.ai/api/v1/chat/completions",
            _env("OPENROUTER_API_KEY") or "", str(models.get("openrouter", "openrouter/free"))
        ))
    if _env("HF_TOKEN"):
        candidates.append(ProviderRuntime(
            "huggingface", "https://router.huggingface.co/v1/chat/completions",
            _env("HF_TOKEN") or "", str(models.get("huggingface", "openai/gpt-oss-120b:fastest"))
        ))
    if _env("OPENAI_COMPAT_API_KEY") and _env("OPENAI_COMPAT_BASE_URL") and _env("OPENAI_COMPAT_MODEL"):
        endpoint = (_env("OPENAI_COMPAT_BASE_URL") or "").rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint += "/chat/completions"
        candidates.append(ProviderRuntime(
            "custom-openai-compatible", endpoint,
            _env("OPENAI_COMPAT_API_KEY") or "", _env("OPENAI_COMPAT_MODEL") or ""
        ))
    return candidates


def resolve_providers(config: Config) -> list[ProviderRuntime]:
    s = config.section("semantic")
    if not bool(s.get("enabled", True)):
        return []
    requested = str(os.getenv("SEMANTIC_PROVIDER") or s.get("provider", "auto")).strip().lower()
    all_candidates = _provider_candidates(config)
    if requested != "auto":
        aliases = {"hf": "huggingface", "typesafe": "jev-official"}
        requested = aliases.get(requested, requested)
        return [p for p in all_candidates if p.name == requested][:1]

    preferred = s.get(
        "provider_priority",
        ["jev", "jev-official", "groq", "cerebras", "gemini", "cohere", "openrouter", "huggingface", "custom-openai-compatible"],
    )
    order = {name: i for i, name in enumerate(preferred)}
    all_candidates.sort(key=lambda p: order.get(p.name, 999))
    max_providers = max(1, int(s.get("ensemble_max_providers", 1)))
    return all_candidates[:max_providers]


def _safe(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, float) and np.isnan(v):
        return None
    try:
        if not isinstance(v, (list, dict, str, bool)) and pd.isna(v):
            return None
    except Exception:
        pass
    return v


def _state(row: pd.Series) -> dict[str, Any]:
    fields = [
        "title", "description", "neighborhood", "area_m2", "rooms", "year_built_shamsi",
        "deposit_toman", "rent_monthly_toman", "equivalent_deposit_per_m2",
        "data_quality_score", "market_anomaly_score", "price_model_ratio",
        "price_model_anomaly_score", "lof_anomaly_score", "duplicate_similarity",
        "duplicate_cluster_size", "duplicate_cluster_neighborhoods",
        "duplicate_cluster_price_span", "duplicate_bait_score",
        "area_text_conflict", "rooms_text_conflict", "amenity_text_conflict",
    ]
    return {k: _safe(row.get(k)) for k in fields if k in row.index}


def _schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "classification": {"type": "string", "enum": CLASSES},
            "suspicion_probability": {"type": "number", "minimum": 0, "maximum": 1},
            "manual_review_probability": {"type": "number", "minimum": 0, "maximum": 1},
            "consistency_score": {"type": "number", "minimum": 0, "maximum": 1},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "evidence": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
        },
        "required": [
            "classification", "suspicion_probability", "manual_review_probability",
            "consistency_score", "confidence", "evidence",
        ],
        "additionalProperties": False,
    }


def _system_prompt() -> str:
    return (
        "You are a bounded real-estate listing risk classifier, not a copywriter. "
        "Assess only the supplied evidence. A market outlier is NOT automatically fraud. "
        "Separate parser/data errors from legitimate unusual properties and from misleading/bait-like ads. "
        "Use price-model residuals as evidence, not truth. Return only schema-compliant JSON."
    )


def _user_prompt(row: pd.Series) -> str:
    state = json.dumps(_state(row), ensure_ascii=False, separators=(",", ":"))
    return (
        "Classify this Tehran rental listing for review triage. "
        "suspicion_probability means evidence of misleading/bait/fabricated behavior; "
        "manual_review_probability means a human should inspect it. "
        "consistency_score=1 means internally consistent.\nSTATE:\n" + state
    )


def _post(runtime: ProviderRuntime, payload: dict[str, Any], timeout: int) -> requests.Response:
    headers = {"Authorization": f"Bearer {runtime.api_key}", "Content-Type": "application/json"}
    if runtime.name == "openrouter":
        headers["HTTP-Referer"] = "https://github.com/prestigegitserp/divar_scanner"
        headers["X-Title"] = "Divar Scanner"
    return requests.post(runtime.endpoint, headers=headers, json=payload, timeout=timeout)


def _extract_json_text(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        value = json.loads(m.group(0))
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        return {}


def _validate_result(data: dict[str, Any]) -> dict[str, Any]:
    cls = str(data.get("classification", "needs_review"))
    if cls not in CLASSES:
        cls = "needs_review"

    def p(name: str, default: float) -> float:
        try:
            return float(np.clip(float(data.get(name, default)), 0, 1))
        except Exception:
            return default

    evidence = data.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = [str(evidence)]
    return {
        "classification": cls,
        "suspicion_probability": p("suspicion_probability", 0.5),
        "manual_review_probability": p("manual_review_probability", 0.5),
        "consistency_score": p("consistency_score", 0.5),
        "confidence": p("confidence", 0.5),
        "evidence": [str(x)[:300] for x in evidence[:6]],
    }


def _decide_openai_compatible(
    row: pd.Series, runtime: ProviderRuntime, timeout: int, retries: int
) -> dict[str, Any]:
    base = {
        "model": runtime.model,
        "messages": [
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": _user_prompt(row)},
        ],
        "temperature": 0,
        "max_tokens": 450,
    }
    formats = [
        {
            "type": "json_schema",
            "json_schema": {"name": "listing_risk", "strict": True, "schema": _schema()},
        },
        {"type": "json_object"},
        None,
    ]
    last_error: Exception | None = None
    for fmt in formats:
        payload = dict(base)
        if fmt is not None:
            payload["response_format"] = fmt
        for attempt in range(max(1, retries)):
            try:
                r = _post(runtime, payload, timeout)
                # Unsupported response_format should fall through to the next compatibility mode.
                if r.status_code in {400, 404, 422} and fmt is not None:
                    break
                r.raise_for_status()
                body = r.json()
                content = (((body.get("choices") or [{}])[0].get("message") or {}).get("content"))
                if isinstance(content, list):
                    content = "".join(str(x.get("text", "")) if isinstance(x, dict) else str(x) for x in content)
                parsed = _extract_json_text(str(content or ""))
                if parsed:
                    return _validate_result(parsed)
                raise ValueError("Provider returned no parseable JSON object")
            except (requests.RequestException, ValueError, KeyError) as exc:
                last_error = exc
                if attempt + 1 < retries:
                    time.sleep(min(2 ** attempt, 6))
    if last_error:
        raise last_error
    raise RuntimeError("No compatible response mode succeeded")


def _decide_jev(row: pd.Series, runtime: ProviderRuntime, timeout: int, retries: int) -> dict[str, Any]:
    payload = {
        "model": runtime.model,
        "state": _state(row),
        "questions": {
            "classification": {
                "type": "choice",
                "instructions": "Classify using only supplied evidence; price outlier alone is not fraud.",
                "criteria": {
                    "plausible": "coherent and plausibly genuine",
                    "data_error": "likely parser/structured-data mistake",
                    "market_outlier": "unusual versus peers without misleading evidence",
                    "misleading_or_bait": "materially inconsistent, bait-like, or misleading",
                    "needs_review": "ambiguous evidence requiring human review",
                },
            },
            "semantic_suspicion": {
                "type": "noul",
                "instructions": "Meaningful evidence of misleading, bait-like, or fabricated behavior?",
            },
            "manual_review": {
                "type": "noul",
                "instructions": "Should a human review this listing before trusting it?",
            },
            "consistency": {
                "type": "score",
                "instructions": "Internal consistency of text and structured fields.",
                "criteria": [
                    "major contradictions", "concerning inconsistencies", "ambiguous",
                    "mostly consistent", "strongly consistent",
                ],
            },
        },
    }
    last: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            r = _post(runtime, payload, timeout)
            r.raise_for_status()
            body = r.json()
            ans = body.get("answers", {})
            cls = str((ans.get("classification") or {}).get("choice", "needs_review"))
            susp = float((ans.get("semantic_suspicion") or {}).get("noul", 0.5) or 0.5)
            rev = float((ans.get("manual_review") or {}).get("noul", 0.5) or 0.5)
            raw_cons = (ans.get("consistency") or {}).get("score", 2)
            try:
                consistency = float(raw_cons) / 4.0 if float(raw_cons) > 1 else float(raw_cons)
            except Exception:
                consistency = 0.5
            confidence = max(abs(susp - 0.5) * 2, abs(rev - 0.5) * 2)
            return _validate_result({
                "classification": cls,
                "suspicion_probability": susp,
                "manual_review_probability": rev,
                "consistency_score": consistency,
                "confidence": confidence,
                "evidence": [],
            })
        except (requests.RequestException, ValueError, KeyError) as exc:
            last = exc
            if attempt + 1 < retries:
                time.sleep(min(2 ** attempt, 6))
    if last:
        raise last
    raise RuntimeError("Jev request failed")


def _num(row: pd.Series, name: str) -> float:
    try:
        value = row.get(name, 0)
        if value is None or pd.isna(value):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def _heuristic(row: pd.Series) -> dict[str, Any]:
    conflict = max(
        _num(row, "area_text_conflict"),
        _num(row, "rooms_text_conflict"),
        _num(row, "amenity_text_conflict"),
    )
    bait = _num(row, "duplicate_bait_score")
    quality = _num(row, "data_quality_score")
    price = _num(row, "price_model_anomaly_score")
    suspicion = float(np.clip(0.46 * conflict + 0.30 * bait + 0.14 * quality + 0.10 * price, 0, 1))
    review = float(np.clip(max(suspicion, 0.55 * quality + 0.45 * price), 0, 1))
    if conflict >= 0.9:
        cls = "data_error"
    elif bait >= 0.6:
        cls = "misleading_or_bait"
    elif price >= 0.65:
        cls = "market_outlier"
    elif review >= 0.55:
        cls = "needs_review"
    else:
        cls = "plausible"
    return {
        "classification": cls,
        "suspicion_probability": suspicion,
        "manual_review_probability": review,
        "consistency_score": float(np.clip(1 - conflict, 0, 1)),
        "confidence": 0.35,
        "evidence": ["deterministic fallback; no external semantic provider"],
    }


def apply_semantic(df: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    providers = resolve_providers(config)
    s = config.section("semantic")
    fallback = [_heuristic(row) for _, row in out.iterrows()]

    out["semantic_score"] = [x["suspicion_probability"] for x in fallback]
    out["semantic_review_probability"] = [x["manual_review_probability"] for x in fallback]
    out["semantic_consistency_score"] = [x["consistency_score"] for x in fallback]
    out["semantic_confidence"] = [x["confidence"] for x in fallback]
    out["semantic_classification"] = [x["classification"] for x in fallback]
    out["semantic_provider"] = "heuristic"
    out["semantic_evidence_json"] = [json.dumps(x["evidence"], ensure_ascii=False) for x in fallback]
    out["semantic_raw_json"] = ""

    # Backward-compatible columns for older notebooks/reports.
    out["jev_classification"] = out["semantic_classification"]
    out["jev_review_probability"] = out["semantic_review_probability"]
    out["jev_consistency_score"] = out["semantic_consistency_score"]
    out["jev_raw_json"] = ""

    if not providers or out.empty:
        return out, {
            "providers_available": [p.name for p in providers],
            "providers_used": [],
            "evaluated": 0,
            "reason": "no configured API key; deterministic semantic fallback used",
        }

    top_k = min(int(s.get("top_k", 80)), len(out))
    min_prefilter = float(s.get("min_prefilter_score", 0.28))
    candidates = out[out["prefilter_score"] >= min_prefilter].nlargest(top_k, "prefilter_score")
    timeout = int(s.get("timeout_seconds", 30))
    retries = int(s.get("retries", 2))
    errors: list[str] = []
    evaluated = 0

    for idx, row in candidates.iterrows():
        results: list[tuple[ProviderRuntime, dict[str, Any]]] = []
        for provider in providers:
            try:
                result = (
                    _decide_jev(row, provider, timeout, retries)
                    if provider.native_jev
                    else _decide_openai_compatible(row, provider, timeout, retries)
                )
                results.append((provider, result))
            except Exception as exc:
                errors.append(f"{provider.name}:{row.get('token', idx)}:{type(exc).__name__}:{exc}")

        if not results:
            continue
        suspicion = float(np.mean([r["suspicion_probability"] for _, r in results]))
        review = float(np.mean([r["manual_review_probability"] for _, r in results]))
        consistency = float(np.mean([r["consistency_score"] for _, r in results]))
        confidence = float(np.mean([r["confidence"] for _, r in results]))
        # Prefer the highest-confidence provider's class while averaging probabilities.
        best_provider, best = max(results, key=lambda pr: pr[1]["confidence"])
        evidence = []
        for p, r in results:
            evidence.extend([f"{p.name}: {e}" for e in r["evidence"]])

        out.at[idx, "semantic_score"] = suspicion
        out.at[idx, "semantic_review_probability"] = review
        out.at[idx, "semantic_consistency_score"] = consistency
        out.at[idx, "semantic_confidence"] = confidence
        out.at[idx, "semantic_classification"] = best["classification"]
        out.at[idx, "semantic_provider"] = ",".join(p.name for p, _ in results)
        out.at[idx, "semantic_evidence_json"] = json.dumps(evidence[:10], ensure_ascii=False)
        out.at[idx, "semantic_raw_json"] = json.dumps(
            {p.name: r for p, r in results}, ensure_ascii=False
        )
        out.at[idx, "jev_classification"] = best["classification"]
        out.at[idx, "jev_review_probability"] = review
        out.at[idx, "jev_consistency_score"] = consistency
        out.at[idx, "jev_raw_json"] = out.at[idx, "semantic_raw_json"]
        evaluated += 1

    return out, {
        "providers_available": [p.name for p in providers],
        "providers_used": sorted(set(",".join(out["semantic_provider"]).split(",")) - {"heuristic", ""}),
        "models": {p.name: p.model for p in providers},
        "evaluated": evaluated,
        "errors": errors[:20],
    }
