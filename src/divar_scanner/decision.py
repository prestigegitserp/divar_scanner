from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import Config


DEFAULT_NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"


def _softmax(values: Iterable[float], temperature: float = 1.0) -> np.ndarray:
    x = np.asarray(list(values), dtype=float)
    if x.size == 0:
        return x
    t = max(float(temperature), 1e-4)
    z = x / t
    z = z - np.nanmax(z)
    ex = np.exp(z)
    denom = ex.sum()
    if not np.isfinite(denom) or denom <= 0:
        return np.full_like(x, 1.0 / len(x), dtype=float)
    return ex / denom


def _normalized_entropy_confidence(probs: np.ndarray) -> float:
    p = np.asarray(probs, dtype=float)
    if len(p) <= 1:
        return 1.0
    p = np.clip(p, 1e-12, 1.0)
    entropy = -float(np.sum(p * np.log(p)))
    return float(np.clip(1.0 - entropy / math.log(len(p)), 0.0, 1.0))


def typed_answer_from_evidence(
    question: dict[str, Any],
    candidate_rows: list[dict[str, float]],
    *,
    temperature: float = 1.0,
) -> dict[str, Any]:
    """Convert candidate-level NLI evidence into Jev-style typed decisions.

    candidate_rows must align with the candidate order compiled for the question and contain:
      key, entailment, contradiction, neutral

    We use entailment-vs-contradiction log-odds as the bounded candidate evidence:
        e_i = log P(entail_i) - log P(contradict_i)
    and softmax across the *provided* candidates. No text/token generation is involved.
    """
    if not candidate_rows:
        raise ValueError("candidate_rows cannot be empty")

    evidence = []
    for row in candidate_rows:
        p_ent = float(np.clip(row["entailment"], 1e-9, 1.0))
        p_con = float(np.clip(row["contradiction"], 1e-9, 1.0))
        evidence.append(math.log(p_ent) - math.log(p_con))

    probs = _softmax(evidence, temperature=temperature)
    keys = [str(r["key"]) for r in candidate_rows]
    probability_map = {k: float(v) for k, v in zip(keys, probs)}
    neutral_mass = float(
        np.sum(probs * np.asarray([float(r["neutral"]) for r in candidate_rows], dtype=float))
    )
    concentration = _normalized_entropy_confidence(probs)
    # Neutral NLI mass means "the state does not decide this hypothesis"; reduce certainty.
    confidence = float(np.clip(concentration * (1.0 - 0.65 * neutral_mass), 0.0, 1.0))

    raw_nli = {
        str(r["key"]): {
            "entailment": float(r["entailment"]),
            "neutral": float(r["neutral"]),
            "contradiction": float(r["contradiction"]),
            "evidence_log_odds": float(e),
        }
        for r, e in zip(candidate_rows, evidence)
    }

    qtype = str(question.get("type", "")).lower()
    if qtype == "choice":
        winner = keys[int(np.argmax(probs))]
        return {
            "type": "choice",
            "choice": winner,
            "probabilities": probability_map,
            "confidence": confidence,
            "diagnostics": {
                "neutral_mass": neutral_mass,
                "probability_method": "softmax_over_nli_entailment_vs_contradiction_log_odds",
                "raw_nli": raw_nli,
            },
        }

    if qtype == "noul":
        if "yes" not in probability_map or "no" not in probability_map:
            raise ValueError("noul requires candidate keys 'no' and 'yes'")
        return {
            "type": "noul",
            "noul": float(probability_map["yes"]),
            "probabilities": probability_map,
            "diagnostics": {
                "decision_confidence": confidence,
                "neutral_mass": neutral_mass,
                "probability_method": "two_candidate_nli_softmax",
                "raw_nli": raw_nli,
            },
        }

    if qtype == "score":
        n = len(probs)
        expected_level = float(np.dot(np.arange(n, dtype=float), probs))
        normalized = expected_level / max(n - 1, 1)
        return {
            "type": "score",
            "score": expected_level,
            "normalized_score": normalized,
            "probabilities": probability_map,
            "legend": [str(x) for x in question.get("criteria", [])],
            "confidence": confidence,
            "diagnostics": {
                "neutral_mass": neutral_mass,
                "probability_method": "expected_level_over_nli_option_distribution",
                "raw_nli": raw_nli,
            },
        }

    raise ValueError(f"Unsupported decision question type: {qtype!r}")


@dataclass(frozen=True)
class CompiledCandidate:
    request_index: int
    question_id: str
    key: str
    premise: str
    hypothesis: str


def _candidate_hypothesis(instructions: str, key: str, description: str, qtype: str) -> str:
    instructions = instructions.strip()
    description = description.strip()
    if qtype == "choice":
        return (
            f"برای پرسش «{instructions}»، گزینهٔ «{key}» درست است: {description}"
        )
    if qtype == "score":
        return (
            f"برای ارزیابی «{instructions}»، سطح مناسب این است: {description}"
        )
    if qtype == "noul":
        answer = "بله" if key == "yes" else "خیر"
        return (
            f"پاسخ درست به پرسش «{instructions}» {answer} است. {description}"
        )
    raise ValueError(qtype)


def _question_candidates(question: dict[str, Any]) -> list[tuple[str, str]]:
    qtype = str(question.get("type", "")).lower()
    if qtype == "choice":
        criteria = question.get("criteria")
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise ValueError("choice requires a criteria mapping with at least two options")
        return [(str(k), str(v)) for k, v in criteria.items()]

    if qtype == "score":
        criteria = question.get("criteria")
        if not isinstance(criteria, list) or len(criteria) < 2:
            raise ValueError("score requires an ordered criteria list with at least two levels")
        return [(str(i), str(v)) for i, v in enumerate(criteria)]

    if qtype == "noul":
        criteria = question.get("criteria") or {}
        if not isinstance(criteria, dict):
            criteria = {}
        return [
            ("no", str(criteria.get("no", "شواهد موجود این گزاره را تأیید نمی‌کند."))),
            ("yes", str(criteria.get("yes", "شواهد موجود این گزاره را تأیید می‌کند."))),
        ]

    raise ValueError(f"Unsupported question type: {qtype!r}")


def compile_requests(
    requests: list[tuple[str, dict[str, dict[str, Any]]]]
) -> tuple[list[CompiledCandidate], dict[tuple[int, str], dict[str, Any]]]:
    compiled: list[CompiledCandidate] = []
    questions: dict[tuple[int, str], dict[str, Any]] = {}
    for request_index, (state, qmap) in enumerate(requests):
        if not isinstance(qmap, dict):
            raise TypeError("questions must be a mapping")
        for qid, question in qmap.items():
            qid = str(qid)
            questions[(request_index, qid)] = question
            qtype = str(question.get("type", "")).lower()
            instructions = str(question.get("instructions", qid))
            for key, description in _question_candidates(question):
                compiled.append(
                    CompiledCandidate(
                        request_index=request_index,
                        question_id=qid,
                        key=key,
                        premise=state,
                        hypothesis=_candidate_hypothesis(
                            instructions, key, description, qtype
                        ),
                    )
                )
    return compiled, questions


class NLIDecisionEngine:
    """A non-generative Jev-style decision engine backed by an NLI encoder classifier."""

    def __init__(
        self,
        model_name: str = DEFAULT_NLI_MODEL,
        *,
        device: str = "auto",
        max_length: int = 512,
        batch_size: int = 24,
        temperatures: dict[str, float] | None = None,
    ):
        self.model_name = model_name
        self.requested_device = device
        self.max_length = int(max_length)
        self.batch_size = max(1, int(batch_size))
        self.temperatures = dict(temperatures or {})
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._device = None
        self._label_indices: tuple[int, int, int] | None = None

    @property
    def backend_name(self) -> str:
        return "mdeberta-nli"

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "The decision engine needs the optional 'decision' dependencies. "
                "Install with: python -m pip install -e '.[decision]'"
            ) from exc

        if self.requested_device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.requested_device

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model.to(device)
        model.eval()

        id2label = {
            int(k): str(v).lower()
            for k, v in getattr(model.config, "id2label", {}).items()
        }
        entail = next((i for i, v in id2label.items() if "entail" in v), None)
        neutral = next((i for i, v in id2label.items() if "neutral" in v), None)
        contradiction = next((i for i, v in id2label.items() if "contrad" in v), None)
        if None in (entail, neutral, contradiction):
            # Upstream mDeBERTa NLI checkpoints use this documented ordering.
            entail, neutral, contradiction = 0, 1, 2

        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        self._label_indices = (int(entail), int(neutral), int(contradiction))

    def _score_pairs(self, pairs: list[CompiledCandidate]) -> list[dict[str, float]]:
        self._load()
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._torch is not None
        assert self._label_indices is not None

        torch = self._torch
        entail_idx, neutral_idx, contradiction_idx = self._label_indices
        results: list[dict[str, float]] = []

        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            encoded = self._tokenizer(
                [x.premise for x in batch],
                [x.hypothesis for x in batch],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self._device) for k, v in encoded.items()}
            with torch.inference_mode():
                logits = self._model(**encoded).logits.float()
                probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            for row in probs:
                results.append(
                    {
                        "entailment": float(row[entail_idx]),
                        "neutral": float(row[neutral_idx]),
                        "contradiction": float(row[contradiction_idx]),
                    }
                )
        return results

    def decide_many(
        self,
        requests: list[tuple[str, dict[str, dict[str, Any]]]],
    ) -> list[dict[str, dict[str, Any]]]:
        if not requests:
            return []
        compiled, question_lookup = compile_requests(requests)
        scored = self._score_pairs(compiled)
        grouped: dict[tuple[int, str], list[dict[str, float]]] = {}

        for item, nli in zip(compiled, scored):
            grouped.setdefault((item.request_index, item.question_id), []).append(
                {"key": item.key, **nli}
            )

        outputs: list[dict[str, dict[str, Any]]] = [dict() for _ in requests]
        for route, rows in grouped.items():
            request_index, qid = route
            question = question_lookup[route]
            qtype = str(question.get("type", "")).lower()
            temperature = float(self.temperatures.get(qtype, 1.0))
            outputs[request_index][qid] = typed_answer_from_evidence(
                question,
                rows,
                temperature=temperature,
            )
        return outputs

    def decide(
        self,
        state: str,
        questions: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        return self.decide_many([(state, questions)])[0]


def _num(row: pd.Series, key: str) -> float | None:
    try:
        value = row.get(key)
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _fmt_num(value: float | None, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "ناموجود"
    return f"{value:.{digits}f}"


def build_listing_state(row: pd.Series) -> str:
    description = str(row.get("description", "") or "").strip()
    if len(description) > 1800:
        description = description[:1800] + "…"

    title = str(row.get("title", "") or "").strip()
    neighborhood = str(row.get("neighborhood", "") or "").strip()
    fields = [
        f"عنوان آگهی: {title}",
        f"توضیحات آگهی: {description}",
        f"محله: {neighborhood or 'نامشخص'}",
        f"متراژ ساختاریافته: {_fmt_num(_num(row, 'area_m2'), 0)} متر",
        f"اتاق ساختاریافته: {_fmt_num(_num(row, 'rooms'), 0)}",
        f"سال ساخت: {_fmt_num(_num(row, 'year_built_shamsi'), 0)}",
        f"ودیعه: {_fmt_num(_num(row, 'deposit_toman'), 0)} تومان",
        f"اجاره ماهانه: {_fmt_num(_num(row, 'rent_monthly_toman'), 0)} تومان",
        "",
        "شواهد محاسباتی مستقل (اینها حکم نهایی نیستند):",
        f"- data_quality_score: {_fmt_num(_num(row, 'data_quality_score'))}",
        f"- area_text_conflict: {_fmt_num(_num(row, 'area_text_conflict'))}",
        f"- rooms_text_conflict: {_fmt_num(_num(row, 'rooms_text_conflict'))}",
        f"- amenity_text_conflict: {_fmt_num(_num(row, 'amenity_text_conflict'))}",
        f"- market_anomaly_score: {_fmt_num(_num(row, 'market_anomaly_score'))}",
        f"- OOF price actual/expected ratio: {_fmt_num(_num(row, 'price_model_ratio'))}",
        f"- OOF price anomaly: {_fmt_num(_num(row, 'price_model_anomaly_score'))}",
        f"- LOF anomaly: {_fmt_num(_num(row, 'lof_anomaly_score'))}",
        f"- nearest duplicate similarity: {_fmt_num(_num(row, 'duplicate_similarity'))}",
        f"- duplicate cluster size: {_fmt_num(_num(row, 'duplicate_cluster_size'), 0)}",
        f"- duplicate cluster neighborhood count: {_fmt_num(_num(row, 'duplicate_cluster_neighborhoods'), 0)}",
        f"- duplicate cluster price inconsistency: {_fmt_num(_num(row, 'duplicate_cluster_price_span'))}",
        f"- duplicate bait score: {_fmt_num(_num(row, 'duplicate_bait_score'))}",
        "",
        "قاعدهٔ تصمیم: قیمت غیرعادی به‌تنهایی نشانهٔ فریب نیست. "
        "خطای داده/پارسینگ را از آگهی واقعاً نامعمول و از شواهد گمراه‌کنندگی جدا کن.",
    ]
    return "\n".join(fields)


def listing_questions() -> dict[str, dict[str, Any]]:
    return {
        "disposition": {
            "type": "choice",
            "instructions": "نوع اصلی وضعیت این آگهی چیست؟",
            "criteria": {
                "plausible": (
                    "متن و فیلدها عمدتاً سازگارند و شواهد معناداری برای مشکل وجود ندارد."
                ),
                "data_error": (
                    "تناقض‌ها بیشتر با خطای پارسینگ، دادهٔ ساختاریافتهٔ خراب یا اشتباه ثبت توضیح داده می‌شوند."
                ),
                "market_outlier": (
                    "آگهی از نظر داده منسجم است اما قیمت یا ترکیب ویژگی‌ها نسبت به بازار غیرعادی است؛ "
                    "بدون شواهد کافی از فریب."
                ),
                "misleading_or_bait": (
                    "فراتر از صرفاً قیمت پرت، تناقض یا الگوی تکراری معناداری وجود دارد که با "
                    "آگهی گمراه‌کننده/طعمه‌ای سازگار است."
                ),
                "needs_review": (
                    "شواهد چندپهلو یا متعارض است و برای نتیجهٔ قابل اتکا بازبینی انسانی لازم است."
                ),
            },
        },
        "consistency": {
            "type": "score",
            "instructions": "سازگاری داخلی متن آگهی با فیلدهای ساختاریافته را ارزیابی کن.",
            "criteria": [
                "تناقض‌های جدی و چندگانه بین متن و فیلدها",
                "چند ناسازگاری مهم",
                "ابهام یا ناسازگاری محدود",
                "عمدتاً سازگار",
                "کاملاً و به‌وضوح سازگار",
            ],
        },
        "bait_evidence": {
            "type": "noul",
            "instructions": (
                "آیا شواهدی فراتر از صرفاً قیمت غیرعادی وجود دارد که این آگهی "
                "گمراه‌کننده، طعمه‌ای یا بازنمایی نادرست ملک باشد؟"
            ),
            "criteria": {
                "no": "شواهد فعلی برای چنین نتیجه‌ای کافی نیست.",
                "yes": "تناقض‌های معنایی، تکرار مشکوک یا ادعاهای ناسازگار چنین احتمالی را تقویت می‌کند.",
            },
        },
        "data_error_evidence": {
            "type": "noul",
            "instructions": (
                "آیا مشکل اصلی این ردیف احتمالاً خطای داده، parser یا ناسازگاری فیلدهای استخراج‌شده است؟"
            ),
            "criteria": {
                "no": "دادهٔ ساختاریافته مشکل آشکار فنی/استخراجی ندارد.",
                "yes": "شواهد به خطای استخراج، mismatch فیلدها یا دادهٔ خراب اشاره می‌کند.",
            },
        },
        "manual_review": {
            "type": "noul",
            "instructions": "آیا این آگهی قبل از اعتماد باید توسط انسان بازبینی شود؟",
            "criteria": {
                "no": "شواهد فعلی برای بازبینی دستی فوری کافی نیست.",
                "yes": "ابهام، تناقض یا ریسک ترکیبی بازبینی انسانی را توجیه می‌کند.",
            },
        },
    }


def _empty_decision_columns(out: pd.DataFrame) -> pd.DataFrame:
    out = out.copy()
    out["decision_evaluated"] = False
    out["decision_backend"] = ""
    out["decision_disposition"] = ""
    out["decision_disposition_confidence"] = np.nan
    out["decision_disposition_probs_json"] = ""
    out["decision_bait_probability"] = np.nan
    out["decision_data_error_probability"] = np.nan
    out["decision_manual_review_probability"] = np.nan
    out["decision_consistency_score"] = np.nan
    out["decision_consistency_confidence"] = np.nan
    out["decision_raw_json"] = ""
    return out


def apply_decisions(
    df: pd.DataFrame,
    config: Config,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = _empty_decision_columns(df)
    d = config.section("decision")
    if not bool(d.get("enabled", True)) or out.empty:
        return out, {
            "enabled": False,
            "backend": None,
            "evaluated": 0,
            "reason": "decision layer disabled",
        }

    backend = str(os.getenv("DECISION_BACKEND") or d.get("backend", "mdeberta-nli")).lower()
    if backend not in {"mdeberta-nli", "nli", "mdeberta"}:
        raise ValueError(
            f"Unsupported non-generative decision backend {backend!r}. "
            "Use 'mdeberta-nli'."
        )

    min_prefilter = float(d.get("min_prefilter_score", 0.24))
    top_k = min(int(d.get("top_k", 60)), len(out))
    candidates = out[out["prefilter_score"] >= min_prefilter].nlargest(
        top_k, "prefilter_score"
    )
    if candidates.empty:
        return out, {
            "enabled": True,
            "backend": "mdeberta-nli",
            "model": str(d.get("model", DEFAULT_NLI_MODEL)),
            "evaluated": 0,
            "reason": "no rows passed decision prefilter",
        }

    engine = NLIDecisionEngine(
        model_name=str(d.get("model", DEFAULT_NLI_MODEL)),
        device=str(d.get("device", "auto")),
        max_length=int(d.get("max_length", 512)),
        batch_size=int(d.get("batch_size", 24)),
        temperatures=d.get("temperatures", {}),
    )

    requests = [
        (build_listing_state(row), listing_questions())
        for _, row in candidates.iterrows()
    ]

    strict = bool(d.get("strict", True))
    try:
        answers = engine.decide_many(requests)
    except Exception as exc:
        if strict:
            raise
        return out, {
            "enabled": True,
            "backend": engine.backend_name,
            "model": engine.model_name,
            "evaluated": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }

    for (idx, _row), answer_map in zip(candidates.iterrows(), answers):
        disp = answer_map["disposition"]
        consistency = answer_map["consistency"]
        bait = answer_map["bait_evidence"]
        data_error = answer_map["data_error_evidence"]
        review = answer_map["manual_review"]

        out.at[idx, "decision_evaluated"] = True
        out.at[idx, "decision_backend"] = engine.backend_name
        out.at[idx, "decision_disposition"] = disp["choice"]
        out.at[idx, "decision_disposition_confidence"] = disp["confidence"]
        out.at[idx, "decision_disposition_probs_json"] = json.dumps(
            disp["probabilities"], ensure_ascii=False
        )
        out.at[idx, "decision_bait_probability"] = bait["noul"]
        out.at[idx, "decision_data_error_probability"] = data_error["noul"]
        out.at[idx, "decision_manual_review_probability"] = review["noul"]
        out.at[idx, "decision_consistency_score"] = consistency["normalized_score"]
        out.at[idx, "decision_consistency_confidence"] = consistency["confidence"]
        out.at[idx, "decision_raw_json"] = json.dumps(answer_map, ensure_ascii=False)

    return out, {
        "enabled": True,
        "backend": engine.backend_name,
        "model": engine.model_name,
        "evaluated": int(out["decision_evaluated"].sum()),
        "candidate_count": int(len(candidates)),
        "probability_note": (
            "These are NLI-derived softmax probabilities, not TypeSafe Jev's proprietary RLCD-calibrated probabilities. "
            "Tune temperatures on labelled Persian data before treating thresholds as calibrated."
        ),
    }
