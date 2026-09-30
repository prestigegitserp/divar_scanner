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


@dataclass(frozen=True)
class NLIModelSpec:
    backend: str
    model_name: str
    license: str
    note: str
    forced_label_indices: tuple[int, int, int] | None = None  # entail, neutral, contradiction


NLI_MODEL_REGISTRY: dict[str, NLIModelSpec] = {
    "mdeberta-nli": NLIModelSpec(
        backend="mdeberta-nli",
        model_name=DEFAULT_NLI_MODEL,
        license="MIT",
        note="Multilingual encoder NLI; permissive default.",
    ),
    "parsbert-parsinlu": NLIModelSpec(
        backend="parsbert-parsinlu",
        model_name="persiannlp/parsbert-base-parsinlu-entailment",
        license="CC-BY-NC-SA-4.0",
        note="Persian-specialist ParsiNLU NLI classifier; non-commercial/share-alike license.",
        forced_label_indices=(0, 2, 1),
    ),
    "mbert-parsinlu": NLIModelSpec(
        backend="mbert-parsinlu",
        model_name="persiannlp/mbert-base-parsinlu-entailment",
        license="CC-BY-NC-SA-4.0",
        note="Multilingual BERT fine-tuned on Persian ParsiNLU entailment.",
        forced_label_indices=(0, 2, 1),
    ),
}


def resolve_model_specs(
    backend: str,
    decision_config: dict[str, Any],
) -> list[NLIModelSpec]:
    backend = backend.lower().strip()
    if backend in {"nli", "mdeberta"}:
        backend = "mdeberta-nli"

    if backend == "persian-ensemble":
        names = decision_config.get(
            "ensemble_backends",
            ["mdeberta-nli", "parsbert-parsinlu"],
        )
        if not isinstance(names, list) or not names:
            raise ValueError("decision.ensemble_backends must be a non-empty list")
        specs = []
        for name in names:
            key = str(name).lower().strip()
            if key not in NLI_MODEL_REGISTRY:
                raise ValueError(f"Unknown ensemble NLI backend: {key!r}")
            specs.append(NLI_MODEL_REGISTRY[key])
        return specs

    if backend not in NLI_MODEL_REGISTRY:
        raise ValueError(
            f"Unsupported non-generative decision backend {backend!r}. "
            f"Choose from {sorted(NLI_MODEL_REGISTRY)} or 'persian-ensemble'."
        )

    spec = NLI_MODEL_REGISTRY[backend]
    # Allow an explicit model override only for a single backend.
    configured_model = decision_config.get("model")
    if configured_model and backend == str(decision_config.get("backend", backend)).lower():
        spec = NLIModelSpec(
            backend=spec.backend,
            model_name=str(configured_model),
            license=spec.license,
            note=spec.note,
            forced_label_indices=spec.forced_label_indices,
        )
    return [spec]


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
        if "evidence_log_odds" in row:
            evidence.append(float(row["evidence_log_odds"]))
        else:
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
    disagreement = float(
        np.sum(
            probs
            * np.asarray(
                [float(r.get("model_disagreement", 0.0)) for r in candidate_rows],
                dtype=float,
            )
        )
    )
    disagreement_penalty = float(np.exp(-0.60 * max(disagreement, 0.0)))
    confidence = float(
        np.clip(
            concentration
            * (1.0 - 0.65 * neutral_mass)
            * disagreement_penalty,
            0.0,
            1.0,
        )
    )

    raw_nli = {
        str(r["key"]): {
            "entailment": float(r["entailment"]),
            "neutral": float(r["neutral"]),
            "contradiction": float(r["contradiction"]),
            "evidence_log_odds": float(e),
            "model_disagreement": float(r.get("model_disagreement", 0.0)),
            "per_model": r.get("per_model", {}),
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
                "model_disagreement": disagreement,
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
                "model_disagreement": disagreement,
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
                "model_disagreement": disagreement,
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
        backend_name: str = "mdeberta-nli",
        forced_label_indices: tuple[int, int, int] | None = None,
        device: str = "auto",
        max_length: int = 512,
        batch_size: int = 24,
        temperatures: dict[str, float] | None = None,
    ):
        self.model_name = model_name
        self._backend_name = backend_name
        self.forced_label_indices = forced_label_indices
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
        return self._backend_name

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

        if self.forced_label_indices is not None:
            entail, neutral, contradiction = self.forced_label_indices
        else:
            id2label = {
                int(k): str(v).lower()
                for k, v in getattr(model.config, "id2label", {}).items()
            }
            entail = next((i for i, v in id2label.items() if "entail" in v), None)
            neutral = next((i for i, v in id2label.items() if "neutral" in v), None)
            contradiction = next((i for i, v in id2label.items() if "contrad" in v), None)
            if None in (entail, neutral, contradiction):
                # mDeBERTa's documented ordering. Other checkpoints with opaque LABEL_* IDs
                # must provide forced_label_indices through the model registry.
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



class EnsembleNLIDecisionEngine:
    """Fuse several non-generative NLI encoders before the bounded option softmax."""

    def __init__(
        self,
        specs: list[NLIModelSpec],
        *,
        device: str = "auto",
        max_length: int = 512,
        batch_size: int = 24,
        temperatures: dict[str, float] | None = None,
        model_weights: dict[str, float] | None = None,
    ):
        if len(specs) < 2:
            raise ValueError("EnsembleNLIDecisionEngine requires at least two models")
        self.specs = specs
        self.temperatures = dict(temperatures or {})
        configured_weights = dict(model_weights or {})
        raw_weights = np.asarray(
            [float(configured_weights.get(spec.backend, 1.0)) for spec in specs],
            dtype=float,
        )
        raw_weights = np.clip(raw_weights, 0.0, None)
        if raw_weights.sum() <= 0:
            raw_weights[:] = 1.0
        self.weights = raw_weights / raw_weights.sum()
        self.engines = [
            NLIDecisionEngine(
                model_name=spec.model_name,
                backend_name=spec.backend,
                forced_label_indices=spec.forced_label_indices,
                device=device,
                max_length=max_length,
                batch_size=batch_size,
                temperatures=self.temperatures,
            )
            for spec in specs
        ]

    @property
    def backend_name(self) -> str:
        return "persian-ensemble"

    @property
    def model_name(self) -> str:
        return " + ".join(spec.model_name for spec in self.specs)

    def decide_many(
        self,
        requests: list[tuple[str, dict[str, dict[str, Any]]]],
    ) -> list[dict[str, dict[str, Any]]]:
        if not requests:
            return []

        compiled, question_lookup = compile_requests(requests)
        per_model_scores = [engine._score_pairs(compiled) for engine in self.engines]
        grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}

        for pair_index, item in enumerate(compiled):
            model_rows = [scores[pair_index] for scores in per_model_scores]
            log_odds = np.asarray(
                [
                    math.log(max(r["entailment"], 1e-9))
                    - math.log(max(r["contradiction"], 1e-9))
                    for r in model_rows
                ],
                dtype=float,
            )
            fused_log_odds = float(np.dot(self.weights, log_odds))
            neutral = float(
                np.dot(
                    self.weights,
                    np.asarray([r["neutral"] for r in model_rows], dtype=float),
                )
            )
            # Preserve the fused entail-vs-contradict ratio while allocating the
            # remaining mass after neutral.
            ent_share = 1.0 / (1.0 + math.exp(-np.clip(fused_log_odds, -30, 30)))
            entailment = (1.0 - neutral) * ent_share
            contradiction = (1.0 - neutral) * (1.0 - ent_share)
            disagreement = float(np.sqrt(np.dot(self.weights, (log_odds - fused_log_odds) ** 2)))
            per_model = {
                spec.backend: {
                    "entailment": float(row["entailment"]),
                    "neutral": float(row["neutral"]),
                    "contradiction": float(row["contradiction"]),
                    "evidence_log_odds": float(lo),
                }
                for spec, row, lo in zip(self.specs, model_rows, log_odds)
            }
            grouped.setdefault((item.request_index, item.question_id), []).append(
                {
                    "key": item.key,
                    "entailment": entailment,
                    "neutral": neutral,
                    "contradiction": contradiction,
                    "evidence_log_odds": fused_log_odds,
                    "model_disagreement": disagreement,
                    "per_model": per_model,
                }
            )

        outputs: list[dict[str, dict[str, Any]]] = [dict() for _ in requests]
        for route, rows in grouped.items():
            request_index, qid = route
            question = question_lookup[route]
            qtype = str(question.get("type", "")).lower()
            outputs[request_index][qid] = typed_answer_from_evidence(
                question,
                rows,
                temperature=float(self.temperatures.get(qtype, 1.0)),
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

    backend = str(
        os.getenv("DECISION_BACKEND") or d.get("backend", "mdeberta-nli")
    ).lower()
    specs = resolve_model_specs(backend, d)

    min_prefilter = float(d.get("min_prefilter_score", 0.24))
    top_k = min(int(d.get("top_k", 60)), len(out))
    candidates = out[out["prefilter_score"] >= min_prefilter].nlargest(
        top_k, "prefilter_score"
    )
    if candidates.empty:
        return out, {
            "enabled": True,
            "backend": backend,
            "model": " + ".join(spec.model_name for spec in specs),
            "evaluated": 0,
            "reason": "no rows passed decision prefilter",
        }

    common_engine_args = {
        "device": str(d.get("device", "auto")),
        "max_length": int(d.get("max_length", 512)),
        "batch_size": int(d.get("batch_size", 24)),
        "temperatures": d.get("temperatures", {}),
    }
    if len(specs) == 1:
        spec = specs[0]
        engine = NLIDecisionEngine(
            model_name=spec.model_name,
            backend_name=spec.backend,
            forced_label_indices=spec.forced_label_indices,
            **common_engine_args,
        )
    else:
        engine = EnsembleNLIDecisionEngine(
            specs,
            model_weights=d.get("ensemble_weights", {}),
            **common_engine_args,
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
        "models": [
            {
                "backend": spec.backend,
                "model": spec.model_name,
                "license": spec.license,
                "note": spec.note,
            }
            for spec in specs
        ],
        "evaluated": int(out["decision_evaluated"].sum()),
        "candidate_count": int(len(candidates)),
        "probability_note": (
            "These are NLI-derived bounded probabilities, not TypeSafe Jev's proprietary RLCD-calibrated probabilities. "
            "For ensembles, model evidence log-odds are fused before the option softmax. "
            "Tune temperatures on labelled Persian data before treating thresholds as calibrated."
        ),
    }
