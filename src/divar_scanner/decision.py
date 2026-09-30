from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
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
    engine: str = "nli"
    subfolder: str | None = None


NLI_MODEL_REGISTRY: dict[str, NLIModelSpec] = {
    "laya-multilingual": NLIModelSpec(
        backend="laya-multilingual",
        model_name="convaiinnovations/laya",
        license="Apache-2.0",
        note=(
            "Native non-autoregressive System-1 model: mmBERT-base encoder + "
            "Laya typed-decision head trained with RLCD."
        ),
        engine="laya",
        subfolder="multilingual",
    ),
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
    configured_model = decision_config.get("model_override")
    if configured_model and backend == str(decision_config.get("backend", backend)).lower():
        spec = NLIModelSpec(
            backend=spec.backend,
            model_name=str(configured_model),
            license=spec.license,
            note=spec.note,
            forced_label_indices=spec.forced_label_indices,
            engine=spec.engine,
            subfolder=spec.subfolder,
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
            f"برای پرسش «{instructions}»، این توصیف پاسخ درست است: {description}"
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
        # Keep float32 for maximum NLI checkpoint compatibility. In particular,
        # the upstream mDeBERTa model card warns about FP16 support; the base model
        # is small enough that float32 remains Colab-friendly.
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
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
            temperature = float(
                self.temperatures.get(qid, self.temperatures.get(qtype, 1.0))
            )
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
                temperature=float(
                    self.temperatures.get(qid, self.temperatures.get(qtype, 1.0))
                ),
            )
        return outputs

    def decide(
        self,
        state: str,
        questions: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        return self.decide_many([(state, questions)])[0]



def load_decision_temperatures(decision_config: dict[str, Any]) -> dict[str, float]:
    temperatures = {
        str(k): float(v)
        for k, v in dict(decision_config.get("temperatures", {}) or {}).items()
    }
    calibration_file = decision_config.get("calibration_file")
    if calibration_file:
        path = Path(str(calibration_file))
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            calibrated = payload.get("temperatures", {})
            if isinstance(calibrated, dict):
                temperatures.update(
                    {
                        str(k): float(v)
                        for k, v in calibrated.items()
                        if isinstance(v, (int, float))
                    }
                )
    return temperatures


def create_decision_engine(
    decision_config: dict[str, Any],
    *,
    backend_override: str | None = None,
):
    backend = str(
        backend_override
        or os.getenv("DECISION_BACKEND")
        or decision_config.get("backend", "laya-multilingual")
    ).lower()
    specs = resolve_model_specs(backend, decision_config)
    temperatures = load_decision_temperatures(decision_config)

    if len(specs) == 1 and specs[0].engine == "laya":
        from .laya_backend import LayaDecisionEngine

        spec = specs[0]
        engine = LayaDecisionEngine(
            model_id=str(decision_config.get("laya_model_id", spec.model_name)),
            subfolder=str(decision_config.get("laya_subfolder", spec.subfolder or "multilingual")),
            device=str(decision_config.get("device", "auto")),
            max_length=int(decision_config.get("max_length", 1024)),
            batch_size=int(decision_config.get("batch_size", 24)),
            temperatures=temperatures,
            permutation_passes=int(decision_config.get("permutation_passes", 2)),
        )
        return engine, specs, temperatures

    common_engine_args = {
        "device": str(decision_config.get("device", "auto")),
        "max_length": int(decision_config.get("max_length", 512)),
        "batch_size": int(decision_config.get("batch_size", 24)),
        "temperatures": temperatures,
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
        if any(spec.engine != "nli" for spec in specs):
            raise ValueError("Only NLI backends can currently be used in persian-ensemble")
        engine = EnsembleNLIDecisionEngine(
            specs,
            model_weights=decision_config.get("ensemble_weights", {}),
            **common_engine_args,
        )
    return engine, specs, temperatures


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


def build_listing_state(row: pd.Series, *, view: str = "full") -> str:
    description = str(row.get("description", "") or "").strip()
    if len(description) > 1800:
        description = description[:1800] + "…"

    title = str(row.get("title", "") or "").strip()
    neighborhood = str(row.get("neighborhood", "") or "").strip()

    structured_identity = [
        f"محله: {neighborhood or 'نامشخص'}",
        f"متراژ ساختاریافته: {_fmt_num(_num(row, 'area_m2'), 0)} متر",
        f"اتاق ساختاریافته: {_fmt_num(_num(row, 'rooms'), 0)}",
        f"سال ساخت: {_fmt_num(_num(row, 'year_built_shamsi'), 0)}",
        f"پارکینگ ساختاریافته: {row.get('parking', 'ناموجود')}",
        f"آسانسور ساختاریافته: {row.get('elevator', 'ناموجود')}",
        f"انباری ساختاریافته: {row.get('storage', 'ناموجود')}",
    ]
    text_identity = [
        f"عنوان آگهی: {title}",
        f"توضیحات آگهی: {description}",
        *structured_identity,
    ]
    consistency = [
        "شواهد سازگاری/کیفیت داده:",
        f"- data_quality_score: {_fmt_num(_num(row, 'data_quality_score'))}",
        f"- area_text_conflict: {_fmt_num(_num(row, 'area_text_conflict'))}",
        f"- rooms_text_conflict: {_fmt_num(_num(row, 'rooms_text_conflict'))}",
        f"- amenity_text_conflict: {_fmt_num(_num(row, 'amenity_text_conflict'))}",
        f"- missing_fraction: {_fmt_num(_num(row, 'missing_fraction'))}",
    ]
    duplicate = [
        "شواهد تکرار/بازنمایی:",
        f"- nearest duplicate similarity: {_fmt_num(_num(row, 'duplicate_similarity'))}",
        f"- duplicate cluster size: {_fmt_num(_num(row, 'duplicate_cluster_size'), 0)}",
        f"- cluster neighborhood count: {_fmt_num(_num(row, 'duplicate_cluster_neighborhoods'), 0)}",
        f"- cluster price inconsistency: {_fmt_num(_num(row, 'duplicate_cluster_price_span'))}",
        f"- duplicate bait score: {_fmt_num(_num(row, 'duplicate_bait_score'))}",
    ]
    market = [
        "شواهد بازار؛ فقط برای تشخیص outlier و نه استنباط قصد فریب:",
        f"- سبک قرارداد: {row.get('contract_style', 'unknown')}",
        f"- ودیعه: {_fmt_num(_num(row, 'deposit_toman'), 0)} تومان",
        f"- اجاره ماهانه: {_fmt_num(_num(row, 'rent_monthly_toman'), 0)} تومان",
        f"- market_anomaly_score: {_fmt_num(_num(row, 'market_anomaly_score'))}",
        f"- robust_market_anomaly: {_fmt_num(_num(row, 'robust_market_anomaly'))}",
        f"- OOF price actual/expected ratio: {_fmt_num(_num(row, 'price_model_ratio'))}",
        f"- OOF price anomaly: {_fmt_num(_num(row, 'price_model_anomaly_score'))}",
        f"- LOF anomaly: {_fmt_num(_num(row, 'lof_anomaly_score'))}",
        f"- equivalence sensitivity: {_fmt_num(_num(row, 'equivalence_sensitivity'))}",
    ]

    if view == "content":
        sections = text_identity + [""] + consistency
    elif view in {"bait", "duplicate"}:
        # Evidence firewall: do not expose single-listing market anomaly to deception decisions.
        sections = text_identity + [""] + consistency + [""] + duplicate
    elif view == "market":
        # Another firewall: market-position judgment sees structured facts/statistics, not persuasive copy.
        sections = structured_identity + [""] + market
    elif view == "full":
        sections = text_identity + [""] + consistency + [""] + duplicate + [""] + market
    else:
        raise ValueError(f"Unknown listing-state view: {view!r}")

    sections += [
        "",
        "قاعدهٔ تصمیم: outlier بودن، ارزان بودن یا گران بودن به‌تنهایی نشانهٔ تقلب نیست. "
        "خطای استخراج، ناسازگاری ادعاها، الگوی بازنشر و وضعیت بازار را مستقل ارزیابی کن.",
    ]
    return "\n".join(str(x) for x in sections)


def listing_questions() -> dict[str, dict[str, Any]]:
    return {
        "disposition": {
            "type": "choice",
            "instructions": "با جمع‌بندی همهٔ شواهد، وضعیت اصلی این آگهی چیست؟",
            "criteria": {
                "plausible": "شواهد غالب با یک آگهی عادی و منسجم سازگار است.",
                "data_error": "شواهد غالب بیشتر خطای داده، استخراج یا mismatch فیلدها را نشان می‌دهد.",
                "market_outlier": "آگهی منسجم است اما نسبت به بازار/ویژگی‌های مشابه outlier محسوسی دارد.",
                "misleading_or_bait": (
                    "شواهدی مستقل از صرف قیمت پرت برای بازنمایی نادرست، bait یا گمراه‌کنندگی وجود دارد."
                ),
                "ambiguous_mixed": "چند توضیح رقیب باقی مانده و تصمیم قابل اتکایی بین آن‌ها نیست.",
            },
        },
        "integrity_class": {
            "type": "choice",
            "instructions": "مشکل سازگاری متن و فیلدهای ساختاریافته را چگونه طبقه‌بندی می‌کنی؟",
            "criteria": {
                "consistent": "متن و فیلدهای اصلی با هم سازگارند.",
                "extraction_error": "اختلاف‌ها بیشتر شبیه خطای parser/crawler یا extraction هستند.",
                "listing_claim_conflict": "خود ادعاهای آگهی با فیلدهای ساختاریافته تضاد معنادار دارند.",
                "insufficient_evidence": "اطلاعات برای تعیین نوع ناسازگاری کافی نیست.",
            },
        },
        "consistency": {
            "type": "score",
            "instructions": "سازگاری داخلی متن آگهی با فیلدهای ساختاریافته را ارزیابی کن.",
            "criteria": [
                "تناقض‌های جدی و چندگانه",
                "چند ناسازگاری مهم",
                "ابهام یا ناسازگاری محدود",
                "عمدتاً سازگار",
                "کاملاً و به‌وضوح سازگار",
            ],
        },
        "duplicate_pattern": {
            "type": "choice",
            "instructions": "الگوی تکرار این آگهی با نزدیک‌ترین نمونه‌ها چگونه تفسیر می‌شود؟",
            "criteria": {
                "no_duplicate_evidence": "شباهت کافی برای نتیجه‌گیری دربارهٔ بازنشر وجود ندارد.",
                "normal_template_reuse": "شباهت بیشتر با متن قالبی/رایج سازگار است و تضاد مهمی دیده نمی‌شود.",
                "likely_same_property_repost": "احتمالاً همان ملک یا همان آگهی با تغییرات محدود دوباره منتشر شده است.",
                "cross_property_conflict": (
                    "متن بسیار مشابه همراه با اختلاف مهم در محله/قیمت/مشخصات دیده می‌شود."
                ),
            },
        },
        "market_status": {
            "type": "choice",
            "instructions": "با توجه به شواهد آماری ارائه‌شده، جایگاه بازار این ملک چیست؟",
            "criteria": {
                "typical": "با peerهای محلی و مدل قیمت سازگار یا نزدیک به محدودهٔ معمول است.",
                "moderate_outlier": "انحراف قابل توجه اما نه بسیار شدید از peerها/مدل قیمت دیده می‌شود.",
                "extreme_outlier": "چند سیگنال مستقل بازار انحراف شدید را تأیید می‌کنند.",
                "insufficient_context": "peer/context کافی برای قضاوت قابل اتکا وجود ندارد.",
            },
        },
        "bait_evidence": {
            "type": "noul",
            "instructions": (
                "آیا شواهدی مستقل از صرفاً قیمت غیرعادی وجود دارد که آگهی "
                "گمراه‌کننده، bait یا بازنمایی نادرست ملک باشد؟"
            ),
            "criteria": {
                "no": "شواهد فعلی برای چنین نتیجه‌ای کافی نیست.",
                "yes": "تناقض ادعاها یا الگوی بازنشر متناقض چنین احتمالی را تقویت می‌کند.",
            },
        },
        "data_error_evidence": {
            "type": "noul",
            "instructions": "آیا شواهد فعلی به خطای داده، parser یا extraction اشاره می‌کند؟",
            "criteria": {
                "no": "شواهد کافی برای خطای فنی/استخراجی وجود ندارد.",
                "yes": "mismatchها با خطای استخراج یا دادهٔ خراب سازگارند.",
            },
        },
        "manual_review": {
            "type": "noul",
            "instructions": "آیا این آگهی قبل از اعتماد/استفاده باید توسط انسان بازبینی شود؟",
            "criteria": {
                "no": "شواهد و عدم‌قطعیت فعلی بازبینی فوری را توجیه نمی‌کند.",
                "yes": "شدت شواهد یا عدم‌قطعیت بازبینی انسانی را توجیه می‌کند.",
            },
        },
    }


def _empty_decision_columns(out: pd.DataFrame) -> pd.DataFrame:
    out = out.copy()
    defaults: dict[str, Any] = {
        "decision_evaluated": False,
        "decision_sampling_reason": "",
        "decision_backend": "",
        "decision_disposition": "",
        "decision_disposition_confidence": np.nan,
        "decision_answer_confidence": np.nan,
        "decision_disposition_probs_json": "",
        "decision_integrity_class": "",
        "decision_integrity_probs_json": "",
        "decision_duplicate_pattern": "",
        "decision_duplicate_pattern_probs_json": "",
        "decision_market_status": "",
        "decision_market_status_probs_json": "",
        "decision_bait_probability": np.nan,
        "decision_data_error_probability": np.nan,
        "decision_manual_review_probability": np.nan,
        "decision_consistency_score": np.nan,
        "decision_consistency_confidence": np.nan,
        "decision_order_stability": np.nan,
        "decision_coherence_score": np.nan,
        "decision_effective_confidence": np.nan,
        "decision_abstain": False,
        "decision_abstain_reasons": "",
        "decision_raw_json": "",
    }
    for column, value in defaults.items():
        out[column] = value
    return out


def _stratified_exploration(
    frame: pd.DataFrame,
    n: int,
    *,
    score_col: str,
    seed: int,
    bins: int = 5,
) -> pd.DataFrame:
    if n <= 0 or frame.empty:
        return frame.iloc[0:0]
    n = min(int(n), len(frame))
    ordered = frame.sort_values(score_col)
    groups = [g for g in np.array_split(ordered.index.to_numpy(), min(bins, len(ordered))) if len(g)]
    rng = np.random.default_rng(seed)
    selected: list[Any] = []
    while len(selected) < n:
        changed = False
        for group in groups:
            remaining = [idx for idx in group.tolist() if idx not in selected]
            if remaining and len(selected) < n:
                selected.append(remaining[int(rng.integers(0, len(remaining)))])
                changed = True
        if not changed:
            break
    return frame.loc[selected]


def _answer_stability(answer: dict[str, Any]) -> float:
    try:
        return float(answer.get("diagnostics", {}).get("order_stability", 1.0))
    except Exception:
        return 1.0


def _answer_effective_confidence(answer: dict[str, Any]) -> float:
    for key in ("effective_confidence", "answer_confidence", "confidence"):
        try:
            value = answer.get(key)
            if value is not None:
                return float(np.clip(float(value), 0.0, 1.0))
        except Exception:
            pass
    return 0.0


def _decision_quality(answer_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    disp = answer_map["disposition"]
    integrity = answer_map["integrity_class"]
    duplicate = answer_map["duplicate_pattern"]
    market = answer_map["market_status"]
    consistency = answer_map["consistency"]
    bait = answer_map["bait_evidence"]
    data_error = answer_map["data_error_evidence"]

    disp_p = disp["probabilities"]
    int_p = integrity["probabilities"]
    dup_p = duplicate["probabilities"]
    market_p = market["probabilities"]

    p_data = float(data_error["noul"])
    p_bait = float(bait["noul"])
    p_integrity_problem = float(
        int_p.get("extraction_error", 0.0) + int_p.get("listing_claim_conflict", 0.0)
    )
    p_dup_conflict = float(dup_p.get("cross_property_conflict", 0.0))
    p_market = float(
        market_p.get("moderate_outlier", 0.0) + market_p.get("extreme_outlier", 0.0)
    )
    consistency_bad = float(1.0 - consistency["normalized_score"])

    agreements = [
        1.0 - abs(float(disp_p.get("data_error", 0.0)) - 0.5 * (p_data + p_integrity_problem)),
        1.0 - abs(float(disp_p.get("misleading_or_bait", 0.0)) - max(p_bait, p_dup_conflict)),
        1.0 - abs(float(disp_p.get("market_outlier", 0.0)) - p_market),
        1.0 - abs(p_data - 0.5 * (consistency_bad + p_integrity_problem)),
    ]
    coherence = float(np.clip(np.mean(agreements), 0.0, 1.0))

    answers = list(answer_map.values())
    stability = float(np.mean([_answer_stability(a) for a in answers]))
    model_confidence = float(np.mean([_answer_effective_confidence(a) for a in answers]))
    effective = float(np.clip(0.50 * model_confidence + 0.25 * stability + 0.25 * coherence, 0, 1))
    return {
        "coherence": coherence,
        "order_stability": stability,
        "model_confidence": model_confidence,
        "effective_confidence": effective,
    }


def apply_decisions(
    df: pd.DataFrame,
    config: Config,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = _empty_decision_columns(df)
    d = config.section("decision")
    calibration_file = d.get("calibration_file")
    if calibration_file:
        requested = Path(str(calibration_file))
        if not requested.is_absolute():
            candidates_for_path = [config.source.parent / requested, Path.cwd() / requested]
            resolved = next((p for p in candidates_for_path if p.exists()), requested)
            d["calibration_file"] = str(resolved)

    if not bool(d.get("enabled", True)) or out.empty:
        return out, {
            "enabled": False,
            "backend": None,
            "evaluated": 0,
            "reason": "decision layer disabled",
        }

    backend = str(os.getenv("DECISION_BACKEND") or d.get("backend", "laya-multilingual")).lower()
    specs = resolve_model_specs(backend, d)

    min_prefilter = float(d.get("min_prefilter_score", 0.24))
    top_k = min(int(d.get("top_k", 60)), len(out))
    priority_candidates = out[out["prefilter_score"] >= min_prefilter].nlargest(
        top_k, "prefilter_score"
    )

    remaining = out.drop(index=priority_candidates.index, errors="ignore")
    exploration = _stratified_exploration(
        remaining,
        max(0, int(d.get("exploration_sample", 20))),
        score_col="prefilter_score",
        seed=int(config.get("project.random_seed", 42)),
        bins=int(d.get("exploration_bins", 5)),
    )

    candidates = pd.concat([priority_candidates, exploration], axis=0)
    candidates = candidates.loc[~candidates.index.duplicated(keep="first")]
    if candidates.empty:
        return out, {
            "enabled": True,
            "backend": backend,
            "model": " + ".join(spec.model_name for spec in specs),
            "evaluated": 0,
            "reason": "no rows selected for bounded decisions",
        }

    out.loc[priority_candidates.index, "decision_sampling_reason"] = "priority"
    out.loc[exploration.index, "decision_sampling_reason"] = "stratified_exploration"

    engine, specs, effective_temperatures = create_decision_engine(d, backend_override=backend)
    all_questions = listing_questions()

    requests: list[tuple[str, dict[str, dict[str, Any]]]] = []
    for _, row in candidates.iterrows():
        requests.extend(
            [
                (
                    build_listing_state(row, view="full"),
                    {
                        "disposition": all_questions["disposition"],
                        "manual_review": all_questions["manual_review"],
                    },
                ),
                (
                    build_listing_state(row, view="content"),
                    {
                        "integrity_class": all_questions["integrity_class"],
                        "consistency": all_questions["consistency"],
                        "data_error_evidence": all_questions["data_error_evidence"],
                    },
                ),
                (
                    build_listing_state(row, view="bait"),
                    {
                        "duplicate_pattern": all_questions["duplicate_pattern"],
                        "bait_evidence": all_questions["bait_evidence"],
                    },
                ),
                (
                    build_listing_state(row, view="market"),
                    {"market_status": all_questions["market_status"]},
                ),
            ]
        )

    strict = bool(d.get("strict", True))
    try:
        flat_answers = engine.decide_many(requests)
        answers: list[dict[str, dict[str, Any]]] = []
        pack_count = 4
        for i in range(len(candidates)):
            merged: dict[str, dict[str, Any]] = {}
            for part in flat_answers[i * pack_count : i * pack_count + pack_count]:
                merged.update(part)
            answers.append(merged)
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

    min_conf = float(d.get("min_effective_confidence", 0.58))
    min_coherence = float(d.get("min_coherence", 0.55))
    min_stability = float(d.get("min_order_stability", 0.80))

    for (idx, _row), answer_map in zip(candidates.iterrows(), answers):
        disp = answer_map["disposition"]
        integrity = answer_map["integrity_class"]
        consistency = answer_map["consistency"]
        duplicate = answer_map["duplicate_pattern"]
        bait = answer_map["bait_evidence"]
        data_error = answer_map["data_error_evidence"]
        market = answer_map["market_status"]
        review = answer_map["manual_review"]
        quality = _decision_quality(answer_map)

        abstain_reasons: list[str] = []
        if quality["effective_confidence"] < min_conf:
            abstain_reasons.append("low_effective_confidence")
        if quality["coherence"] < min_coherence:
            abstain_reasons.append("cross_question_incoherence")
        if quality["order_stability"] < min_stability:
            abstain_reasons.append("option_order_instability")

        out.at[idx, "decision_evaluated"] = True
        out.at[idx, "decision_backend"] = engine.backend_name
        out.at[idx, "decision_disposition"] = disp["choice"]
        out.at[idx, "decision_disposition_confidence"] = disp["confidence"]
        out.at[idx, "decision_answer_confidence"] = disp.get(
            "answer_confidence", max(disp["probabilities"].values())
        )
        out.at[idx, "decision_disposition_probs_json"] = json.dumps(
            disp["probabilities"], ensure_ascii=False
        )

        out.at[idx, "decision_integrity_class"] = integrity["choice"]
        out.at[idx, "decision_integrity_probs_json"] = json.dumps(
            integrity["probabilities"], ensure_ascii=False
        )
        out.at[idx, "decision_duplicate_pattern"] = duplicate["choice"]
        out.at[idx, "decision_duplicate_pattern_probs_json"] = json.dumps(
            duplicate["probabilities"], ensure_ascii=False
        )
        out.at[idx, "decision_market_status"] = market["choice"]
        out.at[idx, "decision_market_status_probs_json"] = json.dumps(
            market["probabilities"], ensure_ascii=False
        )

        out.at[idx, "decision_bait_probability"] = bait["noul"]
        out.at[idx, "decision_data_error_probability"] = data_error["noul"]
        out.at[idx, "decision_manual_review_probability"] = review["noul"]
        out.at[idx, "decision_consistency_score"] = consistency["normalized_score"]
        out.at[idx, "decision_consistency_confidence"] = consistency["confidence"]
        out.at[idx, "decision_order_stability"] = quality["order_stability"]
        out.at[idx, "decision_coherence_score"] = quality["coherence"]
        out.at[idx, "decision_effective_confidence"] = quality["effective_confidence"]
        out.at[idx, "decision_abstain"] = bool(abstain_reasons)
        out.at[idx, "decision_abstain_reasons"] = ",".join(abstain_reasons)
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
                "engine": spec.engine,
                "subfolder": spec.subfolder,
            }
            for spec in specs
        ],
        "evaluated": int(out["decision_evaluated"].sum()),
        "candidate_count": int(len(candidates)),
        "priority_candidate_count": int(len(priority_candidates)),
        "exploration_candidate_count": int(len(exploration)),
        "sampling": "priority + stratified exploration across prefilter score",
        "permutation_passes": int(d.get("permutation_passes", 2)) if backend == "laya-multilingual" else 1,
        "temperatures": effective_temperatures,
        "calibration_file": d.get("calibration_file"),
        "abstention_thresholds": {
            "effective_confidence": min_conf,
            "coherence": min_coherence,
            "order_stability": min_stability,
        },
        "probability_note": (
            "Laya is the native non-autoregressive System-One engine. All logical primitives "
            "are transported through opaque bounded choices, pooled across deterministic option "
            "permutations, then optionally target-domain temperature calibrated. Low confidence "
            "or cross-question disagreement causes abstention/review, not a fraud assertion."
            if engine.backend_name == "laya-multilingual"
            else
            "NLI backends are retained only as non-generative research baselines."
        ),
    }
