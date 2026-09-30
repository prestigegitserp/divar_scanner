from __future__ import annotations

import hashlib
import json
import math
import os
from collections import defaultdict
from typing import Any

import numpy as np


def _entropy_confidence(probs: dict[str, float]) -> float:
    p = np.asarray(list(probs.values()), dtype=float)
    if len(p) <= 1:
        return 1.0
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    h = -float(np.sum(p * np.log(p)))
    return float(np.clip(1.0 - h / math.log(len(p)), 0.0, 1.0))


def _temperature_scale(
    probs: dict[str, float],
    temperature: float,
) -> tuple[dict[str, float], dict[str, float]]:
    keys = list(probs)
    raw = np.asarray([float(probs[k]) for k in keys], dtype=float)
    raw = np.clip(raw, 1e-12, 1.0)
    raw = raw / raw.sum()
    logits = np.log(raw)
    t = max(float(temperature), 1e-4)
    z = logits / t
    z -= np.max(z)
    exp = np.exp(z)
    calibrated = exp / exp.sum()
    return (
        {k: float(v) for k, v in zip(keys, calibrated)},
        {k: float(v) for k, v in zip(keys, logits)},
    )


def _opaque_labels(n: int) -> list[str]:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if n <= len(alphabet):
        return list(alphabet[:n])
    return [f"O{i}" for i in range(n)]


def _stable_permutation(keys: list[str], qid: str, pass_index: int) -> list[str]:
    """Deterministic option permutation for position-bias averaging."""
    if pass_index <= 0 or len(keys) <= 1:
        return list(keys)
    if pass_index == 1:
        return list(reversed(keys))
    seed_bytes = hashlib.sha256(f"{qid}:{pass_index}".encode("utf-8")).digest()[:8]
    seed = int.from_bytes(seed_bytes, "big", signed=False)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(keys))
    return [keys[int(i)] for i in order]


def _canonical_candidates(qid: str, question: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    qtype = str(question.get("type", "")).lower()
    if qtype == "choice":
        criteria = question.get("criteria")
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise ValueError(f"choice {qid!r} needs at least two criteria")
        keys = [str(k) for k in criteria]
        return keys, {str(k): str(v) for k, v in criteria.items()}

    if qtype == "score":
        criteria = question.get("criteria")
        if not isinstance(criteria, list) or len(criteria) < 2:
            raise ValueError(f"score {qid!r} needs at least two ordered levels")
        keys = [str(i) for i in range(len(criteria))]
        return keys, {str(i): str(v) for i, v in enumerate(criteria)}

    if qtype == "noul":
        criteria = question.get("criteria") if isinstance(question.get("criteria"), dict) else {}
        return ["no", "yes"], {
            "no": str(criteria.get("no", "the statement is not supported by the evidence")),
            "yes": str(criteria.get("yes", "the statement is supported by the evidence")),
        }

    raise ValueError(f"Unsupported Laya question type: {qtype!r}")


def adapt_questions_for_laya(
    questions: dict[str, dict[str, Any]],
    *,
    pass_index: int = 0,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Compile all logical primitives to opaque closed-choice transport.

    Laya documents position bias for multilingual score questions and label sensitivity
    for noul. We therefore use the same native decision head but transport choice,
    score, and noul through opaque choice markers, then map the probabilities back.
    Nothing is generated.
    """
    adapted: dict[str, dict[str, Any]] = {}
    metadata: dict[str, dict[str, Any]] = {}

    for qid, q in questions.items():
        qid = str(qid)
        qtype = str(q.get("type", "")).lower()
        instructions = str(q.get("instructions", qid))
        canonical_keys, descriptions = _canonical_candidates(qid, q)
        ordered_keys = _stable_permutation(canonical_keys, qid, pass_index)
        opaque = _opaque_labels(len(ordered_keys))
        forward = dict(zip(ordered_keys, opaque))
        reverse = {v: k for k, v in forward.items()}

        adapted[qid] = {
            "type": "choice",
            "instructions": instructions,
            "criteria": {forward[key]: descriptions[key] for key in ordered_keys},
        }
        metadata[qid] = {
            "type": qtype,
            "transport": "opaque_choice",
            "pass_index": int(pass_index),
            "canonical_keys": canonical_keys,
            "ordered_keys": ordered_keys,
            "forward": forward,
            "reverse": reverse,
            "original": q,
        }

    return adapted, metadata


def _normalize_choice_transport(
    qid: str,
    raw: dict[str, Any],
    meta: dict[str, Any],
) -> dict[str, float]:
    reverse = meta["reverse"]
    raw_probs = raw.get("probabilities") or {}
    mapped = {
        reverse[str(k)]: float(v)
        for k, v in raw_probs.items()
        if str(k) in reverse
    }
    canonical = list(meta["canonical_keys"])
    if set(mapped) != set(canonical):
        missing = sorted(set(canonical) - set(mapped))
        raise ValueError(f"Laya question {qid!r} missed bounded options: {missing}")
    arr = np.asarray([max(mapped[k], 1e-12) for k in canonical], dtype=float)
    arr /= arr.sum()
    return {k: float(v) for k, v in zip(canonical, arr)}


def _js_divergence(probability_maps: list[dict[str, float]]) -> float:
    if len(probability_maps) <= 1:
        return 0.0
    keys = list(probability_maps[0])
    matrix = np.asarray(
        [[max(float(p[k]), 1e-12) for k in keys] for p in probability_maps],
        dtype=float,
    )
    matrix = matrix / matrix.sum(axis=1, keepdims=True)
    mean = matrix.mean(axis=0)
    js = float(
        np.mean(
            np.sum(matrix * (np.log(matrix) - np.log(np.clip(mean, 1e-12, 1.0))), axis=1)
        )
    )
    return float(np.clip(js / math.log(max(len(keys), 2)), 0.0, 1.0))


def _log_pool(probability_maps: list[dict[str, float]]) -> dict[str, float]:
    if not probability_maps:
        raise ValueError("probability_maps cannot be empty")
    keys = list(probability_maps[0])
    logs = np.asarray(
        [[math.log(max(float(p[k]), 1e-12)) for k in keys] for p in probability_maps],
        dtype=float,
    )
    mean_log = logs.mean(axis=0)
    mean_log -= mean_log.max()
    pooled = np.exp(mean_log)
    pooled /= pooled.sum()
    return {k: float(v) for k, v in zip(keys, pooled)}


def _logical_answer(
    qid: str,
    question: dict[str, Any],
    pooled_raw: dict[str, float],
    *,
    temperature: float,
    permutation_maps: list[dict[str, float]],
    raw_confidences: list[float],
    raw_answer_confidences: list[float],
) -> dict[str, Any]:
    probs, calibration_logits = _temperature_scale(pooled_raw, temperature)
    qtype = str(question.get("type", "")).lower()
    winner = max(probs, key=probs.get)
    order_jsd = _js_divergence(permutation_maps)
    stability = float(np.clip(1.0 - order_jsd, 0.0, 1.0))
    entropy_conf = _entropy_confidence(probs)
    answer_conf = float(probs[winner])
    effective_conf = float(np.clip(answer_conf * (0.65 + 0.35 * stability), 0.0, 1.0))

    diagnostics = {
        "backend": "laya-multilingual",
        "transport": "opaque_choice_for_all_primitives",
        "calibration_logits": calibration_logits,
        "permutation_probabilities": permutation_maps,
        "permutation_jsd": order_jsd,
        "order_stability": stability,
        "raw_laya_confidence_mean": float(np.mean(raw_confidences)) if raw_confidences else 0.0,
        "raw_laya_answer_confidence_mean": (
            float(np.mean(raw_answer_confidences)) if raw_answer_confidences else 0.0
        ),
        "native_action_head_used": False,
    }

    if qtype == "choice":
        return {
            "type": "choice",
            "choice": winner,
            "probabilities": probs,
            "confidence": entropy_conf,
            "answer_confidence": answer_conf,
            "effective_confidence": effective_conf,
            "diagnostics": diagnostics,
        }

    if qtype == "noul":
        if "yes" not in probs or "no" not in probs:
            raise ValueError(f"noul {qid!r} did not map to no/yes")
        p_yes = float(probs["yes"])
        binary_conf = max(p_yes, 1.0 - p_yes)
        return {
            "type": "noul",
            "noul": p_yes,
            "probabilities": probs,
            "confidence": binary_conf,
            "answer_confidence": binary_conf,
            "effective_confidence": float(
                np.clip(binary_conf * (0.65 + 0.35 * stability), 0.0, 1.0)
            ),
            "diagnostics": diagnostics,
        }

    if qtype == "score":
        criteria = list(question.get("criteria", []))
        expected = float(sum(i * probs[str(i)] for i in range(len(criteria))))
        return {
            "type": "score",
            "score": expected,
            "normalized_score": expected / max(len(criteria) - 1, 1),
            "probabilities": probs,
            "legend": [str(x) for x in criteria],
            "confidence": entropy_conf,
            "answer_confidence": answer_conf,
            "effective_confidence": effective_conf,
            "diagnostics": diagnostics,
        }

    raise ValueError(qtype)


class LayaDecisionEngine:
    """Native non-autoregressive System-One adapter with order-bias averaging."""

    def __init__(
        self,
        *,
        model_id: str = "convaiinnovations/laya",
        subfolder: str = "multilingual",
        device: str = "auto",
        max_length: int = 1024,
        batch_size: int = 24,
        temperatures: dict[str, float] | None = None,
        permutation_passes: int = 2,
    ):
        self.model_id = model_id
        self.subfolder = subfolder
        self.requested_device = device
        self.max_length = int(max_length)
        self.batch_size = max(1, int(batch_size))
        self.temperatures = dict(temperatures or {})
        self.permutation_passes = max(1, min(int(permutation_passes), 6))
        self._agent = None

    @property
    def backend_name(self) -> str:
        return "laya-multilingual"

    @property
    def model_name(self) -> str:
        return f"{self.model_id}:{self.subfolder}" if self.subfolder else self.model_id

    def _load(self):
        if self._agent is not None:
            return self._agent
        os.environ.setdefault("USE_TF", "0")
        try:
            import laya
        except ImportError as exc:
            raise RuntimeError(
                "Laya backend requires optional decision dependencies. "
                "Install with: python -m pip install -e '.[decision]'"
            ) from exc

        device = None if self.requested_device == "auto" else self.requested_device
        self._agent = laya.load(
            self.model_id,
            subfolder=self.subfolder or None,
            device=device,
        )
        return self._agent

    def _run_pass(
        self,
        requests: list[tuple[str, dict[str, dict[str, Any]]]],
        pass_index: int,
    ) -> list[dict[str, dict[str, Any]]]:
        agent = self._load()
        outputs: list[dict[str, dict[str, Any]]] = [dict() for _ in requests]
        groups: dict[str, list[tuple[int, str, dict[str, Any], dict[str, Any]]]] = defaultdict(list)

        for request_index, (state, questions) in enumerate(requests):
            laya_questions, metadata = adapt_questions_for_laya(
                questions,
                pass_index=pass_index,
            )
            signature = json.dumps(laya_questions, ensure_ascii=False, sort_keys=True)
            groups[signature].append((request_index, state, laya_questions, metadata))

        for rows in groups.values():
            laya_questions = rows[0][2]
            states = [row[1] for row in rows]
            raw_results = agent.predict_batch(
                states,
                laya_questions,
                batch_size=min(self.batch_size, len(states)),
                max_len=self.max_length,
            )
            if len(raw_results) != len(rows):
                raise RuntimeError(
                    f"Laya returned {len(raw_results)} results for {len(rows)} states"
                )

            for row_info, raw_result in zip(rows, raw_results):
                request_index, _state, _questions, metadata = row_info
                raw_answers = raw_result.get("answers", {})
                normalized: dict[str, dict[str, Any]] = {}
                for qid, meta in metadata.items():
                    if qid not in raw_answers:
                        raise KeyError(f"Laya response missing question {qid!r}")
                    raw = raw_answers[qid]
                    normalized[qid] = {
                        "probabilities": _normalize_choice_transport(qid, raw, meta),
                        "raw_confidence": float(raw.get("confidence", 0.0) or 0.0),
                        "raw_answer_confidence": float(
                            raw.get("answer_confidence", 0.0) or 0.0
                        ),
                    }
                outputs[request_index] = normalized

        return outputs

    def decide_many(
        self,
        requests: list[tuple[str, dict[str, dict[str, Any]]]],
    ) -> list[dict[str, dict[str, Any]]]:
        if not requests:
            return []

        per_pass = [
            self._run_pass(requests, pass_index)
            for pass_index in range(self.permutation_passes)
        ]
        outputs: list[dict[str, dict[str, Any]]] = [dict() for _ in requests]

        for request_index, (_state, questions) in enumerate(requests):
            for qid, question in questions.items():
                pass_rows = [p[request_index][qid] for p in per_pass]
                maps = [row["probabilities"] for row in pass_rows]
                pooled = _log_pool(maps)
                qtype = str(question.get("type", "")).lower()
                temperature = float(
                    self.temperatures.get(qid, self.temperatures.get(qtype, 1.0))
                )
                outputs[request_index][qid] = _logical_answer(
                    qid,
                    question,
                    pooled,
                    temperature=temperature,
                    permutation_maps=maps,
                    raw_confidences=[row["raw_confidence"] for row in pass_rows],
                    raw_answer_confidences=[
                        row["raw_answer_confidence"] for row in pass_rows
                    ],
                )
        return outputs

    def decide(
        self,
        state: str,
        questions: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        return self.decide_many([(state, questions)])[0]
