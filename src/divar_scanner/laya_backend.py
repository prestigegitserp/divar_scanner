from __future__ import annotations

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
    # Opaque caller-facing labels reduce accidental semantic leakage from internal IDs.
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if n <= len(alphabet):
        return list(alphabet[:n])
    return [f"O{i}" for i in range(n)]


def adapt_questions_for_laya(
    questions: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    adapted: dict[str, dict[str, Any]] = {}
    metadata: dict[str, dict[str, Any]] = {}

    for qid, q in questions.items():
        qtype = str(q.get("type", "")).lower()
        instructions = str(q.get("instructions", qid))
        if qtype == "choice":
            criteria = q.get("criteria")
            if not isinstance(criteria, dict) or len(criteria) < 2:
                raise ValueError(f"choice {qid!r} needs at least two criteria")
            original_keys = [str(k) for k in criteria]
            opaque = _opaque_labels(len(original_keys))
            forward = dict(zip(original_keys, opaque))
            reverse = {v: k for k, v in forward.items()}
            adapted[qid] = {
                "type": "choice",
                "instructions": instructions,
                "criteria": {
                    forward[str(k)]: str(v)
                    for k, v in criteria.items()
                },
            }
            metadata[qid] = {
                "type": "choice",
                "forward": forward,
                "reverse": reverse,
                "original": q,
            }
        elif qtype == "score":
            criteria = q.get("criteria")
            if not isinstance(criteria, list) or len(criteria) < 2:
                raise ValueError(f"score {qid!r} needs at least two ordered levels")
            adapted[qid] = {
                "type": "score",
                "instructions": instructions,
                "criteria": [str(x) for x in criteria],
            }
            metadata[qid] = {"type": "score", "original": q}
        elif qtype == "noul":
            criteria = q.get("criteria") if isinstance(q.get("criteria"), dict) else {}
            no_desc = str(criteria.get("no", "the statement is not supported"))
            yes_desc = str(criteria.get("yes", "the statement is supported"))
            adapted[qid] = {
                "type": "noul",
                "instructions": instructions,
                "criteria": {
                    "false": no_desc,
                    "true": yes_desc,
                },
                # Opaque model-facing labels; semantic meaning remains in descriptions.
                "labels": {"false": "B", "true": "A"},
            }
            metadata[qid] = {"type": "noul", "original": q}
        else:
            raise ValueError(f"Unsupported Laya question type: {qtype!r}")

    return adapted, metadata


def _normalize_laya_answer(
    qid: str,
    raw: dict[str, Any],
    meta: dict[str, Any],
    *,
    temperature: float,
) -> dict[str, Any]:
    qtype = meta["type"]
    raw_conf = float(raw.get("confidence", 0.0) or 0.0)
    raw_answer_conf = float(raw.get("answer_confidence", 0.0) or 0.0)

    if qtype == "choice":
        reverse = meta["reverse"]
        raw_probs = raw.get("probabilities") or {}
        mapped = {
            reverse[str(k)]: float(v)
            for k, v in raw_probs.items()
            if str(k) in reverse
        }
        if len(mapped) != len(reverse):
            missing = sorted(set(reverse.values()) - set(mapped))
            raise ValueError(f"Laya choice {qid!r} missed options: {missing}")
        probs, logits = _temperature_scale(mapped, temperature)
        winner = max(probs, key=probs.get)
        return {
            "type": "choice",
            "choice": winner,
            "probabilities": probs,
            "confidence": _entropy_confidence(probs),
            "answer_confidence": float(probs[winner]),
            "diagnostics": {
                "backend": "laya-multilingual",
                "calibration_logits": logits,
                "raw_laya_confidence": raw_conf,
                "raw_laya_answer_confidence": raw_answer_conf,
                "raw_laya_choice": raw.get("choice"),
                "action_ignored": True,
            },
        }

    if qtype == "score":
        criteria = list(meta["original"]["criteria"])
        raw_probs = raw.get("probabilities") or {}
        mapped = {
            str(i): float(raw_probs.get(str(i), 0.0))
            for i in range(len(criteria))
        }
        if sum(mapped.values()) <= 0:
            raise ValueError(f"Laya score {qid!r} returned no usable distribution")
        probs, logits = _temperature_scale(mapped, temperature)
        expected = float(
            sum(i * probs[str(i)] for i in range(len(criteria)))
        )
        winner = max(probs, key=probs.get)
        return {
            "type": "score",
            "score": expected,
            "normalized_score": expected / max(len(criteria) - 1, 1),
            "probabilities": probs,
            "legend": [str(x) for x in criteria],
            "confidence": _entropy_confidence(probs),
            "answer_confidence": float(probs[winner]),
            "diagnostics": {
                "backend": "laya-multilingual",
                "calibration_logits": logits,
                "raw_laya_confidence": raw_conf,
                "raw_laya_answer_confidence": raw_answer_conf,
                "raw_laya_score": raw.get("score"),
                "action_ignored": True,
            },
        }

    if qtype == "noul":
        p_true = float(np.clip(float(raw.get("noul", 0.5)), 1e-9, 1 - 1e-9))
        raw_probs = {"no": 1.0 - p_true, "yes": p_true}
        probs, logits = _temperature_scale(raw_probs, temperature)
        p_yes = float(probs["yes"])
        return {
            "type": "noul",
            "noul": p_yes,
            "probabilities": probs,
            "confidence": max(p_yes, 1.0 - p_yes),
            "answer_confidence": max(p_yes, 1.0 - p_yes),
            "diagnostics": {
                "backend": "laya-multilingual",
                "calibration_logits": logits,
                "raw_laya_confidence": raw_conf,
                "raw_laya_answer_confidence": raw_answer_conf,
                "raw_laya_noul": p_true,
                "action_ignored": True,
            },
        }

    raise ValueError(qtype)


class LayaDecisionEngine:
    """Adapter around Laya's native non-autoregressive typed decision model."""

    def __init__(
        self,
        *,
        model_id: str = "convaiinnovations/laya",
        subfolder: str = "multilingual",
        device: str = "auto",
        max_length: int = 1024,
        batch_size: int = 24,
        temperatures: dict[str, float] | None = None,
    ):
        self.model_id = model_id
        self.subfolder = subfolder
        self.requested_device = device
        self.max_length = int(max_length)
        self.batch_size = max(1, int(batch_size))
        self.temperatures = dict(temperatures or {})
        self._agent = None

    @property
    def backend_name(self) -> str:
        return "laya-multilingual"

    @property
    def model_name(self) -> str:
        return (
            f"{self.model_id}:{self.subfolder}"
            if self.subfolder
            else self.model_id
        )

    def _load(self):
        if self._agent is not None:
            return self._agent
        # Avoid optional TensorFlow imports inside transformers stacks.
        os.environ.setdefault("USE_TF", "0")
        try:
            import laya
        except ImportError as exc:
            raise RuntimeError(
                "Laya backend requires the optional decision dependencies. "
                "Install with: python -m pip install -e '.[decision]'"
            ) from exc

        device = None if self.requested_device == "auto" else self.requested_device
        self._agent = laya.load(
            self.model_id,
            subfolder=self.subfolder or None,
            device=device,
        )
        return self._agent

    def decide_many(
        self,
        requests: list[tuple[str, dict[str, dict[str, Any]]]],
    ) -> list[dict[str, dict[str, Any]]]:
        if not requests:
            return []

        agent = self._load()
        outputs: list[dict[str, dict[str, Any]]] = [dict() for _ in requests]

        # Laya predict_batch shares forward passes when question schemas are identical.
        groups: dict[str, list[tuple[int, str, dict[str, Any], dict[str, Any]]]] = defaultdict(list)
        for request_index, (state, questions) in enumerate(requests):
            laya_questions, metadata = adapt_questions_for_laya(questions)
            signature = json.dumps(laya_questions, ensure_ascii=False, sort_keys=True)
            groups[signature].append(
                (request_index, state, laya_questions, metadata)
            )

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
                    qtype = meta["type"]
                    temperature = float(
                        self.temperatures.get(
                            qid,
                            self.temperatures.get(qtype, 1.0),
                        )
                    )
                    normalized[qid] = _normalize_laya_answer(
                        qid,
                        raw_answers[qid],
                        meta,
                        temperature=temperature,
                    )
                outputs[request_index] = normalized

        return outputs

    def decide(
        self,
        state: str,
        questions: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        return self.decide_many([(state, questions)])[0]
