from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .calibration import LABEL_COLUMNS, _target_key


def _read_frame(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    return pd.read_parquet(p) if p.suffix.lower() == '.parquet' else pd.read_csv(p)


def _answer_probabilities(raw_json: Any, question_id: str) -> dict[str, float] | None:
    try:
        payload = json.loads(str(raw_json or ''))
        probs = payload[question_id]['probabilities']
        if not isinstance(probs, dict) or len(probs) < 2:
            return None
        clean = {str(k): float(v) for k, v in probs.items()}
        total = sum(max(v, 0.0) for v in clean.values())
        if total <= 0:
            return None
        return {k: max(v, 0.0) / total for k, v in clean.items()}
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def _ece(confidences: list[float], correct: list[int], bins: int = 10) -> float:
    if not confidences:
        return float('nan')
    conf = np.asarray(confidences, dtype=float)
    acc = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(conf)
    ece = 0.0
    for i in range(bins):
        if i == bins - 1:
            mask = (conf >= edges[i]) & (conf <= edges[i + 1])
        else:
            mask = (conf >= edges[i]) & (conf < edges[i + 1])
        if not mask.any():
            continue
        ece += (mask.sum() / total) * abs(float(acc[mask].mean()) - float(conf[mask].mean()))
    return float(ece)


def _question_metrics(df: pd.DataFrame, question_id: str, label_col: str) -> dict[str, Any]:
    samples: list[tuple[str, dict[str, float], bool]] = []
    for _, row in df.iterrows():
        target = _target_key(question_id, row.get(label_col))
        if target is None:
            continue
        probs = _answer_probabilities(row.get('decision_raw_json', ''), question_id)
        if probs is None or target not in probs:
            continue
        abstain = bool(row.get('decision_abstain', False))
        samples.append((target, probs, abstain))

    if not samples:
        return {'n': 0}

    losses = []
    briers = []
    correct = []
    confidences = []
    selective_correct = []
    for target, probs, abstain in samples:
        keys = list(probs)
        pred = max(probs, key=probs.get)
        p_target = max(float(probs[target]), 1e-12)
        losses.append(-math.log(p_target))
        one_hot = np.asarray([1.0 if k == target else 0.0 for k in keys])
        pv = np.asarray([probs[k] for k in keys], dtype=float)
        briers.append(float(np.mean((pv - one_hot) ** 2)))
        ok = int(pred == target)
        correct.append(ok)
        confidences.append(float(max(probs.values())))
        if not abstain:
            selective_correct.append(ok)

    n = len(samples)
    non_abstained = len(selective_correct)
    return {
        'n': n,
        'accuracy': float(np.mean(correct)),
        'nll': float(np.mean(losses)),
        'multiclass_brier': float(np.mean(briers)),
        'ece_10bin': _ece(confidences, correct, bins=10),
        'coverage_non_abstained': non_abstained / n,
        'selective_accuracy_non_abstained': (
            float(np.mean(selective_correct)) if selective_correct else None
        ),
    }


def evaluate_frame(df: pd.DataFrame) -> dict[str, Any]:
    questions: dict[str, Any] = {}
    for qid, label_col in LABEL_COLUMNS.items():
        if label_col in df:
            questions[qid] = _question_metrics(df, qid, label_col)

    evaluated = df[df.get('decision_evaluated', False) == True] if 'decision_evaluated' in df else df.iloc[0:0]
    abstained = (
        int(evaluated.get('decision_abstain', pd.Series(False, index=evaluated.index)).fillna(False).sum())
        if len(evaluated) else 0
    )
    return {
        'rows': int(len(df)),
        'decision_evaluated_rows': int(len(evaluated)),
        'decision_abstained_rows': abstained,
        'decision_abstention_rate': (abstained / len(evaluated) if len(evaluated) else None),
        'questions': questions,
        'notes': [
            'Metrics use human labels and bounded option probabilities from decision_raw_json.',
            'ECE/Brier/NLL evaluate probability quality; accuracy alone is insufficient.',
            'Selective accuracy is measured only on non-abstained rows.',
            'Keep an untouched held-out set for final claims.',
        ],
    }


def evaluate_file(input_path: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    result = evaluate_frame(_read_frame(input_path))
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result