from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LABEL_COLUMNS = {
    "disposition": "human_disposition",
    "integrity_class": "human_integrity_class",
    "duplicate_pattern": "human_duplicate_pattern",
    "market_status": "human_market_status",
    "bait_evidence": "human_bait",
    "data_error_evidence": "human_data_error",
    "manual_review": "human_manual_review",
    "consistency": "human_consistency_level",
}


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    z = np.asarray(logits, dtype=float) / max(float(temperature), 1e-6)
    z = z - np.max(z)
    ex = np.exp(z)
    return ex / max(float(ex.sum()), 1e-12)


def _parse_bool_label(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, bool):
        return "yes" if value else "no"
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "بله", "بلی", "آره"}:
        return "yes"
    if s in {"0", "false", "no", "n", "خیر", "نه"}:
        return "no"
    return None


def _target_key(question_id: str, value: Any) -> str | None:
    if question_id in {
        "disposition",
        "integrity_class",
        "duplicate_pattern",
        "market_status",
    }:
        s = str(value or "").strip()
        return s or None
    if question_id in {"bait_evidence", "data_error_evidence", "manual_review"}:
        return _parse_bool_label(value)
    if question_id == "consistency":
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        try:
            level = int(float(value))
        except (TypeError, ValueError):
            return None
        return str(level) if 0 <= level <= 4 else None
    return None


def _extract_logits(raw_json: str, question_id: str) -> tuple[list[str], np.ndarray] | None:
    try:
        payload = json.loads(str(raw_json or ""))
        answer = payload[question_id]
        diagnostics = answer["diagnostics"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None

    # Native Laya backend: use log(raw bounded probabilities) before this
    # project's target-domain temperature scaling.
    calibration_logits = diagnostics.get("calibration_logits")
    if isinstance(calibration_logits, dict) and len(calibration_logits) >= 2:
        try:
            keys = list(calibration_logits.keys())
            return keys, np.asarray(
                [float(calibration_logits[k]) for k in keys],
                dtype=float,
            )
        except (TypeError, ValueError):
            return None

    # NLI baselines: use entailment-vs-contradiction evidence.
    raw_nli = diagnostics.get("raw_nli")
    if not isinstance(raw_nli, dict) or len(raw_nli) < 2:
        return None
    keys = list(raw_nli.keys())
    logits = []
    for key in keys:
        try:
            logits.append(float(raw_nli[key]["evidence_log_odds"]))
        except (KeyError, TypeError, ValueError):
            return None
    return keys, np.asarray(logits, dtype=float)


def _nll(samples: list[tuple[list[str], np.ndarray, str]], temperature: float) -> float:
    losses = []
    for keys, logits, target in samples:
        if target not in keys:
            continue
        probs = _softmax(logits, temperature)
        p = float(probs[keys.index(target)])
        losses.append(-math.log(max(p, 1e-12)))
    return float(np.mean(losses)) if losses else float("inf")


def fit_temperature(
    samples: list[tuple[list[str], np.ndarray, str]],
) -> dict[str, Any]:
    if not samples:
        raise ValueError("No calibration samples")

    # Broad log-grid followed by two local refinements. One scalar cannot change argmax;
    # it only changes probability sharpness.
    grid = np.geomspace(0.10, 10.0, 240)
    losses = np.asarray([_nll(samples, float(t)) for t in grid])
    best = float(grid[int(np.argmin(losses))])
    for _ in range(2):
        lo = max(0.03, best / 1.8)
        hi = min(30.0, best * 1.8)
        local = np.geomspace(lo, hi, 160)
        local_losses = np.asarray([_nll(samples, float(t)) for t in local])
        best = float(local[int(np.argmin(local_losses))])

    correct = 0
    for keys, logits, target in samples:
        correct += int(keys[int(np.argmax(logits))] == target)

    return {
        "temperature": best,
        "n": len(samples),
        "raw_nll": _nll(samples, 1.0),
        "calibrated_nll": _nll(samples, best),
        "argmax_accuracy": correct / len(samples),
    }


def fit_calibration_from_frame(
    df: pd.DataFrame,
    *,
    min_samples: int = 12,
) -> dict[str, Any]:
    if "decision_raw_json" not in df:
        raise ValueError("Input must contain decision_raw_json from a scored/review CSV")

    temperatures: dict[str, float] = {}
    questions: dict[str, Any] = {}
    warnings: list[str] = []

    for question_id, label_col in LABEL_COLUMNS.items():
        if label_col not in df:
            warnings.append(f"missing label column: {label_col}")
            continue

        samples: list[tuple[list[str], np.ndarray, str]] = []
        for _, row in df.iterrows():
            target = _target_key(question_id, row.get(label_col))
            if target is None:
                continue
            extracted = _extract_logits(row.get("decision_raw_json", ""), question_id)
            if extracted is None:
                continue
            keys, logits = extracted
            if target in keys:
                samples.append((keys, logits, target))

        if len(samples) < min_samples:
            warnings.append(
                f"{question_id}: only {len(samples)} labelled usable rows; need >= {min_samples}"
            )
            continue

        result = fit_temperature(samples)
        temperatures[question_id] = float(result["temperature"])
        questions[question_id] = result

    return {
        "method": "temperature_scaling_on_bounded_option_evidence",
        "temperatures": temperatures,
        "questions": questions,
        "warnings": warnings,
        "notes": [
            "Fit on held-out human labels from the target Persian housing domain.",
            "Temperature scaling changes probability calibration but not the winning argmax option.",
            "Do not fit and report final performance on the same rows; keep a separate evaluation set.",
        ],
    }


def calibrate_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    min_samples: int = 12,
) -> dict[str, Any]:
    p = Path(input_path)
    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)

    result = fit_calibration_from_frame(df, min_samples=min_samples)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
