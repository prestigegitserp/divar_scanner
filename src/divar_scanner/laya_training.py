from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .decision import build_listing_state, listing_questions
from .laya_backend import adapt_questions_for_laya


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value)) or str(value).strip() == ""


def _bool_target(value: Any) -> str | None:
    if _missing(value):
        return None
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "بله", "بلی", "آره"}:
        return "true"
    if s in {"0", "false", "no", "n", "خیر", "نه"}:
        return "false"
    return None


def _confidence(row: pd.Series) -> float:
    value = row.get("human_label_confidence")
    if _missing(value):
        return 1.0
    try:
        return float(np.clip(float(value), 0.5, 1.0))
    except Exception:
        return 1.0


def _soft_target(keys: list[str], winner: str, confidence: float) -> dict[str, float]:
    if winner not in keys:
        raise ValueError(f"winner {winner!r} not in {keys!r}")
    if len(keys) == 1:
        return {keys[0]: 1.0}
    rest = max(0.0, 1.0 - confidence) / (len(keys) - 1)
    return {key: (confidence if key == winner else rest) for key in keys}


def _choice_gold(question: dict[str, Any], label: Any, confidence: float) -> dict[str, Any] | None:
    if _missing(label):
        return None
    label = str(label).strip()
    keys = [str(k) for k in question["criteria"]]
    if label not in keys:
        return None
    return {
        "label": label,
        "probabilities": _soft_target(keys, label, confidence),
    }


def _noul_gold(label: Any, confidence: float) -> dict[str, Any] | None:
    target = _bool_target(label)
    if target is None:
        return None
    probs = _soft_target(["false", "true"], target, confidence)
    return {
        "label": 1 if target == "true" else 0,
        "probabilities": probs,
    }


def _score_gold(question: dict[str, Any], label: Any, confidence: float) -> dict[str, Any] | None:
    if _missing(label):
        return None
    try:
        level = int(float(label))
    except (TypeError, ValueError):
        return None
    n = len(question["criteria"])
    if not 0 <= level < n:
        return None
    keys = [str(i) for i in range(n)]
    winner = str(level)
    return {
        "label": level,
        "score": float(level),
        "probabilities": _soft_target(keys, winner, confidence),
    }


def _adapt_gold_for_laya(
    gold: dict[str, dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for qid, g in gold.items():
        meta = metadata[qid]
        qtype = meta["type"]
        if qtype == "choice":
            forward = meta["forward"]
            label = str(g["label"])
            probs = {
                forward[str(k)]: float(v)
                for k, v in g["probabilities"].items()
            }
            out[qid] = {
                **g,
                "label": forward[label],
                "probabilities": probs,
            }
        else:
            # score keys are already "0".."K-1"; noul gold already uses false/true.
            out[qid] = g
    return out


def _case(
    *,
    case_id: str,
    state: str,
    questions: dict[str, dict[str, Any]],
    gold: dict[str, dict[str, Any]],
    split_hint: str,
) -> dict[str, Any] | None:
    if not gold:
        return None

    laya_questions, metadata = adapt_questions_for_laya(questions)
    laya_gold = _adapt_gold_for_laya(gold, metadata)

    # This mirrors the public Laya typed-decisions dataset schema: each field is
    # JSON-serialized and the official preprocessor calls json.loads on it.
    return {
        "case_id": case_id,
        "split_hint": split_hint,
        "state": json.dumps(state, ensure_ascii=False),
        "questions": json.dumps(laya_questions, ensure_ascii=False),
        "gold": json.dumps(laya_gold, ensure_ascii=False),
    }


def export_laya_training_frame(
    df: pd.DataFrame,
    *,
    seed: int = 42,
) -> list[dict[str, Any]]:
    questions = listing_questions()
    cases: list[dict[str, Any]] = []

    for row_index, row in df.iterrows():
        token = str(row.get("token") or row_index)
        confidence = _confidence(row)

        full_q = {
            "disposition": questions["disposition"],
            "manual_review": questions["manual_review"],
        }
        full_gold: dict[str, dict[str, Any]] = {}
        g = _choice_gold(questions["disposition"], row.get("human_disposition"), confidence)
        if g:
            full_gold["disposition"] = g
        g = _noul_gold(row.get("human_manual_review"), confidence)
        if g:
            full_gold["manual_review"] = g
        case = _case(
            case_id=f"{token}:full",
            state=build_listing_state(row, view="full"),
            questions=full_q,
            gold=full_gold,
            split_hint="full",
        )
        if case:
            cases.append(case)

        content_q = {
            "consistency": questions["consistency"],
            "data_error_evidence": questions["data_error_evidence"],
        }
        content_gold: dict[str, dict[str, Any]] = {}
        g = _score_gold(
            questions["consistency"],
            row.get("human_consistency_level"),
            confidence,
        )
        if g:
            content_gold["consistency"] = g
        g = _noul_gold(row.get("human_data_error"), confidence)
        if g:
            content_gold["data_error_evidence"] = g
        case = _case(
            case_id=f"{token}:content",
            state=build_listing_state(row, view="content"),
            questions=content_q,
            gold=content_gold,
            split_hint="content",
        )
        if case:
            cases.append(case)

        bait_q = {"bait_evidence": questions["bait_evidence"]}
        bait_gold: dict[str, dict[str, Any]] = {}
        g = _noul_gold(row.get("human_bait"), confidence)
        if g:
            bait_gold["bait_evidence"] = g
        case = _case(
            case_id=f"{token}:bait",
            state=build_listing_state(row, view="bait"),
            questions=bait_q,
            gold=bait_gold,
            split_hint="bait",
        )
        if case:
            cases.append(case)

    # Stable shuffle avoids ordered case families leaking into training batches.
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(cases)) if cases else []
    return [cases[int(i)] for i in order]


def export_laya_training_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    p = Path(input_path)
    df = pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)
    cases = export_laya_training_frame(df, seed=seed)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for case in cases:
            fh.write(json.dumps(case, ensure_ascii=False) + "\n")

    q_count = 0
    by_view: dict[str, int] = {}
    for case in cases:
        gold = json.loads(case["gold"])
        q_count += len(gold)
        view = case["split_hint"]
        by_view[view] = by_view.get(view, 0) + 1

    return {
        "output": str(out),
        "rows_input": int(len(df)),
        "cases_output": int(len(cases)),
        "typed_decisions_output": int(q_count),
        "cases_by_evidence_view": by_view,
        "schema": "Laya state/questions/gold JSON-string fields",
        "next_step": (
            "Use this JSONL in Laya's official fine-tuning notebook by replacing "
            "the load_dataset calls with load_dataset('json', data_files=...)."
        ),
    }
