from __future__ import annotations

import numpy as np

from divar_scanner.decision import (
    compile_requests,
    typed_answer_from_evidence,
)


def test_choice_is_bounded_and_normalized():
    q = {
        "type": "choice",
        "instructions": "نوع آگهی؟",
        "criteria": {
            "plausible": "سازگار",
            "data_error": "خطای داده",
            "needs_review": "نیازمند بازبینی",
        },
    }
    rows = [
        {"key": "plausible", "entailment": 0.75, "neutral": 0.15, "contradiction": 0.10},
        {"key": "data_error", "entailment": 0.10, "neutral": 0.10, "contradiction": 0.80},
        {"key": "needs_review", "entailment": 0.30, "neutral": 0.45, "contradiction": 0.25},
    ]
    out = typed_answer_from_evidence(q, rows)
    assert out["choice"] == "plausible"
    assert set(out["probabilities"]) == set(q["criteria"])
    assert np.isclose(sum(out["probabilities"].values()), 1.0)
    assert 0 <= out["confidence"] <= 1


def test_noul_returns_probability_of_yes():
    q = {
        "type": "noul",
        "instructions": "آیا آگهی گمراه‌کننده است؟",
        "criteria": {"no": "خیر", "yes": "بله"},
    }
    rows = [
        {"key": "no", "entailment": 0.08, "neutral": 0.10, "contradiction": 0.82},
        {"key": "yes", "entailment": 0.83, "neutral": 0.09, "contradiction": 0.08},
    ]
    out = typed_answer_from_evidence(q, rows)
    assert out["noul"] > 0.9
    assert np.isclose(out["noul"], out["probabilities"]["yes"])


def test_score_is_expected_level():
    q = {
        "type": "score",
        "instructions": "سازگاری را بسنج",
        "criteria": ["بد", "متوسط", "خوب"],
    }
    rows = [
        {"key": "0", "entailment": 0.05, "neutral": 0.10, "contradiction": 0.85},
        {"key": "1", "entailment": 0.20, "neutral": 0.20, "contradiction": 0.60},
        {"key": "2", "entailment": 0.80, "neutral": 0.10, "contradiction": 0.10},
    ]
    out = typed_answer_from_evidence(q, rows)
    assert 1.5 < out["score"] <= 2.0
    assert 0.75 < out["normalized_score"] <= 1.0


def test_compile_requests_never_invents_candidates():
    questions = {
        "route": {
            "type": "choice",
            "instructions": "مسیر؟",
            "criteria": {"a": "گزینه الف", "b": "گزینه ب"},
        }
    }
    compiled, _ = compile_requests([("state", questions)])
    assert [x.key for x in compiled] == ["a", "b"]
