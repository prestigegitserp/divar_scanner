from __future__ import annotations

import numpy as np

from divar_scanner.laya_backend import (
    _normalize_laya_answer,
    adapt_questions_for_laya,
)

from divar_scanner.decision import (
    compile_requests,
    resolve_model_specs,
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


def test_persian_specialist_registry_has_explicit_label_order():
    spec = resolve_model_specs("parsbert-parsinlu", {})[0]
    assert spec.model_name == "persiannlp/parsbert-base-parsinlu-entailment"
    assert spec.forced_label_indices == (0, 2, 1)
    assert "NC" in spec.license


def test_persian_ensemble_is_encoder_only_pair():
    specs = resolve_model_specs(
        "persian-ensemble",
        {"ensemble_backends": ["mdeberta-nli", "parsbert-parsinlu"]},
    )
    assert [s.backend for s in specs] == ["mdeberta-nli", "parsbert-parsinlu"]
    assert all("Qwen" not in s.model_name for s in specs)


def test_model_disagreement_reduces_confidence():
    q = {
        "type": "choice",
        "instructions": "کدام؟",
        "criteria": {"a": "الف", "b": "ب"},
    }
    clean = [
        {"key": "a", "entailment": 0.90, "neutral": 0.03, "contradiction": 0.07, "model_disagreement": 0.0},
        {"key": "b", "entailment": 0.07, "neutral": 0.03, "contradiction": 0.90, "model_disagreement": 0.0},
    ]
    disagree = [
        {**clean[0], "model_disagreement": 2.0},
        {**clean[1], "model_disagreement": 2.0},
    ]
    assert (
        typed_answer_from_evidence(q, disagree)["confidence"]
        < typed_answer_from_evidence(q, clean)["confidence"]
    )


def test_laya_registry_is_native_decision_engine():
    spec = resolve_model_specs("laya-multilingual", {})[0]
    assert spec.engine == "laya"
    assert spec.model_name == "convaiinnovations/laya"
    assert spec.subfolder == "multilingual"
    assert spec.license == "Apache-2.0"


def test_laya_question_adapter_uses_closed_opaque_choice_labels_and_boolean_slots():
    questions = {
        "route": {
            "type": "choice",
            "instructions": "نوع؟",
            "criteria": {"normal": "عادی", "error": "خطای داده"},
        },
        "flag": {
            "type": "noul",
            "instructions": "مشکوک است؟",
            "criteria": {"no": "شواهد کافی نیست", "yes": "شواهد کافی است"},
        },
    }
    adapted, meta = adapt_questions_for_laya(questions)
    assert set(adapted["route"]["criteria"]) == {"A", "B"}
    assert meta["route"]["reverse"] == {"A": "normal", "B": "error"}
    assert set(adapted["flag"]["criteria"]) == {"false", "true"}
    assert adapted["flag"]["labels"] == {"false": "B", "true": "A"}


def test_laya_choice_normalization_maps_back_and_temperature_preserves_argmax():
    original = {
        "type": "choice",
        "instructions": "نوع؟",
        "criteria": {"normal": "عادی", "error": "خطا"},
    }
    _, metadata = adapt_questions_for_laya({"q": original})
    raw = {
        "type": "choice",
        "choice": "A",
        "probabilities": {"A": 0.8, "B": 0.2},
        "confidence": 0.4,
        "answer_confidence": 0.8,
        "action": {"act_probability": 1.0},
    }
    out = _normalize_laya_answer("q", raw, metadata["q"], temperature=2.0)
    assert out["choice"] == "normal"
    assert out["probabilities"]["normal"] > out["probabilities"]["error"]
    assert np.isclose(sum(out["probabilities"].values()), 1.0)
    assert "calibration_logits" in out["diagnostics"]
    assert out["diagnostics"]["action_ignored"] is True


def test_laya_noul_normalization_returns_yes_probability():
    q = {
        "type": "noul",
        "instructions": "آیا؟",
        "criteria": {"no": "خیر", "yes": "بله"},
    }
    _, metadata = adapt_questions_for_laya({"q": q})
    raw = {"type": "noul", "noul": 0.9, "confidence": 0.9, "answer_confidence": 0.9}
    out = _normalize_laya_answer("q", raw, metadata["q"], temperature=1.0)
    assert out["noul"] > 0.89
    assert np.isclose(out["probabilities"]["yes"], out["noul"])
