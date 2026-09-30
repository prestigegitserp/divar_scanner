from __future__ import annotations

import numpy as np

from divar_scanner.decision import (
    compile_requests,
    resolve_model_specs,
    typed_answer_from_evidence,
)
from divar_scanner.laya_backend import (
    _logical_answer,
    _log_pool,
    _normalize_choice_transport,
    _stable_permutation,
    adapt_questions_for_laya,
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


def test_nli_baseline_noul_returns_probability_of_yes():
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


def test_laya_registry_is_native_system_one_engine():
    spec = resolve_model_specs("laya-multilingual", {})[0]
    assert spec.engine == "laya"
    assert spec.model_name == "convaiinnovations/laya"
    assert spec.subfolder == "multilingual"
    assert spec.license == "Apache-2.0"
    assert "Qwen" not in spec.note


def test_nli_backends_are_only_research_baselines():
    spec = resolve_model_specs("mdeberta-nli", {})[0]
    assert spec.engine == "nli"
    ensemble = resolve_model_specs(
        "persian-ensemble",
        {"ensemble_backends": ["mdeberta-nli", "parsbert-parsinlu"]},
    )
    assert all(s.engine == "nli" for s in ensemble)


def test_laya_adapter_transports_all_primitives_as_opaque_choice():
    questions = {
        "route": {
            "type": "choice",
            "instructions": "نوع؟",
            "criteria": {"normal": "عادی", "error": "خطای داده"},
        },
        "consistency": {
            "type": "score",
            "instructions": "سازگاری؟",
            "criteria": ["بد", "متوسط", "خوب"],
        },
        "flag": {
            "type": "noul",
            "instructions": "مشکوک است؟",
            "criteria": {"no": "شواهد کافی نیست", "yes": "شواهد کافی است"},
        },
    }
    adapted, meta = adapt_questions_for_laya(questions, pass_index=0)

    assert all(q["type"] == "choice" for q in adapted.values())
    assert set(adapted["route"]["criteria"]) == {"A", "B"}
    assert set(adapted["consistency"]["criteria"]) == {"A", "B", "C"}
    assert set(adapted["flag"]["criteria"]) == {"A", "B"}
    assert meta["route"]["type"] == "choice"
    assert meta["consistency"]["type"] == "score"
    assert meta["flag"]["type"] == "noul"
    assert meta["flag"]["canonical_keys"] == ["no", "yes"]


def test_laya_permutation_is_deterministic_and_reverses_second_pass():
    keys = ["a", "b", "c", "d"]
    assert _stable_permutation(keys, "q", 0) == keys
    assert _stable_permutation(keys, "q", 1) == list(reversed(keys))
    assert _stable_permutation(keys, "q", 3) == _stable_permutation(keys, "q", 3)


def test_laya_transport_maps_permuted_markers_back_to_canonical_keys():
    q = {
        "type": "choice",
        "instructions": "نوع؟",
        "criteria": {"normal": "عادی", "error": "خطا"},
    }
    _, meta0 = adapt_questions_for_laya({"q": q}, pass_index=0)
    _, meta1 = adapt_questions_for_laya({"q": q}, pass_index=1)

    p0 = _normalize_choice_transport(
        "q",
        {"probabilities": {"A": 0.8, "B": 0.2}},
        meta0["q"],
    )
    # pass 1 reverses logical option order, so A maps to error and B to normal.
    p1 = _normalize_choice_transport(
        "q",
        {"probabilities": {"A": 0.2, "B": 0.8}},
        meta1["q"],
    )
    assert p0 == p1 == {"normal": 0.8, "error": 0.2}


def test_laya_noul_is_derived_from_closed_choice_probability():
    q = {
        "type": "noul",
        "instructions": "آیا؟",
        "criteria": {"no": "خیر", "yes": "بله"},
    }
    maps = [{"no": 0.10, "yes": 0.90}, {"no": 0.15, "yes": 0.85}]
    pooled = _log_pool(maps)
    out = _logical_answer(
        "q",
        q,
        pooled,
        temperature=1.0,
        permutation_maps=maps,
        raw_confidences=[0.7, 0.7],
        raw_answer_confidences=[0.9, 0.85],
    )
    assert out["type"] == "noul"
    assert out["noul"] > 0.85
    assert np.isclose(out["noul"], out["probabilities"]["yes"])
    assert out["diagnostics"]["transport"] == "opaque_choice_for_all_primitives"


def test_laya_score_is_expected_value_over_closed_options():
    q = {
        "type": "score",
        "instructions": "سازگاری؟",
        "criteria": ["بد", "متوسط", "خوب"],
    }
    maps = [
        {"0": 0.05, "1": 0.15, "2": 0.80},
        {"0": 0.07, "1": 0.18, "2": 0.75},
    ]
    out = _logical_answer(
        "q",
        q,
        _log_pool(maps),
        temperature=1.0,
        permutation_maps=maps,
        raw_confidences=[0.8, 0.8],
        raw_answer_confidences=[0.8, 0.75],
    )
    assert out["type"] == "score"
    assert 1.6 < out["score"] <= 2.0
    assert 0.8 < out["normalized_score"] <= 1.0


def test_option_order_instability_reduces_effective_confidence():
    q = {
        "type": "choice",
        "instructions": "نوع؟",
        "criteria": {"a": "الف", "b": "ب"},
    }
    stable_maps = [{"a": 0.9, "b": 0.1}, {"a": 0.88, "b": 0.12}]
    unstable_maps = [{"a": 0.9, "b": 0.1}, {"a": 0.1, "b": 0.9}]

    stable = _logical_answer(
        "q", q, _log_pool(stable_maps), temperature=1.0,
        permutation_maps=stable_maps, raw_confidences=[0.8, 0.8],
        raw_answer_confidences=[0.9, 0.88],
    )
    unstable = _logical_answer(
        "q", q, _log_pool(unstable_maps), temperature=1.0,
        permutation_maps=unstable_maps, raw_confidences=[0.8, 0.8],
        raw_answer_confidences=[0.9, 0.9],
    )
    assert unstable["diagnostics"]["order_stability"] < stable["diagnostics"]["order_stability"]
    assert unstable["effective_confidence"] < stable["effective_confidence"]
