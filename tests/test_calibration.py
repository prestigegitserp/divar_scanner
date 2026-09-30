from __future__ import annotations

import json

import numpy as np
import pandas as pd

from divar_scanner.calibration import fit_calibration_from_frame


def _answer(option_logits: dict[str, float]) -> dict:
    raw_nli = {}
    for key, log_odds in option_logits.items():
        # Any entail/contradiction pair with the requested log ratio is sufficient
        # because calibration reads evidence_log_odds directly.
        raw_nli[key] = {
            "entailment": 0.5,
            "neutral": 0.1,
            "contradiction": 0.4,
            "evidence_log_odds": float(log_odds),
        }
    return {"diagnostics": {"raw_nli": raw_nli}}


def test_temperature_calibration_improves_nll_on_overconfident_sample():
    rows = []
    for i in range(24):
        # 18 correct but overly sharp, 6 confidently wrong -> T > 1 should help NLL.
        target = "plausible" if i % 2 == 0 else "data_error"
        logits = {"plausible": -5.0, "data_error": -5.0, "needs_review": -6.0}
        if i < 18:
            logits[target] = 5.0
        else:
            wrong = "data_error" if target == "plausible" else "plausible"
            logits[wrong] = 5.0
        raw = {
            "disposition": _answer(logits),
            "bait_evidence": _answer({"no": 2.0, "yes": -2.0}),
            "data_error_evidence": _answer({"no": 0.0, "yes": 0.0}),
            "manual_review": _answer({"no": 0.0, "yes": 0.0}),
            "consistency": _answer({str(k): float(k) for k in range(5)}),
        }
        rows.append(
            {
                "decision_raw_json": json.dumps(raw),
                "human_disposition": target,
                "human_bait": "no",
                "human_data_error": np.nan,
                "human_manual_review": np.nan,
                "human_consistency_level": np.nan,
            }
        )

    result = fit_calibration_from_frame(pd.DataFrame(rows), min_samples=12)
    disp = result["questions"]["disposition"]
    assert disp["n"] == 24
    assert disp["calibrated_nll"] <= disp["raw_nll"]
    assert result["temperatures"]["disposition"] > 1.0



def test_calibration_accepts_native_laya_calibration_logits():
    rows = []
    for i in range(16):
        target = "plausible" if i % 2 == 0 else "data_error"
        logits = {
            "plausible": 1.5 if target == "plausible" else -1.5,
            "data_error": 1.5 if target == "data_error" else -1.5,
            "market_outlier": -2.0,
            "misleading_or_bait": -2.0,
            "ambiguous_mixed": -2.0,
        }
        raw = {
            "disposition": {
                "diagnostics": {
                    "calibration_logits": logits,
                    "backend": "laya-multilingual",
                }
            }
        }
        rows.append(
            {
                "decision_raw_json": json.dumps(raw),
                "human_disposition": target,
            }
        )

    result = fit_calibration_from_frame(pd.DataFrame(rows), min_samples=12)
    assert "disposition" in result["temperatures"]
    assert result["questions"]["disposition"]["n"] == 16
