from __future__ import annotations

from typing import Any

from .decision import NLIDecisionEngine, listing_questions


CASES: list[dict[str, Any]] = [
    {
        "name": "clear_data_mismatch",
        "state": (
            "عنوان: آپارتمان ۱۴۰ متری سه خواب. "
            "فیلد ساختاریافته: متراژ ۶۵ متر، یک اتاق، بدون پارکینگ. "
            "توضیحات: ۱۴۰ متر، سه خواب، دارای پارکینگ. "
            "data_quality_score=0.92; market_anomaly_score=0.20; duplicate_bait_score=0.05. "
            "قیمت به‌تنهایی معیار فریب نیست."
        ),
        "expected_disposition": "data_error",
    },
    {
        "name": "coherent_normal",
        "state": (
            "عنوان: ۸۵ متر دو خواب فاطمی. فیلدها: ۸۵ متر، دو اتاق، پارکینگ دارد. "
            "توضیحات نیز ۸۵ متر دو خواب و پارکینگ را ذکر می‌کند. "
            "data_quality_score=0.03; market_anomaly_score=0.11; "
            "price_model_ratio=1.04; duplicate_bait_score=0.00."
        ),
        "expected_disposition": "plausible",
    },
    {
        "name": "coherent_market_outlier",
        "state": (
            "آگهی از نظر متن و فیلدها کاملاً سازگار است: ۹۰ متر، دو خواب، فاطمی. "
            "هیچ تناقض متنی یا خوشهٔ تکراری ندارد. "
            "data_quality_score=0.02; market_anomaly_score=0.88; "
            "price_model_ratio=0.54; duplicate_bait_score=0.00. "
            "قیمت غیرعادی به‌تنهایی فریب نیست."
        ),
        "expected_disposition": "market_outlier",
    },
    {
        "name": "duplicate_bait_pattern",
        "state": (
            "متن آگهی تقریباً عیناً در ۶ آگهی دیگر تکرار شده است. "
            "همان متن برای ۳ محلهٔ متفاوت و قیمت‌های بسیار متفاوت استفاده شده. "
            "duplicate_similarity=0.97; duplicate_cluster_size=7; "
            "duplicate_cluster_neighborhoods=3; duplicate_cluster_price_inconsistency=0.95; "
            "duplicate_bait_score=0.91; data_quality_score=0.15."
        ),
        "expected_disposition": "misleading_or_bait",
    },
]


def run_diagnostic(engine: NLIDecisionEngine) -> dict[str, Any]:
    questions = listing_questions()
    requests = [(case["state"], questions) for case in CASES]
    answers = engine.decide_many(requests)

    rows = []
    correct = 0
    for case, answer in zip(CASES, answers):
        predicted = answer["disposition"]["choice"]
        ok = predicted == case["expected_disposition"]
        correct += int(ok)
        rows.append(
            {
                "case": case["name"],
                "expected": case["expected_disposition"],
                "predicted": predicted,
                "correct": ok,
                "choice_confidence": answer["disposition"]["confidence"],
                "bait_probability": answer["bait_evidence"]["noul"],
                "data_error_probability": answer["data_error_evidence"]["noul"],
                "manual_review_probability": answer["manual_review"]["noul"],
                "consistency_score": answer["consistency"]["normalized_score"],
            }
        )
    return {
        "model": engine.model_name,
        "backend": engine.backend_name,
        "correct": correct,
        "total": len(CASES),
        "accuracy_on_diagnostic_only": correct / max(len(CASES), 1),
        "rows": rows,
        "warning": (
            "This tiny hand-written diagnostic only checks gross wiring/Persian behavior. "
            "It is not an estimate of real-world fraud-detection accuracy or calibration."
        ),
    }
