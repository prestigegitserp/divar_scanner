from __future__ import annotations

import json

import pandas as pd

from divar_scanner.laya_training import export_laya_training_frame


def test_laya_training_export_preserves_evidence_firewall_and_gold():
    df = pd.DataFrame(
        [
            {
                "token": "abc",
                "title": "آپارتمان ۸۰ متری",
                "description": "۸۰ متر دو خواب دارای پارکینگ",
                "neighborhood": "فاطمی",
                "area_m2": 80,
                "rooms": 2,
                "year_built_shamsi": 1400,
                "parking": True,
                "elevator": True,
                "storage": True,
                "deposit_toman": 800_000_000,
                "rent_monthly_toman": 15_000_000,
                "contract_style": "mixed",
                "data_quality_score": 0.1,
                "area_text_conflict": 0.0,
                "rooms_text_conflict": 0.0,
                "amenity_text_conflict": 0.0,
                "market_anomaly_score": 0.8,
                "price_model_ratio": 0.55,
                "price_model_anomaly_score": 0.8,
                "lof_anomaly_score": 0.4,
                "equivalence_sensitivity": 0.1,
                "duplicate_similarity": 0.95,
                "duplicate_cluster_size": 5,
                "duplicate_cluster_neighborhoods": 3,
                "duplicate_cluster_price_span": 0.9,
                "duplicate_bait_score": 0.85,
                "human_disposition": "misleading_or_bait",
                "human_bait": "yes",
                "human_data_error": "no",
                "human_manual_review": "yes",
                "human_consistency_level": 3,
                "human_label_confidence": 0.8,
            }
        ]
    )

    cases = export_laya_training_frame(df, seed=42)
    assert len(cases) == 3
    by_view = {case["split_hint"]: case for case in cases}

    full_gold = json.loads(by_view["full"]["gold"])
    assert full_gold["disposition"]["probabilities"]["misleading_or_bait"] == 0.8
    assert full_gold["manual_review"]["probabilities"]["true"] == 0.8

    content_gold = json.loads(by_view["content"]["gold"])
    assert content_gold["consistency"]["label"] == 3
    assert content_gold["data_error_evidence"]["probabilities"]["false"] == 0.8

    bait_state = json.loads(by_view["bait"]["state"])
    assert "duplicate bait score" in bait_state
    assert "market_anomaly_score" not in bait_state
    assert "OOF price" not in bait_state

    bait_gold = json.loads(by_view["bait"]["gold"])
    assert bait_gold["bait_evidence"]["probabilities"]["true"] == 0.8
