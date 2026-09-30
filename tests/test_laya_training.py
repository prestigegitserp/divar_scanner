from __future__ import annotations

import json

import pandas as pd

from divar_scanner.laya_training import export_laya_training_frame


def test_laya_training_export_matches_opaque_transport_and_evidence_firewalls():
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
                "missing_fraction": 0.0,
                "area_text_conflict": 0.0,
                "rooms_text_conflict": 0.0,
                "amenity_text_conflict": 0.0,
                "market_anomaly_score": 0.8,
                "robust_market_anomaly": 0.7,
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
                "human_integrity_class": "consistent",
                "human_duplicate_pattern": "cross_property_conflict",
                "human_market_status": "extreme_outlier",
                "human_bait": "yes",
                "human_data_error": "no",
                "human_manual_review": "yes",
                "human_consistency_level": 3,
                "human_label_confidence": 0.8,
            }
        ]
    )

    cases = export_laya_training_frame(df, seed=42)
    assert len(cases) == 4
    by_view = {case["split_hint"]: case for case in cases}

    full_gold = json.loads(by_view["full"]["gold"])
    full_questions = json.loads(by_view["full"]["questions"])
    assert set(full_questions["disposition"]["criteria"]) == {"A", "B", "C", "D", "E"}
    assert full_gold["disposition"]["label"] == "D"
    assert full_gold["disposition"]["probabilities"]["D"] == 0.8
    # logical yes is second canonical option -> B under pass 0 opaque transport
    assert full_gold["manual_review"]["label"] == "B"
    assert full_gold["manual_review"]["probabilities"]["B"] == 0.8

    content_gold = json.loads(by_view["content"]["gold"])
    assert content_gold["integrity_class"]["label"] == "A"
    # consistency logical level 3 becomes fourth opaque option -> D
    assert content_gold["consistency"]["label"] == "D"
    # data-error=no is first canonical option -> A
    assert content_gold["data_error_evidence"]["label"] == "A"

    bait_state = json.loads(by_view["bait"]["state"])
    assert "duplicate bait score" in bait_state
    assert "market_anomaly_score" not in bait_state
    assert "OOF price" not in bait_state
    bait_gold = json.loads(by_view["bait"]["gold"])
    assert bait_gold["duplicate_pattern"]["label"] == "D"
    assert bait_gold["bait_evidence"]["label"] == "B"

    market_state = json.loads(by_view["market"]["state"])
    assert "market_anomaly_score" in market_state
    assert "توضیحات آگهی" not in market_state
    market_gold = json.loads(by_view["market"]["gold"])
    assert market_gold["market_status"]["label"] == "C"
