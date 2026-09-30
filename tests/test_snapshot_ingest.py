from __future__ import annotations

import json
from pathlib import Path

import yaml

from divar_scanner.pipeline import run_pipeline


def test_raw_jsonl_snapshot_is_normalized_offline(tmp_path: Path):
    rows = [
        {
            "card": {
                "token": f"tok{i}abcd",
                "title": f"آپارتمان {70+i*5} متری",
                "district": "فاطمی",
                "area_hint": 70 + i * 5,
                "rooms_hint": 2,
                "year_hint": 1398,
                "deposit_toman_hint": 600_000_000 + i * 50_000_000,
                "rent_toman_hint": 15_000_000 + i * 1_000_000,
                "parking_hint": True,
                "elevator_hint": True,
                "floor_hint": 2,
                "url": f"https://divar.ir/v/-/tok{i}abcd",
            },
            "detail": {},
            "crawl_transport": "kenar",
        }
        for i in range(6)
    ]
    raw = tmp_path / "snapshot.jsonl"
    raw.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    cfg = {
        "project": {"name": "snapshot-test", "random_seed": 42},
        "crawl": {"redact_phone_numbers": True},
        "features": {
            "rent_to_deposit_multiplier": 30.0,
            "rent_to_deposit_multipliers": [25.0, 30.0, 35.0],
            "area_min": 10,
            "area_max": 1000,
            "rooms_max": 12,
            "shamsi_year_min": 1300,
            "shamsi_year_max": 1410,
            "area_bucket_size": 20,
        },
        "anomaly": {
            "robust_z_clip": 8,
            "peer_min_count": 3,
            "isolation_contamination": "auto",
            "duplicate_similarity_threshold": 0.88,
            "duplicate_max_features": 1000,
        },
        "decision": {"enabled": False},
        "scoring": {
            "data_rule_weight": 1.0,
            "data_decision_weight": 0.0,
            "market_stat_weight": 1.0,
            "market_decision_weight": 0.0,
            "misleading_duplicate_weight": 1.0,
            "misleading_decision_weight": 0.0,
            "bootstrap_decision_multiplier": 0.25,
            "calibrated_decision_multiplier": 0.45,
            "adapted_decision_multiplier": 1.0,
            "review_threshold": 0.55,
            "high_risk_threshold": 0.75,
        },
        "output": {
            "directory": str(tmp_path / "out"),
            "write_csv": False,
            "write_parquet": False,
            "write_html_report": False,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    df, meta = run_pipeline(config_path, crawl=False, input_path=raw)

    assert len(df) == 6
    assert meta["input_format"] == "raw_jsonl"
    assert set(df["neighborhood"]) == {"فاطمی"}
    assert df["area_m2"].notna().all()
    assert df["deposit_toman"].notna().all()
    assert df["rent_monthly_toman"].notna().all()
