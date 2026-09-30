from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from divar_scanner.pipeline import run_pipeline


def _make_rows(n: int = 64) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n):
        area = float(rng.integers(50, 151))
        rooms = int(np.clip(round(area / 45 + rng.normal(0, 0.35)), 1, 4))
        year = int(rng.integers(1382, 1405))
        neighborhood = "فاطمی" if i % 3 else "میدان فاطمی"
        # synthetic market relationship: property characteristics -> price
        base = area * 8_000_000 + rooms * 100_000_000 + (year - 1380) * 8_000_000
        deposit = max(100_000_000, base + rng.normal(0, 90_000_000))
        rent = max(1_000_000, area * 120_000 + rng.normal(0, 2_500_000))
        rows.append(
            {
                "token": f"tok{i}",
                "url": f"https://divar.ir/v/-/tok{i}",
                "title": f"آپارتمان {int(area)} متری {rooms} خواب در {neighborhood}",
                "description": f"واحد {int(area)} متری {rooms} خواب، مناسب سکونت، بازدید با هماهنگی",
                "neighborhood": neighborhood,
                "area_m2": area,
                "rooms": rooms,
                "year_built_shamsi": year,
                "deposit_toman": float(deposit),
                "rent_monthly_toman": float(rent),
                "parking": bool(i % 2),
                "elevator": True,
                "storage": bool(i % 3),
                "floor": "2",
                "crawl_error": "",
                "structured_fields_json": "{}",
            }
        )

    # Deliberately contradictory / extreme review candidate.
    rows[-1].update(
        {
            "title": "آپارتمان ۱۸۰ متری سه خواب لوکس",
            "description": "۱۸۰ متر سه خواب دارای پارکینگ، قیمت استثنایی فقط امروز",
            "area_m2": 38.0,
            "rooms": 1,
            "deposit_toman": 40_000_000.0,
            "rent_monthly_toman": 1_000_000.0,
            "parking": False,
        }
    )
    return pd.DataFrame(rows)


def test_end_to_end_analysis_pipeline(tmp_path: Path):
    df = _make_rows()
    input_path = tmp_path / "synthetic.csv"
    df.to_csv(input_path, index=False)

    config = {
        "project": {"name": "test", "random_seed": 42},
        "features": {
            "rent_to_deposit_multiplier": 30.0,
            "area_min": 10,
            "area_max": 1000,
            "rooms_max": 12,
            "shamsi_year_min": 1300,
            "shamsi_year_max": 1410,
            "area_bucket_size": 20,
        },
        "anomaly": {
            "robust_z_clip": 8,
            "peer_min_count": 6,
            "isolation_contamination": "auto",
            "duplicate_similarity_threshold": 0.88,
            "duplicate_max_features": 4000,
        },
        "decision": {"enabled": False, "top_k": 10, "min_prefilter_score": 0.2},
        "scoring": {
            "data_quality_weight": 0.22,
            "market_anomaly_weight": 0.38,
            "duplicate_weight": 0.16,
            "decision_weight": 0.24,
            "review_threshold": 0.58,
            "high_risk_threshold": 0.78,
        },
        "output": {
            "directory": str(tmp_path / "out"),
            "write_csv": False,
            "write_parquet": False,
            "write_html_report": False,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")

    scored, meta = run_pipeline(config_path, crawl=False, input_path=input_path)

    required = {
        "price_model_anomaly_score",
        "lof_anomaly_score",
        "duplicate_cluster_size",
        "duplicate_bait_score",
        "decision_evaluated",
        "decision_bait_probability",
        "decision_disposition",
        "uncertainty_score",
        "suspicion_score",
        "review_priority_score",
        "risk_band",
        "flag_reasons",
    }
    assert required.issubset(scored.columns)
    assert len(scored) == len(df)
    assert scored["review_priority_score"].between(0, 1).all()
    assert scored["suspicion_score"].between(0, 1).all()
    assert meta["pipeline_version"] == "0.3.0"
    assert meta["counts"]["total"] == len(df)

    candidate = scored.loc[scored["token"] == f"tok{len(df)-1}"].iloc[0]
    assert candidate["data_quality_score"] > 0.25
    assert candidate["review_priority_score"] >= scored["review_priority_score"].median()
