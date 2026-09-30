import pandas as pd

from divar_scanner.config import Config
from divar_scanner.features import add_features


def test_cross_field_conflict_detection(tmp_path):
    cfg = Config(
        raw={
            "features": {
                "rent_to_deposit_multiplier": 30,
                "area_min": 10,
                "area_max": 1000,
                "rooms_max": 12,
                "shamsi_year_min": 1300,
                "shamsi_year_max": 1410,
                "area_bucket_size": 20,
            }
        },
        source=tmp_path / "x.yaml",
    )
    df = pd.DataFrame(
        [
            {
                "title": "آپارتمان ۱۴۰ متری سه خواب",
                "description": "ملک ۱۴۰ متر سه خواب دارای پارکینگ",
                "area_m2": 65,
                "rooms": 1,
                "year_built_shamsi": 1400,
                "deposit_toman": 500_000_000,
                "rent_monthly_toman": 20_000_000,
                "parking": False,
                "elevator": None,
                "storage": None,
            }
        ]
    )
    out = add_features(df, cfg)
    assert out.loc[0, "area_text_conflict"] == 1
    assert out.loc[0, "rooms_text_conflict"] == 1
    assert out.loc[0, "amenity_text_conflict"] == 1
