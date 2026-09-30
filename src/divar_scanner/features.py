from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .config import Config
from .crawler import normalize_text

_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    out = a.astype(float) / b.astype(float).replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def _extract_area_mentions(text: str) -> list[float]:
    s = normalize_text(text).translate(_DIGITS)
    vals = []
    for m in re.finditer(r"(?<!\d)(\d{2,4})(?:\s*)(?:متر|متری)(?!\w)", s):
        v = float(m.group(1))
        if 10 <= v <= 2000:
            vals.append(v)
    return vals


def _extract_room_mentions(text: str) -> list[int]:
    s = normalize_text(text).translate(_DIGITS)
    vals = []
    for pat in (r"(?<!\d)(\d{1,2})\s*(?:خواب|خوابه)", r"(?<!\d)(\d{1,2})\s*اتاق"):
        for m in re.finditer(pat, s):
            v = int(m.group(1))
            if 0 <= v <= 20:
                vals.append(v)
    words = {
        "یک": 1, "دو": 2, "سه": 3, "چهار": 4, "پنج": 5, "شش": 6,
        "هفت": 7, "هشت": 8, "نه": 9, "ده": 10, "یازده": 11, "دوازده": 12,
    }
    for word, value in words.items():
        if re.search(rf"(?:^|\s){word}\s*(?:خواب|خوابه|اتاق)(?:\s|$)", s):
            vals.append(value)
    return vals


def _amenity_conflict(text: str, name: str, structured: object) -> float:
    if pd.isna(structured) or structured is None:
        return 0.0
    s = normalize_text(text)
    positive = bool(re.search(rf"(?:دارای\s+)?{name}", s)) and not bool(re.search(rf"بدون\s+{name}", s))
    negative = bool(re.search(rf"بدون\s+{name}|{name}\s+ندارد", s))
    val = bool(structured)
    return float((val and negative) or ((not val) and positive))


def add_features(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    out = df.copy()
    f = config.section("features")
    mult = float(f.get("rent_to_deposit_multiplier", 30.0))
    sensitivity_mults = [
        float(x) for x in f.get("rent_to_deposit_multipliers", [25.0, 30.0, 35.0])
    ]
    if mult not in sensitivity_mults:
        sensitivity_mults.append(mult)
    sensitivity_mults = sorted(set(sensitivity_mults))
    bucket = max(5, int(f.get("area_bucket_size", 20)))

    deposit = pd.to_numeric(out["deposit_toman"], errors="coerce")
    rent = pd.to_numeric(out["rent_monthly_toman"], errors="coerce")
    both_missing = deposit.isna() & rent.isna()

    out["deposit_per_m2"] = _safe_div(deposit, out["area_m2"])
    out["rent_per_m2"] = _safe_div(rent, out["area_m2"])

    for m in sensitivity_mults:
        suffix = str(int(m)) if float(m).is_integer() else str(m).replace(".", "_")
        equiv = deposit.fillna(0) + rent.fillna(0) * m
        equiv = equiv.mask(both_missing)
        out[f"equivalent_deposit_{suffix}_toman"] = equiv
        out[f"equivalent_deposit_{suffix}_per_m2"] = _safe_div(equiv, out["area_m2"])

    central_suffix = str(int(mult)) if mult.is_integer() else str(mult).replace(".", "_")
    out["equivalent_deposit_toman"] = out[f"equivalent_deposit_{central_suffix}_toman"]
    out["equivalent_deposit_per_m2"] = out[f"equivalent_deposit_{central_suffix}_per_m2"]

    out["contract_style"] = "unknown"
    out.loc[(deposit > 0) & (rent.fillna(0) <= 0), "contract_style"] = "full_deposit"
    out.loc[(deposit.fillna(0) <= 0) & (rent > 0), "contract_style"] = "rent_only"
    out.loc[(deposit > 0) & (rent > 0), "contract_style"] = "mixed"

    rent_equiv = rent.fillna(0) * mult
    denom_equiv = deposit.fillna(0) + rent_equiv
    out["contract_rent_share"] = (rent_equiv / denom_equiv.replace(0, np.nan)).clip(0, 1)
    out["area_bucket"] = (np.floor(out["area_m2"] / bucket) * bucket).astype("Int64")

    missing_cols = ["area_m2", "rooms", "deposit_toman", "rent_monthly_toman", "description"]
    out["missing_fraction"] = out[missing_cols].isna().mean(axis=1)
    out["description_len"] = out["description"].fillna("").astype(str).str.len()

    area_min = float(f.get("area_min", 10))
    area_max = float(f.get("area_max", 1000))
    rooms_max = float(f.get("rooms_max", 12))
    year_min = float(f.get("shamsi_year_min", 1300))
    year_max = float(f.get("shamsi_year_max", 1410))

    out["impossible_area"] = ((out["area_m2"] < area_min) | (out["area_m2"] > area_max)).fillna(False)
    out["impossible_rooms"] = ((out["rooms"] < 0) | (out["rooms"] > rooms_max)).fillna(False)
    out["impossible_year"] = (
        (out["year_built_shamsi"] < year_min) | (out["year_built_shamsi"] > year_max)
    ).fillna(False)
    out["negative_price"] = (
        (out["deposit_toman"] < 0) | (out["rent_monthly_toman"] < 0)
    ).fillna(False)
    out["rooms_area_ratio_bad"] = ((out["rooms"] >= 5) & (out["area_m2"] < 70)).fillna(False)

    area_conflict = []
    room_conflict = []
    amenity_conflict = []
    for row in out.itertuples(index=False):
        text = f"{getattr(row, 'title', '')} {getattr(row, 'description', '')}"
        areas = _extract_area_mentions(text)
        a = getattr(row, "area_m2", np.nan)
        if areas and pd.notna(a):
            area_conflict.append(float(min(abs(x - float(a)) / max(float(a), 1.0) for x in areas) > 0.18))
        else:
            area_conflict.append(0.0)
        rooms = _extract_room_mentions(text)
        r = getattr(row, "rooms", np.nan)
        room_conflict.append(float(bool(rooms) and pd.notna(r) and all(x != int(r) for x in rooms)))
        conflicts = [
            _amenity_conflict(text, "پارکینگ", getattr(row, "parking", None)),
            _amenity_conflict(text, "آسانسور", getattr(row, "elevator", None)),
            _amenity_conflict(text, "انباری", getattr(row, "storage", None)),
        ]
        amenity_conflict.append(float(max(conflicts)))

    out["area_text_conflict"] = area_conflict
    out["rooms_text_conflict"] = room_conflict
    out["amenity_text_conflict"] = amenity_conflict

    binary_quality = out[
        [
            "impossible_area",
            "impossible_rooms",
            "impossible_year",
            "negative_price",
            "rooms_area_ratio_bad",
            "area_text_conflict",
            "rooms_text_conflict",
            "amenity_text_conflict",
        ]
    ].astype(float)
    out["data_quality_score"] = np.clip(
        0.35 * binary_quality.max(axis=1)
        + 0.35 * binary_quality.mean(axis=1)
        + 0.30 * out["missing_fraction"],
        0,
        1,
    )
    return out
