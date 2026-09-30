from __future__ import annotations

import json
import math
import re
from typing import Any, Iterable

import pandas as pd

from .crawler import normalize_text, redact_phone_numbers

_DIGIT_MAP = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


def walk(obj: Any) -> Iterable[dict[str, Any]]:
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from walk(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from walk(value)


def fa_to_en(value: Any) -> str:
    return str(value or "").translate(_DIGIT_MAP).replace("٬", ",").replace("٫", ".")


def parse_number(value: Any) -> float | None:
    s = normalize_text(fa_to_en(value))
    if not s or any(x in s for x in ("توافقی", "مجانی", "رایگان")):
        return None
    multiplier = 1.0
    if "میلیارد" in s:
        multiplier = 1_000_000_000.0
    elif "میلیون" in s:
        multiplier = 1_000_000.0
    elif "هزار" in s:
        multiplier = 1_000.0
    nums = re.findall(r"-?\d+(?:[.,]\d+)?", s.replace(",", ""))
    if not nums:
        return None
    try:
        return float(nums[0]) * multiplier
    except ValueError:
        return None


def parse_int(value: Any) -> int | None:
    n = parse_number(value)
    return int(round(n)) if n is not None and math.isfinite(n) else None


def _type_name(node: dict[str, Any]) -> str:
    t = node.get("@type")
    if isinstance(t, str):
        return t.lower()
    data = node.get("data")
    if isinstance(data, dict) and isinstance(data.get("@type"), str):
        return data["@type"].lower()
    return ""


def extract_rows(detail: dict[str, Any]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for node in walk(detail):
        data = node.get("data") if isinstance(node.get("data"), dict) else node
        title = data.get("title")
        value = data.get("value")
        if isinstance(title, str) and value not in (None, ""):
            rows.setdefault(normalize_text(title), value)
        items = data.get("items")
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                ititle = item.get("title") or item.get("label")
                ivalue = item.get("value")
                if ivalue in (None, "") and "available" in item:
                    ivalue = bool(item.get("available"))
                if isinstance(ititle, str) and ivalue not in (None, ""):
                    rows.setdefault(normalize_text(ititle), ivalue)
    return rows


def extract_description(detail: dict[str, Any]) -> str:
    candidates: list[str] = []
    for node in walk(detail):
        data = node.get("data") if isinstance(node.get("data"), dict) else node
        t = _type_name(data)
        text = data.get("text")
        if isinstance(text, str) and text.strip():
            bonus = 1000 if "description" in t else 0
            candidates.append((" " * bonus) + text.strip())
    seo = detail.get("seo") if isinstance(detail, dict) else None
    if isinstance(seo, dict) and isinstance(seo.get("description"), str):
        candidates.append(seo["description"].strip())
    if not candidates:
        return ""
    return max(candidates, key=len).lstrip()


def extract_title(detail: dict[str, Any], card: dict[str, Any]) -> str:
    for node in walk(detail):
        data = node.get("data") if isinstance(node.get("data"), dict) else node
        t = _type_name(data)
        title = data.get("title")
        if isinstance(title, str) and title.strip() and ("title" in t or len(title) > 5):
            return title.strip()
    return str(card.get("title") or "").strip()


def find_value(rows: dict[str, Any], aliases: tuple[str, ...]) -> Any:
    for alias in aliases:
        a = normalize_text(alias)
        for key, value in rows.items():
            if key == a or a in key or key in a:
                return value
    return None


def parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    s = normalize_text(value)
    if not s:
        return None
    if any(x in s for x in ("دارد", "بلی", "بله", "موجود", "yes", "true")):
        return True
    if any(x in s for x in ("ندارد", "خیر", "نه", "بدون", "no", "false")):
        return False
    return None


def _extract_neighborhood(detail: dict[str, Any], card: dict[str, Any], rows: dict[str, Any]) -> str:
    direct = find_value(rows, ("محله", "محدوده"))
    if direct:
        return str(direct)
    for node in walk(detail):
        for key in ("district", "district_name", "neighborhood", "neighbourhood"):
            v = node.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
    for text in (card.get("bottom_text"), card.get("subtitle")):
        s = str(text or "")
        m = re.search(r"(?:در|محله)\s+([^،,\-]{2,40})", s)
        if m:
            return m.group(1).strip()
    return ""


def normalize_listing(raw: dict[str, Any], redact_phones: bool = True) -> dict[str, Any]:
    card = raw.get("card") or {}
    detail = raw.get("detail") or {}
    rows = extract_rows(detail)
    desc = extract_description(detail)
    if redact_phones:
        desc = redact_phone_numbers(desc)

    deposit = parse_number(find_value(rows, ("ودیعه", "رهن", "مبلغ ودیعه")))
    rent = parse_number(find_value(rows, ("اجاره ماهانه", "اجارهٔ ماهانه", "اجاره")))
    area = parse_number(find_value(rows, ("متراژ", "مساحت")))
    year = parse_int(find_value(rows, ("ساخت", "سال ساخت")))
    rooms = parse_int(find_value(rows, ("اتاق", "تعداد اتاق")))

    token = str(card.get("token") or "")
    row = {
        "token": token,
        "url": f"https://divar.ir/v/-/{token}" if token else "",
        "title": extract_title(detail, card),
        "description": desc,
        "neighborhood": _extract_neighborhood(detail, card, rows),
        "area_m2": area,
        "rooms": rooms,
        "year_built_shamsi": year,
        "deposit_toman": deposit,
        "rent_monthly_toman": rent,
        "parking": parse_bool(find_value(rows, ("پارکینگ",))),
        "elevator": parse_bool(find_value(rows, ("آسانسور",))),
        "storage": parse_bool(find_value(rows, ("انباری",))),
        "floor": str(find_value(rows, ("طبقه",)) or ""),
        "crawl_error": raw.get("crawl_error", ""),
        "structured_fields_json": json.dumps(rows, ensure_ascii=False),
    }
    return row


def normalize_many(raw_rows: list[dict[str, Any]], redact_phones: bool = True) -> pd.DataFrame:
    df = pd.DataFrame([normalize_listing(x, redact_phones=redact_phones) for x in raw_rows])
    for col in ("area_m2", "rooms", "year_built_shamsi", "deposit_toman", "rent_monthly_toman"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
