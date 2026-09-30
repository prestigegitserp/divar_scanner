from __future__ import annotations

import hashlib
import json
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import requests
from tqdm.auto import tqdm

from .config import Config

DIVAR_BASE = "https://api.divar.ir"
CITIES_URL = f"{DIVAR_BASE}/v8/places/cities"
DISTRICTS_URL = f"{DIVAR_BASE}/v8/places/cities/{{city_id}}/districts"
SEARCH_URL = f"{DIVAR_BASE}/v8/postlist/w/search"
DETAIL_URL = f"{DIVAR_BASE}/v8/posts-v2/web/{{token}}"

_PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?98|0)?9\d{9}(?!\d)")


class DivarBlockedError(RuntimeError):
    pass


@dataclass
class DistrictMatch:
    id: str
    name: str
    score: float


def normalize_text(value: Any) -> str:
    s = str(value or "").translate(_PERSIAN_DIGITS)
    s = s.replace("ي", "ی").replace("ك", "ک").replace("ۀ", "ه")
    s = re.sub(r"[\u200c\u200f\u202a-\u202e]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def redact_phone_numbers(text: str) -> str:
    return _PHONE_RE.sub("[REDACTED_PHONE]", text or "")


def _walk(obj: Any) -> Iterable[dict[str, Any]]:
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)


def _string_values(obj: Any) -> list[str]:
    values: list[str] = []
    for node in _walk(obj):
        for key in ("name", "title", "value", "display", "slug"):
            v = node.get(key)
            if isinstance(v, str) and v.strip():
                values.append(v)
    return values


def _district_candidates(payload: Any) -> list[dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for node in _walk(payload):
        raw_id = node.get("id") or node.get("district_id") or node.get("value")
        name = node.get("name") or node.get("title") or node.get("display_name")
        if isinstance(raw_id, (str, int)) and isinstance(name, str):
            sid = str(raw_id)
            if sid.isdigit() and 1 <= len(sid) <= 8:
                out[sid] = {"id": sid, "name": name}
    return list(out.values())


def _similarity(a: str, b: str) -> float:
    from difflib import SequenceMatcher

    a, b = normalize_text(a), normalize_text(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.92
    return SequenceMatcher(None, a, b).ratio()


class DivarCrawler:
    """Polite crawler for public Divar listing/search responses.

    It intentionally does not use contact-info endpoints, login/OTP, browser automation,
    CAPTCHA bypass, or personal-account endpoints.
    """

    def __init__(self, config: Config):
        self.config = config
        c = config.section("crawl")
        self.city_id = str(c.get("city_id", "1"))
        self.category = str(c.get("category", "apartment-rent"))
        self.max_listings = int(c.get("max_listings", 500))
        self.max_pages = int(c.get("max_pages", 50))
        self.delay = float(c.get("request_delay_seconds", 1.2))
        self.jitter = float(c.get("jitter_seconds", 0.35))
        self.timeout = int(c.get("timeout_seconds", 25))
        self.redact_phones = bool(c.get("redact_phone_numbers", True))
        self.cache_dir = Path(c.get("cache_dir", "data/cache"))
        self.raw_dir = Path(c.get("raw_dir", "data/raw"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": str(c.get("user_agent", "divar-scanner-research/0.1")),
                "Accept": "application/json, text/plain, */*",
                "Content-Type": "application/json",
                "Origin": "https://divar.ir",
                "Referer": "https://divar.ir/",
            }
        )

    def _sleep(self) -> None:
        time.sleep(max(0.0, self.delay + random.uniform(-self.jitter, self.jitter)))

    def _cache_path(self, prefix: str, key: str) -> Path:
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
        return self.cache_dir / f"{prefix}_{h}.json"

    def _request_json(self, method: str, url: str, **kwargs: Any) -> Any:
        last_exc: Exception | None = None
        for attempt in range(4):
            try:
                r = self.session.request(method, url, timeout=self.timeout, **kwargs)
                if r.status_code in {401, 403, 429}:
                    raise DivarBlockedError(
                        f"Divar returned HTTP {r.status_code}. The crawler stops instead of bypassing access controls."
                    )
                r.raise_for_status()
                return r.json()
            except DivarBlockedError:
                raise
            except (requests.RequestException, ValueError) as exc:
                last_exc = exc
                if attempt < 3:
                    time.sleep(min(12.0, (2 ** attempt) + random.random()))
        assert last_exc is not None
        raise last_exc

    def fetch_districts(self, use_cache: bool = True) -> list[dict[str, str]]:
        cache = self._cache_path("districts", self.city_id)
        if use_cache and cache.exists():
            return json.loads(cache.read_text(encoding="utf-8"))
        payload = self._request_json("GET", DISTRICTS_URL.format(city_id=self.city_id))
        rows = _district_candidates(payload)
        cache.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        return rows

    def resolve_districts(self) -> list[DistrictMatch]:
        c = self.config.section("crawl")
        queries = [str(c.get("district_query", "")).strip()] + [
            str(x).strip() for x in c.get("district_aliases", [])
        ]
        queries = [q for q in queries if q]
        districts = self.fetch_districts()
        scored: list[DistrictMatch] = []
        for d in districts:
            score = max((_similarity(q, d["name"]) for q in queries), default=0.0)
            if score >= 0.72:
                scored.append(DistrictMatch(id=d["id"], name=d["name"], score=score))
        scored.sort(key=lambda x: x.score, reverse=True)
        if scored and scored[0].score >= 0.90:
            best = scored[0].score
            return [x for x in scored if x.score >= max(0.86, best - 0.06)]
        return scored[:5]

    @staticmethod
    def _extract_post_cards(payload: Any) -> list[dict[str, Any]]:
        cards: list[dict[str, Any]] = []
        seen: set[str] = set()
        candidates: list[Any] = []
        if isinstance(payload, dict):
            for key in ("list_widgets", "web_widgets", "widget_list"):
                if key in payload:
                    candidates.append(payload[key])
        candidates.append(payload)
        for root in candidates:
            for node in _walk(root):
                data = node.get("data") if isinstance(node.get("data"), dict) else node
                token = data.get("token")
                if not token:
                    action = data.get("action")
                    if isinstance(action, dict):
                        token = ((action.get("payload") or {}) if isinstance(action.get("payload"), dict) else {}).get("token")
                if isinstance(token, str) and token not in seen:
                    seen.add(token)
                    cards.append(
                        {
                            "token": token,
                            "title": data.get("title") or data.get("top_description_text") or "",
                            "subtitle": data.get("subtitle") or data.get("middle_description_text") or "",
                            "bottom_text": data.get("bottom_description_text") or "",
                            "raw_card": data,
                        }
                    )
        return cards

    @staticmethod
    def _pagination(payload: Any) -> dict[str, Any] | None:
        if not isinstance(payload, dict):
            return None
        p = payload.get("pagination")
        if isinstance(p, dict):
            data = p.get("data")
            if isinstance(data, dict):
                return data
        out: dict[str, Any] = {}
        for key in ("last_post_date", "search_uid", "page", "layer_page"):
            if key in payload:
                out[key] = payload[key]
        return out or None

    def _search_payload(self, district_ids: list[str], pagination: dict[str, Any] | None, page: int) -> dict[str, Any]:
        form_data: dict[str, Any] = {
            "category": {"str": {"value": self.category}},
        }
        if district_ids:
            form_data["districts"] = {"repeated_string": {"value": district_ids}}
        p: dict[str, Any] = {
            "@type": "type.googleapis.com/post_list.PaginationData",
            "page": page,
            "layer_page": page,
        }
        if pagination:
            for key in ("last_post_date", "search_uid"):
                if pagination.get(key) is not None:
                    p[key] = pagination[key]
        return {
            "city_ids": [self.city_id],
            "pagination_data": p,
            "search_data": {
                "form_data": {"data": form_data},
                "server_payload": {
                    "@type": "type.googleapis.com/widgets.SearchData.ServerPayload",
                    "additional_form_data": {"data": {"sort": {"str": {"value": "sort_date"}}}},
                },
            },
        }

    def search(self, district_ids: list[str]) -> list[dict[str, Any]]:
        all_cards: list[dict[str, Any]] = []
        seen: set[str] = set()
        pagination: dict[str, Any] | None = None
        for page in range(1, self.max_pages + 1):
            payload = self._search_payload(district_ids, pagination, page)
            response = self._request_json("POST", SEARCH_URL, json=payload)
            cards = self._extract_post_cards(response)
            new_cards = [c for c in cards if c["token"] not in seen]
            for c in new_cards:
                seen.add(c["token"])
                all_cards.append(c)
                if len(all_cards) >= self.max_listings:
                    return all_cards[: self.max_listings]
            new_pagination = self._pagination(response)
            if not new_cards or not new_pagination or new_pagination == pagination:
                break
            pagination = new_pagination
            self._sleep()
        return all_cards[: self.max_listings]

    def fetch_detail(self, token: str, use_cache: bool = True) -> dict[str, Any]:
        cache = self.cache_dir / f"post_{token}.json"
        if use_cache and cache.exists():
            return json.loads(cache.read_text(encoding="utf-8"))
        data = self._request_json("GET", DETAIL_URL.format(token=token))
        cache.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        self._sleep()
        return data

    def crawl(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        matches = self.resolve_districts()
        district_ids = [m.id for m in matches]
        if not district_ids:
            raise RuntimeError(
                "Could not resolve the configured district to a Divar district id. "
                "Refusing to crawl all of Tehran accidentally. Inspect the district cache and update config."
            )
        cards = self.search(district_ids)
        rows: list[dict[str, Any]] = []
        for card in tqdm(cards, desc="Fetching listing details"):
            try:
                detail = self.fetch_detail(card["token"])
                rows.append({"card": card, "detail": detail})
            except (requests.RequestException, ValueError) as exc:
                rows.append({"card": card, "detail": {}, "crawl_error": str(exc)})
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        raw_path = self.raw_dir / f"divar_raw_{stamp}.jsonl"
        with raw_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        meta = {
            "district_matches": [m.__dict__ for m in matches],
            "district_ids": district_ids,
            "listing_count": len(rows),
            "raw_path": str(raw_path),
            "crawled_at_utc": stamp,
        }
        return rows, meta


def public_strings(payload: Any) -> list[str]:
    return _string_values(payload)
