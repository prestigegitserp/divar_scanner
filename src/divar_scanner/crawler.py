from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote, urlparse

import requests
from tqdm.auto import tqdm

from .config import Config
from .kenar import KenarClient, KenarAuthError

DIVAR_API_BASE = "https://api.divar.ir"
DIVAR_WEB_BASE = "https://divar.ir"
CITIES_URL = f"{DIVAR_API_BASE}/v8/places/cities"
DISTRICTS_URL = f"{DIVAR_API_BASE}/v8/places/cities/{{city_id}}/districts"
SEARCH_URL = f"{DIVAR_API_BASE}/v8/postlist/w/search"
DETAIL_URL = f"{DIVAR_API_BASE}/v8/posts-v2/web/{{token}}"

_PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?98|0)?9\d{9}(?!\d)")
_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{6,64}$")
_SLUG_RE = re.compile(r"^[a-z0-9-]+$")


class DivarBlockedError(RuntimeError):
    pass


class DivarTransportError(RuntimeError):
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


def _nested(obj: Any, *keys: str) -> Any:
    cur = obj
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _extract_preloaded_state(document: str) -> dict[str, Any]:
    marker = "window.__PRELOADED_STATE__"
    pos = document.find(marker)
    if pos < 0:
        raise ValueError("Divar preloaded state was not found in the server-rendered page")
    eq = document.find("=", pos + len(marker))
    if eq < 0:
        raise ValueError("Divar preloaded-state assignment is malformed")
    source = document[eq + 1 :].lstrip()
    try:
        value, _ = json.JSONDecoder().raw_decode(source)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Divar preloaded state is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("Divar preloaded state has an unexpected shape")
    return value


def _token_from_url(raw_url: str) -> str:
    path = str(raw_url or "").split("?", 1)[0].rstrip("/")
    token = path.rsplit("/", 1)[-1]
    return token if _TOKEN_RE.fullmatch(token or "") else ""


def _jsonld_details(document: str) -> dict[str, dict[str, Any]]:
    details: dict[str, dict[str, Any]] = {}
    pattern = re.compile(
        r"<script\b[^>]*type=[\"']application/ld\+json[\"'][^>]*>(.*?)</script>",
        re.IGNORECASE | re.DOTALL,
    )

    def visit(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        if not isinstance(value, dict):
            return
        token = _token_from_url(str(value.get("url") or ""))
        if token:
            details[token] = value
        graph = value.get("@graph")
        if graph is not None:
            visit(graph)

    for match in pattern.finditer(document):
        try:
            visit(json.loads(match.group(1)))
        except json.JSONDecodeError:
            continue
    return details


def _find_list_widgets(state: dict[str, Any]) -> list[Any]:
    direct = _nested(state, "nb", "listWidgets")
    if isinstance(direct, list):
        return direct
    for node in _walk(state):
        value = node.get("listWidgets")
        if isinstance(value, list):
            return value
    raise ValueError("Divar listWidgets were not found in the server-rendered page")


def extract_server_rendered_cards(document: str) -> list[dict[str, Any]]:
    """Parse the public server-rendered Divar search page without calling api.divar.ir.

    The web app currently embeds listing cards under window.__PRELOADED_STATE__ and
    supplements them with JSON-LD. This parser intentionally extracts only public
    search-card information and does not touch login/contact endpoints.
    """
    state = _extract_preloaded_state(document)
    details = _jsonld_details(document)
    widgets = _find_list_widgets(state)

    cards: list[dict[str, Any]] = []
    seen: set[str] = set()
    for widget in widgets:
        row = _nested(widget, "data", "dto", "data")
        if not isinstance(row, dict):
            continue
        payload = _nested(row, "action", "payload")
        if not isinstance(payload, dict):
            payload = {}
        token = str(payload.get("token") or "")
        if not _TOKEN_RE.fullmatch(token) or token in seen:
            continue
        seen.add(token)

        detail = details.get(token, {})
        summary_web = payload.get("web_info") if isinstance(payload.get("web_info"), dict) else {}
        detail_web = detail.get("web_info") if isinstance(detail.get("web_info"), dict) else {}
        district = str(
            detail_web.get("district_persian")
            or summary_web.get("district_persian")
            or ""
        ).strip()
        city = str(
            detail_web.get("city_persian")
            or summary_web.get("city_persian")
            or ""
        ).strip()
        floor_size = detail.get("floorSize") if isinstance(detail.get("floorSize"), dict) else {}
        listing_url = str(detail.get("url") or "").strip()
        if not listing_url:
            listing_url = f"{DIVAR_WEB_BASE}/v/-/{quote(token)}"

        cards.append(
            {
                "token": token,
                "title": row.get("title") or detail.get("name") or "",
                "subtitle": row.get("middle_description_text") or "",
                "bottom_text": row.get("bottom_description_text") or "",
                "deposit_text": row.get("top_description_text") or "",
                "rent_text": row.get("middle_description_text") or "",
                "district": district,
                "city": city,
                "area_hint": floor_size.get("value"),
                "rooms_hint": detail.get("numberOfRooms"),
                "description_hint": detail.get("description") or "",
                "url": listing_url,
                "raw_card": row,
                "search_detail": detail,
            }
        )

    if not cards:
        raise ValueError(
            "No listings were found in Divar's server-rendered page. "
            "The page structure may have changed or a block page may have been returned."
        )
    return cards


class DivarCrawler:
    """Polite crawler for public Divar listing/search responses.

    Preferred cloud transport is Divar's official Kenar/Open Platform when a
    KENAR_API_KEY is available. The public server-rendered page and legacy
    api.divar.ir transports remain optional fallbacks. The crawler
    does not use contact-info endpoints, login/OTP, browser automation, CAPTCHA bypass,
    proxy rotation, or personal-account endpoints.
    """

    def __init__(self, config: Config):
        self.config = config
        c = config.section("crawl")
        self.city_id = str(c.get("city_id", "1"))
        self.city_slug = str(c.get("city_slug", "tehran"))
        self.category = str(c.get("category", "apartment-rent"))
        self.web_category_slug = str(c.get("web_category_slug", "rent-apartment"))
        self.district_slug = str(c.get("district_slug", "")).strip()
        self.search_page_url = str(c.get("search_page_url", "")).strip()
        self.transport = str(c.get("transport", "auto")).strip().lower()
        self.max_listings = int(c.get("max_listings", 500))
        self.max_pages = int(c.get("max_pages", 50))
        self.max_web_pages = int(c.get("max_web_pages", 4))
        self.delay = float(c.get("request_delay_seconds", 1.2))
        self.jitter = float(c.get("jitter_seconds", 0.35))
        self.timeout = int(c.get("timeout_seconds", 25))
        self.web_timeout = int(c.get("web_timeout_seconds", min(self.timeout, 20)))
        self.redact_phones = bool(c.get("redact_phone_numbers", True))
        self.cache_dir = Path(c.get("cache_dir", "data/cache"))
        self.raw_dir = Path(c.get("raw_dir", "data/raw"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": str(c.get("user_agent", "divar-scanner-research/0.4")),
                "Accept-Language": "fa-IR,fa;q=0.9,en;q=0.5",
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
                r = self.session.request(
                    method,
                    url,
                    timeout=self.timeout,
                    headers={"Accept": "application/json, text/plain, */*"},
                    **kwargs,
                )
                if r.status_code in {401, 403, 429}:
                    raise DivarBlockedError(
                        f"Divar returned HTTP {r.status_code}. "
                        "The crawler stops instead of bypassing access controls."
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

    def _request_text(self, url: str) -> str:
        last_exc: Exception | None = None
        # Public-page mode is intentionally conservative: only two attempts.
        for attempt in range(2):
            try:
                r = self.session.get(
                    url,
                    timeout=self.web_timeout,
                    headers={"Accept": "text/html,application/xhtml+xml"},
                )
                if r.status_code in {401, 403, 429}:
                    raise DivarBlockedError(
                        f"Divar web returned HTTP {r.status_code}. "
                        "The crawler stops instead of bypassing access controls."
                    )
                r.raise_for_status()
                return r.text
            except DivarBlockedError:
                raise
            except requests.RequestException as exc:
                last_exc = exc
                if attempt == 0:
                    time.sleep(1.0 + random.random())
        assert last_exc is not None
        raise last_exc

    def _write_raw(self, rows: list[dict[str, Any]]) -> tuple[str, str]:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        raw_path = self.raw_dir / f"divar_raw_{stamp}.jsonl"
        with raw_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return stamp, str(raw_path)

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
        explicit_ids = [str(x) for x in (c.get("district_ids") or []) if str(x).strip()]
        if explicit_ids:
            name = str(c.get("district_query", self.district_slug or "configured district"))
            return [DistrictMatch(id=x, name=name, score=1.0) for x in explicit_ids]

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
                        token = (
                            (action.get("payload") or {})
                            if isinstance(action.get("payload"), dict)
                            else {}
                        ).get("token")
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

    def _search_payload(
        self,
        district_ids: list[str],
        pagination: dict[str, Any] | None,
        page: int,
    ) -> dict[str, Any]:
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
                    "additional_form_data": {
                        "data": {"sort": {"str": {"value": "sort_date"}}}
                    },
                },
            },
        }

    def search_api(self, district_ids: list[str]) -> list[dict[str, Any]]:
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

    def _web_search_base_url(self) -> str:
        if self.search_page_url:
            parsed = urlparse(self.search_page_url)
            if parsed.scheme != "https" or parsed.netloc not in {"divar.ir", "www.divar.ir"}:
                raise ValueError(
                    "crawl.search_page_url must be an https://divar.ir public search URL"
                )
            if not parsed.path.startswith("/s/"):
                raise ValueError("crawl.search_page_url must point to a Divar /s/ search page")
            return self.search_page_url.rstrip("/")

        if not _SLUG_RE.fullmatch(self.city_slug):
            raise ValueError(f"Unsafe/invalid city_slug: {self.city_slug!r}")
        if not _SLUG_RE.fullmatch(self.web_category_slug):
            raise ValueError(f"Unsafe/invalid web_category_slug: {self.web_category_slug!r}")
        if not self.district_slug or not _SLUG_RE.fullmatch(self.district_slug):
            raise ValueError(
                "crawl.district_slug is required for server-rendered web mode "
                "(for this project use 'fatemi')."
            )
        return (
            f"{DIVAR_WEB_BASE}/s/{self.city_slug}/"
            f"{self.web_category_slug}/{self.district_slug}"
        )

    def search_web(self) -> tuple[list[dict[str, Any]], list[str]]:
        base = self._web_search_base_url()
        all_cards: list[dict[str, Any]] = []
        seen: set[str] = set()
        source_urls: list[str] = []

        for page in range(1, max(1, self.max_web_pages) + 1):
            url = base if page == 1 else f"{base}?page={page}"
            document = self._request_text(url)
            source_urls.append(url)
            cards = extract_server_rendered_cards(document)
            new_cards = [card for card in cards if card["token"] not in seen]
            if not new_cards:
                break
            for card in new_cards:
                seen.add(card["token"])
                all_cards.append(card)
                if len(all_cards) >= self.max_listings:
                    return all_cards[: self.max_listings], source_urls
            self._sleep()

        return all_cards[: self.max_listings], source_urls

    def fetch_detail(self, token: str, use_cache: bool = True) -> dict[str, Any]:
        cache = self.cache_dir / f"post_{token}.json"
        if use_cache and cache.exists():
            return json.loads(cache.read_text(encoding="utf-8"))
        data = self._request_json("GET", DETAIL_URL.format(token=token))
        cache.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        self._sleep()
        return data


    def _crawl_kenar(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        client = KenarClient(self.config)
        if not client.available:
            raise KenarAuthError(
                "crawl.transport='kenar' requires KENAR_API_KEY in the runtime environment."
            )
        result = client.search()
        raw_path = client.write_raw_snapshot(result)
        rows = result.rows
        meta = {
            "crawl_transport": "kenar",
            "listing_count": len(rows),
            "district_slug": self.district_slug,
            "district_ids": [],
            "district_matches": [
                {
                    "id": f"slug:{self.district_slug}",
                    "name": str(self.config.get("crawl.district_query", self.district_slug)),
                    "score": 1.0,
                }
            ],
            "raw_path": raw_path,
            "crawled_at_utc": result.searched_at_utc,
            "source_urls": [
                "https://open-api.divar.ir/v2/open-platform/finder/post"
            ],
            "official_api": True,
            "kenar_detail_enrichment": bool(client.enrich_details),
            "coverage_note": (
                "Kenar SEARCH_POST is an official bounded snapshot endpoint. "
                "Divar documents a strict overall call quota and no pagination; "
                "the response contains at most 100 recent matching posts."
            ),
        }
        return rows, meta

    def _crawl_web(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        cards, source_urls = self.search_web()
        rows = [
            {
                "card": card,
                "detail": card.get("search_detail") or {},
                "crawl_transport": "server_rendered_web",
            }
            for card in cards
        ]
        stamp, raw_path = self._write_raw(rows)
        c = self.config.section("crawl")
        district_name = str(c.get("district_query", self.district_slug))
        meta = {
            "crawl_transport": "server_rendered_web",
            "district_matches": [
                {"id": f"slug:{self.district_slug}", "name": district_name, "score": 1.0}
            ],
            "district_ids": [],
            "district_slug": self.district_slug,
            "listing_count": len(rows),
            "raw_path": raw_path,
            "source_urls": source_urls,
            "crawled_at_utc": stamp,
            "detail_enrichment": "search-page JSON-LD only; api.divar.ir not required",
            "coverage_note": (
                "Server-rendered mode is resilient when api.divar.ir is unreachable. "
                "Divar may expose only a bounded number of cards per rendered page; "
                "the crawler stops if ?page=N yields no new tokens."
            ),
        }
        return rows, meta

    def _crawl_api(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        matches = self.resolve_districts()
        district_ids = [m.id for m in matches]
        if not district_ids:
            raise RuntimeError(
                "Could not resolve the configured district to a Divar district id. "
                "Refusing to crawl all of Tehran accidentally. "
                "Use crawl.transport='web' with district_slug='fatemi' when the "
                "district API is unavailable."
            )
        cards = self.search_api(district_ids)
        rows: list[dict[str, Any]] = []
        for card in tqdm(cards, desc="Fetching listing details"):
            try:
                detail = self.fetch_detail(card["token"])
                rows.append(
                    {
                        "card": card,
                        "detail": detail,
                        "crawl_transport": "api",
                    }
                )
            except (requests.RequestException, ValueError) as exc:
                rows.append(
                    {
                        "card": card,
                        "detail": {},
                        "crawl_error": str(exc),
                        "crawl_transport": "api",
                    }
                )
        stamp, raw_path = self._write_raw(rows)
        meta = {
            "crawl_transport": "api",
            "district_matches": [m.__dict__ for m in matches],
            "district_ids": district_ids,
            "listing_count": len(rows),
            "raw_path": raw_path,
            "crawled_at_utc": stamp,
        }
        return rows, meta

    def crawl(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if self.transport not in {"snapshot", "kenar", "web", "api", "auto"}:
            raise ValueError("crawl.transport must be one of: snapshot, kenar, web, api, auto")

        if self.transport == "snapshot":
            raise DivarTransportError(
                "crawl.transport='snapshot' performs no network acquisition. "
                "Use run_pipeline(..., crawl=False, input_path='saved_page.html') "
                "or upload HTML/JSONL/CSV/Parquet/ZIP in the Colab snapshot mode."
            )
        if self.transport == "kenar":
            return self._crawl_kenar()
        if self.transport == "web":
            return self._crawl_web()
        if self.transport == "api":
            return self._crawl_api()

        errors: list[str] = []

        if os.getenv("KENAR_API_KEY"):
            try:
                return self._crawl_kenar()
            except (requests.RequestException, ValueError, RuntimeError) as exc:
                errors.append(f"kenar:{type(exc).__name__}:{exc}")

        try:
            return self._crawl_web()
        except (requests.RequestException, ValueError, DivarBlockedError) as exc:
            errors.append(f"web:{type(exc).__name__}:{exc}")
            if isinstance(exc, DivarBlockedError):
                raise

        try:
            rows, meta = self._crawl_api()
            meta["transport_fallback_errors"] = errors
            return rows, meta
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            errors.append(f"api:{type(exc).__name__}:{exc}")
            raise DivarTransportError(
                "All configured Divar transports failed. No proxy/access-control bypass "
                "was attempted. Errors: "
                + " | ".join(errors)
                + " Direct Divar access appears unavailable from this runtime. "
                "Use Divar's official Kenar/Open Platform with KENAR_API_KEY + SEARCH_POST, "
                "or acquire a snapshot in a network that can reach Divar and analyze it in Colab."
            ) from exc


def public_strings(payload: Any) -> list[str]:
    return _string_values(payload)
