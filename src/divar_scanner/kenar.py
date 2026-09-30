from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from .config import Config

KENAR_BASE = "https://open-api.divar.ir"
KENAR_SEARCH_URL = f"{KENAR_BASE}/v2/open-platform/finder/post"
KENAR_GET_POST_URL = f"{KENAR_BASE}/v1/open-platform/finder/post/{{token}}"


class KenarAuthError(RuntimeError):
    pass


class KenarPermissionError(RuntimeError):
    pass


@dataclass(frozen=True)
class KenarSearchResult:
    rows: list[dict[str, Any]]
    raw_response: dict[str, Any]
    searched_at_utc: str


def _number_value(value: Any) -> Any:
    if isinstance(value, dict):
        return value.get("value")
    return value


def _rooms_value(value: Any) -> Any:
    if value is None:
        return None
    text = str(value).strip()
    mapping = {
        "بدون اتاق": 0,
        "یک": 1,
        "دو": 2,
        "سه": 3,
        "چهار": 4,
        "پنج": 5,
        "شش": 6,
    }
    if text in mapping:
        return mapping[text]
    return value


class KenarClient:
    """Thin client for Divar's official Kenar/Open Platform Finder API.

    This client intentionally uses only the public Finder endpoints documented by Divar.
    It does not use OAuth/user-private data, chat, phone-number, or contact endpoints.
    """

    def __init__(self, config: Config):
        self.config = config
        c = config.section("crawl")
        self.api_key = str(os.getenv("KENAR_API_KEY") or c.get("kenar_api_key") or "").strip()
        self.timeout = int(c.get("kenar_timeout_seconds", 20))
        self.enrich_details = bool(c.get("kenar_enrich_details", False))
        self.max_listings = min(int(c.get("max_listings", 100)), 100)
        self.category = str(c.get("category", "apartment-rent"))
        self.city = str(c.get("city_slug", "tehran"))
        self.district_slug = str(c.get("district_slug", "fatemi"))
        self.raw_dir = Path(c.get("raw_dir", "data/raw"))
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": str(c.get("user_agent", "divar-scanner-research/0.4.2")),
            }
        )

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise KenarAuthError(
                "KENAR_API_KEY is not set. Create an API key in Divar's Kenar developer "
                "panel with SEARCH_POST permission and set it only in the runtime environment."
            )
        return {"X-API-Key": self.api_key}

    def _request(
        self,
        method: str,
        url: str,
        *,
        json_body: dict[str, Any] | None = None,
        retries: int = 2,
    ) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(max(1, retries)):
            try:
                response = self.session.request(
                    method,
                    url,
                    headers=self._headers(),
                    json=json_body,
                    timeout=self.timeout,
                )
                if response.status_code in {401, 403}:
                    body = response.text[:500]
                    if response.status_code == 401:
                        raise KenarAuthError(
                            f"Kenar rejected the API key (HTTP 401): {body}"
                        )
                    raise KenarPermissionError(
                        f"Kenar denied this operation (HTTP 403). "
                        f"Check SEARCH_POST / GET_POST permissions. Response: {body}"
                    )
                if response.status_code == 429:
                    raise RuntimeError(
                        "Kenar rate limit/quota reached (HTTP 429). "
                        "The official SEARCH_POST endpoint has a strict usage limit."
                    )
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    raise ValueError("Kenar returned a non-object JSON response")
                return data
            except (KenarAuthError, KenarPermissionError):
                raise
            except (requests.RequestException, ValueError, RuntimeError) as exc:
                last_exc = exc
                if attempt + 1 < max(1, retries):
                    time.sleep(1.5 * (attempt + 1))
        assert last_exc is not None
        raise last_exc

    def search(self) -> KenarSearchResult:
        payload: dict[str, Any] = {
            "category": self.category,
            "city": self.city,
            "districts": [self.district_slug],
        }
        raw = self._request("POST", KENAR_SEARCH_URL, json_body=payload, retries=2)
        posts = raw.get("posts") or []
        if not isinstance(posts, list):
            raise ValueError("Kenar search response has an unexpected 'posts' shape")

        rows: list[dict[str, Any]] = []
        for post in posts[: self.max_listings]:
            if not isinstance(post, dict):
                continue
            token = str(post.get("token") or "").strip()
            if not token:
                continue
            real = post.get("real_estate_fields")
            if not isinstance(real, dict):
                real = {}
            credit = real.get("credit")
            rent = real.get("rent")
            price = post.get("price")
            card = {
                "token": token,
                "title": post.get("title") or "",
                "district": self.district_slug,
                "city": post.get("city") or self.city,
                "area_hint": real.get("size"),
                "rooms_hint": _rooms_value(real.get("rooms")),
                "year_hint": real.get("year"),
                "parking_hint": real.get("has_parking"),
                "elevator_hint": real.get("has_elevator"),
                "floor_hint": real.get("floor"),
                "deposit_toman_hint": _number_value(credit),
                "rent_toman_hint": _number_value(rent),
                "price_hint": _number_value(price),
                "url": f"https://divar.ir/v/-/{token}",
                "raw_card": post,
                "search_detail": {
                    "category": post.get("category"),
                    "city": post.get("city"),
                    "real_estate_fields": real,
                    "price": price,
                    "last_modified_at": post.get("last_modified_at"),
                },
            }
            detail: dict[str, Any] = {}
            if self.enrich_details:
                try:
                    detail = self.get_post(token)
                except KenarPermissionError:
                    # SEARCH_POST-only keys should still produce a usable snapshot.
                    self.enrich_details = False
                    detail = {}
            rows.append(
                {
                    "card": card,
                    "detail": detail,
                    "crawl_transport": "kenar",
                }
            )

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return KenarSearchResult(rows=rows, raw_response=raw, searched_at_utc=stamp)

    def get_post(self, token: str) -> dict[str, Any]:
        return self._request(
            "GET",
            KENAR_GET_POST_URL.format(token=token),
            retries=1,
        )

    def write_raw_snapshot(self, result: KenarSearchResult) -> str:
        path = self.raw_dir / f"kenar_raw_{result.searched_at_utc}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in result.rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return str(path)
