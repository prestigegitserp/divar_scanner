from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

from .config import Config
from .crawler import (
    DIVAR_WEB_BASE,
    DivarBlockedError,
    _jsonld_details,
    _token_from_url,
)

_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")

_BLOCK_HINTS = (
    "captcha",
    "کپچا",
    "دسترسی شما محدود",
    "تعداد درخواست",
    "خطای 403",
)


@dataclass
class BrowserCard:
    token: str
    url: str
    title: str
    text: str


def _first_title_line(text: str) -> str:
    lines = [x.strip() for x in str(text or "").splitlines() if x.strip()]
    for line in lines:
        if len(line) >= 4 and not any(
            marker in line for marker in ("ودیعه", "اجاره", "لحظاتی پیش", "دقایقی پیش")
        ):
            return line
    return lines[0] if lines else ""


def _line_with(text: str, needles: tuple[str, ...]) -> str:
    for line in [x.strip() for x in str(text or "").splitlines() if x.strip()]:
        if any(n in line for n in needles):
            return line
    return ""


def _area_hint(text: str) -> int | None:
    s = str(text or "").translate(_DIGITS)
    m = re.search(r"(?<!\d)(\d{2,4})\s*(?:متر|متری)", s)
    return int(m.group(1)) if m else None


def _rooms_hint(text: str) -> int | None:
    s = str(text or "").translate(_DIGITS)
    m = re.search(r"(?<!\d)(\d{1,2})\s*(?:خواب|اتاق)", s)
    if m:
        return int(m.group(1))
    words = {"بدون اتاق": 0, "یک خواب": 1, "دو خواب": 2, "سه خواب": 3, "چهار خواب": 4}
    for phrase, value in words.items():
        if phrase in s:
            return value
    return None


def normalize_dom_cards(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize visible Divar listing anchors collected by the browser."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        href = str(row.get("href") or "").strip()
        if not href:
            continue
        url = urljoin(DIVAR_WEB_BASE, href)
        token = _token_from_url(url)
        if not token or token in seen:
            continue
        seen.add(token)
        text = str(row.get("text") or "").strip()
        title = str(row.get("title") or "").strip() or _first_title_line(text)
        out.append(
            {
                "token": token,
                "url": url,
                "title": title,
                "description_hint": text,
                "area_hint": _area_hint(title + "\n" + text),
                "rooms_hint": _rooms_hint(title + "\n" + text),
                "deposit_text": _line_with(text, ("ودیعه", "رهن")),
                "rent_text": _line_with(text, ("اجاره",)),
                "bottom_text": text,
                "subtitle": text,
                "raw_card": {"browser_text": text},
            }
        )
    return out


def _safe_public_search_url(config: Config) -> str:
    raw = str(config.get("crawl.search_page_url", "") or "").strip()
    if not raw:
        city = str(config.get("crawl.city_slug", "tehran"))
        category = str(config.get("crawl.web_category_slug", "rent-apartment"))
        district = str(config.get("crawl.district_slug", "fatemi"))
        raw = f"{DIVAR_WEB_BASE}/s/{city}/{category}/{district}"
    parsed = urlparse(raw)
    if parsed.scheme != "https" or parsed.netloc not in {"divar.ir", "www.divar.ir"}:
        raise ValueError("Browser crawler only accepts public https://divar.ir search URLs")
    if not parsed.path.startswith("/s/"):
        raise ValueError("Browser crawler search URL must point to a Divar /s/ page")
    return raw


class BrowserDivarCrawler:
    """Automatic public-web crawler driven by a real Chromium browser.

    It uses the same public search and listing pages a normal browser sees. It does not
    log in, request contact information, solve CAPTCHAs, rotate proxies, spoof browser
    fingerprints, or continue after explicit access-control/rate-limit responses.
    """

    def __init__(self, config: Config):
        self.config = config
        c = config.section("crawl")
        self.search_url = _safe_public_search_url(config)
        self.max_listings = int(c.get("max_listings", 200))
        self.max_scrolls = int(c.get("browser_max_scrolls", 80))
        self.stall_rounds = int(c.get("browser_stall_rounds", 5))
        self.scroll_delay = float(c.get("browser_scroll_delay_seconds", 1.0))
        self.detail_delay = float(c.get("browser_detail_delay_seconds", 1.4))
        self.detail_jitter = float(c.get("browser_detail_jitter_seconds", 0.35))
        self.navigation_timeout_ms = int(c.get("browser_navigation_timeout_ms", 35_000))
        self.enrich_details = bool(c.get("browser_enrich_details", True))
        self.max_detail_pages = min(
            int(c.get("browser_max_detail_pages", self.max_listings)),
            self.max_listings,
        )
        self.headless = bool(c.get("browser_headless", True))
        self.channel = str(c.get("browser_channel", "") or "").strip() or None
        self.raw_dir = Path(c.get("raw_dir", "data/raw"))
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def _launch(self, playwright):
        kwargs: dict[str, Any] = {"headless": self.headless}
        if self.channel:
            kwargs["channel"] = self.channel
        return playwright.chromium.launch(**kwargs)

    @staticmethod
    def _assert_not_blocked(status: int | None, body_text: str) -> None:
        if status in {401, 403, 429}:
            raise DivarBlockedError(
                f"Divar returned HTTP {status}. Browser crawling stopped; no bypass attempted."
            )
        normalized = str(body_text or "").lower()
        if any(hint in normalized for hint in _BLOCK_HINTS):
            raise DivarBlockedError(
                "Divar appears to have returned an access-control/rate-limit page. "
                "Browser crawling stopped; no CAPTCHA/access-control bypass attempted."
            )

    @staticmethod
    def _collect_dom_rows(page) -> list[dict[str, Any]]:
        return page.eval_on_selector_all(
            'a[href*="/v/"]',
            """
            els => els.map(a => {
              const root = a.closest('article, [class*="post-card"], [class*="kt-post"]') || a;
              return {
                href: a.getAttribute('href') || a.href || '',
                title: (a.getAttribute('title') || '').trim(),
                text: ((root.innerText || a.innerText || '') + '').trim()
              };
            })
            """,
        )

    def _collect_search_cards(self, page) -> list[dict[str, Any]]:
        response = page.goto(
            self.search_url,
            wait_until="domcontentloaded",
            timeout=self.navigation_timeout_ms,
        )
        status = response.status if response else None
        try:
            page.wait_for_selector('a[href*="/v/"]', timeout=self.navigation_timeout_ms)
        except Exception:
            body = page.locator("body").inner_text(timeout=5_000)
            self._assert_not_blocked(status, body)
            raise RuntimeError(
                "Divar search page loaded but no public listing links were found. "
                "The DOM may have changed."
            )

        cards: dict[str, dict[str, Any]] = {}
        stalled = 0
        last_count = 0

        for _ in range(max(1, self.max_scrolls)):
            body = page.locator("body").inner_text(timeout=5_000)
            self._assert_not_blocked(status, body)

            for card in normalize_dom_cards(self._collect_dom_rows(page)):
                card["district"] = str(self.config.get("crawl.district_query", "فاطمی"))
                cards[card["token"]] = card
                if len(cards) >= self.max_listings:
                    return list(cards.values())[: self.max_listings]

            if len(cards) == last_count:
                stalled += 1
            else:
                stalled = 0
                last_count = len(cards)
            if stalled >= self.stall_rounds:
                break

            page.evaluate(
                """
                () => window.scrollTo({
                  top: Math.max(document.body.scrollHeight, document.documentElement.scrollHeight),
                  behavior: 'instant'
                })
                """
            )
            page.wait_for_timeout(int(max(self.scroll_delay, 0.2) * 1000))

        return list(cards.values())[: self.max_listings]

    def _enrich_card(self, page, card: dict[str, Any]) -> dict[str, Any]:
        response = page.goto(
            card["url"],
            wait_until="domcontentloaded",
            timeout=self.navigation_timeout_ms,
        )
        status = response.status if response else None
        page.wait_for_timeout(350)
        body = page.locator("body").inner_text(timeout=5_000)
        self._assert_not_blocked(status, body)

        document = page.content()
        details = _jsonld_details(document)
        detail = details.get(card["token"], {})
        if not detail:
            # Preserve visible public text as a fallback for the decision layer.
            detail = {"description": body[:12_000]}
        elif not detail.get("description"):
            detail = dict(detail)
            detail["description"] = body[:12_000]

        return {
            "card": card,
            "detail": detail,
            "crawl_transport": "browser",
        }

    def crawl(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError(
                "Browser crawling requires the optional browser dependencies. Install with "
                "python -m pip install -e '.[browser]' && python -m playwright install chromium"
            ) from exc

        rows: list[dict[str, Any]] = []
        started = datetime.now(timezone.utc)
        with sync_playwright() as p:
            browser = self._launch(p)
            context = browser.new_context(
                locale="fa-IR",
                timezone_id="Asia/Tehran",
                viewport={"width": 1440, "height": 1100},
            )
            search_page = context.new_page()
            search_page.set_default_timeout(self.navigation_timeout_ms)
            cards = self._collect_search_cards(search_page)

            if self.enrich_details and cards:
                detail_page = context.new_page()
                detail_page.set_default_timeout(self.navigation_timeout_ms)
                for i, card in enumerate(cards):
                    if i >= self.max_detail_pages:
                        rows.append(
                            {
                                "card": card,
                                "detail": {},
                                "crawl_transport": "browser",
                            }
                        )
                        continue
                    try:
                        rows.append(self._enrich_card(detail_page, card))
                    except DivarBlockedError:
                        raise
                    except Exception as exc:
                        rows.append(
                            {
                                "card": card,
                                "detail": {},
                                "crawl_error": f"{type(exc).__name__}: {exc}",
                                "crawl_transport": "browser",
                            }
                        )
                    if i + 1 < min(len(cards), self.max_detail_pages):
                        delay = max(
                            0.25,
                            self.detail_delay
                            + random.uniform(-self.detail_jitter, self.detail_jitter),
                        )
                        time.sleep(delay)
            else:
                rows = [
                    {"card": card, "detail": {}, "crawl_transport": "browser"}
                    for card in cards
                ]

            context.close()
            browser.close()

        stamp = started.strftime("%Y%m%dT%H%M%SZ")
        raw_path = self.raw_dir / f"browser_raw_{stamp}.jsonl"
        with raw_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        meta = {
            "crawl_transport": "browser",
            "listing_count": len(rows),
            "raw_path": str(raw_path),
            "source_urls": [self.search_url],
            "crawled_at_utc": stamp,
            "browser_headless": self.headless,
            "detail_enrichment": self.enrich_details,
            "detail_pages_attempted": min(len(rows), self.max_detail_pages)
            if self.enrich_details
            else 0,
            "elapsed_seconds": round(
                (datetime.now(timezone.utc) - started).total_seconds(), 2
            ),
            "safety_note": (
                "Public pages only; stops on explicit block/rate-limit signals; "
                "no login/contact/CAPTCHA/proxy bypass."
            ),
        }
        return rows, meta
