from __future__ import annotations

from pathlib import Path

import pytest

from divar_scanner.browser_crawler import (
    _safe_public_search_url,
    normalize_dom_cards,
)
from divar_scanner.config import Config


def _cfg(tmp_path: Path, url: str) -> Config:
    return Config(
        raw={"crawl": {"search_page_url": url}},
        source=tmp_path / "config.yaml",
    )


def test_normalize_dom_cards_extracts_public_card_hints():
    rows = [
        {
            "href": "/v/test/AbCd1234",
            "title": "",
            "text": (
                "آپارتمان ۸۵ متری دو خواب\n"
                "ودیعه ۸۰۰ میلیون تومان\n"
                "اجاره ۱۵ میلیون تومان\n"
                "فاطمی"
            ),
        }
    ]
    cards = normalize_dom_cards(rows)
    assert len(cards) == 1
    card = cards[0]
    assert card["token"] == "AbCd1234"
    assert card["area_hint"] == 85
    assert card["rooms_hint"] == 2
    assert "۸۰۰ میلیون" in card["deposit_text"]
    assert "۱۵ میلیون" in card["rent_text"]


def test_normalize_dom_cards_deduplicates_tokens():
    rows = [
        {"href": "/v/a/AbCd1234", "text": "اول"},
        {"href": "/v/b/AbCd1234", "text": "دوم"},
    ]
    cards = normalize_dom_cards(rows)
    assert len(cards) == 1


def test_browser_search_url_rejects_non_divar_hosts(tmp_path):
    with pytest.raises(ValueError):
        _safe_public_search_url(_cfg(tmp_path, "https://example.com/s/x"))


def test_browser_search_url_accepts_public_divar_search(tmp_path):
    url = "https://divar.ir/s/tehran/rent-apartment/fatemi"
    assert _safe_public_search_url(_cfg(tmp_path, url)) == url
