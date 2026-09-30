from __future__ import annotations

import json
from pathlib import Path

from divar_scanner.config import Config
from divar_scanner.crawler import DivarCrawler, extract_server_rendered_cards
from divar_scanner.normalize import normalize_listing


def _html() -> str:
    state = {
        "nb": {
            "listWidgets": [
                {
                    "data": {
                        "dto": {
                            "data": {
                                "title": "آپارتمان ۸۵ متری دو خواب",
                                "top_description_text": "ودیعه ۸۰۰ میلیون تومان",
                                "middle_description_text": "اجاره ۱۵ میلیون تومان",
                                "bottom_description_text": "فاطمی",
                                "action": {
                                    "payload": {
                                        "token": "ABCdef123",
                                        "web_info": {
                                            "district_persian": "فاطمی",
                                            "city_persian": "تهران",
                                        },
                                    }
                                },
                            }
                        }
                    }
                }
            ]
        }
    }
    ld = {
        "@type": "Apartment",
        "url": "https://divar.ir/v/test/ABCdef123",
        "name": "آپارتمان ۸۵ متری دو خواب",
        "description": "واحد ۸۵ متری دو خواب در فاطمی",
        "floorSize": {"value": 85},
        "numberOfRooms": 2,
        "web_info": {
            "district_persian": "فاطمی",
            "city_persian": "تهران",
        },
    }
    return (
        "<html><head>"
        f"<script>window.__PRELOADED_STATE__ = {json.dumps(state, ensure_ascii=False)};</script>"
        f'<script type="application/ld+json">{json.dumps(ld, ensure_ascii=False)}</script>'
        "</head><body></body></html>"
    )


def test_server_rendered_card_parser_and_normalizer():
    cards = extract_server_rendered_cards(_html())
    assert len(cards) == 1
    card = cards[0]
    assert card["token"] == "ABCdef123"
    assert card["district"] == "فاطمی"
    assert card["area_hint"] == 85
    assert card["rooms_hint"] == 2

    row = normalize_listing({"card": card, "detail": card["search_detail"]})
    assert row["neighborhood"] == "فاطمی"
    assert row["area_m2"] == 85
    assert row["rooms"] == 2
    assert row["deposit_toman"] == 800_000_000
    assert row["rent_monthly_toman"] == 15_000_000
    assert "۸۵ متری" in row["description"]


def test_web_transport_never_calls_district_api(monkeypatch, tmp_path: Path):
    cfg = Config(
        raw={
            "crawl": {
                "transport": "web",
                "city_id": "1",
                "city_slug": "tehran",
                "category": "apartment-rent",
                "web_category_slug": "rent-apartment",
                "district_slug": "fatemi",
                "district_query": "فاطمی",
                "max_listings": 20,
                "max_web_pages": 2,
                "request_delay_seconds": 0,
                "jitter_seconds": 0,
                "web_timeout_seconds": 1,
                "cache_dir": str(tmp_path / "cache"),
                "raw_dir": str(tmp_path / "raw"),
            }
        },
        source=tmp_path / "config.yaml",
    )
    crawler = DivarCrawler(cfg)
    monkeypatch.setattr(crawler, "_request_text", lambda _url: _html())

    def forbidden(*_args, **_kwargs):
        raise AssertionError("district API must not be used in web transport")

    monkeypatch.setattr(crawler, "fetch_districts", forbidden)
    rows, meta = crawler.crawl()

    assert len(rows) == 1
    assert meta["crawl_transport"] == "server_rendered_web"
    assert meta["district_slug"] == "fatemi"
    assert meta["district_ids"] == []
    assert meta["listing_count"] == 1
