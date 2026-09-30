from __future__ import annotations

import json
import zipfile
from pathlib import Path

from divar_scanner.snapshot import load_snapshot


def _html(token: str, area: int, rooms: int) -> str:
    state = {
        "nb": {
            "listWidgets": [
                {
                    "data": {
                        "dto": {
                            "data": {
                                "title": f"آپارتمان {area} متری {rooms} خواب",
                                "top_description_text": "ودیعه ۸۰۰ میلیون تومان",
                                "middle_description_text": "اجاره ۱۵ میلیون تومان",
                                "bottom_description_text": "فاطمی",
                                "action": {
                                    "payload": {
                                        "token": token,
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
        "url": f"https://divar.ir/v/test/{token}",
        "name": f"آپارتمان {area} متری {rooms} خواب",
        "description": f"واحد {area} متری {rooms} خواب در فاطمی",
        "floorSize": {"value": area},
        "numberOfRooms": rooms,
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


def test_saved_browser_html_is_a_first_class_snapshot(tmp_path: Path):
    page = tmp_path / "fatemi.html"
    page.write_text(_html("ABCdef123", 85, 2), encoding="utf-8")

    df, meta = load_snapshot(page)

    assert meta["format"] == "browser_html"
    assert len(df) == 1
    row = df.iloc[0]
    assert row["token"] == "ABCdef123"
    assert row["neighborhood"] == "فاطمی"
    assert row["area_m2"] == 85
    assert row["rooms"] == 2
    assert row["deposit_toman"] == 800_000_000
    assert row["rent_monthly_toman"] == 15_000_000


def test_snapshot_directory_merges_and_deduplicates_pages(tmp_path: Path):
    (tmp_path / "p1.html").write_text(_html("ABCdef123", 85, 2), encoding="utf-8")
    (tmp_path / "p2.html").write_text(_html("XYZghi456", 110, 3), encoding="utf-8")
    (tmp_path / "duplicate.html").write_text(_html("ABCdef123", 85, 2), encoding="utf-8")

    df, meta = load_snapshot(tmp_path)

    assert meta["format"] == "snapshot_directory"
    assert len(df) == 2
    assert set(df["token"]) == {"ABCdef123", "XYZghi456"}


def test_snapshot_zip_accepts_multiple_browser_pages(tmp_path: Path):
    archive = tmp_path / "fatemi_pages.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("page1.html", _html("ABCdef123", 85, 2))
        zf.writestr("page2.html", _html("XYZghi456", 110, 3))

    df, meta = load_snapshot(archive)

    assert meta["format"] == "snapshot_zip"
    assert len(df) == 2
    assert set(df["token"]) == {"ABCdef123", "XYZghi456"}
