from __future__ import annotations

from pathlib import Path

import pytest

from divar_scanner.config import Config
from divar_scanner.kenar import KenarAuthError, KenarClient
from divar_scanner.normalize import normalize_listing


class FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.text = ""

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def _cfg(tmp_path: Path) -> Config:
    return Config(
        raw={
            "crawl": {
                "category": "apartment-rent",
                "city_slug": "tehran",
                "district_slug": "fatemi",
                "max_listings": 100,
                "kenar_timeout_seconds": 3,
                "kenar_enrich_details": False,
                "raw_dir": str(tmp_path / "raw"),
            }
        },
        source=tmp_path / "config.yaml",
    )


def test_kenar_requires_runtime_key(monkeypatch, tmp_path):
    monkeypatch.delenv("KENAR_API_KEY", raising=False)
    client = KenarClient(_cfg(tmp_path))
    assert client.available is False
    with pytest.raises(KenarAuthError):
        client.search()


def test_kenar_search_maps_real_estate_fields(monkeypatch, tmp_path):
    monkeypatch.setenv("KENAR_API_KEY", "secret")
    client = KenarClient(_cfg(tmp_path))

    payload = {
        "posts": [
            {
                "token": "AbCd1234",
                "title": "آپارتمان ۹۰ متری دو خواب",
                "city": "tehran",
                "category": "apartment-rent",
                "real_estate_fields": {
                    "credit": {"mode": "fixed", "value": "900000000"},
                    "rent": {"mode": "fixed", "value": "18000000"},
                    "size": 90,
                    "rooms": "دو",
                    "year": 1398,
                    "floor": 3,
                    "has_parking": True,
                    "has_elevator": True,
                },
            }
        ]
    }

    def fake_request(method, url, headers=None, json=None, timeout=None):
        assert method == "POST"
        assert headers["X-API-Key"] == "secret"
        assert json["category"] == "apartment-rent"
        assert json["city"] == "tehran"
        assert json["districts"] == ["fatemi"]
        return FakeResponse(200, payload)

    monkeypatch.setattr(client.session, "request", fake_request)
    result = client.search()

    assert len(result.rows) == 1
    raw = result.rows[0]
    assert raw["crawl_transport"] == "kenar"

    row = normalize_listing(raw)
    assert row["area_m2"] == 90
    assert row["rooms"] == 2
    assert row["year_built_shamsi"] == 1398
    assert row["deposit_toman"] == 900_000_000
    assert row["rent_monthly_toman"] == 18_000_000
    assert row["parking"] is True
    assert row["elevator"] is True
    assert row["floor"] == "3"
