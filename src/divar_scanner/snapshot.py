from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .crawler import extract_server_rendered_cards
from .normalize import normalize_many


SUPPORTED_EXTENSIONS = {".html", ".htm", ".jsonl", ".ndjson", ".csv", ".parquet"}


def _raw_from_html_text(document: str) -> list[dict[str, Any]]:
    cards = extract_server_rendered_cards(document)
    return [
        {
            "card": card,
            "detail": card.get("search_detail") or {},
            "crawl_transport": "browser_snapshot_html",
        }
        for card in cards
    ]


def _raw_from_jsonl_text(text: str, *, source: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {source}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(
                f"Raw JSONL row {line_number} in {source} must be an object, "
                f"got {type(value).__name__}"
            )
        rows.append(value)
    return rows


def _dedupe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    key = None
    for candidate in ("token", "url"):
        if candidate in out and out[candidate].astype(str).str.len().gt(0).any():
            key = candidate
            break
    if key:
        out = out.drop_duplicates(subset=[key], keep="last")
    else:
        out = out.drop_duplicates()
    return out.reset_index(drop=True)


def _normalize_raw(
    raw: list[dict[str, Any]],
    *,
    redact_phones: bool,
) -> pd.DataFrame:
    return _dedupe(normalize_many(raw, redact_phones=redact_phones))


def _read_one_path(
    path: Path,
    *,
    redact_phones: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return _dedupe(pd.read_parquet(path)), {"format": "parquet", "raw": False}
    if suffix == ".csv":
        return _dedupe(pd.read_csv(path)), {"format": "csv", "raw": False}
    if suffix in {".jsonl", ".ndjson"}:
        raw = _raw_from_jsonl_text(path.read_text(encoding="utf-8"), source=str(path))
        return _normalize_raw(raw, redact_phones=redact_phones), {
            "format": "raw_jsonl",
            "raw": True,
        }
    if suffix in {".html", ".htm"}:
        raw = _raw_from_html_text(path.read_text(encoding="utf-8", errors="replace"))
        return _normalize_raw(raw, redact_phones=redact_phones), {
            "format": "browser_html",
            "raw": True,
        }
    raise ValueError(
        f"Unsupported snapshot file: {path}. "
        f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}, .zip"
    )


def _read_zip(
    path: Path,
    *,
    redact_phones: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    members_meta: list[dict[str, Any]] = []
    with zipfile.ZipFile(path) as zf:
        for name in sorted(zf.namelist()):
            if name.endswith("/"):
                continue
            suffix = Path(name).suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                continue
            data = zf.read(name)
            if suffix == ".parquet":
                frame = pd.read_parquet(io.BytesIO(data))
                meta = {"name": name, "format": "parquet", "raw": False}
            elif suffix == ".csv":
                frame = pd.read_csv(io.BytesIO(data))
                meta = {"name": name, "format": "csv", "raw": False}
            elif suffix in {".jsonl", ".ndjson"}:
                raw = _raw_from_jsonl_text(
                    data.decode("utf-8", errors="replace"),
                    source=f"{path}!{name}",
                )
                frame = normalize_many(raw, redact_phones=redact_phones)
                meta = {"name": name, "format": "raw_jsonl", "raw": True}
            else:
                raw = _raw_from_html_text(data.decode("utf-8", errors="replace"))
                frame = normalize_many(raw, redact_phones=redact_phones)
                meta = {"name": name, "format": "browser_html", "raw": True}
            frames.append(frame)
            members_meta.append(meta)

    if not frames:
        raise ValueError(
            f"No supported snapshot files were found inside {path.name}. "
            "Put HTML/JSONL/CSV/Parquet files in the ZIP."
        )
    return _dedupe(pd.concat(frames, ignore_index=True, sort=False)), {
        "format": "snapshot_zip",
        "raw": any(x["raw"] for x in members_meta),
        "members": members_meta,
    }


def load_snapshot(
    input_path: str | Path,
    *,
    redact_phones: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load offline acquisition artifacts without contacting Divar.

    Accepted inputs:
      - a saved Divar search page (.html/.htm)
      - raw crawler snapshots (.jsonl/.ndjson)
      - normalized CSV/Parquet
      - a ZIP containing any mixture of the above
      - a directory containing supported files

    Multiple artifacts are merged and deduplicated by token (or URL).
    """
    path = Path(input_path)
    if path.is_dir():
        frames: list[pd.DataFrame] = []
        items: list[dict[str, Any]] = []
        for child in sorted(path.iterdir()):
            if child.is_file() and (
                child.suffix.lower() in SUPPORTED_EXTENSIONS or child.suffix.lower() == ".zip"
            ):
                frame, meta = load_snapshot(child, redact_phones=redact_phones)
                frames.append(frame)
                items.append({"path": str(child), **meta})
        if not frames:
            raise ValueError(f"No supported snapshot files found in directory: {path}")
        return _dedupe(pd.concat(frames, ignore_index=True, sort=False)), {
            "format": "snapshot_directory",
            "items": items,
        }

    if path.suffix.lower() == ".zip":
        return _read_zip(path, redact_phones=redact_phones)

    frame, meta = _read_one_path(path, redact_phones=redact_phones)
    return frame, {"path": str(path), **meta}
