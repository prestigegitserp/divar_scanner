from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .pipeline import run_pipeline


def _patch_config(path: str, max_listings: int | None, no_jev: bool) -> str:
    if max_listings is None and not no_jev:
        return path
    import yaml

    cfg = load_config(path).raw
    if max_listings is not None:
        cfg.setdefault("crawl", {})["max_listings"] = int(max_listings)
    if no_jev:
        cfg.setdefault("semantic", {})["enabled"] = False
    tmp = Path(".divar_scanner.runtime.yaml")
    tmp.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return str(tmp)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Divar rental anomaly/suspicion scanner")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="crawl Divar and run the full analysis pipeline")
    run.add_argument("--config", default="config/fatemi.yaml")
    run.add_argument("--max-listings", type=int, default=None)
    run.add_argument("--no-jev", action="store_true", help="disable optional Jev semantic review")

    analyze = sub.add_parser("analyze", help="analyze an existing normalized CSV/Parquet file")
    analyze.add_argument("--config", default="config/fatemi.yaml")
    analyze.add_argument("--input", required=True)
    analyze.add_argument("--no-jev", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = _patch_config(args.config, getattr(args, "max_listings", None), args.no_jev)
    if args.command == "run":
        df, meta = run_pipeline(cfg, crawl=True)
    else:
        df, meta = run_pipeline(cfg, crawl=False, input_path=args.input)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    if not df.empty:
        cols = [c for c in ["suspicion_score", "risk_band", "title", "area_m2", "deposit_toman", "rent_monthly_toman", "flag_reasons"] if c in df]
        print("\nTop review candidates:\n")
        print(df[cols].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
