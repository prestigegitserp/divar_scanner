from __future__ import annotations

import argparse
import json
from pathlib import Path

from .calibration import calibrate_file
from .config import load_config
from .pipeline import run_pipeline


def _patch_config(
    path: str,
    max_listings: int | None,
    no_decision: bool,
    decision_backend: str | None,
) -> str:
    if max_listings is None and not no_decision and decision_backend is None:
        return path
    import yaml

    cfg = load_config(path).raw
    if max_listings is not None:
        cfg.setdefault("crawl", {})["max_listings"] = int(max_listings)
    if no_decision:
        cfg.setdefault("decision", {})["enabled"] = False
    if decision_backend:
        cfg.setdefault("decision", {})["backend"] = decision_backend
    tmp = Path(".divar_scanner.runtime.yaml")
    tmp.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return str(tmp)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="config/fatemi.yaml")
    parser.add_argument(
        "--no-decision",
        "--no-jev",
        dest="no_decision",
        action="store_true",
        help="disable the bounded Jev-style decision layer (legacy --no-jev alias kept)",
    )
    parser.add_argument(
        "--decision-backend",
        default=None,
        choices=[
            "laya-multilingual",
            "mdeberta-nli",
            "parsbert-parsinlu",
            "mbert-parsinlu",
            "persian-ensemble",
        ],
        help="bounded non-generative decision backend",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Divar rental anomaly scanner with bounded non-generative decisions"
    )
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="crawl Divar and run the full analysis pipeline")
    _add_common_args(run)
    run.add_argument("--max-listings", type=int, default=None)

    analyze = sub.add_parser("analyze", help="analyze an existing normalized CSV/Parquet file")
    _add_common_args(analyze)
    analyze.add_argument("--input", required=True)

    calibrate = sub.add_parser(
        "calibrate",
        help="fit decision temperatures from a human-labelled review CSV/Parquet",
    )
    calibrate.add_argument("--input", required=True)
    calibrate.add_argument("--output", default="config/fatemi_calibration.json")
    calibrate.add_argument("--min-samples", type=int, default=12)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "calibrate":
        result = calibrate_file(
            args.input,
            args.output,
            min_samples=args.min_samples,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    cfg = _patch_config(
        args.config,
        getattr(args, "max_listings", None),
        args.no_decision,
        args.decision_backend,
    )
    if args.command == "run":
        df, meta = run_pipeline(cfg, crawl=True)
    else:
        df, meta = run_pipeline(cfg, crawl=False, input_path=args.input)

    print(json.dumps(meta, ensure_ascii=False, indent=2))
    if not df.empty:
        cols = [
            c
            for c in [
                "review_priority_score",
                "suspicion_score",
                "risk_band",
                "decision_disposition",
                "decision_bait_probability",
                "title",
                "area_m2",
                "deposit_toman",
                "rent_monthly_toman",
                "flag_reasons",
            ]
            if c in df
        ]
        print("\nTop review candidates:\n")
        print(df[cols].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
