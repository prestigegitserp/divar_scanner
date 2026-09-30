from __future__ import annotations

import importlib
import json
import platform
import sys
from pathlib import Path

import requests


def collect() -> dict:
    import divar_scanner

    packages = {}
    for name in (
        "numpy",
        "pandas",
        "sklearn",
        "pyarrow",
        "plotly",
        "yaml",
        "requests",
        "torch",
        "transformers",
        "sentencepiece",
        "laya",
    ):
        try:
            mod = importlib.import_module(name)
            packages[name] = getattr(mod, "__version__", "installed")
        except Exception as exc:
            packages[name] = f"NOT_AVAILABLE: {type(exc).__name__}: {exc}"

    package_path = str(Path(divar_scanner.__file__).resolve())
    decision_ready = all(
        not str(packages[name]).startswith("NOT_AVAILABLE")
        for name in ("torch", "transformers", "sentencepiece", "laya")
    )
    gpu = {}
    if decision_ready:
        import torch

        gpu = {
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "sys_path_head": sys.path[:6],
        "divar_scanner_file": package_path,
        "package_origin_ok": "/src/divar_scanner/" in package_path.replace("\\", "/"),
        "decision_dependencies_ready": decision_ready,
        "packages": packages,
        "gpu": gpu,
    }


def network_check(timeout: int = 10) -> dict:
    out = {}
    checks = {
        "divar_web_fatemi": (
            "https://divar.ir/s/tehran/rent-apartment/fatemi",
            min(timeout, 10),
        ),
        # Optional legacy transport. Keep this probe short: cloud/Colab IPs may
        # simply be unable to connect to api.divar.ir.
        "divar_api_districts_optional": (
            "https://api.divar.ir/v8/places/cities/1/districts",
            min(timeout, 4),
        ),
        "laya_model_card": (
            "https://huggingface.co/convaiinnovations/laya/resolve/main/multilingual/config.json",
            min(timeout, 10),
        ),
    }
    for name, (url, check_timeout) in checks.items():
        try:
            r = requests.get(
                url,
                timeout=check_timeout,
                headers={"User-Agent": "divar-scanner-doctor/0.4.1"},
            )
            out[name] = {
                "status": r.status_code,
                "content_type": r.headers.get("content-type", ""),
                "timeout_seconds": check_timeout,
            }
        except Exception as exc:
            out[name] = {
                "error": f"{type(exc).__name__}: {exc}",
                "timeout_seconds": check_timeout,
            }
    out["interpretation"] = {
        "preferred_crawl_transport": "server-rendered web page",
        "api_note": (
            "api.divar.ir is optional for the default Fatemi Colab flow. "
            "A timeout there does not imply that the public divar.ir search page is unavailable."
        ),
    }
    return out


def main() -> None:
    data = collect()
    if "--network" in sys.argv:
        data["network"] = network_check()
    print(json.dumps(data, ensure_ascii=False, indent=2))
    if not data["package_origin_ok"]:
        raise SystemExit(
            "\nERROR: divar_scanner is being imported from the wrong path. "
            "In Colab clone the repo to /content/divar_scanner_repo (not /content/divar_scanner) "
            "and put /content/divar_scanner_repo/src at the front of sys.path."
        )


if __name__ == "__main__":
    main()
