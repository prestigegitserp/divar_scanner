from __future__ import annotations

import importlib
import json
import os
import platform
import sys
from pathlib import Path

import requests


def collect() -> dict:
    import divar_scanner

    packages = {}
    for name in ("numpy", "pandas", "sklearn", "pyarrow", "plotly", "yaml", "requests"):
        try:
            mod = importlib.import_module(name)
            packages[name] = getattr(mod, "__version__", "installed")
        except Exception as exc:
            packages[name] = f"ERROR: {type(exc).__name__}: {exc}"

    provider_env = {
        name: bool(os.getenv(name))
        for name in (
            "JEV_API_KEY", "TYPESAFE_API_KEY", "GROQ_API_KEY", "CEREBRAS_API_KEY",
            "GEMINI_API_KEY", "OPENROUTER_API_KEY", "HF_TOKEN",
            "OPENAI_COMPAT_API_KEY",
        )
    }
    package_path = str(Path(divar_scanner.__file__).resolve())
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "sys_path_head": sys.path[:6],
        "divar_scanner_file": package_path,
        "package_origin_ok": "/src/divar_scanner/" in package_path.replace("\\", "/"),
        "packages": packages,
        "provider_keys_present": provider_env,
    }


def network_check(timeout: int = 12) -> dict:
    out = {}
    for name, url in {
        "divar_districts": "https://api.divar.ir/v8/places/cities/1/districts",
    }.items():
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": "divar-scanner-doctor/0.2"})
            out[name] = {"status": r.status_code, "content_type": r.headers.get("content-type", "")}
        except Exception as exc:
            out[name] = {"error": f"{type(exc).__name__}: {exc}"}
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
