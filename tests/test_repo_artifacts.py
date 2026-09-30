from __future__ import annotations

import json
from pathlib import Path

from divar_scanner.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_repository_fatemi_config_parses_cleanly():
    cfg = load_config(ROOT / "config" / "fatemi.yaml")
    assert cfg.get("semantic.models.local_qwen") == "Qwen/Qwen3-4B"
    assert cfg.get("semantic.models.local_aya") == "CohereLabs/aya-expanse-8b"
    priority = cfg.get("semantic.provider_priority")
    assert isinstance(priority, list)
    assert "cohere" in priority
    assert all("\\n" not in str(x) for x in priority)


def test_colab_notebook_avoids_package_namespace_collision():
    path = ROOT / "colab" / "Divar_Fatemi_Anomaly_Scanner.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    assert "/content/divar_scanner_repo" in code
    assert "REPO = '/content/divar_scanner_repo'" in code
    assert "endswith('/src/divar_scanner/__init__.py')" in code
    assert "local-qwen" in code
