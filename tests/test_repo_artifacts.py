from __future__ import annotations

import json
from pathlib import Path

from divar_scanner.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_repository_fatemi_config_parses_cleanly():
    cfg = load_config(ROOT / "config" / "fatemi.yaml")
    assert cfg.get("crawl.transport") == "snapshot"
    assert cfg.get("crawl.district_slug") == "fatemi"
    assert cfg.get("crawl.web_category_slug") == "rent-apartment"
    assert cfg.get("decision.backend") == "laya-multilingual"
    assert cfg.get("decision.model_override") is None
    assert cfg.get("decision.permutation_passes") == 2
    assert cfg.get("decision.min_order_stability") == 0.80
    assert cfg.get("decision.trust_stage") == "bootstrap"
    assert cfg.get("scoring.bootstrap_decision_multiplier") == 0.25
    assert cfg.get("scoring.misleading_decision_weight") == 0.58
    assert cfg.get("decision.ensemble_backends") == [
        "mdeberta-nli",
        "parsbert-parsinlu",
    ]
    assert cfg.get("features.rent_to_deposit_multipliers") == [25.0, 30.0, 35.0]
    assert cfg.get("semantic") is None


def test_colab_notebook_avoids_package_namespace_collision_and_compiles():
    path = ROOT / "colab" / "Divar_Fatemi_Anomaly_Scanner.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]
    code = "\n".join(code_cells)
    assert "/content/divar_scanner_repo" in code
    assert "REPO = '/content/divar_scanner_repo'" in code
    assert "endswith('/src/divar_scanner/__init__.py')" in code
    assert "laya-multilingual" in code
    assert "ACQUISITION_MODE = 'snapshot'" in code
    assert "snapshot" in code
    assert ".html" in code
    assert ".zip" in code
    assert "mdeberta-nli" in code
    assert "local-qwen" not in code
    for i, source in enumerate(code_cells):
        compile(source, f"colab-cell-{i}", "exec")
