from __future__ import annotations

from pathlib import Path

from divar_scanner.config import Config
from divar_scanner.semantic import resolve_providers


def _cfg(provider="auto"):
    return Config(
        raw={
            "semantic": {
                "enabled": True,
                "provider": provider,
                "ensemble_max_providers": 1,
                "models": {"groq": "openai/gpt-oss-20b", "openrouter": "openrouter/free"},
            }
        },
        source=Path("test.yaml"),
    )


def test_auto_provider_prefers_groq_when_only_groq_key(monkeypatch):
    for name in [
        "JEV_API_KEY", "TYPESAFE_API_KEY", "CEREBRAS_API_KEY", "GEMINI_API_KEY",
        "COHERE_API_KEY", "OPENROUTER_API_KEY", "HF_TOKEN", "OPENAI_COMPAT_API_KEY",
        "OPENAI_COMPAT_BASE_URL", "OPENAI_COMPAT_MODEL", "SEMANTIC_PROVIDER",
    ]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    providers = resolve_providers(_cfg())
    assert [p.name for p in providers] == ["groq"]


def test_explicit_provider_does_not_fall_through(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "g")
    monkeypatch.setenv("OPENROUTER_API_KEY", "o")
    monkeypatch.setenv("SEMANTIC_PROVIDER", "openrouter")
    providers = resolve_providers(_cfg())
    assert [p.name for p in providers] == ["openrouter"]
