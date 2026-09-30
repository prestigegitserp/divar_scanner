"""Backward-compatibility shim.

The project no longer uses generative semantic providers. The decision layer is a
non-generative Jev-style NLI classifier implemented in :mod:`divar_scanner.decision`.
"""

from .decision import apply_decisions as apply_semantic

__all__ = ["apply_semantic"]
