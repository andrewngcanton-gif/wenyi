"""Isolated WP1a evaluation harness for Wenyi variants.

The package is intentionally independent from the repository workspace: it owns
its uv environment and lock, reads product state only through public storage
APIs, and never imports the current working tree as the old A baseline.
"""

__all__ = ["__version__"]

__version__ = "0.0.0"
