"""Fixture third-party package: one pack, one bundle."""
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def packs_dir() -> Path:
    return _HERE / "packs"


def bundles_dir() -> Path:
    return _HERE / "bundles"
