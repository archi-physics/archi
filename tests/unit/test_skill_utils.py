"""Unit tests for skill_utils: load_skill and save_skill."""

import importlib.util
import os
import sys
import types
import pytest
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Bootstrap: same pattern as test_oracle_tools.py
# ---------------------------------------------------------------------------

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

_PACKAGES_TO_STUB = [
    "src",
    "src.archi",
    "src.archi.pipelines",
    "src.archi.pipelines.agents",
    "src.archi.pipelines.agents.utils",
    "src.utils",
]

for _pkg in _PACKAGES_TO_STUB:
    if _pkg not in sys.modules:
        mod = types.ModuleType(_pkg)
        mod.__path__ = [os.path.join(_ROOT, *_pkg.split("."))]
        mod.__package__ = _pkg
        sys.modules[_pkg] = mod


def _load_module(name: str, filepath: str):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_src = os.path.join(_ROOT, "src")

_load_module("src.utils.logging", os.path.join(_src, "utils", "logging.py"))

_skill_utils = _load_module(
    "src.archi.pipelines.agents.utils.skill_utils",
    os.path.join(_src, "archi", "pipelines", "agents", "utils", "skill_utils.py"),
)

load_skill = _skill_utils.load_skill
save_skill = _skill_utils.save_skill


# =============================================================================
# Tests for save_skill
# =============================================================================


class TestSaveSkill:
    """Test save_skill function."""

    def test_save_to_writable_directory(self, tmp_path):
        config = {"services": {"chat_app": {"skills_dir": str(tmp_path)}}}
        result = save_skill("test_skill", "# Test\nContent here.", config)
        assert result is True
        saved = (tmp_path / "test_skill.md").read_text(encoding="utf-8")
        assert saved == "# Test\nContent here."

    def test_creates_directory_if_missing(self, tmp_path):
        skills_dir = tmp_path / "nested" / "skills"
        config = {"services": {"chat_app": {"skills_dir": str(skills_dir)}}}
        result = save_skill("oracle_databases", "content", config)
        assert result is True
        assert (skills_dir / "oracle_databases.md").exists()

    def test_no_skills_dir_configured(self):
        config = {"services": {"chat_app": {}}}
        result = save_skill("test_skill", "content", config)
        assert result is False

    def test_empty_config(self):
        result = save_skill("test_skill", "content", {})
        assert result is False

    def test_write_failure(self, tmp_path):
        """Simulate a write failure by making skills_dir a file instead of directory."""
        fake_dir = tmp_path / "not_a_dir"
        fake_dir.write_text("I'm a file")
        config = {"services": {"chat_app": {"skills_dir": str(fake_dir / "subdir")}}}
        result = save_skill("test_skill", "content", config)
        assert result is False

    def test_overwrite_existing(self, tmp_path):
        config = {"services": {"chat_app": {"skills_dir": str(tmp_path)}}}
        (tmp_path / "my_skill.md").write_text("old content")
        result = save_skill("my_skill", "new content", config)
        assert result is True
        assert (tmp_path / "my_skill.md").read_text() == "new content"


class TestLoadSkill:
    """Test load_skill function."""

    def test_load_existing_skill(self, tmp_path):
        (tmp_path / "rucio_events.md").write_text("# Rucio\nEvents schema.")
        config = {"services": {"chat_app": {"skills_dir": str(tmp_path)}}}
        result = load_skill("rucio_events", config)
        assert result == "# Rucio\nEvents schema."

    def test_skill_not_found(self, tmp_path):
        config = {"services": {"chat_app": {"skills_dir": str(tmp_path)}}}
        result = load_skill("nonexistent", config)
        assert result is None

    def test_no_skills_dir(self):
        config = {"services": {"chat_app": {}}}
        result = load_skill("test", config)
        assert result is None
