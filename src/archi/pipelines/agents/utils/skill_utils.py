"""
Utility for loading skill files that provide domain-specific knowledge to agent tools.

Skills are markdown files stored in a configurable directory.  They are appended to
tool descriptions so the LLM has context about field names, query patterns, and
domain conventions.

The skills directory is resolved from ``services.chat_app.skills_dir`` in the
runtime config.  If not set, the skill cannot be loaded and ``None`` is returned.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.logging import get_logger

logger = get_logger(__name__)


def _resolve_skills_dir(config: Dict[str, Any]) -> Optional[Path]:
    """Resolve the skills directory from the services.chat_app config."""
    chat_cfg = config.get("services", {}).get("chat_app", {})
    skills_dir = chat_cfg.get("skills_dir")
    if skills_dir:
        return Path(skills_dir)

    return None


def load_skill(skill_name: str, config: Dict[str, Any]) -> Optional[str]:
    """
    Load a skill markdown file by name from the configured skills directory.

    The skills directory is read from ``services.chat_app.skills_dir`` in the
    runtime config.  Returns ``None`` if the directory is not configured, the
    skill file doesn't exist, or it cannot be read.

    Args:
        skill_name: Name of the skill file (without .md extension).
        config: Runtime config dict (from ``get_full_config()``).

    Returns:
        Skill content as string, or ``None`` if not found.
    """
    skills_dir = _resolve_skills_dir(config)
    if skills_dir is None:
        logger.warning(
            "No skills_dir configured in services.chat_app.skills_dir; "
            "cannot load skill '%s'",
            skill_name,
        )
        return None

    skill_path = skills_dir / f"{skill_name}.md"
    if not skill_path.exists():
        logger.warning("Skill file not found: %s", skill_path)
        return None

    try:
        content = skill_path.read_text(encoding="utf-8")
        logger.info("Loaded skill '%s' from %s (%d chars)", skill_name, skill_path, len(content))
        return content
    except Exception as e:
        logger.error("Failed to read skill file %s: %s", skill_path, e)
        return None
