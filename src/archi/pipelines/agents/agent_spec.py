from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import re
import yaml


@dataclass(frozen=True)
class AgentSpec:
    name: str
    tools: List[str]
    prompt: str
    source_path: Path


class AgentSpecError(ValueError):
    pass


def list_agent_files(agents_dir: Path) -> List[Path]:
    if not agents_dir.exists():
        raise AgentSpecError(f"Agents directory not found: {agents_dir}")
    if not agents_dir.is_dir():
        raise AgentSpecError(f"Agents path is not a directory: {agents_dir}")
    return sorted(p for p in agents_dir.iterdir() if p.is_file() and p.suffix.lower() == ".md")


def load_agent_spec(path: Path) -> AgentSpec:
    text = path.read_text()
    frontmatter, prompt = _parse_frontmatter(text, path)
    name, tools = _extract_metadata(frontmatter, path)
    return AgentSpec(
        name=name,
        tools=tools,
        prompt=prompt,
        source_path=path,
    )


def load_agent_spec_from_text(text: str) -> AgentSpec:
    frontmatter, prompt = _parse_frontmatter(text, Path("<memory>"))
    name, tools = _extract_metadata(frontmatter, Path("<memory>"))
    return AgentSpec(
        name=name,
        tools=tools,
        prompt=prompt,
        source_path=Path("<memory>"),
    )


def slugify_agent_name(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", name.strip().lower()).strip("-")
    if not slug:
        slug = "agent"
    return f"{slug}.md"


def select_agent_spec(agents_dir: Path, agent_name: Optional[str] = None) -> AgentSpec:
    agent_files = list_agent_files(agents_dir)
    if not agent_files:
        raise AgentSpecError(f"No agent markdown files found in {agents_dir}")
    if agent_name:
        for path in agent_files:
            spec = load_agent_spec(path)
            if spec.name == agent_name:
                return spec
        raise AgentSpecError(f"Agent name '{agent_name}' not found in {agents_dir}")
    return load_agent_spec(agent_files[0])


def _parse_frontmatter(text: str, path: Path) -> Tuple[dict, str]:
    lines = text.splitlines()
    if not lines:
        raise AgentSpecError(f"{path} is empty.")
    idx = 0
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    if idx >= len(lines) or lines[idx].strip() != "---":
        raise AgentSpecError(f"{path} missing YAML frontmatter (---).")
    idx += 1
    frontmatter_lines: List[str] = []
    while idx < len(lines):
        if lines[idx].strip() == "---":
            idx += 1
            break
        frontmatter_lines.append(lines[idx])
        idx += 1
    else:
        raise AgentSpecError(f"{path} frontmatter missing closing '---'.")

    try:
        frontmatter = yaml.safe_load("\n".join(frontmatter_lines)) or {}
    except Exception as exc:
        raise AgentSpecError(f"{path} invalid YAML frontmatter: {exc}") from exc

    prompt = "\n".join(lines[idx:]).strip()
    if not prompt:
        raise AgentSpecError(f"{path} prompt body is empty.")
    return frontmatter, prompt


def _extract_metadata(frontmatter: dict, path: Path) -> Tuple[str, List[str]]:
    if not isinstance(frontmatter, dict):
        raise AgentSpecError(f"{path} frontmatter must be a mapping.")
    name = frontmatter.get("name")
    tools = frontmatter.get("tools")
    if not name or not isinstance(name, str):
        raise AgentSpecError(f"{path} frontmatter must include a string 'name'.")
    if not tools or not isinstance(tools, list) or not all(isinstance(t, str) and t.strip() for t in tools):
        raise AgentSpecError(f"{path} frontmatter must include a list 'tools'.")
    return name.strip(), [t.strip() for t in tools]
