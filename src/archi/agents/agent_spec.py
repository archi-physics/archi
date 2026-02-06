from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import re


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
    lines = text.splitlines()
    _reject_front_matter(lines, path)
    name = _extract_name(lines, path)
    tools = _extract_tools(lines, path)
    prompt = _extract_prompt(lines, path)
    return AgentSpec(
        name=name,
        tools=tools,
        prompt=prompt,
        source_path=path,
    )


def load_agent_spec_from_text(text: str) -> AgentSpec:
    lines = text.splitlines()
    _reject_front_matter(lines, Path("<memory>"))
    name = _extract_name(lines, Path("<memory>"))
    tools = _extract_tools(lines, Path("<memory>"))
    prompt = _extract_prompt(lines, Path("<memory>"))
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


def _reject_front_matter(lines: List[str], path: Path) -> None:
    for line in lines:
        if not line.strip():
            continue
        if line.strip() == "---":
            raise AgentSpecError(
                f"{path} uses YAML front matter; use Markdown sections (# Name, ## Tools, ## Prompt) instead."
            )
        break


def _extract_name(lines: List[str], path: Path) -> str:
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("##"):
            continue
        if stripped.startswith("# "):
            name = stripped[2:].strip()
            if name.lower().startswith("name:"):
                name = name.split(":", 1)[1].strip()
            if not name:
                break
            return name
    raise AgentSpecError(f"{path} missing required '# Name' heading.")


def _extract_tools(lines: List[str], path: Path) -> List[str]:
    start, end = _find_section(lines, "## Tools", path)
    tools: List[str] = []
    for line in lines[start:end]:
        if not line.strip():
            continue
        stripped = line.lstrip()
        if stripped.startswith("- "):
            tool = stripped[2:].strip()
        elif stripped.startswith("* "):
            tool = stripped[2:].strip()
        else:
            raise AgentSpecError(f"{path} Tools section must be a bullet list.")
        if not tool:
            raise AgentSpecError(f"{path} Tools section contains an empty bullet.")
        tools.append(tool)
    if not tools:
        raise AgentSpecError(f"{path} Tools section is empty.")
    return tools


def _extract_prompt(lines: List[str], path: Path) -> str:
    start, end = _find_section(lines, "## Prompt", path)
    prompt = "\n".join(lines[start:end]).strip()
    if not prompt:
        raise AgentSpecError(f"{path} Prompt section is empty.")
    return prompt


def _find_section(lines: List[str], header: str, path: Path) -> tuple[int, int]:
    header_idx = None
    for idx, line in enumerate(lines):
        if line.strip() == header:
            header_idx = idx
            break
    if header_idx is None:
        raise AgentSpecError(f"{path} missing required '{header}' section.")
    start = header_idx + 1
    end = len(lines)
    for idx in range(start, len(lines)):
        line = lines[idx].strip()
        if line.startswith("## "):
            end = idx
            break
    return start, end
