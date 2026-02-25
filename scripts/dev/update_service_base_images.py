#!/usr/bin/env python3
"""Update service Dockerfiles to reference a specific base image tag."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILES_DIR = PROJECT_ROOT / "src" / "cli" / "templates" / "dockerfiles"

BASE_IMAGE_MAP = {
    "python": "a2rchi-python-base",
    "pytorch": "a2rchi-pytorch-base",
}

SOURCE_PREFIXES = {
    "localhost": "localhost/a2rchi/",
    "dockerhub": "docker.io/a2rchi/",
}


@dataclass
class UpdateOptions:
    tag: Optional[str]
    orig_tag: Optional[str]
    switch_source: Optional[str]
    bases: Iterable[str]


def _normalize_prefix(prefix: str) -> str:
    """Return a registry/image prefix with a single trailing slash or empty."""
    cleaned = "/".join(filter(None, prefix.split("/")))
    if cleaned:
        return cleaned.rstrip("/") + "/"
    return ""


def _split_image_spec(image_spec: str) -> Tuple[str, str, Optional[str]]:
    """Split "<prefix><image>:<tag>" into (prefix, image, tag)."""
    if ":" in image_spec:
        repo_part, tag = image_spec.rsplit(":", 1)
    else:
        repo_part, tag = image_spec, None

    repo_part = repo_part.replace("//", "/")
    segments = [seg for seg in repo_part.split("/") if seg]
    if not segments:
        return "", repo_part, tag

    image = segments[-1]
    prefix = "/".join(segments[:-1])
    if prefix:
        prefix += "/"

    return prefix, image, tag


def _build_image_spec(prefix: str, image: str, tag: Optional[str]) -> str:
    prefix = _normalize_prefix(prefix)
    repo = f"{prefix}{image}" if prefix else image
    if tag:
        return f"{repo}:{tag}"
    return repo


def _split_line_ending(line: str) -> Tuple[str, str]:
    if line.endswith("\r\n"):
        return line[:-2], "\r\n"
    if line.endswith("\n"):
        return line[:-1], "\n"
    if line.endswith("\r"):
        return line[:-1], "\r"
    return line, ""


def _update_line(line: str, base_name: str, options: UpdateOptions) -> Tuple[str, bool]:
    core, newline = _split_line_ending(line)
    stripped = core.lstrip()
    if not stripped.startswith("FROM "):
        return line, False

    match = re.match(r"(?P<intro>\s*FROM\s+(?:--platform=\S+\s+)?)(?P<image>\S+)(?P<suffix>.*)", core)
    if not match:
        return line, False

    intro, image_spec, suffix = match.group("intro"), match.group("image"), match.group("suffix")
    prefix, image, current_tag = _split_image_spec(image_spec)

    if image != base_name:
        return line, False

    if options.orig_tag is not None and current_tag != options.orig_tag:
        return line, False

    target_tag = options.tag if options.tag is not None else current_tag
    target_prefix = prefix
    if options.switch_source:
        target_prefix = SOURCE_PREFIXES[options.switch_source]

    updated_spec = _build_image_spec(target_prefix, image, target_tag)
    if updated_spec == image_spec:
        return line, False

    return f"{intro}{updated_spec}{suffix}{newline}", True


def update_base_tags(options: UpdateOptions) -> None:
    bases = list(options.bases)
    invalid = set(bases) - set(BASE_IMAGE_MAP)
    if invalid:
        raise SystemExit(f"Unknown base types requested: {', '.join(sorted(invalid))}")

    for path in sorted(DOCKERFILES_DIR.glob("Dockerfile*")):
        original = path.read_text()
        updated = original
        changed = False

        for base in bases:
            image_name = BASE_IMAGE_MAP[base]
            new_lines = []
            file_changed = False
            for line in updated.splitlines(keepends=True):
                new_line, line_changed = _update_line(line, image_name, options)
                new_lines.append(new_line)
                file_changed = file_changed or line_changed
            updated = ''.join(new_lines)
            changed = changed or file_changed

        if changed and updated != original:
            path.write_text(updated)
            rel_path = path.relative_to(PROJECT_ROOT)
            print(f"Updated {rel_path}")


def parse_args() -> UpdateOptions:
    parser = argparse.ArgumentParser(description="Point service Dockerfiles at the given base image tag.")
    parser.add_argument("--tag", help="Base image tag to reference, e.g. v1.2.3")
    parser.add_argument("--orig-tag", help="Only update lines using this tag (use 'all' to match any)", default="latest")
    parser.add_argument(
        "--switch-source",
        choices=sorted(SOURCE_PREFIXES),
        help="Switch base image registry/source",
    )
    parser.add_argument(
        "--bases",
        choices=sorted(BASE_IMAGE_MAP),
        nargs="+",
        default=sorted(BASE_IMAGE_MAP),
        help="Base images to update",
    )
    args = parser.parse_args()
    orig_tag = args.orig_tag
    if orig_tag in ("all", ""):
        orig_tag = None
    return UpdateOptions(
        tag=args.tag,
        orig_tag=orig_tag,
        switch_source=args.switch_source,
        bases=args.bases,
    )


def main() -> None:
    options = parse_args()
    update_base_tags(options)


if __name__ == "__main__":
    main()
