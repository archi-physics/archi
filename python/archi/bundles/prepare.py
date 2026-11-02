"""Pack preparation steps: what a pack needs on disk before okg's first ingest.

A pack declares `prepare:` in pack.yaml (docs/docs/reference/pack-format.md). Each step
is `python -m <module> <args>`, run by `archi init` and `archi add` after the
pack's answers are known and before okg reads the pack's sources. The docsite
pack uses it to write its records cache.

A step runs without a shell, in archi's own Python, with ARCHI_DATA_ROOT set
to the instance's data root and the values of the instance's `.env` added to its
environment (a value exported in the shell wins). A step that fails or times out raises
PrepareError; the caller stops. Nothing is skipped and no empty result is
substituted, because a source reading a missing or empty cache would publish
nothing or fail the whole batch.
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from archi.bundles.compose import question_prefix
from archi.bundles.manifest import TEMPLATE_REF, Pack

DEFAULT_TIMEOUT = 600
DATA_ROOT_ENV = "ARCHI_DATA_ROOT"
#: How much of a failed step's output the error carries.
TAIL = 3000


class PrepareError(Exception):
    """A preparation step failed; the message names the pack and step."""


@dataclass(frozen=True)
class Step:
    pack: str
    name: str
    command: tuple[str, ...]
    timeout: int

    @property
    def label(self) -> str:
        return f"{self.pack}:{self.name}"


def _substitute(arg: str, pack: Pack, values: Mapping[str, Any]) -> str:
    local = {q["id"] for q in pack.questions()}
    prefix = question_prefix(pack)

    def fill(match) -> str:
        ref = match.group(1)
        key = prefix + ref if ref in local else ref
        if key not in values or values[key] is None:
            raise PrepareError(
                f"{pack.name}: prepare step needs the answer {key!r}, which the install "
                "neither gives nor defaults"
            )
        return str(values[key])

    return TEMPLATE_REF.sub(fill, arg)


def steps_for(packs: Iterable[Pack], values: Mapping[str, Any]) -> list[Step]:
    """Every selected pack's steps, in the order the packs are given.

    `values` maps rendered question ids (the pack-prefixed ones included) to
    their answers or defaults.
    """
    out: list[Step] = []
    for pack in packs:
        for step in pack.prepare:
            args = [_substitute(a, pack, values) for a in step.get("args", ())]
            out.append(Step(
                pack=pack.name, name=step["name"],
                command=(sys.executable, "-m", step["module"], *args),
                timeout=int(step.get("timeout", DEFAULT_TIMEOUT)),
            ))
    return out


def question_values(
    questions: Sequence[Mapping[str, Any]], answers: Mapping[str, Any]
) -> dict[str, Any]:
    """Rendered question id -> the given answer, else the question's default."""
    values = {q["id"]: q.get("default") for q in questions}
    values.update({k: v for k, v in answers.items()})
    return values


def run_steps(
    steps: Sequence[Step],
    data_root: Path,
    *,
    echo: Callable[[str], None] = print,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    env_values: Optional[Mapping[str, str]] = None,
) -> None:
    """Run the steps in order; the first failure raises PrepareError.

    `env_values` is the instance's `.env`: a step (a downloader) reads its
    credential from its environment, so those values are added to it. A value
    exported in the shell wins.
    """
    from archi.stack.instance import instance_env

    env = instance_env(env_values or {})
    env[DATA_ROOT_ENV] = str(data_root)
    Path(data_root).mkdir(parents=True, exist_ok=True)
    for step in steps:
        echo(f"preparing {step.label} ...")
        try:
            result = runner(
                list(step.command), cwd=str(data_root), env=env, capture_output=True,
                text=True, timeout=step.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise PrepareError(
                f"{step.label}: stopped after {step.timeout}s; nothing was ingested"
            ) from exc
        except OSError as exc:
            raise PrepareError(f"{step.label}: could not start: {exc}") from exc
        if result.returncode != 0:
            detail = "\n".join(t for t in (result.stderr, result.stdout) if t).strip()[-TAIL:]
            raise PrepareError(
                f"{step.label}: failed (exit {result.returncode}); nothing was ingested.\n{detail}"
            )
