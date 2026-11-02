"""Run okg for an instance: from the instance directory, with its environment."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from archi.bundles.lock import LockError, _okg_executable
from archi.stack.instance import Instance, host_env


class OkgError(Exception):
    """An okg command failed."""

    def __init__(self, message: str, result: Optional[subprocess.CompletedProcess] = None):
        super().__init__(message)
        self.result = result


def okg_version() -> Optional[str]:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("okg")
    except PackageNotFoundError:
        return None


def executable() -> str:
    try:
        return _okg_executable()
    except LockError as exc:
        raise OkgError(str(exc)) from exc


@dataclass
class Okg:
    instance: Instance
    secrets: Mapping[str, str]
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run
    echo: Callable[[str], None] = print

    def env(self, extra: Optional[Mapping[str, str]] = None) -> dict[str, str]:
        env = host_env(self.instance, self.secrets)
        env.update(extra or {})
        return env

    def run(
        self,
        *args: str,
        ok_codes: Sequence[int] = (0,),
        extra_env: Optional[Mapping[str, str]] = None,
        quiet: bool = False,
    ) -> subprocess.CompletedProcess:
        cmd = [executable(), *args]
        if not quiet:
            self.echo("$ okg " + " ".join(args))
        result = self.runner(
            cmd, cwd=self.instance.dir, env=self.env(extra_env), capture_output=True, text=True,
        )
        if result.returncode not in ok_codes:
            tail = (result.stderr or "")[-2500:] + (result.stdout or "")[-1500:]
            raise OkgError(f"`okg {' '.join(args)}` failed ({result.returncode}):\n{tail}", result)
        return result

    def json(self, *args: str, ok_codes: Sequence[int] = (0,), **kw: Any) -> Any:
        """Run with --json and parse stdout; a failing code still returns its JSON."""
        result = self.run(*args, "--json", ok_codes=range(256), quiet=True, **kw)
        try:
            data = json.loads(result.stdout) if result.stdout.strip() else None
        except json.JSONDecodeError:
            data = None
        if result.returncode not in ok_codes and data is None:
            raise OkgError(
                f"`okg {' '.join(args)} --json` failed ({result.returncode}):\n"
                + (result.stderr or result.stdout)[-2500:],
                result,
            )
        return {"returncode": result.returncode, "data": data}


def answer_flags(answers: Mapping[str, Any]) -> list[str]:
    """okg install takes every profile question as a flag: id `a_b` -> `--a-b`."""
    flags: list[str] = []
    for qid, value in answers.items():
        flags += [f"--{qid.replace('_', '-')}", str(value)]
    return flags


def role_password(path: Path, role: str) -> str:
    """A role's password from the file `okg db role-passwords --apply` wrote."""
    from urllib.parse import unquote, urlsplit

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    dsn = (data.get("roles") or {}).get(role)
    if not dsn:
        raise OkgError(f"{path} records no password for role {role!r}")
    password = urlsplit(dsn).password
    if not password:
        raise OkgError(f"{path}: the {role} connection string carries no password")
    return unquote(password)
