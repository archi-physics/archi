"""The optional `cern` extra carries the SSO helper's package.

okg core removed its own `cern` extra; `archi.downloaders.sso_login` imports
`requests_kerberos` at run time, so this distribution declares it.
"""

import tomllib
from pathlib import Path

from packaging.requirements import Requirement

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _extras() -> dict[str, list[str]]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return data["project"]["optional-dependencies"]


def test_cern_extra_names_the_sso_package() -> None:
    names = {Requirement(spec).name for spec in _extras()["cern"]}
    assert names == {"requests-kerberos"}


def test_cern_extra_is_not_a_core_dependency() -> None:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    core = " ".join(data["project"]["dependencies"]).lower()
    assert "kerberos" not in core
