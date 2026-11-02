"""acc.r3-docs.quickstart: the quickstart page shows the commands the nightly build runs.

The page (docs/docs/quickstart.md) has one fenced `bash` block. The nightly
workflow (.github/workflows/nightly-e2e.yml) runs the same path as five steps
named "user command N". This test turns the steps into the commands an
operator would type and compares them with the block, so a change to one
without the other fails.

The documented differences, and only these, are normalised away:

- Command 1 installs `archi-kb==<version>` from PyPI on the page, where
  `<version>` is the package's own version. The nightly installs the checkout
  it is testing (`./python`). Both get okg from PyPI as archi-kb's dependency.
- The nightly's instance is `gp`; the page calls it `myteam`.
- The nightly sets a working directory per step; the page uses `cd myteam`.
- The nightly pipes output to `tee` and reads its question from `$QUESTION`;
  the page leaves out the `tee` and writes the question out. The script path
  starts at `$GITHUB_WORKSPACE` there and `$ARCHI_CHECKOUT` on the page.
"""
from __future__ import annotations

import re
import shlex
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
PAGE = REPO / "docs" / "docs" / "quickstart.md"
WORKFLOW = REPO / ".github" / "workflows" / "nightly-e2e.yml"

VERSION = re.search(r'^version = "([^"]+)"', (REPO / "python" / "pyproject.toml").read_text(encoding="utf-8"), re.M).group(1)
ARCHI_PYPI = f"archi-kb=={VERSION}"


def page_commands() -> list[list[str]]:
    text = PAGE.read_text(encoding="utf-8")
    blocks = re.findall(r"^```bash\n(.*?)^```", text, re.M | re.S)
    assert len(blocks) == 1, "the quickstart page must have exactly one bash block"
    lines = [ln for ln in blocks[0].splitlines() if ln.strip()]
    assert not any(ln.rstrip().endswith("\\") for ln in lines), "keep each command on one line"
    return [shlex.split(ln) for ln in lines]


def nightly_steps() -> tuple[dict[str, str], list[dict]]:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    job = next(iter(workflow["jobs"].values()))
    steps = [s for s in job["steps"] if str(s.get("name", "")).startswith("user command ")]
    assert [s["name"].split(":")[0] for s in steps] == [f"user command {i}" for i in range(1, 6)]
    return job["env"], steps


def nightly_as_typed() -> list[list[str]]:
    env, steps = nightly_steps()
    commands: list[list[str]] = []
    cwd = None
    for step in steps:
        run = step["run"].strip()
        assert "\n" not in run, f"{step['name']}: expected one command"
        run = run.split(" | tee ")[0]
        where = step.get("working-directory")
        if where is not None and where != cwd and cwd is not None:
            # moving from the runner's temp directory into the instance
            instance = where.rstrip("/").rsplit("/", 1)[-1]
            commands.append(["cd", "myteam" if instance == "gp" else instance])
        if where is not None:
            cwd = where
        run = run.replace('"$QUESTION"', shlex.quote(env["QUESTION"]))
        run = run.replace("./python", ARCHI_PYPI).replace("$GITHUB_WORKSPACE", "$ARCHI_CHECKOUT")
        run = re.sub(r"\bgp\b", "myteam", run)
        commands.append(shlex.split(run))
    return commands


def test_page_commands_are_the_nightly_commands():
    assert page_commands() == nightly_as_typed()


def test_nightly_runs_five_user_commands():
    _, steps = nightly_steps()
    assert len(steps) == 5


def test_page_installs_from_pypi_alone_and_names_the_script():
    text = PAGE.read_text(encoding="utf-8")
    assert "okg @ " not in text  # okg comes with archi-kb from PyPI; no wheel URL to copy
    assert "python/tests/cli/mcp_cited_answer.py" in text
