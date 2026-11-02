"""Render to an okg profile, and let okg itself judge the result.

Real input: the shipped cern-team bundle and its packs (bundles/cern-team/,
bundles/packs/), copied by `build_cern_team_catalog`. Two properties:

- acc.r3-bundle-format.okg-accepts: every rendered profile in the matrix
  passes `okg install --dry-run`, run from the instance directory with no
  OKG_PROFILES_DIR, and its scaffold passes `okg deployment lint` with no
  blocker. The two template blockers the cern-team profile carried (docsite's
  reserved `params.source_name`, git_history's drifted admission blocks) were
  fixed when the packs were split out (PACT r3-packs-split).
- acc.r3-bundle-format.no-trace: rendering base+P with P removed is
  byte-identical to rendering base, and nothing P owns appears in it.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from archi.bundles.render import PROFILES_DIR, render
from archi.bundles.resolve import resolve

from bundles.bundle_support import (
    CATALOG,
    CERN_TEAM_DEFAULT,
    CERN_TEAM_OPTIONAL,
    build_cern_team_catalog,
    load_catalog,
)

OKG = str(Path(sys.executable).with_name("okg"))
OPTIONAL = CERN_TEAM_OPTIONAL

#: Answers for the questions a full render asks, by rendered id.
ANSWERS = {
    "deployment_name": "qt",
    "postgres_dsn": "postgresql://localhost:1/qt",
    "archi_data_root": "/srv/archi-data",
    "git_code_repo_url": "https://example.org/demo.git",
    "twiki_eos_root": "/srv/twiki",
}


def _requires(cern, pack):
    """`pack` preceded by the packs it requires (a --with list that resolves)."""
    out = []

    def visit(name):
        for dep in cern.packs[name].requires_packs:
            visit(dep)
        if name not in out:
            out.append(name)

    visit(pack)
    return out


@pytest.fixture(scope="module")
def cern(tmp_path_factory):
    root = tmp_path_factory.mktemp("cern-team-packs")
    build_cern_team_catalog(root)
    return load_catalog(root)


def _tree(directory: Path) -> dict[str, bytes]:
    return {
        p.relative_to(directory).as_posix(): p.read_bytes()
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }


def _rendered_answers(rendered) -> dict[str, str]:
    """ANSWERS for the questions this render asks."""
    asked = {q["id"] for q in rendered.composition.questions}
    return {qid: value for qid, value in ANSWERS.items() if qid in asked}


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


def test_render_writes_an_okg_profile(cern, tmp_path):
    out = render(resolve(cern, ["cern-team"]), tmp_path)
    assert out.profile_dir == tmp_path / PROFILES_DIR / "cern-team"
    profile = yaml.safe_load((out.profile_dir / "profile.yaml").read_text())
    assert profile["name"] == "cern-team"
    assert profile["source_defaults"] == "source-defaults/"
    assert sorted(p.name for p in (out.profile_dir / "source-defaults").iterdir()) == [
        "docsite.docsite.yaml", "indico.indico_live.yaml"
    ]
    assert (out.profile_dir / "skills" / "chat-system-prompt.md").is_file()


def test_render_starts_from_an_empty_directory(cern, tmp_path):
    render(resolve(cern, ["cern-team"], with_packs=["twiki"]), tmp_path)
    out = render(resolve(cern, ["cern-team"]), tmp_path)
    assert not any(p.startswith("source-defaults/twiki.") for p in _tree(out.profile_dir))


def test_render_is_deterministic(cern, tmp_path):
    a = render(resolve(cern, ["cern-team"], with_packs=OPTIONAL), tmp_path / "a")
    b = render(resolve(cern, ["cern-team"], with_packs=OPTIONAL), tmp_path / "b")
    assert _tree(a.profile_dir) == _tree(b.profile_dir)


def test_rendered_deployment_defaults_carry_the_bundle_chat(cern, tmp_path):
    out = render(resolve(cern, ["cern-team"], with_packs=["jira"]), tmp_path)
    defaults = yaml.safe_load((out.profile_dir / "deployment-defaults.yaml").read_text())
    assert defaults["search"]["profiles"]["content"]["subtypes"] == [
        "documentation_page", "software_repository", "document_chunk",
        "meeting_minutes", "document", "jira_issue",
    ]
    assert defaults["nomos"]["rollout"]["owner"]["name"] == "archi-cern-team"
    bundle = cern.bundles["cern-team"]
    assert defaults["chat"]["preset"]["suggestions"] == bundle.chat["suggestions"]


def test_doc_text_nodes_is_declared_only_when_a_pack_writes_doc_records(cern, tmp_path):
    # okg 0.4.0 does not recognise archi's code_text module as a document source,
    # so without the key its MCP instructions leave out the doc: records.
    plain = render(resolve(cern, ["cern-team"]), tmp_path / "plain")
    with_text = render(resolve(cern, ["cern-team"], with_packs=["git-code", "code-text"]),
                       tmp_path / "text")
    read = lambda out: yaml.safe_load((out.profile_dir / "deployment-defaults.yaml").read_text())
    assert "mcp" not in read(plain)
    assert read(with_text)["mcp"] == {"doc_text_nodes": True}


@pytest.mark.parametrize("line", [
    "  module: archi.sources.code_text",
    '  module: "archi.sources.code_text"',
    "  module: 'archi.sources.code_text'  # file texts",
])
def test_doc_text_modules_reads_quoted_and_commented_module_lines(line):
    from archi.bundles.compose import Composition, doc_text_modules
    comp = Composition(sources={"x": f"x:\n{line}\n", "y": "y:\n  module: archi.sources.docs\n"})
    assert doc_text_modules(comp) == ["archi.sources.code_text"]


# ---------------------------------------------------------------------------
# No trace
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pack", OPTIONAL)
def test_base_plus_pack_minus_pack_equals_base(cern, tmp_path, pack):
    deps = _requires(cern, pack)[:-1]  # code-text and code-symbols need git-code
    base = render(resolve(cern, ["cern-team"], with_packs=deps), tmp_path / "base")
    plus = render(resolve(cern, ["cern-team"], with_packs=deps + [pack]), tmp_path / "plus")
    minus = render(
        resolve(cern, ["cern-team"], with_packs=deps + [pack], without_packs=[pack]),
        tmp_path / "minus",
    )
    assert _tree(minus.profile_dir) == _tree(base.profile_dir)
    assert _tree(plus.profile_dir) != _tree(base.profile_dir)
    # Nothing the pack owns alone survives in the base render.
    owned = [
        key for key, owners in plus.composition.owners.items()
        if owners == [pack]
    ]
    assert owned
    base_text = "\n".join(
        data.decode("utf-8", "replace") for data in _tree(base.profile_dir).values()
    )
    for key in owned:
        kind, _, name = key.partition(":")
        if kind in ("source", "question"):
            assert name not in base_text, f"{key} left a trace"


def test_fixture_base_plus_pack_minus_pack_equals_base(tmp_path):
    """The same property on the fixture library, where the pack carries every
    optional part (questions, invariants, playbooks, triggers, fragment)."""
    catalog = load_catalog(CATALOG)
    plus = render(resolve(catalog, ["base-bundle"], with_packs=["beta"]), tmp_path / "p",
                  name="x")
    minus = render(resolve(catalog, ["base-bundle"], with_packs=["beta"], without_packs=["beta"]),
                   tmp_path / "m", name="x")
    base = render(resolve(catalog, ["base-bundle"]), tmp_path / "b", name="x")
    assert _tree(minus.profile_dir) == _tree(base.profile_dir)
    assert "source-defaults/beta.issues.yaml" in _tree(plus.profile_dir)


def test_a_bare_render_composes_only_the_modules_its_packs_need(cern, tmp_path):
    """The old cern-team profile composed git_graph even with no git source;
    the rendered one composes it only when a git pack is selected."""
    out = render(resolve(cern, ["cern-team"]), tmp_path)
    modules = yaml.safe_load((out.profile_dir / "modules.yaml").read_text())["modules"]
    assert "git_graph" not in modules
    out = render(resolve(cern, ["cern-team"], with_packs=["git-code"]), tmp_path)
    assert "git_graph" in yaml.safe_load((out.profile_dir / "modules.yaml").read_text())["modules"]


# ---------------------------------------------------------------------------
# okg accepts every rendered profile in the matrix
# ---------------------------------------------------------------------------

def _matrix():
    """Bare, each optional pack (with what it requires), and all of them."""
    from bundles.bundle_support import SHIPPED_PACKS

    def requires(name):
        data = yaml.safe_load((SHIPPED_PACKS / name / "pack.yaml").read_text())
        return data.get("requires", {}).get("packs", [])

    rows = [[]]
    for pack in OPTIONAL:
        rows.append([d for d in requires(pack) if d != pack] + [pack])
    return rows + [OPTIONAL]


MATRIX = _matrix()


def _rendered_plan(rendered, instance, tmp_path):
    from okg.substrate.deployment_bootstrap.profile_init import build_plan, discover_profile

    return build_plan(
        profile=discover_profile(rendered.name, profiles_root=instance / PROFILES_DIR),
        deployment_name="qt",
        deployments_root=tmp_path / "deployments-rendered",
        raw_answers=_rendered_answers(rendered),
    )


def _write_scaffold(plan):
    from okg.substrate.deployment_bootstrap.profile_init import planned_writes

    for write in planned_writes(plan):
        write.path.parent.mkdir(parents=True, exist_ok=True)
        if write.data is not None:
            write.path.write_bytes(write.data)
        else:
            write.path.write_text(write.text, encoding="utf-8")


def _blockers(deployment_dir, cwd):
    """{(source id, code)} for every lint blocker; source id from the pointer."""
    lint = _okg(["deployment", "lint", str(deployment_dir), "--json"], cwd=cwd)
    report = json.loads(lint.stdout)
    out = set()
    for f in report["findings"]:
        if f["severity"] in ("blocker", "error"):
            pointer = f["location"].get("pointer", "")
            source = pointer.split("/")[2] if pointer.startswith("/sources/") else None
            out.add((source, f["code"]))
    return out


def _okg(args, cwd):
    env = {k: v for k, v in os.environ.items() if k != "OKG_PROFILES_DIR"}
    env["HOME"] = str(cwd)  # keep okg's user-state lookups inside the test dir
    return subprocess.run([OKG, *args], cwd=cwd, env=env, capture_output=True, text=True,
                          timeout=120)


@pytest.mark.parametrize("with_packs", MATRIX, ids=["bare"] + OPTIONAL + ["all"])
def test_okg_accepts_the_rendered_profile(cern, tmp_path, with_packs):
    instance = tmp_path / "instance"
    rendered = render(resolve(cern, ["cern-team"], with_packs=with_packs), instance)
    answers = _rendered_answers(rendered)

    flags = []
    for qid, value in answers.items():
        if qid != "deployment_name":
            flags += [f"--{qid.replace('_', '-')}", value]
    result = _okg(
        ["install", "--profile", rendered.name, "--deployment-name", "qt",
         "--deployments-root", str(tmp_path / "deployments"),
         "--dry-run", "--json", "--non-interactive", *flags],
        cwd=instance,
    )
    assert result.returncode == 0, result.stderr[-2000:] + result.stdout[-2000:]
    report = json.loads(result.stdout)
    assert report["status"] == "dry_run_ok" and report["blockers"] == [], report
    assert report["profile"]["name"] == rendered.name

    plan = _rendered_plan(rendered, instance, tmp_path)
    _write_scaffold(plan)
    assert _blockers(plan.deployment_dir, instance) == set()


def test_render_command(cern, tmp_path, capsys):
    from archi.bundles.cli import main as archi_main

    root = next(iter(cern.packs.values())).dir.parents[1]
    rc = archi_main(["render", "--bundle", "cern-team", "--with", "twiki,jira",
                     "--instance", str(tmp_path), "--pack-root", str(root / "packs"),
                     "--bundle-root", str(root)])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "docsite, indico, core-playbooks, twiki, jira" in out
    assert (tmp_path / "profiles" / "cern-team" / "source-defaults" / "twiki.twiki_eos.yaml").is_file()


def test_render_command_reports_the_rule_that_failed(cern, tmp_path, capsys):
    from archi.bundles.cli import main as archi_main

    root = next(iter(cern.packs.values())).dir.parents[1]
    rc = archi_main(["render", "--bundle", "cern-team", "--without", "jira",
                     "--instance", str(tmp_path), "--pack-root", str(root / "packs"),
                     "--bundle-root", str(root)])
    assert rc == 1
    assert "error (without_not_selected)" in capsys.readouterr().err
