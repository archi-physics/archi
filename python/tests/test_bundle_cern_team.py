"""cern-team bundle structural guards (W6; okg#1185 release-claim shape).

The install demo (docs/cern-team-demo.md) proves the bundle live; these
tests keep its structure honest offline: strict-admission blocks on
every connector default, resolvable playbook symlinks, no operator
paths or credential values in bundle files.
"""
import importlib.util
import re
from pathlib import Path

import yaml

BUNDLE = Path(__file__).resolve().parents[2] / "bundles" / "cern-team"


def test_profile_parses_with_required_questions():
    profile = yaml.safe_load((BUNDLE / "profile.yaml").read_text())
    assert profile["name"] == "cern-team"
    ids = {q["id"] for q in profile["init_questions"]}
    assert {"deployment_name", "postgres_dsn", "archi_data_root"} <= ids


def test_source_defaults_carry_strict_admission_shape():
    defaults = sorted((BUNDLE / "source-defaults").glob("*.yaml"))
    assert defaults, "no connector defaults in the bundle"
    for path in defaults:
        entries = yaml.safe_load(path.read_text())
        for name, entry in entries.items():
            policy = entry.get("admission_policy", {})
            assert policy.get("output_signature"), f"{path.name}:{name} missing output_signature"
            assert policy.get("output_scope_summary"), f"{path.name}:{name} missing output_scope_summary"
            assert entry.get("sync"), f"{path.name}:{name} missing sync block"
            # A bundle legitimately composes BOTH archi connectors and
            # substrate-resident okg sources. What must not appear is a
            # module from anywhere else: a bundle reaching outside these two
            # homes ships a dependency its consumers have no way to install.
            module = entry["module"]
            assert module.startswith(
                ("archi.sources.", "okg.substrate.library.sources.")
            ), f"{path.name}:{name} is neither an archi connector nor a substrate source"
            # A prefix check alone is weaker than what it replaced. `archi.*`
            # modules ship in this wheel, so a rename breaks the import tests
            # next door; `okg.*` modules do not — okg is the host environment
            # and pyproject declares no dependency on it, so nothing else here
            # would notice okg renaming or relocating a source. Resolve it for
            # real: an unimportable module is a bundle that fails at ingest
            # time on an operator's machine, which is the worst place to find
            # out.
            if module.startswith("okg."):
                assert importlib.util.find_spec(module) is not None, (
                    f"{path.name}:{name} names {module}, which does not resolve "
                    "in this environment — the host okg has renamed, moved or "
                    "dropped it"
                )


def test_playbook_symlinks_resolve():
    skills = BUNDLE / "skills"
    links = list(skills.iterdir())
    assert len(links) >= 20
    for link in links:
        assert link.resolve().exists(), f"dangling playbook symlink: {link}"


def test_no_operator_paths_or_secrets_in_bundle():
    pattern = re.compile(r"(/Users/|/root/|/work/submit/|/home/submit/|password\s*[:=]\s*\S|token\s*[:=]\s*[A-Za-z0-9]{16,})")
    for path in BUNDLE.rglob("*"):
        if path.is_file() and not path.is_symlink():
            for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                assert not pattern.search(line), f"{path}:{lineno}: {line.strip()[:80]}"


def test_bundle_ships_schema_slices_matching_the_package():
    """The `schemas:` slot (okg#1367) needs real files, so they are duplicated.

    okg refuses symlinked schema assets — `profile_invalid: schema assets may
    not be symlinks` — which is why these are copies rather than links into
    `python/archi/schemas/` the way `skills/` links into `skills/`. Duplication
    without a guard drifts, and a drifted bridge is the exact failure W3 spent
    a wave on: the instance composes something the distribution did not ship.
    """
    declared = yaml.safe_load((BUNDLE / "profile.yaml").read_text()).get("schemas")
    assert declared == "schemas/", "bundle must declare the schemas: slot"

    bundled = BUNDLE / "schemas"
    packaged = Path(__file__).resolve().parents[1] / "archi" / "schemas"
    expected = {"operations.yaml", "sources.yaml",
                "bridges/operations.yaml", "bridges/sources.yaml"}
    present = {str(p.relative_to(bundled)) for p in bundled.rglob("*.yaml")}
    assert present == expected, f"bundle schemas drifted: {present ^ expected}"
    for rel in sorted(expected):
        assert not (bundled / rel).is_symlink(), f"{rel} is a symlink; okg refuses those"
        assert (bundled / rel).read_bytes() == (packaged / rel).read_bytes(), (
            f"{rel} differs from python/archi/schemas/{rel} — the bundle copy and "
            "the package copy must stay byte-identical"
        )


def test_default_sources_need_no_credentials():
    """A bare install must publish, so nothing selected by default may be gated.

    ADR 0001 W6: "A deployment with no optional connector configured must start
    cleanly." The completeness gate fails the whole batch when any *selected*
    source fails, so a credential-gated default silently makes a fresh install
    unable to publish at all. Sources needing credentials, a prebuilt cache or
    a URL ship as `.yaml.example` and are opted into by renaming.

    github_repo and gitlab_repo used to be defaults with blank URL answers, on
    the theory that a blank URL meant "install without this source". A live run
    disproved it: both installed, both failed admission with
    `source_health_not_healthy: checkout unavailable`, and that blocked the
    publish for cmssw_releases too. An installed source is always a selected
    source, so a source needing an operator-supplied input cannot be a default.
    """
    selected = sorted(p.name for p in (BUNDLE / "source-defaults").glob("*.yaml"))
    assert selected == ["cmssw_releases.yaml"], (
        f"default source set changed: {selected}. Anything needing a credential, "
        "a prebuilt cache or a URL belongs in a .yaml.example."
    )
    for path in (BUNDLE / "source-defaults").glob("*.yaml"):
        entry = next(iter(yaml.safe_load(path.read_text()).values()))
        assert not entry.get("credential_refs"), (
            f"{path.name} is selected by default but declares credential_refs; "
            "a fresh install would fail to publish"
        )


def test_no_default_source_depends_on_an_answer_that_defaults_to_blank():
    """The general form of the blank-URL defect.

    An install answer that defaults to blank is one the operator may never be
    asked about. A source selected by default must not depend on one, because
    it will then install with an empty parameter, fail to read its input, and
    take the publish down with it -- which is exactly what github_repo and
    gitlab_repo did with `${github_repo_url}` / `${gitlab_repo_url}`.

    Examples are exempt: renaming one is the operator saying they will supply
    its inputs.
    """
    profile = yaml.safe_load((BUNDLE / "profile.yaml").read_text())
    blank = {
        q["id"]
        for q in profile["init_questions"]
        if not q.get("required") and not (q.get("default") or "")
    }
    offenders = {}
    for path in sorted((BUNDLE / "source-defaults").glob("*.yaml")):
        used = set(re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", path.read_text()))
        if used & blank:
            offenders[path.name] = sorted(used & blank)
    assert not offenders, (
        "these default sources depend on answers that default to blank, so a "
        f"plain install would scaffold them empty and block the publish: {offenders}"
    )


def test_the_repository_sources_need_both_a_rename_and_a_url():
    """Enabling a repository source is two steps, and the file says so."""
    for host in ("github", "gitlab"):
        path = BUNDLE / "source-defaults" / f"{host}_repo.yaml.example"
        text = path.read_text()
        assert f"${{{host}_repo_url}}" in text
        assert "Both steps are needed." in text, (
            f"{path.name} must say that renaming alone is not enough"
        )


def test_default_install_declares_standalone_release_authority():
    defaults = yaml.safe_load((BUNDLE / "deployment-defaults.yaml").read_text())
    assert defaults["release"] == {"kind": "standalone"}


def test_blocking_floor_matches_the_default_source_set():
    """Optional sources must not block a bare bundle publish."""
    invariants = yaml.safe_load((BUNDLE / "invariants.yaml").read_text())
    floor = next(
        item
        for item in invariants["invariants"]
        if item["name"] == "cern_team_core_pages_floor"
    )
    assert floor["severity"] == "error"
    assert "('cmssw_release')" in floor["sql"]
    assert "documentation_page" not in floor["sql"]
    assert "jira_issue" not in floor["sql"]


def test_chat_declares_a_system_prompt_that_ships():
    """`okg chat sync` REFUSES a deployment with no prompt source.

    Open WebUI never shows the model the MCP server's own `instructions`, so
    without a system prompt the assistant gets graph tools registered and no
    idea it has them — a preset that looks configured and cannot answer. okg
    refuses rather than allow that, so this is a hard requirement, not a
    nicety, and the file has to be one the bundle actually materialises.
    """
    defaults = yaml.safe_load((BUNDLE / "deployment-defaults.yaml").read_text())
    ref = defaults["chat"]["preset"]["system_prompt_ref"]
    assert ref, "chat.preset.system_prompt_ref must be declared"

    # The ref is deployment-relative; the bundle's skills/ becomes
    # <deployment>/skills/, so a skills/-rooted ref must exist there.
    assert ref.startswith("skills/"), (
        f"system_prompt_ref {ref!r} is not under skills/, which is the only "
        "bundle directory materialised into the deployment as loose files"
    )
    shipped = BUNDLE / ref
    assert shipped.is_file(), f"{ref} is declared but not shipped in the bundle"
    text = shipped.read_text(encoding="utf-8").strip()
    assert text, "a declared but empty prompt is refused by okg chat sync"
    # The prompt exists to tell the model it has graph tools. If it stops
    # naming them, it has stopped doing its job.
    for operator in ("search", "inspect", "expand"):
        assert operator in text, f"prompt no longer mentions the {operator} tool"
