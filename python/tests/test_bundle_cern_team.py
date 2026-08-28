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
    unable to publish at all. Sources needing credentials or a prebuilt cache
    ship as `.yaml.example` and are opted into by renaming.
    """
    selected = sorted(p.name for p in (BUNDLE / "source-defaults").glob("*.yaml"))
    assert selected == ["cmssw_releases.yaml", "github_repo.yaml", "gitlab_repo.yaml"], (
        f"default source set changed: {selected}. Anything needing a credential "
        "or a prebuilt cache belongs in a .yaml.example."
    )
    for path in (BUNDLE / "source-defaults").glob("*.yaml"):
        entry = next(iter(yaml.safe_load(path.read_text()).values()))
        assert not entry.get("credential_refs"), (
            f"{path.name} is selected by default but declares credential_refs; "
            "a fresh install would fail to publish"
        )


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
