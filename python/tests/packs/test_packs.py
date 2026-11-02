"""The packs split out of cern-team and the CMS deployment (PACT r3-packs-split).

- acc.r3-packs-split.validate: `archi bundles validate` passes on every pack.
- Each context pack owns exactly one coverage invariant, over subtypes its
  own sources emit, and declares every credential its templates read.
- needs_credentials is honest: a pack that says false publishes in the
  isolation test with no credential in its environment
  (test_pack_isolation.py strips them); one that says true says why.
- Every schema path is owned by one pack or byte-identical across packs.
"""
import importlib.util
import re
from pathlib import Path

import pytest
import yaml

from packs.pack_support import (
    FIXTURE_ENV,
    PACKS_DIR,
    SKILLS_DIR,
    context_packs,
    emitted_subtypes,
    invariants,
    manifest,
    pack_dirs,
    required_subtypes,
    templates,
)

#: The packs PACT r3-packs-split names.
PACT_PACKS = {
    "docsite", "indico", "twiki", "cds", "jira",
    "git-code", "code-text", "code-symbols",
    "cms-catalogs", "cmssst", "cmssw-releases", "dqm", "hypernews", "monit-live",
}


def test_every_pact_pack_exists():
    assert PACT_PACKS <= {p.name for p in pack_dirs()}


def test_archi_bundles_validate_passes_on_every_pack(capsys):
    from archi.bundles.cli import main

    for directory in pack_dirs():
        rc = main(["bundles", "validate", str(directory)])
        out = capsys.readouterr()
        assert rc == 0, f"{directory.name}: {out.out}{out.err}"
    rc = main(["bundles", "validate"])
    out = capsys.readouterr()
    assert rc == 0, out.out + out.err
    assert "valid" in out.out


@pytest.mark.parametrize("pack", context_packs())
def test_each_context_pack_owns_one_coverage_invariant(pack):
    """One invariant, named for the pack, requiring only subtypes it emits.

    The invariant leaves with its pack, so dropping a pack can no longer leave
    a floor for data that never arrives (the cern-team "core pages floor").
    """
    owned = invariants(pack)
    assert owned[0]["name"] == pack.replace("-", "_") + "_coverage"
    # Any further invariant is advisory (a pack may warn about a wiring step).
    assert all(i["severity"] == "warn" for i in owned[1:])
    inv = owned[0]
    assert inv["severity"] == "error"
    required = required_subtypes(inv)
    assert required, f"{pack}: the invariant requires no subtype"
    assert set(required) <= emitted_subtypes(pack), (
        f"{pack}: {sorted(set(required) - emitted_subtypes(pack))} are not emitted by its sources"
    )


def _credential_names(entry: dict) -> set[str]:
    names = set(entry.get("credential_refs") or [])
    params = entry.get("params") or {}
    for key in ("token_env", "cookie_file_env"):
        if isinstance(params.get(key), str) and re.fullmatch(r"[A-Z][A-Z0-9_]*", params[key]):
            names.add(params[key])
    return names


@pytest.mark.parametrize("pack", context_packs())
def test_each_pack_declares_the_credentials_its_templates_read(pack):
    read = set().union(*(_credential_names(e) for e in templates(pack).values()))
    declared = set(manifest(pack).get("credentials", []))
    assert read <= declared, f"{pack} reads {sorted(read - declared)} but does not declare them"


@pytest.mark.parametrize("pack", context_packs())
def test_needs_credentials_is_explained_and_matches_the_fixture_env(pack):
    """A pack that publishes credential-free gets no credential in the
    isolation test; one that does not says why in its manifest."""
    text = (PACKS_DIR / pack / "pack.yaml").read_text(encoding="utf-8")
    needs = manifest(pack)["needs_credentials"]
    if not needs:
        assert not FIXTURE_ENV.get(pack), (
            f"{pack} says needs_credentials: false but its isolation run sets {FIXTURE_ENV[pack]}"
        )
    comment = text[: text.index("needs_credentials:")].rstrip().splitlines()[-1]
    assert comment.lstrip().startswith("#"), f"{pack}: say why needs_credentials is {needs}"


def test_each_schema_path_is_owned_once_or_identical():
    """acc.r3-packs-split.matrix, the schema half: a schema path two packs
    ship is byte-identical, so composing them can never conflict."""
    seen: dict[str, tuple[str, bytes]] = {}
    for directory in pack_dirs():
        schemas = directory / "schemas"
        if not schemas.is_dir():
            continue
        for path in sorted(schemas.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(schemas).as_posix()
            data = path.read_bytes()
            if rel in seen:
                owner, first = seen[rel]
                assert data == first, f"schemas/{rel} differs between {owner} and {directory.name}"
            else:
                seen[rel] = (directory.name, data)
    assert seen


def test_shared_schema_slices_are_the_package_slices():
    """The shared vocabulary is python/archi/schemas/, linked, not copied."""
    package = Path(__file__).resolve().parents[2] / "archi" / "schemas"
    for directory in pack_dirs():
        schemas = directory / "schemas"
        if not schemas.is_dir():
            continue
        for path in sorted(schemas.rglob("*.yaml")):
            rel = path.relative_to(schemas)
            if not (package / rel).is_file():
                continue  # a pack's own slice (ops-record's), owned once
            assert path.is_symlink(), f"{directory.name}: {rel} is a copy, not a link"
            assert path.resolve() == (package / rel).resolve(), f"{directory.name}: {rel}"


@pytest.mark.parametrize("pack", context_packs())
def test_templates_carry_the_strict_admission_shape(pack):
    for source, entry in templates(pack).items():
        where = f"{pack}/{source}"
        policy = entry.get("admission_policy", {})
        assert policy.get("output_signature"), f"{where} missing output_signature"
        assert policy.get("output_scope_summary"), f"{where} missing output_scope_summary"
        assert entry.get("sync"), f"{where} missing sync block"
        # The scope summary lists every node subtype the signature does, or
        # okg lint reports admission_policy_block_drift (git_history did).
        signature = {n["subtype"] for n in policy["output_signature"]["nodes"]}
        assert signature == set(policy["output_scope_summary"]["nodes"]), where
        assert policy["authority_scope"]["source_family"] == pack, where
        module = entry["module"]
        assert module.startswith(("archi.sources.", "okg.substrate.library.sources.")), where
        # okg is the host environment, not a declared dependency, so resolve
        # its modules for real: a rename there would break an install.
        assert importlib.util.find_spec(module) is not None, f"{where}: {module}"
        assert "source_name" not in (entry.get("params") or {}), (
            f"{where}: params.source_name is reserved by okg (param_reserved)"
        )


def test_predecessors_name_ownership_ids():
    """A bare source id is refused once ids are pack-qualified; an ownership
    id survives the renaming (`PredecessorNotRun` on a staging deployment)."""
    for pack in context_packs():
        for source, entry in templates(pack).items():
            pred = entry.get("predecessor_source") or (entry.get("params") or {}).get(
                "predecessor_source"
            )
            if pred is not None:
                assert pred.strip("'").startswith("${deployment_name}."), f"{pack}/{source}"


def test_no_operator_paths_or_secrets_in_packs():
    pattern = re.compile(
        r"(/Users/|/root/|/work/submit/|/home/submit/|password\s*[:=]\s*\S|"
        r"token\s*[:=]\s*[A-Za-z0-9]{16,})"
    )
    for path in PACKS_DIR.rglob("*"):
        if path.is_file() and not path.is_symlink():
            for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                assert not pattern.search(line), f"{path}:{lineno}: {line.strip()[:80]}"


def test_pack_playbooks_resolve_into_the_skills_library():
    links = [p for p in PACKS_DIR.glob("*/playbooks/*.md")]
    assert links
    for link in links:
        assert link.is_symlink(), f"{link} should link into skills/, not copy it"
        assert link.resolve().parent == SKILLS_DIR.resolve(), link
        assert link.resolve().is_file(), f"dangling playbook symlink: {link}"


def test_pack_triggers_are_the_library_triggers():
    library = yaml.safe_load((SKILLS_DIR / "skill-triggers.yaml").read_text())["intents"]
    for path in PACKS_DIR.glob("*/playbooks/skill-triggers.yaml"):
        body = yaml.safe_load(path.read_text())
        pack_dir = path.parent
        for intent, value in body["intents"].items():
            assert library.get(intent) == value, f"{path}: {intent} drifted from the library"
            assert (pack_dir / f"{value['skill']}.md").exists(), f"{path}: {value['skill']}"


def test_git_history_follows_the_repo_source():
    """History reads the checkout and the repo name the repo source uses, so
    commits attach to the file nodes that source emits."""
    sources = templates("git-code")
    history, repo = sources["git_history"], sources["repo"]
    assert history["predecessor_source"] == repo["ownership_id"]
    assert history["params"]["repo"] == repo["params"]["repo"]
    assert history["params"]["repo_root"] == repo["params"]["repo_root"]
    assert history["params"]["store_diffs"] is True
    assert {"git_commit", "git_file_diff", "git_tag"} <= set(manifest("git-code")["search_subtypes"])


def _emitted_subtypes(pack: str) -> set[str]:
    """Subtypes a pack emits: its sources' output signatures plus the
    subtypes it declares for search (ops-record has no sources)."""
    out = set(manifest(pack).get("search_subtypes", []))
    for profile in manifest(pack).get("search_profiles", {}).values():
        out |= set(profile.get("subtypes", []))
    for entry in templates(pack).values():
        out |= {n["subtype"] for n in entry["admission_policy"]["output_signature"]["nodes"]}
    return out


def test_every_pack_trigger_names_a_subtype_its_pack_can_see():
    """A trigger fires on subtypes. At least one must be emitted by the owning
    pack or a pack it requires; others may come from optional siblings."""
    for path in PACKS_DIR.glob("*/*/skill-triggers.yaml"):
        pack = path.parent.parent.name
        if manifest(pack)["kind"] == "tool":
            continue  # core-playbooks emits nothing; its triggers are generic
        visible, todo = set(), [pack]
        while todo:
            name = todo.pop()
            visible |= _emitted_subtypes(name)
            todo += manifest(name).get("requires", {}).get("packs", [])
        for intent, value in yaml.safe_load(path.read_text())["intents"].items():
            wanted = set(value["subtypes"])
            assert wanted & visible or "*" in wanted, f"{path}: {intent} names no subtype {pack} emits"
