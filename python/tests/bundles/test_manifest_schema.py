"""acc.r3-bundle-format.schema: pack.yaml / bundle.yaml validation.

The published JSON schemas fix the shape; `validate_pack` / `validate_bundle`
add what a schema cannot see. Each invalid case below starts from the valid
fixture catalog and breaks one thing, so a pass proves that one rule.
"""
import json
import shutil
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from archi.bundles import discovery
from archi.bundles.cli import main as archi_main
from archi.bundles.manifest import (
    QUESTION_TYPES,
    SCHEMA_DIR,
    validate_bundle,
    validate_pack,
)

from bundles.bundle_support import CATALOG, edit_yaml


def _catalog(root: Path) -> discovery.Catalog:
    return discovery.from_roots([root / "packs"], [root])


# ---------------------------------------------------------------------------
# The published schemas
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["pack", "bundle"])
def test_published_schema_is_a_valid_json_schema(kind):
    schema = json.loads((SCHEMA_DIR / f"{kind}.schema.json").read_text())
    Draft202012Validator.check_schema(schema)
    assert schema["additionalProperties"] is False


def test_pack_schema_covers_every_adr_key():
    """Every key of the ADR 0002 section 3.1 example is declared."""
    adr_keys = {
        "name", "kind", "version", "description", "requires", "schemas",
        "sources", "init_questions", "invariants", "freshness", "playbooks",
        "skill_triggers", "search_subtypes", "live_tools", "prompt_fragment",
        "credentials", "needs_credentials",
    }
    schema = json.loads((SCHEMA_DIR / "pack.schema.json").read_text())
    assert adr_keys <= set(schema["properties"])


def test_bundle_schema_covers_every_adr_key():
    adr_keys = {"name", "version", "extends", "packs", "chat", "eval", "okg"}
    schema = json.loads((SCHEMA_DIR / "bundle.schema.json").read_text())
    assert adr_keys <= set(schema["properties"])


def test_question_types_match_okg():
    from okg.substrate.catalog.profile import INIT_QUESTION_TYPES

    assert tuple(QUESTION_TYPES) == tuple(INIT_QUESTION_TYPES)


# ---------------------------------------------------------------------------
# Valid fixtures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pack", ["alpha", "beta"])
def test_valid_fixture_packs_pass(pack):
    assert validate_pack(CATALOG / "packs" / pack) == []


def test_valid_fixture_catalog_loads_clean():
    catalog = _catalog(CATALOG)
    assert catalog.problems == []
    assert sorted(catalog.packs) == ["alpha", "beta"]
    assert sorted(catalog.bundles) == ["base-bundle", "child-bundle"]


# ---------------------------------------------------------------------------
# Invalid packs: one broken thing each
# ---------------------------------------------------------------------------


def _pack(root, name="alpha"):
    return root / "packs" / name


INVALID_PACKS = {
    "unknown top-level key": (
        lambda root: edit_yaml(_pack(root) / "pack.yaml", lambda d: d.update(colour="blue")),
        "Additional properties are not allowed",
    ),
    "unknown nested key": (
        lambda root: edit_yaml(
            _pack(root) / "pack.yaml", lambda d: d["requires"].update(services=["x"])
        ),
        "Additional properties are not allowed",
    ),
    "missing required key": (
        lambda root: edit_yaml(_pack(root) / "pack.yaml", lambda d: d.pop("needs_credentials")),
        "'needs_credentials' is a required property",
    ),
    "bad kind": (
        lambda root: edit_yaml(_pack(root) / "pack.yaml", lambda d: d.update(kind="magic")),
        "is not one of",
    ),
    "name differs from directory": (
        lambda root: edit_yaml(_pack(root) / "pack.yaml", lambda d: d.update(name="gamma")),
        "must equal the directory name",
    ),
    "missing sources directory": (
        lambda root: edit_yaml(_pack(root) / "pack.yaml", lambda d: d.update(sources="nowhere/")),
        "sources: directory 'nowhere/' is missing",
    ),
    "missing invariants file": (
        lambda root: (_pack(root) / "invariants.yaml").unlink(),
        "invariants: file 'invariants.yaml' is missing",
    ),
    "missing prompt fragment": (
        lambda root: (_pack(root) / "prompt.md").unlink(),
        "prompt_fragment: file 'prompt.md' is missing",
    ),
    "path escaping the pack": (
        lambda root: edit_yaml(
            _pack(root) / "pack.yaml", lambda d: d.update(schemas="../beta/sources")
        ),
        "should not be valid",
    ),
    "credential value instead of a name": (
        lambda root: edit_yaml(
            _pack(root, "beta") / "pack.yaml", lambda d: d.update(credentials=["s3cr3t-value"])
        ),
        "does not match",
    ),
    "credential value in a source template": (
        lambda root: edit_yaml(
            _pack(root, "beta") / "sources" / "issues.yaml",
            lambda d: d["issues"]["params"].update(api_token="abc123def"),
        ),
        "looks like a credential value",
    ),
    "credential question ships a default": (
        lambda root: edit_yaml(
            _pack(root) / "questions.yaml",
            lambda d: d["init_questions"].append(
                {"id": "api_token", "prompt": "token", "default": "abc123"}
            ),
        ),
        "a credential question may not ship a value",
    ),
    "template references an unknown question": (
        lambda root: edit_yaml(
            _pack(root) / "sources" / "pages.yaml",
            lambda d: d["pages"]["params"].update(extra="${nowhere}"),
        ),
        "references ${nowhere}",
    ),
    "prepare module is not a dotted name": (
        lambda root: edit_yaml(
            _pack(root) / "pack.yaml",
            lambda d: d.update(prepare=[{"name": "fetch", "module": "rm -rf /"}]),
        ),
        "does not match",
    ),
    "prepare step with an unknown key": (
        lambda root: edit_yaml(
            _pack(root) / "pack.yaml",
            lambda d: d.update(prepare=[{"name": "fetch", "module": "a.b", "shell": "ls"}]),
        ),
        "Additional properties are not allowed",
    ),
    "prepare timeout out of range": (
        lambda root: edit_yaml(
            _pack(root) / "pack.yaml",
            lambda d: d.update(prepare=[{"name": "fetch", "module": "a.b", "timeout": 0}]),
        ),
        "less than the minimum",
    ),
    "prepare references an unknown question": (
        lambda root: edit_yaml(
            _pack(root) / "pack.yaml",
            lambda d: d.update(prepare=[{"name": "fetch", "module": "a.b", "args": ["${nowhere}"]}]),
        ),
        "prepare[0].args[0]: references ${nowhere}",
    ),
    "prepare repeats a step name": (
        lambda root: edit_yaml(
            _pack(root) / "pack.yaml",
            lambda d: d.update(prepare=[
                {"name": "fetch", "module": "a.b"}, {"name": "fetch", "module": "a.c"},
            ]),
        ),
        "duplicate step 'fetch'",
    ),
    "template key differs from file name": (
        lambda root: (_pack(root) / "sources" / "pages.yaml").rename(
            _pack(root) / "sources" / "other.yaml"
        ),
        "must hold exactly one top-level key, 'other'",
    ),
    "freshness for another pack's source": (
        lambda root: edit_yaml(
            _pack(root) / "pack.yaml",
            lambda d: d.update(freshness={"beta.issues": {"max_staleness": "1h"}}),
        ),
        "is not one of this pack's sources",
    ),
    "source without max_staleness": (
        lambda root: edit_yaml(_pack(root) / "pack.yaml", lambda d: d.pop("freshness")),
        "declares no max_staleness",
    ),
    "link target for an edge no source declares": (
        lambda root: edit_yaml(
            _pack(root) / "pack.yaml",
            lambda d: d.update(link_coverage=[
                {"name": "x", "from": "page", "edge": "cites", "to": "page", "min_fraction": 0.5}
            ]),
        ),
        "no source of this pack declares the edge",
    ),
    "link target above 100 percent": (
        lambda root: edit_yaml(
            _pack(root) / "pack.yaml",
            lambda d: d.update(link_coverage=[
                {"name": "x", "from": "page", "edge": "cites", "to": "page", "min_fraction": 1.5}
            ]),
        ),
        "greater than the maximum",
    ),
    "bad staleness": (
        lambda root: edit_yaml(
            _pack(root) / "pack.yaml",
            lambda d: d.update(freshness={"alpha.pages": {"max_staleness": "soon"}}),
        ),
        "does not match",
    ),
    "unknown question key": (
        lambda root: edit_yaml(
            _pack(root) / "questions.yaml",
            lambda d: d["init_questions"][0].update(secret=True),
        ),
        "unknown keys ['secret']",
    ),
    "bad question type": (
        lambda root: edit_yaml(
            _pack(root) / "questions.yaml",
            lambda d: d["init_questions"][0].update(type="url"),
        ),
        "is not one of",
    ),
    "duplicate invariant names": (
        lambda root: edit_yaml(
            _pack(root) / "invariants.yaml",
            lambda d: d["invariants"].append(dict(d["invariants"][0])),
        ),
        "invariant names must be unique",
    ),
    "context pack without sources": (
        lambda root: edit_yaml(_pack(root, "beta") / "pack.yaml", lambda d: d.pop("sources")),
        "a context pack must declare `sources:` or `schemas:`",
    ),
}


#: Cases that break beta; every other case breaks alpha.
PACK_OF = {
    "credential value instead of a name": "beta",
    "credential value in a source template": "beta",
    "context pack without sources": "beta",
}


@pytest.mark.parametrize("case", sorted(INVALID_PACKS))
def test_invalid_pack_is_rejected(case, catalog_copy):
    breaker, expected = INVALID_PACKS[case]
    breaker(catalog_copy)
    pack = PACK_OF.get(case, "alpha")
    problems = [str(p) for p in validate_pack(catalog_copy / "packs" / pack)]
    assert any(expected in p for p in problems), problems


# ---------------------------------------------------------------------------
# Invalid bundles
# ---------------------------------------------------------------------------

INVALID_BUNDLES = {
    "unknown key": (
        "base-bundle",
        lambda d: d.update(schedules=["x"]),
        "Additional properties are not allowed",
    ),
    "unknown chat key": (
        "base-bundle",
        lambda d: d["chat"].update(theme="dark"),
        "Additional properties are not allowed",
    ),
    "missing okg range": ("base-bundle", lambda d: d.pop("okg"), "'okg' is a required property"),
    "malformed okg range": (
        "base-bundle",
        lambda d: d.update(okg="0.2 or later"),
        "is not a PEP 440 version range",
    ),
    "unknown pack": (
        "base-bundle",
        lambda d: d["packs"]["optional"].append("nope"),
        "unknown pack 'nope'",
    ),
    "unknown parent": (
        "child-bundle",
        lambda d: d.update(extends=["nope"]),
        "unknown bundle 'nope'",
    ),
    "pack both default and optional": (
        "base-bundle",
        lambda d: d["packs"]["optional"].append("alpha"),
        "both default and optional",
    ),
    "missing eval file": (
        "base-bundle",
        lambda d: d.update(eval="eval/smoke.jsonl"),
        "eval: file 'eval/smoke.jsonl' is missing",
    ),
    "no credential-free default pack": (
        "base-bundle",
        lambda d: d["packs"].update(default=["beta"], optional=[]),
        "no credential-free default context pack",
    ),
}


@pytest.mark.parametrize("case", sorted(INVALID_BUNDLES))
def test_invalid_bundle_is_rejected(case, catalog_copy):
    name, change, expected = INVALID_BUNDLES[case]
    edit_yaml(catalog_copy / name / "bundle.yaml", change)
    catalog = _catalog(CATALOG)  # the clean packs and bundles
    problems = [str(p) for p in validate_bundle(catalog_copy / name, catalog.packs, catalog.bundles)]
    assert any(expected in p for p in problems), problems


def test_extends_cycle_is_rejected(catalog_copy):
    edit_yaml(catalog_copy / "base-bundle" / "bundle.yaml", lambda d: d.update(extends=["child-bundle"]))
    catalog = _catalog(catalog_copy)
    assert any("cycle" in str(p) for p in catalog.problems), catalog.problems


def test_inherited_credential_free_default_satisfies_a_child():
    """child-bundle's own default (beta) needs a token; base-bundle's alpha does not."""
    catalog = _catalog(CATALOG)
    assert validate_bundle(CATALOG / "child-bundle", catalog.packs, catalog.bundles) == []


def test_a_tool_pack_does_not_count_as_a_credential_free_default(catalog_copy):
    edit_yaml(catalog_copy / "packs" / "alpha" / "pack.yaml", lambda d: d.update(kind="tool"))
    catalog = _catalog(catalog_copy)
    assert any("no credential-free default" in str(p) for p in catalog.problems)


def test_duplicate_pack_names_across_roots_are_reported(catalog_copy, tmp_path):
    other = tmp_path / "other"
    shutil.copytree(catalog_copy / "packs" / "alpha", other / "alpha")
    catalog = discovery.from_roots([catalog_copy / "packs", other], [catalog_copy])
    assert any("also provided by" in str(p) for p in catalog.problems)


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------


def test_validate_command_passes_on_valid_fixtures(capsys):
    rc = archi_main(
        ["bundles", "validate", str(CATALOG / "packs" / "alpha"), str(CATALOG / "child-bundle"),
         "--pack-root", str(CATALOG / "packs"), "--bundle-root", str(CATALOG)]
    )
    assert rc == 0, capsys.readouterr().err


def test_validate_command_fails_and_lists_every_problem(catalog_copy, capsys):
    edit_yaml(catalog_copy / "packs" / "alpha" / "pack.yaml", lambda d: d.update(colour="blue", kind="x"))
    rc = archi_main(["bundles", "validate", str(catalog_copy / "packs" / "alpha" / "pack.yaml")])
    err = capsys.readouterr().err
    assert rc == 1
    assert "Additional properties" in err and "is not one of" in err


def test_validate_command_with_no_paths_checks_everything_found(catalog_copy, capsys):
    edit_yaml(catalog_copy / "packs" / "beta" / "pack.yaml", lambda d: d.update(colour="blue"))
    rc = archi_main(
        ["bundles", "validate", "--pack-root", str(catalog_copy / "packs"),
         "--bundle-root", str(catalog_copy)]
    )
    assert rc == 1
    assert "colour" in capsys.readouterr().err


def test_shipped_catalog_validates():
    """Whatever archi itself ships must pass its own validator."""
    assert archi_main(["bundles", "validate"]) == 0


def test_a_schema_only_context_pack_validates_but_does_not_count_as_a_publishing_default(
    catalog_copy,
):
    """ops-record's shape: a context pack with a schema slice and no sources is
    valid, but cannot be what makes a bundle publish with nothing configured."""
    import shutil

    alpha = catalog_copy / "packs" / "alpha"
    shutil.rmtree(alpha / "sources")
    edit_yaml(alpha / "pack.yaml", lambda d: (d.pop("sources"), d.pop("freshness")))
    catalog = _catalog(catalog_copy)
    problems = [str(p) for p in catalog.problems]
    assert not any("alpha/pack.yaml" in p for p in problems), problems
    assert any("no credential-free default" in p for p in problems), problems
