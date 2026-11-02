"""cern-team is a bundle of packs (ADR 0002 section 5; PACT r3-packs-split).

The packs themselves are checked in tests/packs/. These tests hold the
bundle: its default and optional packs, what a bare render selects, and the
chat site a render produces.
"""
import json
import re
from pathlib import Path

import pytest
import yaml

from archi.bundles.render import render
from archi.bundles.resolve import resolve

BUNDLE = Path(__file__).resolve().parents[2] / "bundles" / "cern-team"


def _bundle():
    return yaml.safe_load((BUNDLE / "bundle.yaml").read_text())


@pytest.fixture(scope="module")
def catalog():
    from packs.pack_support import shipped_catalog

    return shipped_catalog()


@pytest.fixture(scope="module")
def bare(catalog, tmp_path_factory):
    return render(resolve(catalog, ["cern-team"]), tmp_path_factory.mktemp("bare"))


@pytest.fixture(scope="module")
def full(catalog, tmp_path_factory):
    optional = _bundle()["packs"]["optional"]
    return render(resolve(catalog, ["cern-team"], with_packs=optional),
                  tmp_path_factory.mktemp("full"))


def test_the_packs_are_the_adr_selection():
    """docsite and indico by default; twiki, cds, jira and the three code
    packs optional; cmssw-releases moved to the CMS bundles."""
    packs = _bundle()["packs"]
    assert packs["default"] == ["docsite", "indico", "core-playbooks"]
    assert packs["optional"] == ["twiki", "cds", "jira", "git-code", "code-text", "code-symbols"]
    assert "cmssw-releases" not in packs["default"] + packs["optional"]


def test_the_bundle_is_only_a_manifest():
    """The old profile files are gone: sources, schemas, invariants and
    playbooks now come from the packs, so a dropped pack leaves no trace."""
    # eval/ holds the smoke questions (PACT r3-bundle-evals), not profile files.
    assert sorted(p.name for p in BUNDLE.iterdir()) == ["README.md", "bundle.yaml", "eval"]


def test_the_bundle_validates_with_a_credential_free_default(catalog):
    assert "cern-team" in catalog.bundles
    defaults = [catalog.packs[p] for p in _bundle()["packs"]["default"]]
    assert any(p.kind == "context" and not p.needs_credentials for p in defaults)


def test_a_bare_render_selects_the_default_packs_only(bare):
    assert sorted(bare.composition.sources) == ["docsite.docsite", "indico.indico_live"]
    sources = bare.profile_dir / "source-defaults"
    assert sorted(p.name for p in sources.iterdir()) == [
        "docsite.docsite.yaml", "indico.indico_live.yaml",
    ]
    names = [i["name"] for i in bare.composition.invariants]
    assert names == ["docsite_coverage", "indico_coverage"]


def test_default_sources_need_no_credentials(bare):
    """A bare install must not depend on a credential: the publish gate fails
    the whole batch when any selected source fails."""
    for sid, text in bare.composition.sources.items():
        entry = yaml.safe_load(text)[sid]
        assert not entry.get("credential_refs"), sid


def test_no_default_source_depends_on_an_answer_that_defaults_to_blank(bare):
    """An answer that defaults to blank is one the operator may never be asked
    about; a default source that needs one installs empty and blocks the
    publish (what github_repo and gitlab_repo once did)."""
    blank = {
        q["id"] for q in bare.composition.questions
        if not q.get("required") and not str(q.get("default") or "").strip()
    }
    for sid, text in bare.composition.sources.items():
        used = set(re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", text))
        assert not used & blank, f"{sid} depends on blank answers {sorted(used & blank)}"


def test_default_install_declares_standalone_release_authority(bare):
    defaults = yaml.safe_load((bare.profile_dir / "deployment-defaults.yaml").read_text())
    assert defaults["release"] == {"kind": "standalone"}
    assert defaults["nomos"]["rollout"]["owner"]["name"] == "archi-cern-team"


def test_chat_declares_a_system_prompt_that_ships(bare):
    """`okg chat sync` refuses a deployment with no prompt source: Open WebUI
    never shows the model the MCP server's own instructions."""
    defaults = yaml.safe_load((bare.profile_dir / "deployment-defaults.yaml").read_text())
    ref = defaults["chat"]["preset"]["system_prompt_ref"]
    assert ref.startswith("skills/")
    text = (bare.profile_dir / ref).read_text(encoding="utf-8").strip()
    assert text
    for operator in ("search", "inspect", "expand"):
        assert operator in text, f"prompt no longer mentions the {operator} tool"


# `inspect`'s input schema (the chat server's tool list): `target.kind` picks
# the subject and each kind accepts only these keys inside `target`.
_INSPECT_TARGET_KEYS = {
    "graph": set(),
    "subtype": {"subtype", "continuation_token", "detail", "page_size"},
    "node": {"node_id", "path", "source", "hydrate"},
    "nodes": {"node_ids", "ref"},
    "text": {"node_id", "path", "source", "at", "lines", "continuation_token"},
    "texts": {"items"},
}


def _prompt(rendered) -> str:
    return (rendered.profile_dir / "skills" / "chat-system-prompt.md").read_text(encoding="utf-8")


def test_chat_prompt_is_generic(full):
    """No deployment name, no tool-name prefix, no `query` operator, with
    every optional pack's fragment appended."""
    text = _prompt(full)
    assert "cms-kb" not in text and "okg-" not in text
    assert not re.search(r"\bquery\b", text)
    assert len(text.split()) <= 900


def test_chat_prompt_carries_each_selected_packs_fragment(bare, full):
    for heading in ("## Documentation sites", "## Indico"):
        assert heading in _prompt(bare)
    assert "## TWiki" not in _prompt(bare)
    for heading in ("## TWiki", "## CDS", "## JIRA", "## Code repository"):
        assert heading in _prompt(full)


def test_chat_prompt_inspect_examples_match_the_input_schema(full):
    """Every `inspect(target=...)` example parses and fits the tool schema;
    `kind` and its keys sit inside `target`."""
    text = _prompt(full)
    flat = re.sub(r"\s+", " ", text)
    kinds = set()
    for raw in re.findall(r"inspect\(target=(\{.*?\})\)", flat):
        target = json.loads(raw)
        kind = target["kind"]
        assert kind in _INSPECT_TARGET_KEYS, f"unknown inspect kind {kind!r}"
        extra = set(target) - {"kind"} - _INSPECT_TARGET_KEYS[kind]
        assert not extra, f"inspect kind {kind!r} does not take {sorted(extra)}"
        kinds.add(kind)
    assert {"graph", "node", "text"} <= kinds
    assert not re.search(r"inspect\([^)]*target=[^)]*next=", text)


def test_chat_prompt_operators_are_the_declared_chat_tools(full):
    defaults = yaml.safe_load((full.profile_dir / "deployment-defaults.yaml").read_text())
    declared = set(defaults["chat"]["mcp"]["tools"])
    named = set(re.findall(r"\*\*([a-z]+)\*\*", _prompt(full)))
    assert named == declared


def test_the_content_search_profile_is_the_selected_packs_subtypes(bare, full):
    def subtypes(rendered):
        data = yaml.safe_load((rendered.profile_dir / "deployment-defaults.yaml").read_text())
        return data["search"]["profiles"]["content"]["subtypes"]

    assert "cmssw_release" not in subtypes(full)
    assert {"documentation_page", "meeting_minutes", "document_chunk"} <= set(subtypes(bare))
    assert "source_file" not in subtypes(bare)
    assert {"source_file", "jira_issue", "cds_record", "code_symbol"} <= set(subtypes(full))
