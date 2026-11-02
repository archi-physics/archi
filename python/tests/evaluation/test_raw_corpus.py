"""req.r3-context-quality.context-eval, task raw-corpus: the raw-document corpus, built from
fixture connector caches (the pack tests' recorded shapes) through archi.snapshot.

Needs the `zstd` command, as the snapshot builder does; skipped, with the reason, without it.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml

from archi.evaluation.raw_corpus import (
    CorpusError,
    build_corpus,
    corpus_root,
    newest_corpus,
    plan_sources,
    safe_name,
    verify_corpus,
)

pytestmark = pytest.mark.skipif(shutil.which("zstd") is None, reason="the zstd command is not installed")

TESTS = Path(__file__).resolve().parents[1]
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

PLANTED = "jira.reporter@example.org"


def registry_entry(module: str, **params) -> dict:
    return {"module": module, "class": "X", "params": params}


def write_caches(data: Path) -> dict:
    """Fixture caches under a data root, and the registry an instance would carry for them."""
    from packs.cluster_fixtures import write_slurm_cache, write_tickets_cache
    from packs.pack_support import _docsite, _jira

    _jira(data, None)
    records = json.loads((data / "jira" / "records.json").read_text())
    records[0]["description"] += f" Mail {PLANTED} for access."
    (data / "jira" / "records.json").write_text(json.dumps(records))
    _docsite(data, None)
    write_tickets_cache(data)
    write_slurm_cache(data)
    root = "${archi_data_root}"
    return {
        "jira": registry_entry("archi.sources.jira", records_path=f"{root}/jira/records.json",
                               meta_path=f"{root}/jira/meta.json"),
        "docsite": registry_entry("archi.sources.docs", records_path=f"{root}/docsite/records.json"),
        "tickets": registry_entry("archi.sources.tickets", records_path=f"{root}/tickets/records.json"),
        "slurm": registry_entry("archi.sources.slurm", records_path=f"{root}/slurm/records.json"),
        "indico_live": registry_entry("archi.sources.indico_live"),
        "monit_sam": registry_entry("archi.sources.monit", records_path=f"{root}/monit-sam/records.json"),
        "dqm": {**registry_entry("archi.sources.dqm", records_path=f"{root}/dqm/records.json"), "enabled": False},
        "twiki_eos": registry_entry("archi.sources.twiki", eos_root="${eos_root}"),
        "indico": registry_entry("archi.sources.indico_live", records_path=f"{root}/indico/records.json"),
    }


@pytest.fixture
def built(tmp_path):
    data = tmp_path / "data"
    registry = write_caches(data)
    plan = plan_sources(registry, data_root=data)
    corpus = build_corpus(plan, bundle="demo", out_root=corpus_root(data, "demo"))
    return data, plan, corpus


def test_only_enabled_sources_with_a_checked_cache_go_in_and_every_other_one_is_named(built):
    _, plan, corpus = built
    assert sorted(corpus.sources) == ["docsite", "jira", "slurm", "tickets"]
    left = corpus.left_out
    assert "fetched live" in left["indico_live"] and "MONIT" in left["monit_sam"]
    assert "names a value the deployment does not record" in left["twiki_eos"]
    assert "not the cache reader archi.sources.indico" in left["indico"]
    assert "okg admission" in left["ops-record"] and "both agent arms" in left["ops-record"]
    assert "dqm" not in left and "dqm" not in corpus.sources  # not enabled: neither in nor left out


def test_the_corpus_is_one_file_per_record_under_its_lock_sha(built):
    data, _, corpus = built
    lock = (corpus.path / "snapshot.lock.yaml").read_bytes()
    assert corpus.path.parent == data / "eval" / "raw-corpus" / "demo"
    assert corpus.path.name == hashlib.sha256(lock).hexdigest()[:16] and corpus.lock_sha256.startswith(corpus.path.name)
    files = sorted(p.relative_to(corpus.records).as_posix() for p in corpus.records.rglob("*") if p.is_file())
    assert "jira/DEMO-1.json" in files and "jira/DEMO-2.json" in files
    assert f"docsite/{safe_name('https://docs.example.org/start.html')}.json" in files
    assert any(f.startswith("tickets/") for f in files) and "slurm/submit.json" in files
    assert any(f.startswith("slurm/partitions/") for f in files) and any(f.startswith("slurm/nodes/") for f in files)
    issue = json.loads((corpus.records / "jira" / "DEMO-1.json").read_text())
    assert issue["key"] == "DEMO-1" and issue["summary"] == "Transfer failures at the test site"
    for name in ("README.md", "MANIFEST.txt", "keys.json", "corpus.json"):
        assert (corpus.path / name).is_file()
    assert not (corpus.records / "jira" / "meta.json").exists()  # counts, not records


def test_email_addresses_are_removed(built):
    _, _, corpus = built
    every = b"".join(p.read_bytes() for p in corpus.records.rglob("*") if p.is_file())
    assert PLANTED.encode() not in every and b"@example.org" not in every


def test_the_manifest_lists_every_file_and_the_digest_is_its_hash(built):
    _, _, corpus = built
    text = (corpus.path / "MANIFEST.txt").read_text()
    listed = dict(line.split("\t") for line in text.splitlines())
    on_disk = {p.relative_to(corpus.records).as_posix() for p in corpus.records.rglob("*") if p.is_file()}
    assert set(listed) == on_disk and list(listed) == sorted(listed)
    for rel, sha in listed.items():
        assert hashlib.sha256((corpus.records / rel).read_bytes()).hexdigest() == sha
    assert corpus.digest == "sha256:" + hashlib.sha256(text.encode()).hexdigest()
    assert verify_corpus(corpus.path, bundle="demo").digest == corpus.digest


def test_keys_map_natural_keys_and_node_ids_to_files(built):
    _, _, corpus = built
    keys = corpus.keys()
    jira = keys["sources"]["jira"]
    assert jira["lookup"]["DEMO-1"] == "jira/DEMO-1.json" and jira["natural_key"]["jira/DEMO-1.json"] == "DEMO-1"
    slurm = keys["sources"]["slurm"]
    partition = next(k for k in slurm["lookup"] if k.startswith("slurm_partition:submit:"))
    assert slurm["lookup"][partition].startswith("slurm/partitions/")
    assert keys["sources"]["docsite"]["lookup"]["https://docs.example.org/start.html"].startswith("docsite/")
    assert "ops-record" in keys["left_out"]


@pytest.mark.parametrize("tamper, message", [
    (lambda c: (c.records / "jira" / "DEMO-1.json").write_text("{}"), "changed since the corpus was built"),
    (lambda c: (c.records / "jira" / "EXTRA.json").write_text("{}"), "not listed"),
    (lambda c: (c.records / "jira" / "DEMO-2.json").unlink(), "missing"),
    (lambda c: (c.path / "MANIFEST.txt").write_text("x\ty\n"), "has digest"),
])
def test_a_corpus_that_no_longer_matches_its_manifest_is_refused(built, tamper, message):
    _, _, corpus = built
    tamper(corpus)
    with pytest.raises(CorpusError, match=message):
        verify_corpus(corpus.path)


def test_a_corpus_of_another_bundle_is_refused(built):
    _, _, corpus = built
    with pytest.raises(CorpusError, match="not 'cern-team'"):
        verify_corpus(corpus.path, bundle="cern-team")


def test_the_newest_corpus_is_found_by_its_build_time(built):
    data, _, corpus = built
    assert newest_corpus(data, "demo") == corpus.path
    with pytest.raises(CorpusError, match="build one with `archi eval --raw-corpus`"):
        newest_corpus(data, "cern-team")


def test_a_cache_the_snapshot_refuses_refuses_the_corpus_and_writes_nothing(tmp_path):
    data = tmp_path / "data"
    registry = write_caches(data)
    (data / "jira" / "records.json").write_text(json.dumps([{"summary": "no key"}]))
    plan = plan_sources(registry, data_root=data)
    with pytest.raises(CorpusError, match="group 'jira' refused"):
        build_corpus(plan, bundle="demo", out_root=corpus_root(data, "demo"))
    root = corpus_root(data, "demo")
    assert not root.exists() or list(root.iterdir()) == []


def test_an_instance_without_any_cache_source_has_nothing_to_build(tmp_path):
    plan = plan_sources({"indico_live": registry_entry("archi.sources.indico_live")}, data_root=tmp_path)
    with pytest.raises(CorpusError, match="no enabled source"):
        build_corpus(plan, bundle="demo", out_root=tmp_path / "out")


def test_the_command_builds_the_corpus_of_an_instance(tmp_path, monkeypatch, capsys):
    from archi.bundles.cli import main
    from archi.stack.instance import Instance
    from evaluation.test_context import instance_at

    from archi.evaluation import agent as agent_mod
    from evaluation.raw_support import FakeMcp

    class Endpoint(FakeMcp):
        def __init__(self, url, token):
            super().__init__()

    inst = instance_at(tmp_path, ["demo"])
    registry = write_caches(inst.data_root)
    inst.deployment_dir.mkdir(parents=True)
    (inst.deployment_dir / "source_registry.yaml").write_text(yaml.safe_dump({"sources": registry}))
    monkeypatch.setattr(Instance, "load", classmethod(lambda cls, d: inst))
    monkeypatch.setattr(agent_mod, "McpToolBox", Endpoint)
    assert main(["eval", "--raw-corpus"]) == 0
    out = capsys.readouterr().out
    corpus = newest_corpus(inst.data_root, "demo")
    assert f"corpus {corpus}" in out and "left out: ops-record" in out
    assert "generation g-1" in out and "coverage: demo: docsite, jira, slurm, tickets; " in out
    checked = verify_corpus(corpus)
    assert checked.sources.keys() == {"docsite", "jira", "slurm", "tickets"} and checked.generation == "g-1"


def test_without_a_reachable_endpoint_the_generation_is_unknown_and_said_so(tmp_path, monkeypatch, capsys):
    from archi.bundles.cli import main
    from archi.evaluation import agent as agent_mod
    from archi.stack.instance import Instance
    from evaluation.test_context import instance_at

    class Down:
        def __init__(self, url, token):
            pass

        async def __aenter__(self):
            raise agent_mod.ToolError("connection refused")

        async def __aexit__(self, *exc):
            return None

    inst = instance_at(tmp_path, ["demo"])
    registry = write_caches(inst.data_root)
    inst.deployment_dir.mkdir(parents=True)
    (inst.deployment_dir / "source_registry.yaml").write_text(yaml.safe_dump({"sources": registry}))
    monkeypatch.setattr(Instance, "load", classmethod(lambda cls, d: inst))
    monkeypatch.setattr(agent_mod, "McpToolBox", Down)
    assert main(["eval", "--raw-corpus"]) == 0
    assert "the published generation is unknown" in capsys.readouterr().err
    assert verify_corpus(newest_corpus(inst.data_root, "demo")).generation is None


def _topic(parent: str = "") -> str:
    line = f'%META:TOPICPARENT{{name="{parent}"}}%\n' if parent else ""
    return f'%META:TOPICINFO{{author="someone" date="1700000000" version="3"}}%\n{line}---+ Heading\nBody text.\n'


def test_the_twiki_pages_are_the_ones_the_instances_reader_keeps(tmp_path):
    """physics_filter as the instance sets it (a rendered answer, the string "true"): the corpus holds
    the physics pages the graph read, not the whole dump."""
    from archi.sources.twiki import TwikiEOSSource

    data = tmp_path / "data"
    dump = data / "twiki-dump"
    dump.mkdir(parents=True)
    pages = {"HIG-19-001.txt": _topic(), "StatisticsCommittee.txt": _topic("HIG-19-001"),
             "GrandChildPage.txt": _topic("StatisticsCommittee"), "CompOpsTransferTeam.txt": _topic(),
             "HowToRunCrab.txt": _topic(), "RandomUnrelatedPage.txt": _topic()}
    for name, text in pages.items():
        (dump / name).write_text(text)
    registry = {"twiki_eos": registry_entry("archi.sources.twiki", eos_root="${eos_root}", web_root="CMS",
                                            physics_filter="${physics_filter}")}
    plan = plan_sources(registry, data_root=data, answers={"eos_root": str(dump), "physics_filter": "true"})
    corpus = build_corpus(plan, bundle="cms-physics", out_root=corpus_root(data, "cms-physics"))
    kept = sorted(p.name for p in (corpus.records / "twiki-eos").iterdir())
    reader = TwikiEOSSource(eos_root=str(dump), web_root="CMS", physics_filter=True)
    expected = sorted(r.source_path for r in reader._physics_scoped(list(reader._records_from_root(dump).records)))
    assert kept == expected and "CompOpsTransferTeam.txt" not in kept and len(kept) < len(pages)
    assert corpus.sources["twiki_eos"]["scope"] == "physics filter"
    assert corpus.coverage.startswith("cms-physics: twiki_eos (physics filter); ")
    assert "ops-record not included" in corpus.coverage


def test_pack_named_registry_sources_are_planned_and_resolved_by_their_source_name(tmp_path):
    """A rendered instance names its sources `<pack>.<source>` (twiki.twiki_eos, cds.cds). The corpus
    plans them by the source name, and keys.json is keyed so that a node's okg provenance
    (cms-physics.twiki-eos) finds its file."""
    from archi.evaluation import raw_arm

    data = tmp_path / "data"
    dump = data / "twiki-dump"
    dump.mkdir(parents=True)
    (dump / "HIG-19-001.txt").write_text(_topic())
    registry = {
        "twiki.twiki_eos": registry_entry("archi.sources.twiki", eos_root=str(dump), web_root="CMS",
                                          physics_filter="true"),
        "cds.cds": registry_entry("archi.sources.cds"),
        "indico.indico_live": registry_entry("archi.sources.indico_live"),
        "analysis-notes.analysis_notes": registry_entry("archi.sources.analysis_notes"),
    }
    plan = plan_sources(registry, data_root=data)
    assert [p.source for p in plan.included] == ["twiki_eos"]
    assert "harvested live" in plan.left_out["cds"] and "fetched live" in plan.left_out["indico_live"]
    assert "LaTeX" in plan.left_out["analysis_notes"]
    corpus = build_corpus(plan, bundle="cms-physics", out_root=corpus_root(data, "cms-physics"))
    keys = json.loads((corpus.path / "keys.json").read_text())
    info = raw_arm.NodeInfo("twiki:CMS:HIG-19-001", "documentation_page", "cms-physics.twiki-eos", {})
    record = raw_arm.record_for(info, keys)
    assert not isinstance(record, str) and record.path == "twiki-eos/HIG-19-001.txt"
    cds = raw_arm.NodeInfo("cds:1", "cds_record", "cms-physics.cds", {})
    assert raw_arm.record_for(cds, keys).status == "unavailable"


def test_git_code_checkouts_go_in_as_their_tracked_files(tmp_path):
    """The comp-ops gold cites repository files: each enabled git-code checkout's tracked files
    go in under git-repo/<repo>/, emails removed, binaries left out and counted."""
    from packs.pack_support import _git_code, git

    data = tmp_path / "data"
    data.mkdir()
    _git_code(data, None)
    upstream = tmp_path / "upstream" / "demo"
    (upstream / "OWNERS").write_text(f"Maintained by {PLANTED} for the transfer team of the test site.\n")
    (upstream / "logo.png").write_bytes(b"\x89PNG\r\n\x00\x00binary")
    (upstream / "untracked.txt").write_text("not committed")
    git(["add", "OWNERS", "logo.png"], upstream)
    git(["commit", "-q", "-m", "owners"], upstream)
    registry = {
        "repo": {"module": "okg.substrate.library.sources.git_graph_files", "class": "GitFilesSource",
                 "params": {"repo": "${repo_name}", "repo_root": str(upstream)}},
        "git_history": {"module": "okg.substrate.library.sources.git_graph_history", "params": {}},
    }
    plan = plan_sources(registry, data_root=data, answers={"repo_name": "demo"})
    corpus = build_corpus(plan, bundle="cms-comp-ops", out_root=corpus_root(data, "cms-comp-ops"))
    files = sorted(p.relative_to(corpus.records).as_posix() for p in corpus.records.rglob("*") if p.is_file())
    # deploy/db-values.yaml is the fixture's planted secret file (for code-text's screen test).
    assert files == ["git-repo/demo/OWNERS", "git-repo/demo/README.md", "git-repo/demo/deploy/db-values.yaml",
                     "git-repo/demo/pkg/transfer.py"]
    assert PLANTED not in (corpus.records / "git-repo" / "demo" / "OWNERS").read_text()
    row = corpus.sources["git_repo"]
    assert row["repos"]["demo"]["left_out_files"]["binary"] == 1 and len(row["repos"]["demo"]["head"]) == 40
    keys = corpus.keys()["sources"]["git_repo"]
    assert keys["lookup"]["demo/pkg/transfer.py"] == "git-repo/demo/pkg/transfer.py"
    assert keys["lookup"]["pkg/transfer.py"] == "git-repo/demo/pkg/transfer.py"
    assert "commit history" in corpus.left_out["git_history"]
    verify_corpus(corpus.path)


def test_a_twiki_source_whose_filter_is_unknown_is_left_out(tmp_path):
    registry = {"twiki_eos": registry_entry("archi.sources.twiki", eos_root=str(tmp_path),
                                            physics_filter="${physics_filter}")}
    plan = plan_sources(registry, data_root=tmp_path)
    assert plan.included == [] and "physics_filter" in plan.left_out["twiki_eos"]


def test_a_symbolic_link_under_records_is_refused(built, tmp_path):
    _, _, corpus = built
    outside = tmp_path / "secret.txt"
    outside.write_text("host file")
    (corpus.records / "jira" / "link.json").symlink_to(outside)
    with pytest.raises(CorpusError, match="symbolic link"):
        verify_corpus(corpus.path)


def test_the_command_needs_an_instance(tmp_path, monkeypatch, capsys):
    from archi.bundles.cli import main

    monkeypatch.setenv("OKG_CHAT_MCP_TOKEN", "t")
    assert main(["eval", "--raw-corpus", "--mcp-url", "http://127.0.0.1:9/mcp"]) == 2
    assert "reads an instance's caches" in capsys.readouterr().err
