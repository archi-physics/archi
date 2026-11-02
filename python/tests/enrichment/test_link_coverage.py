"""req.r3-context-quality.linking: a pack's link-coverage target becomes an
okg invariant, and a corpus with broken links fails it
(acc.r3-context-quality.coverage).

Evaluation: the generated SQL reads `okg.v_nodes` and `okg.v_edges` and sticks
to syntax SQLite shares with Postgres, so these tests run the generated text
over fixture rows in an in-memory SQLite database (the `okg` schema is an
ATTACHed database). They do not run it on Postgres; the pack-isolation test
does that when a database is available. `coverage()` is the Python statement
of the same arithmetic and is checked against the SQL.
"""
import sqlite3

import pytest
import yaml

from archi.bundles import discovery
from archi.bundles.compose import compose
from archi.bundles.manifest import load_pack, validate_pack
from archi.bundles.resolve import resolve
from archi.enrichment.link_coverage import (
    coverage,
    invariant_name,
    link_coverage_invariant,
    link_coverage_invariants,
)

TARGET = {"name": "storage_endpoint_site", "from": "storage_endpoint", "edge": "contains",
          "to": "site", "direction": "in", "min_fraction": 0.9}


def database(nodes, edges):
    """`okg.v_nodes(node_id, subtype)` and `okg.v_edges(edge_id, src, dst, edge_type)`."""
    db = sqlite3.connect(":memory:")
    db.execute("ATTACH DATABASE ':memory:' AS okg")
    db.execute("CREATE TABLE okg.v_nodes (node_id TEXT, subtype TEXT)")
    db.execute("CREATE TABLE okg.v_edges (edge_id INTEGER, src TEXT, dst TEXT, edge_type TEXT)")
    db.executemany("INSERT INTO okg.v_nodes VALUES (?, ?)", nodes)
    db.executemany("INSERT INTO okg.v_edges VALUES (?, ?, ?, ?)",
                   [(i, s, d, t) for i, (s, d, t) in enumerate(edges)])
    return db


def violations(target, nodes, edges):
    return database(nodes, edges).execute(link_coverage_invariant("cms-catalogs", target)["sql"]).fetchall()


def corpus(n_endpoints, n_linked):
    nodes = [("site:T1", "site")] + [(f"se:{i}", "storage_endpoint") for i in range(n_endpoints)]
    edges = [("site:T1", f"se:{i}", "contains") for i in range(n_linked)]
    return nodes, edges


def test_a_corpus_with_broken_links_fails_the_invariant():
    nodes, edges = corpus(n_endpoints=10, n_linked=5)
    [row] = violations(TARGET, nodes, edges)
    assert row == ("storage_endpoint", 10, 5, 0.9)
    assert coverage((n for n, t in nodes if t == "storage_endpoint"), (d for _, d, _ in edges))[2] == 0.5


def test_a_well_linked_corpus_passes():
    nodes, edges = corpus(n_endpoints=10, n_linked=9)  # 90 percent meets the 0.9 target
    assert violations(TARGET, nodes, edges) == []


def test_the_target_is_a_share_not_a_count():
    nodes, edges = corpus(n_endpoints=10, n_linked=8)
    assert violations(TARGET, nodes, edges) != []
    assert violations({**TARGET, "min_fraction": 0.8}, nodes, edges) == []


def test_only_the_named_edge_between_the_named_subtypes_counts():
    nodes, edges = corpus(n_endpoints=4, n_linked=0)
    wrong_edge = [("site:T1", f"se:{i}", "hosts") for i in range(4)]
    wrong_peer = [(f"se:{(i + 1) % 4}", f"se:{i}", "contains") for i in range(4)]
    assert violations(TARGET, nodes, wrong_edge) != []
    assert violations(TARGET, nodes, wrong_peer) != []


def test_a_dangling_edge_to_a_node_that_is_not_there_does_not_count():
    nodes, _ = corpus(n_endpoints=2, n_linked=0)
    assert violations(TARGET, nodes, [("site:gone", "se:0", "contains"), ("site:gone", "se:1", "contains")]) != []


def test_direction_out_reads_the_edge_from_the_carrier():
    target = {"name": "wf_ds", "from": "workflow", "edge": "consumes", "to": "dataset", "min_fraction": 0.5}
    nodes = [(f"wf:{i}", "workflow") for i in range(4)] + [("ds:1", "dataset")]
    assert violations(target, nodes, [("wf:0", "ds:1", "consumes")]) != []
    assert violations(target, nodes, [("wf:0", "ds:1", "consumes"), ("wf:1", "ds:1", "consumes")]) == []


def test_no_carriers_is_the_coverage_invariants_finding_not_this_ones():
    assert violations(TARGET, [("site:T1", "site")], []) == []


def test_shares_agree_with_the_python_statement_of_the_rule():
    for n, linked in [(10, 0), (10, 5), (10, 9), (10, 10), (3, 2), (7, 7)]:
        nodes, edges = corpus(n, linked)
        total, got, fraction = coverage((i for i, t in nodes if t == "storage_endpoint"),
                                        (d for _, d, _ in edges))
        assert (fraction < TARGET["min_fraction"]) == bool(violations(TARGET, nodes, edges)), (n, linked)
    assert coverage([], []) == (0, 0, 1.0)


# ---------------------------------------------------------------------------
# From pack.yaml to the invariant okg loads
# ---------------------------------------------------------------------------


def test_the_generated_invariant_has_the_packs_invariant_format():
    inv = link_coverage_invariant("cms-catalogs", TARGET)
    assert inv["name"] == invariant_name("cms-catalogs", "storage_endpoint_site") == \
        "cms_catalogs_link_storage_endpoint_site"
    assert (inv["severity"], inv["scope"]) == ("warn", "cross")
    assert {"name", "severity", "scope", "timeout_ms", "description", "sql"} == set(inv)
    assert "90%" in inv["description"]
    assert link_coverage_invariant("p", {**TARGET, "severity": "error"})["severity"] == "error"


@pytest.fixture(scope="module")
def shipped():
    return discovery.discover([], [])


def test_composition_adds_one_invariant_per_target_and_okg_accepts_them(shipped, tmp_path):
    from okg.substrate.catalog.invariants import load_invariants

    comp = compose(resolve(shipped, ["cern-team"], with_packs=["jira", "git-code", "code-symbols"]))
    generated = [i for i in comp.invariants if "_link_" in i["name"]]
    assert {i["name"] for i in generated} == {
        "jira_link_issue_reporter", "git_code_link_tag_commit", "code_symbols_link_symbol_file"}
    for inv in generated:
        assert comp.owners[f"invariant:{inv['name']}"] == [inv["name"].split("_link_")[0].replace("_", "-")]
    path = tmp_path / "invariants.yaml"
    path.write_text(yaml.safe_dump({"invariants": generated}))
    assert [spec.name for spec in load_invariants(path)] == [i["name"] for i in generated]


def test_every_shipped_target_matches_an_edge_the_pack_declares(shipped):
    targets = {}
    for name, pack in shipped.packs.items():
        assert validate_pack(pack.dir) == []  # includes the output_signature edge check
        if pack.link_coverage:
            targets[name] = [t["name"] for t in pack.link_coverage]
            assert len(link_coverage_invariants(name, pack.link_coverage)) == len(pack.link_coverage)
    assert set(targets) >= {"cms-catalogs", "dqm", "monit-live", "jira", "git-code", "code-symbols"}


def test_a_target_for_an_edge_the_pack_does_not_emit_is_refused(tmp_path):
    import shutil

    shutil.copytree(shipped_pack_dir("jira"), tmp_path / "jira")
    manifest = tmp_path / "jira" / "pack.yaml"
    data = yaml.safe_load(manifest.read_text())
    data["link_coverage"] = [{"name": "bad", "from": "jira_issue", "edge": "blocks",
                              "to": "jira_issue", "min_fraction": 0.5}]
    manifest.write_text(yaml.safe_dump(data))
    problems = [str(p) for p in validate_pack(tmp_path / "jira")]
    assert any("no source of this pack declares the edge" in p for p in problems), problems


def shipped_pack_dir(name):
    from pathlib import Path

    return Path(__file__).resolve().parents[3] / "bundles" / "packs" / name
