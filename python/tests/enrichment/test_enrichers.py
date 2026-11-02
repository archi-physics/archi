"""task.w2.enrichment — each SQL enricher's core transform, offline.

The four ported enrichers are SQL-shaped: candidate rows in, derived
edge candidates out. These tests fake the two substrate seams the
modules call (``_chronos.query`` and ``insert_deterministic_edges``)
and pin the transform — attrs, source_record_id, provenance source
string, and the uuid5 dedupe-key recipe — on minimal fixture rows, so
a port drift that would re-key derived edges at cutover fails here
without a database.
"""
import uuid
from types import SimpleNamespace

import pytest

import archi.enrichment.chunk_reference_rollup as chunk_mod
import archi.enrichment.dqm_run_range as dqm_mod
import archi.enrichment.jira_affects as jira_mod
import archi.enrichment.meeting_document_reference_rollup as meeting_mod


class FakeCursor:
    def __init__(self, rows=()):
        self.rows = list(rows)
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchall(self):
        return list(self.rows)

    def fetchone(self):
        return self.rows[0] if self.rows else None


class FakeConn:
    def __init__(self, cursor=None):
        self._cursor = cursor or FakeCursor()

    def cursor(self):
        return self._cursor


class FakeChronos:
    """Returns candidate rows; edge-existence lookups return empty."""

    def __init__(self, candidate_rows):
        self.candidate_rows = list(candidate_rows)
        self.calls = []

    def query(self, cur, sql, params=None):
        self.calls.append((sql, params))
        if "edge_id = ANY(:edge_ids)" in sql:
            return []
        return list(self.candidate_rows)


def _capture_insert(captured):
    def fake_insert(cur, *, source, candidates, observed_at=None, run_id=None):
        candidates = list(candidates)
        captured["source"] = source
        captured["candidates"] = candidates
        return SimpleNamespace(inserted=len(candidates), skipped=0)

    return fake_insert


def _dedupe_key(name, src, edge_type, dst):
    return uuid.uuid5(
        uuid.NAMESPACE_URL, f"{name}|{src}|{edge_type}|{dst}"
    ).hex


def test_chunk_reference_rollup_transform(monkeypatch):
    rows = [
        {
            "parent_id": "jira:CMSCOMPPR-1",
            "parent_subtype": "jira_issue",
            "target_id": "site:T2_US_MIT",
            "target_subtype": "site",
            "chunk_count": 3,
        }
    ]
    captured = {}
    monkeypatch.setattr(chunk_mod, "_chronos", FakeChronos(rows))
    monkeypatch.setattr(
        chunk_mod, "insert_deterministic_edges", _capture_insert(captured)
    )
    enricher = chunk_mod.JiraChunkReferenceRollup()
    result = enricher.enrich(FakeConn(), generation_id=7, incremental=None)

    assert result.n_edges_emitted == 1
    assert captured["source"] == "_enricher:cms_jira_chunk_reference_rollup"
    (cand,) = captured["candidates"]
    assert (cand.src, cand.edge_type, cand.dst) == (
        "jira:CMSCOMPPR-1",
        "references",
        "site:T2_US_MIT",
    )
    assert cand.attrs == {
        "match_type": "inherited_chunk_reference",
        "inherited_from": "document_chunk.references",
        "chunk_count": 3,
        "parent_subtype": "jira_issue",
        "target_subtype": "site",
    }
    assert cand.source_record_id["enricher"] == enricher.name
    assert cand.source_revision == {"generation_id": 7}
    assert cand.dedupe_key == _dedupe_key(
        enricher.name, "jira:CMSCOMPPR-1", "references", "site:T2_US_MIT"
    )


def test_chunk_reference_rollup_no_candidates(monkeypatch):
    monkeypatch.setattr(chunk_mod, "_chronos", FakeChronos([]))
    enricher = chunk_mod.JiraChunkReferenceRollup()
    result = enricher.enrich(FakeConn(), generation_id=1, incremental=None)
    assert result.n_edges_emitted == 0


def test_jira_affects_transform(monkeypatch):
    rows = [
        {
            "jira_id": "jira:CMSCOMPPR-2",
            "target_id": "service:eos",
            "target_subtype": "infrastructure_service",
            "reference_count": 4,
        }
    ]
    captured = {}
    monkeypatch.setattr(jira_mod, "_chronos", FakeChronos(rows))
    monkeypatch.setattr(
        jira_mod, "insert_deterministic_edges", _capture_insert(captured)
    )
    enricher = jira_mod.JiraAffectsFromReferences()
    result = enricher.enrich(FakeConn(), generation_id=9, incremental=None)

    assert result.n_edges_emitted == 1
    (cand,) = captured["candidates"]
    assert (cand.src, cand.edge_type, cand.dst) == (
        "jira:CMSCOMPPR-2",
        "affects",
        "service:eos",
    )
    assert cand.attrs == {
        "match_type": "jira_reference_impact",
        "derived_from": "jira_issue.references",
        "reference_count": 4,
        "target_subtype": "infrastructure_service",
    }
    assert cand.dedupe_key == _dedupe_key(
        enricher.name, "jira:CMSCOMPPR-2", "affects", "service:eos"
    )


def test_jira_affects_empty_incremental_scope_short_circuits(monkeypatch):
    chronos = FakeChronos(
        [{"jira_id": "x", "target_id": "y", "target_subtype": "site",
          "reference_count": 1}]
    )
    monkeypatch.setattr(jira_mod, "_chronos", chronos)
    incremental = SimpleNamespace(
        is_first_run=False,
        dirty_node_ids=["site:T2_US_MIT"],  # not jira:* -> filtered out
        prev_edge_watermark=0,
        prev_fact_watermark=0,
    )
    conn = FakeConn(FakeCursor(rows=[]))  # no new edge facts
    enricher = jira_mod.JiraAffectsFromReferences()
    result = enricher.enrich(conn, generation_id=1, incremental=incremental)
    assert result.n_edges_emitted == 0
    assert chronos.calls == []  # never reached the candidate query


def test_dqm_run_range_transform(monkeypatch):
    rows = [
        {
            "cert_id": "cert:2024-golden",
            "run_id": "run:381000",
            "certification_id": "2024-golden",
            "run_min": 380000,
            "run_max": 382000,
            "run_number": 381000,
        }
    ]
    captured = {}
    monkeypatch.setattr(dqm_mod, "_chronos", FakeChronos(rows))
    monkeypatch.setattr(
        dqm_mod, "insert_deterministic_edges", _capture_insert(captured)
    )
    enricher = dqm_mod.DQMRunRangeLinker()
    result = enricher.enrich(FakeConn(), generation_id=3, incremental=None)

    assert result.n_edges_emitted == 1
    (cand,) = captured["candidates"]
    assert (cand.src, cand.edge_type, cand.dst) == (
        "cert:2024-golden",
        "recorded_during",
        "run:381000",
    )
    assert cand.attrs == {
        "match_type": "dqm_certification_run_range",
        "certification_id": "2024-golden",
        "run_min": 380000,
        "run_max": 382000,
        "run_number": 381000,
    }
    assert cand.dedupe_key == _dedupe_key(
        enricher.name, "cert:2024-golden", "recorded_during", "run:381000"
    )


def test_meeting_document_reference_rollup_transform(monkeypatch):
    rows = [
        {
            "meeting_id": "indico:12345",
            "target_id": "cmssw_release:CMSSW_15_0_15",
            "target_subtype": "cmssw_release",
            "support_count": 2,
            "support_kinds": [
                "document.references",
                "document_chunk.references",
            ],
        }
    ]
    captured = {}
    monkeypatch.setattr(meeting_mod, "_chronos", FakeChronos(rows))
    monkeypatch.setattr(
        meeting_mod, "insert_deterministic_edges", _capture_insert(captured)
    )
    enricher = meeting_mod.MeetingDocumentReferenceRollup()
    result = enricher.enrich(FakeConn(), generation_id=5, incremental=None)

    assert result.n_edges_emitted == 1
    (cand,) = captured["candidates"]
    assert (cand.src, cand.edge_type, cand.dst) == (
        "indico:12345",
        "references",
        "cmssw_release:CMSSW_15_0_15",
    )
    assert cand.attrs == {
        "match_type": "inherited_meeting_document_reference",
        "inherited_from": [
            "document.references",
            "document_chunk.references",
        ],
        "support_count": 2,
        "target_subtype": "cmssw_release",
    }
    assert cand.dedupe_key == _dedupe_key(
        enricher.name,
        "indico:12345",
        "references",
        "cmssw_release:CMSSW_15_0_15",
    )


@pytest.mark.parametrize(
    "enricher_cls, expected_name",
    [
        (chunk_mod.JiraChunkReferenceRollup, "cms_jira_chunk_reference_rollup"),
        (jira_mod.JiraAffectsFromReferences, "cms_jira_affects_from_references"),
        (dqm_mod.DQMRunRangeLinker, "cms_dqm_run_range_linker"),
        (
            meeting_mod.MeetingDocumentReferenceRollup,
            "cms_meeting_document_reference_rollup",
        ),
    ],
)
def test_historical_names_kept_for_cutover_parity(enricher_cls, expected_name):
    """Renaming re-keys progress rows and dedupe keys — pinned on purpose."""
    assert enricher_cls.name == expected_name
    assert enricher_cls.requires_narrowings  # declared write surface
