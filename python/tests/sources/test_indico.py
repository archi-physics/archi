"""req.w2.sources-catalogs — IndicoSource emission, offline."""
import hashlib
import json

from okg.deployment import EdgeFact, NodeFact

from archi.sources.indico import IndicoSource

PDF_TEXT = "Site report for T2_US_MIT: transfers nominal."
RECORDS = [
    {
        "id": "1465000",
        "title": "Comp Ops Weekly",
        "description": "<p>Weekly ops meeting</p>",
        "startDate": {"date": "2026-08-03"},
        "endDate": {"date": "2026-08-03"},
        "type": "meeting",
        "category": "CMS Computing",
        "categoryId": 4776,
        "chairs": [{"fullName": "Ada Lovelace"}],
        "folders": [
            {
                "attachments": [
                    {"download_url": "https://indico.cern.ch/a/1.pdf"},
                ],
            },
        ],
        "_contributions_text": "Round table: site reports.",
        "_pdf_texts": [
            {
                "text": PDF_TEXT,
                "title": "site-report.pdf",
                "url": "https://indico.cern.ch/a/1.pdf",
            },
        ],
    },
    {"id": "1465000", "title": "duplicate ignored"},
]


def _source(tmp_path):
    root = tmp_path / "data" / "indico"
    root.mkdir(parents=True)
    (root / "records.json").write_text(json.dumps(RECORDS))
    return IndicoSource(base=str(tmp_path))


def test_meeting_document_and_chunk_emission(tmp_path):
    source = _source(tmp_path)
    run = source.run("run-1", mode="scope_complete")
    facts = list(run.facts)
    nodes = {f.node_id: f for f in facts if isinstance(f, NodeFact)}
    meeting_id = "meeting_minutes:1465000"
    doc_id = "doc:indico:event/1465000/pdf/0"
    # Chunk ids are salted with the parent doc id + chunk index (the
    # hypernews pattern, literal backslash-zero separator) so identical
    # boilerplate in different events cannot collide onto one node.
    chunk_id = "chunk:" + hashlib.sha256(
        f"{doc_id}\\0{0}\\0{PDF_TEXT}".encode("utf-8")
    ).hexdigest()[:16]
    assert set(nodes) == {meeting_id, doc_id, chunk_id}  # dup event dropped
    meeting = nodes[meeting_id]
    assert meeting.subtype == "meeting_minutes"
    assert meeting.attrs["title"] == "Comp Ops Weekly"
    assert meeting.attrs["chairs"] == ["Ada Lovelace"]
    assert meeting.attrs["pdf_text_count"] == 1
    # description (html stripped) and contributions land in text only
    assert "Weekly ops meeting" in meeting.attrs["text"]
    assert "<p>" not in meeting.attrs["text"]
    assert "Round table: site reports." in meeting.attrs["text"]
    doc = nodes[doc_id]
    assert doc.subtype == "document"
    assert doc.attrs["format"] == "pdf"
    assert doc.attrs["title"] == "site-report.pdf"
    chunk = nodes[chunk_id]
    assert chunk.subtype == "document_chunk"
    assert chunk.attrs["text"] == PDF_TEXT
    assert chunk.attrs["chunker_name"] == "cms_indico_pdf_window_v1"
    edges = {
        (e.src, e.edge_type, e.dst)
        for e in facts
        if isinstance(e, EdgeFact)
    }
    assert edges == {
        (meeting_id, "contains", doc_id),
        (doc_id, "contains", chunk_id),
        (chunk_id, "member_of", doc_id),
    }
    assert run.health.record_count == 1


def test_url_fallback_uses_base_url_and_preflight(tmp_path):
    source = _source(tmp_path)
    facts = list(source.run("run-1").facts)
    meeting = next(
        f for f in facts
        if isinstance(f, NodeFact) and f.subtype == "meeting_minutes"
    )
    assert meeting.attrs["url"] == "https://indico.cern.ch/event/1465000/"
    result = source.preflight()
    assert result.status == "ok"
    assert result.record_count == 1


# --- circleback-fixes regressions ---


def test_identical_pdf_text_in_different_events_gets_distinct_chunks(
    tmp_path,
):
    # A content-hash-only chunk id collided identical boilerplate from
    # different events onto one node with contradictory parents.
    records = [
        {"id": "100", "title": "A", "_pdf_texts": [{"text": PDF_TEXT}]},
        {"id": "200", "title": "B", "_pdf_texts": [{"text": PDF_TEXT}]},
    ]
    root = tmp_path / "data" / "indico"
    root.mkdir(parents=True)
    (root / "records.json").write_text(json.dumps(records))
    source = IndicoSource(base=str(tmp_path))
    facts = list(source.run("run-1", mode="scope_complete").facts)
    chunks = [
        f for f in facts
        if isinstance(f, NodeFact) and f.subtype == "document_chunk"
    ]
    assert len(chunks) == 2
    assert len({c.node_id for c in chunks}) == 2
    parents = {
        (e.src, e.dst)
        for e in facts
        if isinstance(e, EdgeFact) and e.edge_type == "contains"
        and e.dst.startswith("chunk:")
    }
    assert len(parents) == 2  # one distinct parent doc per chunk


def test_skipped_cache_items_never_claim_scope(tmp_path):
    root = tmp_path / "data" / "indico"
    root.mkdir(parents=True)
    (root / "records.json").write_text(
        json.dumps(RECORDS + ["junk", {"title": "no id"}])
    )
    source = IndicoSource(base=str(tmp_path))
    run = source.run("run-1", mode="scope_complete")
    nodes = {f.node_id for f in run.facts if isinstance(f, NodeFact)}
    assert "meeting_minutes:1465000" in nodes  # survivors still emitted
    assert run.completed_scope is False
    assert run.health.status == "ok"
    assert "skipped 2" in run.health.reason


def test_all_items_unparseable_is_endpoint_failed(tmp_path):
    root = tmp_path / "data" / "indico"
    root.mkdir(parents=True)
    (root / "records.json").write_text(json.dumps(["junk"]))
    source = IndicoSource(base=str(tmp_path))
    run = source.run("run-1", mode="scope_complete")
    assert list(run.facts) == []
    assert run.completed_scope is False
    assert run.health.status == "endpoint_failed"
