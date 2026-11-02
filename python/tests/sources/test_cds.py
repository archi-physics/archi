"""pact cds-connector — OAI-PMH harvest of CERN Document Server records.

Fixture pages are carved from a real CDS harvest of cerncds:cms-pas, so
the parsing is exercised against the shape the server actually sends
rather than one invented here.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import requests

from archi.sources._cds_parse import (
    CDSOAIError,
    parse_records,
    parse_report_number,
)
from archi.sources.cds import CDSAdapter, CDSSource

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
PAGE1 = (FIXTURES / "cds_page1.xml").read_text(encoding="utf-8")
PAGE2 = (FIXTURES / "cds_page2.xml").read_text(encoding="utf-8")


class _FakeResponse:
    def __init__(self, text: str, status: int = 200) -> None:
        self.text = text
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")


def _pager(monkeypatch, pages, *, fail_after=None):
    """Serve pages in order; optionally raise once the chain is partway."""
    calls = {"n": 0}

    def fake_get(url, params=None, timeout=None, headers=None):
        index = calls["n"]
        calls["n"] += 1
        if fail_after is not None and index >= fail_after:
            raise requests.ConnectionError("connection reset")
        return _FakeResponse(pages[min(index, len(pages) - 1)])

    monkeypatch.setattr(requests, "get", fake_get)
    return calls


# --- parsing -----------------------------------------------------------

def test_parses_a_real_page():
    records, token, size = parse_records(PAGE1)
    assert len(records) == 2
    assert token == "TOKEN-PAGE-2"
    assert size == 4
    first = records[0]
    assert first.recid.isdigit()
    assert first.node_id == f"cds:{first.recid}"
    assert first.title
    assert first.url.startswith("http")


def test_identifiers_are_classified_by_shape():
    """dc:identifier carries a URL, an OAI id and a report number with
    nothing distinguishing them but their shape."""
    record = parse_records(PAGE1)[0][0]
    assert record.url.startswith("http://cds.cern.ch/record/")
    assert record.oai_id.startswith("oai:cds.cern.ch:")
    assert all(not r.raw.startswith("http") for r in record.report_numbers)


def test_oai_error_element_raises():
    xml = (
        '<?xml version="1.0"?><OAI-PMH '
        'xmlns="http://www.openarchives.org/OAI/2.0/">'
        '<error code="badArgument">no such set</error></OAI-PMH>'
    )
    with pytest.raises(CDSOAIError, match="badArgument"):
        parse_records(xml)


def test_deleted_records_are_visible_not_dropped():
    xml = (
        '<?xml version="1.0"?><OAI-PMH '
        'xmlns="http://www.openarchives.org/OAI/2.0/"><ListRecords>'
        '<record><header status="deleted">'
        '<identifier>oai:cds.cern.ch:999</identifier>'
        '<datestamp>2026-01-01T00:00:00Z</datestamp>'
        '</header></record></ListRecords></OAI-PMH>'
    )
    records, _token, _size = parse_records(xml)
    assert len(records) == 1 and records[0].deleted is True


# --- report numbers ----------------------------------------------------

@pytest.mark.parametrize("raw,kind,code", [
    ("CMS-PAS-SUS-08-005", "PAS", "SUS-08-005"),
    ("CMS-PAS-HIG-19-001", "PAS", "HIG-19-001"),
    ("CMS-AN-2019-123", "AN", ""),
    ("CMS-NOTE-2019-001", "NOTE", ""),
    ("CMS-DP-2021-004", "DP", ""),
])
def test_report_number_parsing(raw, kind, code):
    parsed = parse_report_number(raw)
    assert parsed is not None
    assert parsed.kind == kind
    assert parsed.analysis_code == code


def test_non_report_identifiers_are_not_parsed():
    assert parse_report_number("arXiv:1234.5678") is None
    assert parse_report_number("10.1007/JHEP01(2019)001") is None


def test_analysis_code_is_empty_for_unknown_pag_tokens():
    """A token that looks positional but is not a PAG must not produce a
    join key -- it would match nothing, or worse, something wrong."""
    parsed = parse_report_number("CMS-PAS-ZZZ-19-001")
    assert parsed is not None and parsed.pag_code == "ZZZ"
    assert parsed.analysis_code == ""


def test_primary_report_number_prefers_a_joinable_one():
    from archi.sources._cds_parse import CDSRecord, ReportNumber
    joinable = ReportNumber("CMS-PAS-HIG-19-001", "PAS", "HIG", "19", "001")
    plain = ReportNumber("CMS-NOTE-2019-001", "NOTE", "", "2019", "001")
    record = CDSRecord(
        oai_id="oai:cds.cern.ch:1", datestamp="", sets=(), title="t",
        description="", creators=(), subjects=(), date="", url="",
        report_numbers=(plain, joinable),
    )
    assert record.primary_report_number is joinable


# --- the harvest -------------------------------------------------------

def _facts(source, mode="scope_complete"):
    run = source.run("run-1", mode=mode)
    return list(run.facts), run


def test_harvest_follows_the_resumption_chain(monkeypatch):
    _pager(monkeypatch, [PAGE1, PAGE2])
    source = CDSSource(sets=["cerncds:cms-pas"])
    facts, run = _facts(source)
    assert len(facts) == 4
    assert run.health.record_count == 4
    assert run.completed_scope is True
    assert source.last_harvest_report["pages"] == 2


def test_a_broken_chain_does_not_claim_complete_scope(monkeypatch):
    """The records that did arrive look fine. Claiming complete scope
    would retract the missing tail on the next run."""
    _pager(monkeypatch, [PAGE1, PAGE2], fail_after=1)
    source = CDSSource(sets=["cerncds:cms-pas"])
    facts, run = _facts(source)
    assert len(facts) == 2
    assert run.completed_scope is False
    assert run.health.status == "endpoint_failed"
    assert source.last_harvest_report["incomplete_sets"] == ["cerncds:cms-pas"]


def test_max_records_bounds_and_does_not_claim_complete_scope(monkeypatch):
    _pager(monkeypatch, [PAGE1, PAGE2])
    source = CDSSource(sets=["cerncds:cms-pas"], max_records=2)
    facts, run = _facts(source)
    assert len(facts) == 2
    assert run.completed_scope is False
    # The operator's own bound is not a failure: admission must let the
    # publish through, so health is ok and the reason says it was bounded.
    assert run.health.status == "ok"
    assert run.health.record_count == 2
    assert "bounded by max_records=2" in (run.health.reason or "")
    assert source.last_harvest_report["truncated"] is True
    assert source.last_harvest_report["incomplete_sets"] == []


def test_max_records_larger_than_the_set_is_a_complete_harvest(monkeypatch):
    _pager(monkeypatch, [PAGE1, PAGE2])
    source = CDSSource(sets=["cerncds:cms-pas"], max_records=100)
    _facts_list, run = _facts(source)
    assert run.completed_scope is True
    assert run.health.status == "ok"
    assert source.last_harvest_report["truncated"] is False


def test_a_failure_before_the_bound_is_still_endpoint_failed(monkeypatch):
    """A broken chain is a failure even when max_records is also set."""
    _pager(monkeypatch, [PAGE1, PAGE2], fail_after=1)
    source = CDSSource(sets=["cerncds:cms-pas"], max_records=100)
    _facts_list, run = _facts(source)
    assert run.completed_scope is False
    assert run.health.status == "endpoint_failed"


def test_first_set_fails_and_second_hits_the_bound_is_endpoint_failed(monkeypatch):
    """A set that failed is missing from the harvest. The second set's
    bound must not make the run look merely bounded."""
    calls = {"n": 0}

    def fake_get(url, params=None, timeout=None, headers=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.ConnectionError("connection reset")
        return _FakeResponse(PAGE1)

    monkeypatch.setattr(requests, "get", fake_get)
    source = CDSSource(
        sets=["cerncds:cms-pas", "cerncds:cms-notes"], max_records=2
    )
    facts, run = _facts(source)
    assert len(facts) == 2
    assert source.last_harvest_report["truncated"] is True
    assert source.last_harvest_report["incomplete_sets"] == ["cerncds:cms-pas"]
    assert run.completed_scope is False
    assert run.health.status == "endpoint_failed"
    assert "cerncds:cms-pas" in (run.health.reason or "")


def test_a_record_in_two_sets_is_emitted_once(monkeypatch):
    _pager(monkeypatch, [PAGE2])
    source = CDSSource(sets=["cerncds:cms-pas", "cerncds:cms-notes"])
    facts, _run = _facts(source)
    assert len({fact.node_id for fact in facts}) == len(facts)


def test_emitted_attributes_carry_the_join_key(monkeypatch):
    _pager(monkeypatch, [PAGE1, PAGE2])
    facts, _run = _facts(CDSSource(sets=["cerncds:cms-pas"]))
    with_codes = [f for f in facts if f.attrs.get("analysis_code")]
    assert with_codes, "the fixture page should carry PAS report numbers"
    fact = with_codes[0]
    assert fact.subtype == "cds_record"
    assert fact.attrs["report_kind"] == "PAS"
    assert fact.attrs["pag_code"] in fact.attrs["analysis_code"]
    assert fact.attrs["recid"] in fact.node_id


def test_preflight_rejects_a_non_oai_body(monkeypatch):
    """CDS's anti-bot interstitial answers 200 with HTML. Parsing that as
    an empty corpus is the failure this guards."""
    monkeypatch.setattr(
        requests, "get",
        lambda *a, **k: _FakeResponse("<html><title>Making sure…</title></html>"),
    )
    result = CDSSource().preflight()
    assert result.status == "endpoint_failed"
    assert "interstitial" in (result.reason or "")


def test_preflight_ok_on_an_identify_response(monkeypatch):
    monkeypatch.setattr(
        requests, "get",
        lambda *a, **k: _FakeResponse(
            '<?xml version="1.0"?><OAI-PMH><Identify>'
            "<repositoryName>CDS</repositoryName></Identify></OAI-PMH>"
        ),
    )
    assert CDSSource().preflight().status == "ok"


def test_empty_sets_are_refused():
    with pytest.raises(ValueError, match="at least one OAI set"):
        CDSSource(sets=[])


# --- adapter -----------------------------------------------------------

def test_adapter_binds_and_rejects_typos():
    adapter = CDSAdapter(sets=["cerncds:cms-pas"], required=False)
    assert adapter.reader.sets == ("cerncds:cms-pas",)
    with pytest.raises(TypeError):
        CDSAdapter(set=["cerncds:cms-pas"])


def test_adapter_declares_literal_profile_and_probe_kind():
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(CDSAdapter))
    literals = {
        node.targets[0].id: node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
    }
    assert literals["profile"] == "discovery_crawl"
    assert literals["change_probe_kind"] == "content_hash"
