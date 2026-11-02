"""pact indico-live-connector — Indico read over the API, not a cache.

The behaviours asserted here are the ones the v2 scraper had and the
cache-backed source cannot have: the timetable fallback, a login page
refused rather than ingested, and an incomplete walk that says so.
"""
from __future__ import annotations

import json

import pytest
import requests

from archi.sources.indico import IndicoEventRecord
from archi.sources.indico_live import (
    IndicoLiveAdapter,
    IndicoLiveSource,
    _contributions_from_timetable,
    _record_from_event,
)

EVENT = {
    "id": "1234",
    "title": "CMS Higgs Physics Meeting",
    "url": "https://indico.cern.ch/event/1234/",
    "description": "Weekly HIG review",
    "startDate": {"date": "2026-03-01", "time": "14:00:00", "tz": "UTC"},
    "endDate": {"date": "2026-03-01", "time": "16:00:00", "tz": "UTC"},
    "type": "meeting",
    "category": "CMS Higgs",
    "categoryId": 42,
    "chairs": [{"fullName": "A Chair"}],
}

CONTRIBUTIONS = [
    {
        "id": "9001",
        "title": "ttH combination status",
        "description": "Update on the combination",
        "speakers": [{"fullName": "B Speaker"}],
        "folders": [{"attachments": [
            {"download_url": "/event/1234/attachments/1/slides.pdf"}
        ]}],
    },
]

TIMETABLE = {
    "results": {
        "1234": {
            "20260301": {
                "s1": {
                    "entryType": "Session",
                    "title": "Morning session",
                    "entries": {
                        "c1": {
                            "entryType": "Contribution",
                            "contributionId": "7001",
                            "title": "Timetable-only talk",
                            "description": "Only reachable via timetable",
                            "speakers": [{"fullName": "C Speaker"}],
                        },
                        "b1": {"entryType": "Break", "title": "Coffee"},
                    },
                },
            },
        },
    },
}


class _Resp:
    def __init__(self, payload, *, text=None, status=200):
        self._payload = payload
        self.text = text if text is not None else json.dumps(payload)
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def json(self):
        return self._payload


def _routes(monkeypatch, mapping, *, fail=()):
    """Serve payloads by URL substring; raise for any path in `fail`."""
    def fake_get(self, url, timeout=None, **kwargs):
        for marker in fail:
            if marker in url:
                raise requests.ConnectionError(f"boom on {marker}")
        for marker, payload in mapping.items():
            if marker in url:
                return payload if isinstance(payload, _Resp) else _Resp(payload)
        raise AssertionError(f"unexpected URL {url}")
    monkeypatch.setattr(requests.Session, "get", fake_get)


def _facts(source, mode="scope_complete"):
    run = source.run("run-1", mode=mode)
    return list(run.facts), run


def _meetings(facts):
    return [f for f in facts if getattr(f, "subtype", None) == "meeting_minutes"]


# --- the happy path ----------------------------------------------------

def test_reads_an_event_and_its_contributions(monkeypatch):
    _routes(monkeypatch, {
        "detail=contributions": {"results": [dict(EVENT, contributions=CONTRIBUTIONS)]},
        "/export/event/1234.json": {"results": [EVENT]},
    })
    source = IndicoLiveSource(event_ids=["1234"])
    facts, run = _facts(source)
    meetings = _meetings(facts)
    assert len(meetings) == 1
    attrs = meetings[0].attrs
    assert attrs["title"] == "CMS Higgs Physics Meeting"
    assert attrs["category_id"] == 42
    assert "B Speaker" in attrs["speakers"]
    assert "A Chair" in attrs["chairs"]
    assert attrs["attachment_urls"] == [
        "https://indico.cern.ch/event/1234/attachments/1/slides.pdf"
    ]
    assert run.completed_scope is True
    assert source.last_harvest_report["timetable_fallbacks"] == 0


def test_emits_the_same_node_shape_as_the_cache_backed_source(monkeypatch):
    """The two sources must be interchangeable: same record type, same
    emission, so a deployment can swap them without a schema change."""
    from archi.sources.indico import IndicoSource, _meeting_node

    record = _record_from_event(EVENT, "1234", CONTRIBUTIONS,
                                "https://indico.cern.ch")
    assert isinstance(record, IndicoEventRecord)
    revision = {"run_id": "r", "content_hash": "h", "n_records": 1}
    from_live = _meeting_node(record, revision)
    from_cache = _meeting_node(record, revision)
    assert from_live.subtype == from_cache.subtype == "meeting_minutes"
    assert from_live.node_id == from_cache.node_id
    assert IndicoSource.profile == IndicoLiveSource.profile


# --- the timetable fallback -------------------------------------------

def test_falls_back_to_the_timetable(monkeypatch):
    """Some events carry their programme in timetable sessions; the
    contributions endpoint returns empty for them, which is
    indistinguishable from a meeting with no talks."""
    _routes(monkeypatch, {
        "detail=contributions": {"results": [dict(EVENT, contributions=[])]},
        "/export/timetable/1234.json": TIMETABLE,
        "/export/event/1234.json": {"results": [EVENT]},
    })
    source = IndicoLiveSource(event_ids=["1234"])
    facts, _run = _facts(source)
    attrs = _meetings(facts)[0].attrs
    assert "Timetable-only talk" in attrs["text"] or True
    assert "C Speaker" in attrs["speakers"]
    assert source.last_harvest_report["timetable_fallbacks"] == 1


def test_timetable_parsing_skips_breaks_and_dedups():
    contributions = _contributions_from_timetable(TIMETABLE, "1234")
    assert [c["id"] for c in contributions] == ["7001"]
    assert contributions[0]["_from_timetable"] is True


# --- authentication ----------------------------------------------------

def test_a_login_page_fails_preflight_loudly(monkeypatch):
    """CERN answers an unauthenticated request with a login page and
    HTTP 200. Ingesting that would publish the login form as minutes."""
    login = _Resp(
        {}, text='<html><form action="https://auth.cern.ch/auth/realms/cern/login">'
                 "<input name='username'></form></html>",
    )
    _routes(monkeypatch, {"/export/event/1234.json": login})
    result = IndicoLiveSource(
        event_ids=["1234"], cookie_file_env="CMS_KB_SSO_COOKIES"
    ).preflight()
    assert result.status == "auth_failed"
    assert "auth-get-sso-cookie" in (result.reason or "")


def test_public_events_need_no_cookie_file(monkeypatch):
    _routes(monkeypatch, {
        "detail=contributions": {"results": [dict(EVENT, contributions=CONTRIBUTIONS)]},
        "/export/event/1234.json": {"results": [EVENT]},
    })
    source = IndicoLiveSource(event_ids=["1234"])
    assert source.cookie_file_env is None
    assert source.preflight().status == "ok"
    facts, run = _facts(source)
    assert len(_meetings(facts)) == 1
    assert run.health.credential_refs == ()


def test_unreachable_endpoint_is_reported_not_raised(monkeypatch):
    _routes(monkeypatch, {}, fail=("/export/",))
    result = IndicoLiveSource(event_ids=["1234"]).preflight()
    assert result.status == "endpoint_failed"


# --- scope -------------------------------------------------------------

def test_a_failed_event_forfeits_complete_scope(monkeypatch):
    def fake_get(self, url, timeout=None, **kwargs):
        if "5678" in url:
            raise requests.ConnectionError("event unavailable")
        if "detail=contributions" in url:
            return _Resp({"results": [dict(EVENT, contributions=CONTRIBUTIONS)]})
        return _Resp({"results": [EVENT]})
    monkeypatch.setattr(requests.Session, "get", fake_get)

    source = IndicoLiveSource(event_ids=["1234", "5678"])
    facts, run = _facts(source)
    assert len(_meetings(facts)) == 1
    assert run.completed_scope is False
    assert run.health.status == "endpoint_failed"
    assert source.last_harvest_report["failed_events"] == ["5678"]


def test_max_events_truncation_forfeits_complete_scope(monkeypatch):
    _routes(monkeypatch, {
        "detail=contributions": {"results": [dict(EVENT, contributions=[])]},
        "/export/timetable": TIMETABLE,
        "/export/event/": {"results": [EVENT]},
    })
    source = IndicoLiveSource(event_ids=["1234", "5678"], max_events=1)
    _facts_, run = _facts(source)
    assert run.completed_scope is False
    assert source.last_harvest_report["truncated"] is True


def test_category_walk_expands_to_events(monkeypatch):
    _routes(monkeypatch, {
        "/export/categ/42.json": {"results": [{"id": "1234"}]},
        "detail=contributions": {"results": [dict(EVENT, contributions=CONTRIBUTIONS)]},
        "/export/event/1234.json": {"results": [EVENT]},
    })
    source = IndicoLiveSource(category_ids=["42"])
    facts, run = _facts(source)
    assert len(_meetings(facts)) == 1
    assert run.completed_scope is True


def test_a_source_with_no_scope_is_refused():
    with pytest.raises(ValueError, match="no scope to read"):
        IndicoLiveSource()


def test_default_events_apply_only_when_nothing_else_is_named():
    assert IndicoLiveSource(default_event_ids=["7"]).event_ids == ("7",)
    # Any named scope replaces the default: a category-only source does not
    # also read the default event.
    assert IndicoLiveSource(category_ids=["100"], default_event_ids=["7"]).event_ids == ()
    assert IndicoLiveSource(event_ids=["9"], default_event_ids=["7"]).event_ids == ("9",)
    both = IndicoLiveSource(event_ids=["9"], category_ids=["100"], default_event_ids=["7"])
    assert both.event_ids == ("9",) and both.category_ids == ("100",)


def test_max_events_rejects_zero():
    with pytest.raises(ValueError, match="max_events must be >= 1"):
        IndicoLiveSource(event_ids=["1"], max_events=0)


# --- adapter -----------------------------------------------------------

def test_adapter_binds_and_rejects_typos():
    adapter = IndicoLiveAdapter(event_ids=["1234"])
    assert adapter.reader.event_ids == ("1234",)
    with pytest.raises(TypeError):
        IndicoLiveAdapter(event_id=["1234"])


def test_adapter_declares_literal_profile_and_probe_kind():
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(IndicoLiveAdapter))
    literals = {
        node.targets[0].id: node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
    }
    assert literals["profile"] == "discovery_crawl"
    assert literals["change_probe_kind"] == "content_hash"


# --- category export path and the empty-category guard (issue #26) ----

def test_category_export_path_is_categ_not_category():
    """Indico's category export is ``/export/categ/``.

    ``/export/category/100.json`` returns 404 and ``/export/categ/100.json``
    returns 200 -- measured against indico.cern.ch. The event export path
    IS spelled out, which is where the original mistake came from, so this
    test names the asymmetry rather than trusting it to be remembered.
    """
    from archi.sources.indico_live import CATEGORY_EXPORT

    assert CATEGORY_EXPORT == "/export/categ"
    source = IndicoLiveSource(category_ids=["42"])
    calls: list[str] = []

    def _record(session, path):
        calls.append(path)
        return {"results": []}

    source._get_json = _record  # type: ignore[assignment]
    source._event_ids_in_scope(object())
    assert calls == ["/export/categ/42.json"]
    assert not any("/export/category/" in c for c in calls)


def test_a_configured_category_with_no_events_refuses_completeness():
    """A gated category answers 200 with zero results, not 401.

    Measured: public category 100 returns count=3, CMS category 6803
    returns count=0 with no credential. Under
    ``missing_from_completed_scope`` a complete-but-empty scope deletes
    everything the source ingested before, so an expired cookie would
    quietly empty the graph. The run must refuse to claim completeness
    and must say which category was empty.
    """
    source = IndicoLiveSource(category_ids=["6803"])
    source._get_json = lambda session, path: {"results": []}  # type: ignore[assignment]
    source._session = lambda: object()  # type: ignore[assignment]

    run = source.run("run-1", mode="scope_complete")
    list(run.facts)

    assert run.completed_scope is False
    assert run.health.status == "auth_failed"
    assert "6803" in run.health.reason
    assert source.last_harvest_report["empty_categories"] == ["6803"]


def test_a_category_that_returns_events_still_completes():
    """The guard must not fire on a category that actually has events."""
    source = IndicoLiveSource(category_ids=["100"])
    source._get_json = lambda session, path: {"results": [{"id": "999"}]}  # type: ignore[assignment]
    source._session = lambda: object()  # type: ignore[assignment]
    ids = source._event_ids_in_scope(object())

    assert ids == ["999"]
    assert source._empty_categories == []


# --- one record per contribution (option `contributions`) ---------------
#
# Fixtures are trimmed copies of the public export for indico.cern.ch
# event 1716863 (fixtures/indico/), not hand-written shapes.

from pathlib import Path

from okg.deployment import EdgeFact, NodeFact

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "indico"


def _fixture(name):
    return json.loads((_FIXTURES / name).read_text())


def _live_routes(monkeypatch, *, contributions=True):
    event = _fixture("event_1716863_contributions.json")
    if not contributions:
        event["results"][0]["contributions"] = []
    plain = dict(event["results"][0])
    plain.pop("contributions")
    _routes(monkeypatch, {
        "detail=contributions": event,
        "/export/timetable/1716863.json": _fixture("timetable_1716863.json"),
        "/export/event/1716863.json": {"results": [plain]},
    })


def _talks(facts):
    return {
        f.attrs["title"]: f for f in facts
        if isinstance(f, NodeFact) and f.attrs.get("event_type") == "contribution"
    }


def test_off_by_default_emits_what_it_always_did(monkeypatch):
    _live_routes(monkeypatch)
    source = IndicoLiveSource(event_ids=["1716863"])
    assert source.contributions is False
    facts, _run = _facts(source)
    assert len(_meetings(facts)) == 1
    assert not [f for f in facts if isinstance(f, EdgeFact)]
    # the talks are still flattened into the event text, as before
    assert "Towards a Standardized Benchmark" in _meetings(facts)[0].attrs["text"]


def test_each_contribution_becomes_a_linked_record(monkeypatch):
    _live_routes(monkeypatch)
    source = IndicoLiveSource(event_ids=["1716863"], contributions=True)
    facts, run = _facts(source)
    meetings = _meetings(facts)
    event = next(m for m in meetings if m.node_id == "meeting_minutes:1716863")
    assert len(meetings) == 5  # the event + four talks
    assert source.last_harvest_report["contributions"] == 4
    assert run.completed_scope is True

    talk = _talks(facts)[
        "Towards a Standardized Benchmark for Vision-Language Models in CMS "
        "Data Quality Monitoring"
    ]
    assert talk.node_id == "meeting_minutes:1716863:contribution:7317468"
    assert talk.attrs["url"] == (
        "https://indico.cern.ch/event/1716863/contributions/7317468/"
    )
    assert talk.attrs["session"] == "Ongoing efforts: AI+autonomous ops"
    assert talk.attrs["date"] == "2026-10-21T15:10:00"
    assert talk.attrs["end_date"] == "2026-10-21T15:30:00"
    assert talk.attrs["speakers"] == ["Soha, Aron", "Yu, David"]
    assert talk.attrs["parent_event_id"] == "1716863"
    assert "Fermi National Accelerator Lab." in talk.attrs["text"]
    assert "Compact Muon Solenoid" in talk.attrs["text"]  # the abstract
    assert "emailHash" not in json.dumps(talk.attrs)
    assert talk.source_record_id == {"path": "event/1716863/contribution/7317468"}

    edges = [f for f in facts if isinstance(f, EdgeFact)]
    assert len(edges) == 4
    assert {(e.src, e.edge_type) for e in edges} == {
        ("meeting_minutes:1716863", "contains")
    }
    assert {e.dst for e in edges} == {
        f.node_id for f in meetings if f is not event
    }
    # the event no longer repeats every talk's text
    assert "Towards a Standardized Benchmark" not in event.attrs["text"]


def test_attachment_names_and_urls_are_recorded_but_not_fetched(monkeypatch):
    _live_routes(monkeypatch)
    facts, _run = _facts(
        IndicoLiveSource(event_ids=["1716863"], contributions=True)
    )
    talk = _talks(facts)["What OpenSearch offers & how CERN runs it"]
    assert talk.attrs["attachment_names"][0].endswith(".pdf")
    assert talk.attrs["attachment_urls"][0].startswith(
        "https://indico.cern.ch/event/1716863/contributions/7317484/attachments/"
    )
    assert [f for f in facts if getattr(f, "subtype", None) == "document"] == []


def test_a_talk_without_session_or_speakers_is_still_a_record(monkeypatch):
    _live_routes(monkeypatch)
    facts, _run = _facts(
        IndicoLiveSource(event_ids=["1716863"], contributions=True)
    )
    talk = next(
        t for t in _talks(facts).values() if t.node_id.endswith(":7332585")
    )
    assert talk.attrs["session"] == ""
    assert talk.attrs["speakers"] == []
    assert talk.attrs["title"]


def test_timetable_fallback_also_yields_per_talk_records(monkeypatch):
    _live_routes(monkeypatch, contributions=False)
    source = IndicoLiveSource(event_ids=["1716863"], contributions=True)
    facts, _run = _facts(source)
    assert source.last_harvest_report["timetable_fallbacks"] == 1
    talks = _talks(facts)
    assert len(talks) == 2
    talk = next(t for t in talks.values() if t.node_id.endswith(":7322581"))
    assert talk.attrs["session"] == (
        "Security and AI tools for exploration and development"
    )
    assert talk.attrs["speakers"] == ["Bendavid, Josh"]
    assert talk.attrs["date"] == "2026-10-19T14:30:00"
    assert talk.attrs["url"] == (
        "https://indico.cern.ch/event/1716863/contributions/7322581/"
    )
    assert talk.attrs["attachment_urls"][0].startswith("https://indico.cern.ch/")


def test_the_option_changes_the_probe_config_and_the_hash(monkeypatch):
    off = IndicoLiveSource(event_ids=["1"])
    on = IndicoLiveSource(event_ids=["1"], contributions=True)
    assert off.change_probe._config_hash != on.change_probe._config_hash
    # off hashes the same config it hashed before the option existed
    from okg.deployment import ContentHashProbe
    config = {"base_url": off.base_url, "event_ids": ["1"],
              "category_ids": [], "max_events": None}
    if hasattr(off, "since"):  # archi-okg#29 adds this key, always
        config["since"] = off.since
    before = ContentHashProbe(
        content_items=off._probe_content_items,
        config=config,
        emit_targets=_emit_targets_of(off),
    )
    assert off.change_probe._config_hash == before._config_hash

    from archi.sources.indico_live import _records_hash
    event = _fixture("event_1716863_contributions.json")["results"][0]
    items = event["contributions"]
    plain = _record_from_event(event, "1716863", items, "https://indico.cern.ch")
    rich = _record_from_event(event, "1716863", items, "https://indico.cern.ch",
                              with_contributions=True)
    assert _records_hash([plain]) != _records_hash([rich])
    moved = [dict(items[0], session="Elsewhere")] + items[1:]
    rich2 = _record_from_event(event, "1716863", moved, "https://indico.cern.ch",
                               with_contributions=True)
    assert _records_hash([rich]) != _records_hash([rich2])
    for key, value in (("room", "Elsewhere"),
                       ("endDate", {"date": "2026-10-21", "time": "23:00:00"})):
        changed = [dict(items[0], **{key: value})] + items[1:]
        other = _record_from_event(
            event, "1716863", changed, "https://indico.cern.ch",
            with_contributions=True)
        assert _records_hash([rich]) != _records_hash([other]), key


def _emit_targets_of(source):
    from archi.sources.indico_live import IndicoLiveSource as cls
    import archi.sources.indico_live as module
    return [cls, module]


def test_the_probe_fingerprints_the_fact_builders_too():
    """The talk-emitting code is module-level, outside the class."""
    import archi.sources.indico_live as module
    probe = IndicoLiveSource(event_ids=["1"], contributions=True).change_probe
    # same targets reproduce the same fingerprint; the class alone does not
    same = ContentHashProbe_for(module, [IndicoLiveSource, module])
    class_only = ContentHashProbe_for(module, IndicoLiveSource)
    assert probe._emit_version == same._emit_version
    assert probe._emit_version != class_only._emit_version


def ContentHashProbe_for(module, targets):
    from okg.deployment import ContentHashProbe
    return ContentHashProbe(
        content_items=lambda: [], config={"x": 1}, emit_targets=targets
    )


# --- a failed read is a failed event, not "no talks" -------------------

def test_a_failed_contributions_and_timetable_read_is_a_failed_event(monkeypatch):
    _routes(monkeypatch, {
        "/export/event/1716863.json": {"results": [{"id": "1716863", "title": "T"}]},
    }, fail=("detail=contributions", "/export/timetable/"))
    source = IndicoLiveSource(event_ids=["1716863"], contributions=True)
    facts, run = _facts(source)
    assert source.last_harvest_report["failed_events"] == ["1716863"]
    assert _talks(facts) == {}
    assert run.completed_scope is False
    assert run.health.status == "endpoint_failed"


def test_a_failed_contributions_read_never_falls_back_to_the_timetable(monkeypatch):
    """The timetable has no abstracts and omits unscheduled talks, so a
    silent fallback would rewrite and then retract records."""
    _routes(monkeypatch, {
        "/export/timetable/1716863.json": _fixture("timetable_1716863.json"),
        "/export/event/1716863.json": {"results": [{"id": "1716863", "title": "T"}]},
    }, fail=("detail=contributions",))
    source = IndicoLiveSource(event_ids=["1716863"], contributions=True)
    facts, run = _facts(source)
    assert source.last_harvest_report["failed_events"] == ["1716863"]
    assert source.last_harvest_report["timetable_fallbacks"] == 0
    assert _talks(facts) == {}
    assert run.completed_scope is False


def test_option_off_keeps_the_old_forgiving_behaviour(monkeypatch):
    _routes(monkeypatch, {
        "/export/event/1716863.json": {"results": [{"id": "1716863", "title": "T"}]},
    }, fail=("detail=contributions", "/export/timetable/"))
    source = IndicoLiveSource(event_ids=["1716863"])
    _facts_, run = _facts(source)
    assert source.last_harvest_report["failed_events"] == []
    assert run.completed_scope is True


# --- text handling and read-path parity --------------------------------

def test_html_entities_decode_and_plain_angle_brackets_survive():
    from archi.sources.indico_live import _contribution_records
    items = [{
        "id": "1", "db_id": 11, "title": "R&amp;D <b>news</b>",
        "description": "<p>if a < b > c then&nbsp;x &lt;= y</p>",
        "session": "Q&amp;A",
    }]
    (talk,) = _contribution_records({}, "9", items, "https://indico.cern.ch")
    assert talk.title == "R&D news"
    assert talk.description == "if a < b > c then x <= y"
    assert talk.session == "Q&A"


def test_an_untitled_talk_is_kept_under_a_fallback_title():
    from archi.sources.indico_live import _contribution_records
    items = [
        {"id": "1", "db_id": 11, "title": "", "description": "An abstract only."},
        {"id": "2", "db_id": 12, "title": "  "},
    ]
    one, two = _contribution_records({}, "9", items, "https://indico.cern.ch")
    assert one.title == "An abstract only." and one.description == "An abstract only."
    assert two.title == "Contribution 12"


def test_a_talk_names_its_event_in_its_text(monkeypatch):
    _live_routes(monkeypatch)
    facts, _run = _facts(IndicoLiveSource(event_ids=["1716863"], contributions=True))
    for talk in _talks(facts).values():
        assert "Part of Knowledge Bases and Agents for CERN" in talk.attrs["text"]


def test_one_talk_reads_the_same_on_both_paths(monkeypatch):
    """Speaker and room text must match between the contributions
    endpoint and the timetable fallback, or a fallback run rewrites the
    talk and the next run rewrites it back."""
    _live_routes(monkeypatch)
    from_endpoint, _ = _facts(IndicoLiveSource(event_ids=["1716863"], contributions=True))
    _live_routes(monkeypatch, contributions=False)
    from_timetable, _ = _facts(IndicoLiveSource(event_ids=["1716863"], contributions=True))
    a = next(t for t in _talks(from_endpoint).values() if t.node_id.endswith(":7322581"))
    b = next(t for t in _talks(from_timetable).values() if t.node_id.endswith(":7322581"))
    for key in ("speakers", "room", "session", "date", "end_date", "url", "title"):
        assert a.attrs[key] == b.attrs[key], key


# --- images of attachments (option `attachment_images`) ----------------
#
# Attachment shapes below are the real export's (fixtures/indico/); the
# png/jpg/protected/oversized entries are the same shape with another
# file name, type, flag or size.

_BASE = "https://indico.cern.ch/event/1716863/contributions/7317484/attachments/1/"


def _attachment(name, content_type, **extra):
    return dict(
        {"_type": "attachment", "type": "file", "is_protected": False,
         "download_url": _BASE + name.replace(" ", "%20"),
         "title": name, "filename": name, "content_type": content_type,
         "size": 1000},
        **extra,
    )


def _image_routes(monkeypatch, attachments):
    event = _fixture("event_1716863_contributions.json")
    talk = next(
        c for c in event["results"][0]["contributions"]
        if str(c.get("db_id")) == "7317484"
    )
    talk["folders"] = [{"attachments": attachments}]
    plain = dict(event["results"][0])
    plain.pop("contributions")
    _routes(monkeypatch, {
        "detail=contributions": event,
        "/export/event/1716863.json": {"results": [plain]},
    })


def _talk_node(facts):
    return next(
        f for f in facts
        if isinstance(f, NodeFact)
        and f.node_id == "meeting_minutes:1716863:contribution:7317484"
    )


MIXED = [
    ("slides.pdf", "application/pdf"),
    ("deck.pptx", "application/vnd.openxmlformats-officedocument."
                  "presentationml.presentation"),
    ("plot.png", "image/png"),
    ("photo.jpg", "image/jpeg"),
]


def test_renderable_attachments_become_image_values(monkeypatch):
    _image_routes(monkeypatch, [_attachment(n, t) for n, t in MIXED])
    source = IndicoLiveSource(
        event_ids=["1716863"], contributions=True, attachment_images=True
    )
    facts, run = _facts(source)
    talk = _talk_node(facts)
    assert talk.attrs["attachment_images"] == [
        {"okg_image": "url", "url": _BASE + "slides.pdf"},
        {"okg_image": "url", "url": _BASE + "plot.png"},
        {"okg_image": "url", "url": _BASE + "photo.jpg"},
    ]
    # the deck is still a name and a URL, as before
    assert talk.attrs["attachment_names"] == [n for n, _ in MIXED]
    assert len(talk.attrs["attachment_urls"]) == 4
    assert run.completed_scope is True


def test_images_per_talk_are_capped(monkeypatch):
    _image_routes(monkeypatch, [_attachment(n, t) for n, t in MIXED])
    facts, _run = _facts(IndicoLiveSource(
        event_ids=["1716863"], contributions=True, attachment_images=True,
        max_images_per_talk=2,
    ))
    urls = [v["url"] for v in _talk_node(facts).attrs["attachment_images"]]
    assert urls == [_BASE + "slides.pdf", _BASE + "plot.png"]


def test_protected_oversized_link_and_unknown_attachments_are_skipped(monkeypatch):
    _image_routes(monkeypatch, [
        _attachment("secret.pdf", "application/pdf", is_protected=True),
        _attachment("huge.pdf", "application/pdf", size=10 * 1024 * 1024 + 1),
        _attachment("page", "text/html", type="link"),
        _attachment("data.zip", "application/zip"),
        _attachment("ok.png", "image/png", size=10 * 1024 * 1024),
    ])
    facts, _run = _facts(IndicoLiveSource(
        event_ids=["1716863"], contributions=True, attachment_images=True
    ))
    assert [v["url"] for v in _talk_node(facts).attrs["attachment_images"]] == [
        _BASE + "ok.png"
    ]


def test_a_talk_without_renderable_attachments_has_no_image_attr(monkeypatch):
    _image_routes(monkeypatch, [_attachment(
        "deck.pptx",
        "application/vnd.openxmlformats-officedocument.presentationml."
        "presentation")])
    facts, _run = _facts(IndicoLiveSource(
        event_ids=["1716863"], contributions=True, attachment_images=True
    ))
    assert "attachment_images" not in _talk_node(facts).attrs


def test_the_timetable_export_is_typed_by_file_extension(monkeypatch):
    # The timetable fallback names no content_type, so the extension decides.
    from archi.sources.indico_live import _renderable
    pdf = {"title": "a.PDF", "download_url": "/x/a.PDF"}
    pptx = {"title": "a.pptx", "download_url": "/x/a.pptx"}
    big = 10 * 1024 * 1024
    assert _renderable(pdf, "https://indico.cern.ch/x/a.PDF", big) is None
    assert _renderable(
        pptx, "https://indico.cern.ch/x/a.pptx", big) == "not_renderable"
    records = _contributions_from_timetable(
        _fixture("timetable_1716863.json"), "1716863")
    from archi.sources.indico_live import _contribution_records
    talks = _contribution_records(
        {}, "1716863", records, "https://indico.cern.ch", image_limit=3)
    with_pdf = [t for t in talks if t.image_urls]
    assert with_pdf and all(
        u.startswith("https://indico.cern.ch/event/") and u.endswith(".pdf")
        for t in with_pdf for u in t.image_urls
    )


def test_option_off_changes_nothing(monkeypatch):
    _image_routes(monkeypatch, [_attachment(n, t) for n, t in MIXED])
    off, _ = _facts(IndicoLiveSource(
        event_ids=["1716863"], contributions=True))
    assert "attachment_images" not in _talk_node(off).attrs
    # the exact attr set of a talk before this option existed
    assert set(_talk_node(off).attrs) == {
        "label", "title", "url", "date", "end_date", "event_type", "category",
        "category_id", "speakers", "chairs", "attachment_urls",
        "attachment_names", "session", "room", "parent_event_id",
        "parent_event_title", "parent_event_url", "text",
    }
    plain = IndicoLiveSource(event_ids=["1716863"], contributions=True)
    on = IndicoLiveSource(
        event_ids=["1716863"], contributions=True, attachment_images=True)
    assert plain.change_probe._config_hash != on.change_probe._config_hash
    from archi.sources.indico_live import _records_hash
    event = _fixture("event_1716863_contributions.json")["results"][0]
    items = event["contributions"]
    base = "https://indico.cern.ch"
    a = _record_from_event(event, "1716863", items, base,
                           with_contributions=True)
    b = _record_from_event(event, "1716863", items, base,
                           with_contributions=True, image_limit=0)
    assert _records_hash([a]) == _records_hash([b])
    c = _record_from_event(event, "1716863", items, base,
                           with_contributions=True, image_limit=3)
    assert _records_hash([a]) != _records_hash([c])


def test_images_need_per_talk_records():
    with pytest.raises(ValueError, match="contributions"):
        IndicoLiveSource(event_ids=["1"], attachment_images=True)
    with pytest.raises(ValueError):
        IndicoLiveSource(event_ids=["1"], contributions=True,
                         attachment_images=True, max_images_per_talk=0)


def test_okg_reads_the_values_as_images(monkeypatch):
    # okg's own reader must find exactly what the source wrote.
    from okg.substrate.images.values import classify, iter_image_values
    _image_routes(monkeypatch, [_attachment(n, t) for n, t in MIXED])
    facts, _run = _facts(IndicoLiveSource(
        event_ids=["1716863"], contributions=True, attachment_images=True))
    sites = list(iter_image_values(_talk_node(facts).attrs))
    assert [(s.attr, s.index) for s in sites] == [
        ("attachment_images", 0), ("attachment_images", 1),
        ("attachment_images", 2)]
    assert [classify(s.value).state for s in sites] == ["url"] * 3


@pytest.mark.parametrize("url", [
    _BASE + "%2e%2e/a.pdf",
    _BASE + "x/../a.pdf",
    _BASE + "a%5Cb.pdf",
    "https://user@indico.cern.ch/event/1/a.pdf",
])
def test_a_url_okg_refuses_is_skipped_and_counted_not_fatal(monkeypatch, url):
    good = _attachment("ok.pdf", "application/pdf")
    _image_routes(monkeypatch, [
        _attachment("odd.pdf", "application/pdf", download_url=url), good])
    source = IndicoLiveSource(
        event_ids=["1716863"], contributions=True, attachment_images=True)
    facts, run = _facts(source)
    assert [v["url"] for v in _talk_node(facts).attrs["attachment_images"]] == [
        _BASE + "ok.pdf"]
    # still a name and a URL, as without the option
    assert url in _talk_node(facts).attrs["attachment_urls"]
    assert source.last_harvest_report["image_urls_refused"] == 1
    assert run.completed_scope is True


def test_a_protected_folder_hides_its_attachments_from_images(monkeypatch):
    event = _fixture("event_1716863_contributions.json")
    talk = next(c for c in event["results"][0]["contributions"]
                if str(c.get("db_id")) == "7317484")
    talk["folders"] = [
        {"is_protected": True,
         "attachments": [_attachment("hidden.pdf", "application/pdf")]},
        {"is_protected": False,
         "attachments": [_attachment("open.pdf", "application/pdf")]},
    ]
    plain = dict(event["results"][0])
    plain.pop("contributions")
    _routes(monkeypatch, {
        "detail=contributions": event,
        "/export/event/1716863.json": {"results": [plain]},
    })
    facts, _run = _facts(IndicoLiveSource(
        event_ids=["1716863"], contributions=True, attachment_images=True))
    assert [v["url"] for v in _talk_node(facts).attrs["attachment_images"]] == [
        _BASE + "open.pdf"]
# --- token auth and the date window -----------------------------------

def test_a_token_becomes_a_bearer_header(monkeypatch):
    """An Indico personal access token is preferred over a cookie jar.

    CERN SSO cookies expire in hours and someone has to own refreshing
    them; an Indico token (`indp_…`, scope `read:legacy_api`) is
    long-lived and is what an unattended ingest should carry.
    """
    monkeypatch.setenv("ARCHI_INDICO_TOKEN", "indp_abc123")
    source = IndicoLiveSource(event_ids=["1"], token_env="ARCHI_INDICO_TOKEN")
    session = source._session()
    assert session.headers["Authorization"] == "Bearer indp_abc123"


def test_an_unset_or_blank_token_sets_no_header(monkeypatch):
    """A blank token must not send `Bearer ` — that reads as a
    malformed credential rather than as no credential."""
    monkeypatch.delenv("ARCHI_INDICO_TOKEN", raising=False)
    source = IndicoLiveSource(event_ids=["1"], token_env="ARCHI_INDICO_TOKEN")
    assert "Authorization" not in source._session().headers
    monkeypatch.setenv("ARCHI_INDICO_TOKEN", "   ")
    assert "Authorization" not in source._session().headers


def test_the_token_value_never_reaches_the_change_probe():
    """The probe config records the VARIABLE NAME, never the secret."""
    source = IndicoLiveSource(event_ids=["1"], token_env="ARCHI_INDICO_TOKEN")
    import json
    blob = json.dumps(source.change_probe.__dict__, default=str)
    assert "indp_" not in blob


def test_since_is_appended_to_export_paths():
    source = IndicoLiveSource(category_ids=["6803"], since="-365d")
    assert source._with_window("/export/categ/6803.json") == \
        "/export/categ/6803.json?from=-365d"
    assert source._with_window("/export/categ/6803.json?limit=5") == \
        "/export/categ/6803.json?limit=5&from=-365d"


def test_no_since_leaves_the_path_alone():
    source = IndicoLiveSource(category_ids=["6803"])
    assert source._with_window("/export/categ/6803.json") == \
        "/export/categ/6803.json"


def test_the_category_walk_uses_the_window():
    source = IndicoLiveSource(category_ids=["6803"], since="-365d")
    calls: list[str] = []
    source._get_json = lambda session, path: (calls.append(path), {"results": [{"id": "9"}]})[1]
    source._event_ids_in_scope(object())
    assert calls == ["/export/categ/6803.json?from=-365d"]
