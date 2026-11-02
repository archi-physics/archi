"""task.w2.sources-monit — live MONIT agent tools, offline.

Mirrors what the cms originals (okg-deployments cms/cms_tools/
monit_live.py at main@f33a9c4) consume: the Grafana ``_msearch`` NDJSON
request (meta line + body line) and the ``{"responses": [...]}``
envelope. All HTTP goes through a monkeypatched ``requests.post``.
"""
import inspect
import json

import pytest
import requests

import archi.tools.monit as monit_tools
from archi.tools.monit import (
    archi_monit_condor_aggregate,
    archi_monit_condor_search,
    archi_monit_rucio_aggregate,
    archi_monit_rucio_search,
    monit_aggregate,
    monit_search,
)

TOKEN_ENV = "MONIT_GRAFANA_TOKEN"  # default the wired wrappers read


class _FakeHTTPResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(
                f"HTTP {self.status_code}", response=self
            )

    def json(self):
        return self._payload


def _fake_post(monkeypatch, respond):
    """Patch requests.post; ``respond(url, headers, data)`` -> payload."""
    calls = []

    def _post(url, headers=None, data=None, timeout=None):
        calls.append({
            "url": url, "headers": headers, "data": data, "timeout": timeout,
        })
        payload = respond(url, headers, data)
        if isinstance(payload, Exception):
            raise payload
        return _FakeHTTPResponse(payload)

    monkeypatch.setattr(monit_tools.requests, "post", _post)
    return calls


def _body(call):
    meta_line, body_line, _ = call["data"].split("\n")
    return json.loads(meta_line), json.loads(body_line)


SEARCH_RESPONSE = {
    "responses": [{
        "hits": {
            "total": {"value": 42, "relation": "eq"},
            "hits": [
                {
                    "_id": "doc-1",
                    "_index": "monit_prod_cms_rucio_raw_events-2026",
                    "_score": 1.5,
                    "_source": {
                        "data": {
                            "event_type": "transfer-failed",
                            "dst_rse": "T2_US_MIT",
                            "reason": "CHECKSUM MISMATCH",
                            "bytes": 123,
                        },
                        "metadata": {"timestamp": 1754900000000},
                    },
                },
                {
                    # _source arrives as a JSON string: must still parse.
                    "_id": "doc-2",
                    "_index": "monit_prod_cms_rucio_raw_events-2026",
                    "_score": None,
                    "_source": json.dumps(
                        {"data": {"event_type": "transfer-done"}}
                    ),
                },
            ],
        }
    }]
}

TERMS_AGG_RESPONSE = {
    "responses": [{
        "hits": {"total": {"value": 917, "relation": "eq"}},
        "aggregations": {"result": {
            "buckets": [
                {"key": "CHECKSUM MISMATCH", "doc_count": 500},
                {"key": "TIMEOUT", "doc_count": 400},
            ],
            "sum_other_doc_count": 17,
        }},
    }]
}


# --- wired wrappers ------------------------------------------------------------

def test_rucio_search_wrapper_binding_and_payload(monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, "test-token")
    calls = _fake_post(monkeypatch, lambda *a: SEARCH_RESPONSE)
    payload = archi_monit_rucio_search("data.event_type:transfer-failed")
    (call,) = calls
    assert call["url"] == (
        "https://monit-grafana.cern.ch/api/datasources/proxy/9269/_msearch"
    )
    assert call["headers"]["Authorization"] == "Bearer test-token"
    meta, body = _body(call)
    assert meta == {
        "search_type": "query_then_fetch",
        "ignore_unavailable": True,
        "index": ["monit_prod_cms_rucio_raw_events*"],
    }
    assert body["size"] == 5 and body["from"] == 0
    assert body["_source"] == list(monit_tools.RUCIO_SUMMARY_FIELDS)
    assert body["sort"] == [{"metadata.timestamp": {"order": "desc"}}]
    must = body["query"]["bool"]["must"][0]["query_string"]
    assert must == {
        "query": "data.event_type:transfer-failed",
        "analyze_wildcard": True,
    }
    time_range = body["query"]["bool"]["filter"][0]["range"]
    assert time_range["metadata.timestamp"]["gte"] == "now-24h"
    assert time_range["metadata.timestamp"]["lte"] == "now"

    assert payload["boundary"] == "external_live"
    assert payload["source"] == "monit_opensearch:rucio_raw_events"
    assert payload["observed_at"]
    assert payload["time_window"] == {
        "from": "now-24h", "to": "now", "time_field": "metadata.timestamp",
    }
    assert payload["total"] == {"value": 42, "relation": "eq"}
    assert "error" not in payload
    first, second = payload["results"]
    assert first["document_id"] == "doc-1"
    # Nested extraction of only the summary fields that are present.
    assert first["fields"]["data.event_type"] == "transfer-failed"
    assert first["fields"]["data.reason"] == "CHECKSUM MISMATCH"
    assert first["fields"]["metadata.timestamp"] == 1754900000000
    assert "data.scope" not in first["fields"]
    # String _source is parsed before extraction.
    assert second["fields"] == {"data.event_type": "transfer-done"}


def test_condor_search_wrapper_binding(monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, "test-token")
    calls = _fake_post(monkeypatch, lambda *a: SEARCH_RESPONSE)
    payload = archi_monit_condor_search("data.Status:Held")
    (call,) = calls
    assert "/proxy/8787/_msearch" in call["url"]
    meta, body = _body(call)
    assert meta["index"] == ["monit_prod_condor_raw_metric*"]
    assert body["_source"] == list(monit_tools.CONDOR_SUMMARY_FIELDS)
    assert payload["source"] == "monit_opensearch:condor_raw_metric"


def test_wrappers_do_not_expose_endpoint_knobs():
    # Instance wiring passes only agent-supplied arguments, so wired
    # tools must not let an agent re-point endpoint/datasource/index.
    for func in (archi_monit_rucio_search, archi_monit_condor_search):
        assert set(inspect.signature(func).parameters) == {
            "query", "from_time", "to_time", "max_results", "page",
        }
    for func in (archi_monit_rucio_aggregate, archi_monit_condor_aggregate):
        assert set(inspect.signature(func).parameters) == {
            "query", "group_by", "agg_type", "top_n", "from_time", "to_time",
        }


# --- search core ---------------------------------------------------------------

def test_search_pagination_and_result_bounds(monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, "test-token")
    calls = _fake_post(monkeypatch, lambda *a: SEARCH_RESPONSE)
    payload = archi_monit_rucio_search("*", max_results=50, page=3)
    (call,) = calls
    _, body = _body(call)
    assert body["size"] == 10  # hard limit
    assert body["from"] == 20  # (page 3 - 1) * 10
    assert payload["query"]["max_results"] == 10
    assert payload["query"]["page"] == 3
    assert payload["query"]["offset"] == 20


def test_search_empty_query_short_circuits(monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, "test-token")
    calls = _fake_post(monkeypatch, lambda *a: SEARCH_RESPONSE)
    payload = archi_monit_rucio_search("   ")
    assert calls == []  # no HTTP
    assert payload["error"] == "query must be non-empty"
    assert payload["results"] == []
    assert payload["total"] == {"value": 0, "relation": "eq"}


def test_search_redacts_token_and_truncates_query(monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, "sekrit-token")
    _fake_post(monkeypatch, lambda *a: SEARCH_RESPONSE)
    payload = archi_monit_rucio_search(
        "data.account:sekrit-token AND " + "x" * 900
    )
    lucene = payload["query"]["lucene"]
    assert "sekrit-token" not in lucene
    assert "[redacted]" in lucene
    assert len(lucene) <= monit_tools.MAX_QUERY_CHARS


def test_search_opensearch_error_payload(monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, "test-token")
    _fake_post(monkeypatch, lambda *a: {
        "responses": [{"error": {
            "type": "x_content_parse_exception", "reason": "bad query",
        }}]
    })
    payload = archi_monit_rucio_search("data.event_type:transfer-failed")
    assert payload["error"] == "bad query"
    assert payload["results"] == []


def test_search_timeout_is_structured_error(monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, "test-token")
    _fake_post(
        monkeypatch, lambda *a: requests.exceptions.Timeout("too slow")
    )
    payload = archi_monit_rucio_search("*")
    assert payload["error"] == "MONIT OpenSearch request timed out"
    assert payload["boundary"] == "external_live"


def test_search_connection_error_is_structured(monkeypatch):
    """Finding 7: requests.ConnectionError must come back as the same
    structured error payload as a timeout, not escape as an exception."""
    monkeypatch.setenv(TOKEN_ENV, "test-token")
    _fake_post(
        monkeypatch,
        lambda *a: requests.exceptions.ConnectionError("dns failure"),
    )
    payload = archi_monit_rucio_search("*")
    assert payload["error"] == (
        "MONIT OpenSearch request failed: ConnectionError"
    )
    assert payload["results"] == []
    assert payload["boundary"] == "external_live"


def test_aggregate_connection_error_is_structured(monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, "test-token")
    _fake_post(
        monkeypatch,
        lambda *a: requests.exceptions.ConnectionError("dns failure"),
    )
    payload = archi_monit_rucio_aggregate("*", "data.reason")
    assert payload["aggregation"]["error"] == (
        "MONIT OpenSearch request failed: ConnectionError"
    )
    assert payload["aggregation"]["buckets"] == []


class _NonJSONResponse:
    status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        raise requests.exceptions.JSONDecodeError(
            "Expecting value", "<html>Grafana error page</html>", 0
        )


def test_search_non_json_200_body_is_structured(monkeypatch):
    """Finding 7: a 200 body that is not JSON (Grafana HTML error page)
    must come back as a structured error payload."""
    monkeypatch.setenv(TOKEN_ENV, "test-token")
    monkeypatch.setattr(
        monit_tools.requests, "post", lambda *a, **k: _NonJSONResponse()
    )
    payload = archi_monit_rucio_search("*")
    assert payload["error"] == (
        "MONIT OpenSearch returned a non-JSON response body"
    )
    assert payload["results"] == []


def test_aggregate_non_json_200_body_is_structured(monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, "test-token")
    monkeypatch.setattr(
        monit_tools.requests, "post", lambda *a, **k: _NonJSONResponse()
    )
    payload = archi_monit_rucio_aggregate("*", "data.reason")
    assert payload["aggregation"]["error"] == (
        "MONIT OpenSearch returned a non-JSON response body"
    )


def test_time_window_echo_matches_query_body(monkeypatch):
    """Finding 9: the echoed time_window must reflect exactly the window
    the query body used — blank inputs are defaulted identically in
    both, not just in the echo."""
    monkeypatch.setenv(TOKEN_ENV, "test-token")
    calls = _fake_post(monkeypatch, lambda *a: SEARCH_RESPONSE)
    payload = archi_monit_rucio_search("*", from_time="", to_time="  ")
    (call,) = calls
    _, body = _body(call)
    time_range = body["query"]["bool"]["filter"][0]["range"][
        "metadata.timestamp"
    ]
    assert time_range["gte"] == "now-24h"
    assert time_range["lte"] == "now"
    assert payload["time_window"]["from"] == time_range["gte"]
    assert payload["time_window"]["to"] == time_range["lte"]

    calls.clear()
    payload = archi_monit_rucio_aggregate(
        "*", "data.reason", from_time=" now-7d ", to_time=None,
    )
    _, body = _body(calls[0])
    time_range = body["query"]["bool"]["filter"][0]["range"][
        "metadata.timestamp"
    ]
    assert time_range["gte"] == "now-7d"
    assert time_range["lte"] == "now"
    assert payload["time_window"]["from"] == "now-7d"
    assert payload["time_window"]["to"] == "now"


def test_search_http_401_maps_to_auth_failed_message(monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, "test-token")
    monkeypatch.setattr(
        monit_tools.requests, "post",
        lambda *a, **k: _FakeHTTPResponse({}, status_code=401),
    )
    payload = archi_monit_rucio_search("*")
    assert payload["error"] == (
        "MONIT OpenSearch authentication failed with HTTP 401"
    )
    monkeypatch.setattr(
        monit_tools.requests, "post",
        lambda *a, **k: _FakeHTTPResponse({}, status_code=502),
    )
    payload = archi_monit_rucio_search("*")
    assert payload["error"] == (
        "MONIT OpenSearch request failed with HTTP 502"
    )


def test_missing_token_raises(monkeypatch):
    # As in the cms original: the harness gates via requires_env, so a
    # missing token is a hard RuntimeError, not a soft error payload.
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    calls = _fake_post(monkeypatch, lambda *a: SEARCH_RESPONSE)
    with pytest.raises(RuntimeError, match=TOKEN_ENV):
        archi_monit_rucio_search("*")
    assert calls == []


# --- aggregation core ------------------------------------------------------------

def test_rucio_aggregate_terms_keyword_and_formatting(monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, "test-token")
    calls = _fake_post(monkeypatch, lambda *a: TERMS_AGG_RESPONSE)
    payload = archi_monit_rucio_aggregate(
        "data.event_type:transfer-failed", "data.reason", top_n=2,
    )
    (call,) = calls
    _, body = _body(call)
    assert body["size"] == 0
    assert body["aggs"]["result"]["terms"] == {
        "field": "data.reason.keyword", "size": 2,
    }
    agg = payload["aggregation"]
    assert agg["agg_type"] == "terms"
    assert agg["group_by"] == "data.reason"
    assert agg["total_matching_documents"] == 917
    assert agg["buckets"] == [
        {"key": "CHECKSUM MISMATCH", "doc_count": 500},
        {"key": "TIMEOUT", "doc_count": 400},
    ]
    assert agg["sum_other_doc_count"] == 17
    assert payload["boundary"] == "external_live"
    assert payload["query"]["top_n"] == 2


def test_aggregate_keyword_retry_falls_back_to_raw_field(monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, "test-token")
    # Documents matched but the auto-appended .keyword sub-field is
    # unmapped (e.g. a numeric field): zero buckets with a non-zero
    # total triggers the raw-field retry.
    matched_bucketless = {"responses": [{
        "hits": {"total": {"value": 917}},
        "aggregations": {"result": {"buckets": []}},
    }]}

    def _respond(url, headers, data):
        _, body = json.loads(data.split("\n")[0]), json.loads(
            data.split("\n")[1]
        )
        field = body["aggs"]["result"]["terms"]["field"]
        return (
            TERMS_AGG_RESPONSE
            if field == "data.bytes"
            else matched_bucketless
        )

    calls = _fake_post(monkeypatch, _respond)
    payload = archi_monit_condor_aggregate("*", "data.bytes")
    assert len(calls) == 2  # .keyword first, raw field retry second
    assert payload["aggregation"]["buckets"]
    # An explicit .keyword group_by is not retried.
    calls.clear()
    payload = archi_monit_condor_aggregate("*", "data.reason.keyword")
    assert len(calls) == 1
    assert payload["aggregation"]["buckets"] == []


def test_aggregate_field_error_triggers_raw_retry(monkeypatch):
    """Finding 8: a field-related .keyword error (fielddata disabled,
    missing mapping) retries with the raw field name."""
    monkeypatch.setenv(TOKEN_ENV, "test-token")
    field_error = {"responses": [{"error": {
        "type": "search_phase_execution_exception",
        "reason": "Fielddata is disabled on text fields by default",
    }}]}

    def _respond(url, headers, data):
        body = json.loads(data.split("\n")[1])
        field = body["aggs"]["result"]["terms"]["field"]
        return TERMS_AGG_RESPONSE if field == "data.reason" else field_error

    calls = _fake_post(monkeypatch, _respond)
    payload = archi_monit_rucio_aggregate("*", "data.reason")
    assert len(calls) == 2
    assert payload["aggregation"]["buckets"]
    assert "error" not in payload["aggregation"]


def test_aggregate_zero_match_empty_buckets_is_valid_result(monkeypatch):
    """Finding 8 regression: a legitimate zero-bucket .keyword result
    (zero matching documents) must be returned as-is — the old code
    retried with the raw field and surfaced its fielddata error."""
    monkeypatch.setenv(TOKEN_ENV, "test-token")
    state = {"calls": 0}
    empty = {"responses": [{
        "hits": {"total": {"value": 0}},
        "aggregations": {"result": {"buckets": []}},
    }]}
    fielddata_error = {"responses": [{"error": {
        "type": "search_phase_execution_exception",
        "reason": "Fielddata is disabled on text fields by default",
    }}]}

    def _respond(url, headers, data):
        state["calls"] += 1
        return empty if state["calls"] == 1 else fielddata_error

    calls = _fake_post(monkeypatch, _respond)
    payload = archi_monit_rucio_aggregate(
        "data.event_type:none-match", "data.reason"
    )
    assert len(calls) == 1  # no raw-field retry on a zero-match result
    agg = payload["aggregation"]
    assert agg["buckets"] == []
    assert agg["total_matching_documents"] == 0
    assert "error" not in agg


def test_aggregate_failing_raw_retry_keeps_valid_empty_result(monkeypatch):
    """Finding 8: when the matched-but-bucketless retry itself errors,
    the valid empty .keyword result wins — never the retry's error."""
    monkeypatch.setenv(TOKEN_ENV, "test-token")
    matched_bucketless = {"responses": [{
        "hits": {"total": {"value": 42}},
        "aggregations": {"result": {"buckets": []}},
    }]}
    fielddata_error = {"responses": [{"error": {
        "reason": "Fielddata is disabled on text fields by default",
    }}]}

    def _respond(url, headers, data):
        body = json.loads(data.split("\n")[1])
        field = body["aggs"]["result"]["terms"]["field"]
        return (
            fielddata_error
            if field == "data.reason"
            else matched_bucketless
        )

    calls = _fake_post(monkeypatch, _respond)
    payload = archi_monit_rucio_aggregate("*", "data.reason")
    assert len(calls) == 2
    agg = payload["aggregation"]
    assert agg["buckets"] == []
    assert agg["total_matching_documents"] == 42
    assert "error" not in agg


def test_aggregate_metric_value(monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, "test-token")
    calls = _fake_post(monkeypatch, lambda *a: {"responses": [{
        "hits": {"total": {"value": 10}},
        "aggregations": {"result": {"value": 1234.5}},
    }]})
    payload = archi_monit_rucio_aggregate("*", "data.bytes", agg_type="sum")
    (call,) = calls
    _, body = _body(call)
    # Metric aggregations use the raw field, never .keyword.
    assert body["aggs"]["result"] == {"sum": {"field": "data.bytes"}}
    agg = payload["aggregation"]
    assert agg["value"] == 1234.5
    assert agg["buckets"] == []
    assert agg["total_matching_documents"] == 10


def test_aggregate_validation_errors_short_circuit(monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, "test-token")
    calls = _fake_post(monkeypatch, lambda *a: TERMS_AGG_RESPONSE)
    payload = archi_monit_rucio_aggregate("", "data.reason")
    assert "non-empty" in payload["aggregation"]["error"]
    payload = archi_monit_rucio_aggregate("*", "  ")
    assert payload["aggregation"]["error"] == "group_by must be non-empty"
    payload = archi_monit_rucio_aggregate("*", "data.reason", agg_type="median")
    assert payload["aggregation"]["error"].startswith("agg_type must be one of")
    assert calls == []  # none of the invalid calls hit HTTP


def test_aggregate_opensearch_error_and_timeout(monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, "test-token")
    _fake_post(monkeypatch, lambda *a: {
        "responses": [{"error": {"reason": "no such field"}}]
    })
    payload = archi_monit_rucio_aggregate("*", "data.nope")
    assert payload["aggregation"]["error"] == "no such field"
    assert payload["aggregation"]["buckets"] == []
    _fake_post(
        monkeypatch, lambda *a: requests.exceptions.Timeout("too slow")
    )
    payload = archi_monit_rucio_aggregate("*", "data.reason")
    assert payload["aggregation"]["error"] == (
        "MONIT OpenSearch request timed out"
    )


# --- parameterized cores ---------------------------------------------------------

def test_generic_cores_take_endpoint_and_token_env(monkeypatch):
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    monkeypatch.setenv("ARCHI_T_OTHER_TOKEN", "other-token")
    calls = _fake_post(monkeypatch, lambda *a: SEARCH_RESPONSE)
    payload = monit_search(
        "data.event_type:x",
        source_name="monit_opensearch:my_index",
        datasource_id=4242,
        index="monit_prod_my-*",
        summary_fields=("data.event_type",),
        grafana_base_url="https://grafana.example.org/",
        token_env="ARCHI_T_OTHER_TOKEN",
        timeout=5.0,
    )
    (call,) = calls
    assert call["url"] == (
        "https://grafana.example.org/api/datasources/proxy/4242/_msearch"
    )
    assert call["headers"]["Authorization"] == "Bearer other-token"
    assert call["timeout"] == 5.0
    meta, _ = _body(call)
    assert meta["index"] == ["monit_prod_my-*"]
    assert payload["source"] == "monit_opensearch:my_index"

    calls.clear()
    payload = monit_aggregate(
        "*",
        "data.reason",
        source_name="monit_opensearch:my_index",
        datasource_id=4242,
        index="monit_prod_my-*",
        grafana_base_url="https://grafana.example.org",
        token_env="ARCHI_T_OTHER_TOKEN",
    )
    assert calls and "/proxy/4242/_msearch" in calls[0]["url"]
    assert payload["source"] == "monit_opensearch:my_index"
