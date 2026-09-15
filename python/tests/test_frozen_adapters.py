"""Synthetic consumer fixtures; no real deployment evidence or network."""

import json
from pathlib import Path

import pytest
from okg.deployment import ConnectorAdapter, NodeFact

from archi.install.adapters import (
    FrozenCMSSWAdapter,
    FrozenDocumentationAdapter,
    FrozenJiraAdapter,
)
from archi.install.snapshots import digest, prepare_caches, read_snapshot


def snapshot():
    url = "https://fts3-docs.web.cern.ch/fts3-docs/"
    return {
        "records.json": json.dumps(
            [
                {
                    "key": "CMSPROD-101",
                    "summary": "Fixture issue",
                    "description": "Synthetic transfer fixture",
                    "project": "CMSPROD",
                }
            ]
        ).encode(),
        "fetch-receipt.json": json.dumps(
            {
                "complete_project_scope": False,
                "record_count": 1,
                "fetched_at": "2026-01-01T00:00:00Z",
            }
        ).encode(),
        "documentation.html": b"<html><title>Fixture documentation</title><body><p>Synthetic transfer fixture.</p></body></html>",
        "documentation-fetch-receipt.json": json.dumps(
            {"url": url, "final_url": url, "http_status": 200}
        ).encode(),
        "releases.map": b"architecture=el8_amd64_gcc12;label=CMSSW_15_0_0;type=Production;\narchitecture=el8_amd64_gcc12;label=CMSSW_15_0_X;type=Development;\n",
    }


ALLOWLISTS = {
    "FrozenDocumentationAdapter": ["https://fts3-docs.web.cern.ch/fts3-docs/"],
    "FrozenJiraAdapter": ["CMSPROD-101"],
}
#: The run mode each reader is installed with (archi.install.configuration).
MODES = {
    "FrozenCMSSWAdapter": "release_new",
    "FrozenDocumentationAdapter": "scope_complete",
    "FrozenJiraAdapter": "reconcile",
}


def build(adapter, root, pins, **overrides):
    params = dict(configuration_root=str(root), snapshot_digests=pins)
    if adapter.__name__ in ALLOWLISTS:
        params["record_allowlist"] = list(ALLOWLISTS[adapter.__name__])
    params.update(overrides)
    return adapter(**params)


def cache(tmp_path):
    values = prepare_caches(snapshot())
    for name, body in values.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    return {name: digest(body) for name, body in values.items()}


@pytest.mark.parametrize(
    "adapter,expected",
    [
        (FrozenCMSSWAdapter, "cmssw_release:CMSSW_15_0_0"),
        (FrozenDocumentationAdapter, "documentation_page:"),
        (FrozenJiraAdapter, "jira:CMSPROD-101"),
    ],
)
def test_real_sdk_roundtrip_preserves_reader_semantics(tmp_path, adapter, expected):
    pins = cache(tmp_path)
    source = build(adapter, tmp_path, pins)
    assert isinstance(source, ConnectorAdapter)
    assert (
        isinstance(adapter.profile, str) and adapter.profile == source._reader.profile
    )
    mode = MODES[adapter.__name__]
    actual = source.run("test-run", mode=mode)
    direct = source._reader.run("test-run", mode=mode)
    assert actual.completed_scope == direct.completed_scope
    assert actual.health == direct.health
    assert actual.run_mode == direct.run_mode
    assert actual.facts == list(direct.facts)
    assert any(
        isinstance(fact, NodeFact) and fact.node_id.startswith(expected)
        for fact in actual.facts
    )
    assert not getattr(source, "probe_short_circuit_safe", False)
    assert source.change_probe is source._reader.change_probe
    with pytest.raises(ValueError, match="narrowed"):
        source.run("narrowed", sync_scope={"key": "selected"})


def test_frozen_cmssw_retains_partial_scope(tmp_path):
    source = build(FrozenCMSSWAdapter, tmp_path, cache(tmp_path))
    result = source.run("partial", mode="scope_complete")
    assert result.completed_scope is False  # unsupported family row is not hidden
    assert result.health.status != "ok" or "skip" in result.health.reason.lower()


@pytest.mark.parametrize(
    "adapter", [FrozenCMSSWAdapter, FrozenDocumentationAdapter, FrozenJiraAdapter]
)
def test_changed_cache_refuses_before_each_run(tmp_path, adapter):
    pins = cache(tmp_path)
    source = build(adapter, tmp_path, pins)
    (tmp_path / "snapshots/jira/records.json").write_bytes(b"[]")
    with pytest.raises(ValueError, match="digest"):
        source.run("changed")


def test_jira_is_explicitly_cache_only(tmp_path):
    source = build(FrozenJiraAdapter, tmp_path, cache(tmp_path))
    assert source.requires_live_call_authorization is False
    assert source.change_probe_kind == "mutable_api"


def test_snapshot_manifest_roundtrip_and_tamper(tmp_path):
    values = snapshot()
    for name, body in values.items():
        (tmp_path / name).write_bytes(body)
    manifest = json.dumps(
        {
            "schema": "archi.frozen-snapshot/v1",
            "files": {k: digest(v) for k, v in values.items()},
        }
    ).encode()
    path = tmp_path / "snapshot.json"
    path.write_bytes(manifest)
    assert read_snapshot(path, digest(manifest)) == values
    (tmp_path / "releases.map").write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest"):
        read_snapshot(path, digest(manifest))


def test_symlink_and_missing_inventory_refuse(tmp_path):
    pins = cache(tmp_path)
    file = tmp_path / "snapshots/cmssw/releases.map"
    old = file.read_bytes()
    file.unlink()
    (tmp_path / "map").write_bytes(old)
    file.symlink_to(tmp_path / "map")
    with pytest.raises(ValueError, match="symlink"):
        build(FrozenCMSSWAdapter, tmp_path, pins)
    with pytest.raises(ValueError, match="inventory"):
        build(FrozenJiraAdapter, tmp_path, {})


def test_fifo_refused_without_waiting_for_writer(tmp_path):
    import os

    from archi.install.snapshots import read_regular

    path = tmp_path / "fifo"
    os.mkfifo(path)
    with pytest.raises(ValueError, match="regular"):
        read_regular(path, digest(b""))


@pytest.mark.parametrize(
    "adapter,member,rows",
    [
        (
            FrozenJiraAdapter,
            "snapshots/jira/records.json",
            [{"key": "CMSPROD-101"}, {"key": "CMSPROD-102"}],
        ),
        (
            FrozenDocumentationAdapter,
            "snapshots/docsite/records.json",
            [{"url": "https://example.org/other"}],
        ),
    ],
)
def test_records_outside_the_allowlist_refuse(tmp_path, adapter, member, rows):
    pins = cache(tmp_path)
    body = json.dumps(rows).encode()
    (tmp_path / member).write_bytes(body)
    pins[member] = digest(body)
    with pytest.raises(ValueError, match="allowlist"):
        build(adapter, tmp_path, pins)


@pytest.mark.parametrize("allowlist", [[], ["CMSPROD-101", "CMSPROD-101"], [""]])
def test_record_allowlist_must_name_exactly_the_records(tmp_path, allowlist):
    with pytest.raises(ValueError, match="allowlist"):
        build(FrozenJiraAdapter, tmp_path, cache(tmp_path), record_allowlist=allowlist)


@pytest.mark.parametrize(
    "schema,names",
    [
        (
            "okg.cern-snapshot-input/v1",
            [
                "records.json",
                "fetch-receipt.json",
                "documentation.html",
                "documentation-fetch-receipt.json",
            ],
        ),
        ("okg.cern-snapshot-input/v2", None),
    ],
)
def test_only_the_five_file_snapshot_schema_is_read(tmp_path, schema, names):
    values = snapshot()
    if names is not None:
        values = {name: values[name] for name in names}
    for name, body in values.items():
        (tmp_path / name).write_bytes(body)
    manifest = json.dumps(
        {"schema": schema, "files": {k: digest(v) for k, v in values.items()}}
    ).encode()
    (tmp_path / "snapshot.json").write_bytes(manifest)
    with pytest.raises(ValueError, match="schema"):
        read_snapshot(tmp_path / "snapshot.json", digest(manifest))
