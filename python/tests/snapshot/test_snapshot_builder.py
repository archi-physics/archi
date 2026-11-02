"""req.compops-snapshot-builder -- build and verify a cache snapshot, offline.

Every fixture is synthetic and tiny. Each text-bearing group carries one
planted address (``PLANTED``) that must not survive into any archive.
"""
from __future__ import annotations

import dataclasses
import io
import json
import os
import signal
import subprocess
import tarfile
import tempfile
import time
import unicodedata
from pathlib import Path

import pytest
import yaml

from archi.snapshot import builder as builder_module
from archi.snapshot import groups as groups_module
from archi.snapshot.__main__ import main as cli_main
from archi.snapshot.builder import (
    LOCK_NAME,
    BuildRefused,
    GroupRefused,
    SnapshotError,
    build,
    load_config,
    normalize_fact,
    parse_config,
    read_archive,
    verify,
)

PLANTED = {
    "cric": "site.admin@cern.ch",
    "cric-core": "core.ops@fnal.gov",
    "jira": "jane.doe@cern.ch",
    "indico": "chair.person@cern.ch",
    "dqm": "dqm.shifter@cern.ch",
    "gocdb-downtimes": "grid.admin@example.org",
    "gitlab-docs": "doc.writer@cern.ch",
    "docsite": "web.master@cern.ch",
    "twiki-eos": "twiki.editor@cern.ch",
    "conddb-global-tags": "alca.contact@cern.ch",
    "wmstats": "pdmv.operator@cern.ch",
    "dbs": "dbs.contact@cern.ch",
    "cmssst-site-status": "site.support@cern.ch",
}
DN = "/DC=ch/DC=cern/OU=Organic Units/OU=Users/CN=jdoe/CN=123456/CN=Jane Doe"

JIRA_RECORDS = [
    {
        "key": "CMSPROD-101",
        "summary": "Transfer failures at T2_US_MIT",
        "description": f"Stuck transfers; mail {PLANTED['jira']} or see CMSPROD-100.",
        "project": "CMSPROD",
        "status": "Open",
        "priority": "Critical",
        "issue_type": "Bug",
        "assignee": "Ada Lovelace",
        "reporter": "Grace Hopper",
        "created": "2026-01-05T10:00:00.000+0000",
        "updated": "2026-02-01T09:30:00.000+0000",
        "labels": ["ops"],
        "components": ["transfers"],
        "comment_count": 1,
        "recent_comments": [
            {
                "author": "Ada Lovelace",
                "created": "2026-01-06T08:00:00.000+0000",
                "body": f"Retrying; cc {PLANTED['jira']}",
            }
        ],
        "issue_links": [{"key": "CMSPROD-100", "type": "Relates", "direction": "outward"}],
        "subtasks": [],
        "parent_key": "",
        "environment": "prod",
        "duedate": "2026-03-01",
    },
    {
        "key": "CMSPROD-100",
        "fields": {
            "summary": "Baseline transfer issue",
            "description": "Original report.",
            "project": {"key": "CMSPROD", "name": "CMS Production"},
            "status": {"name": "Closed"},
            "issuetype": {"name": "Task"},
            "assignee": {
                "displayName": "Ada Lovelace",
                "emailAddress": "ada.lovelace@cern.ch",
                "avatarUrls": {"48x48": "https://its.cern.ch/avatar/1"},
            },
            "reporter": {"displayName": "Enrico Fermi"},
            "created": "2025-12-01T10:00:00.000+0000",
            "updated": "2025-12-20T10:00:00.000+0000",
            "comment": {"comments": [], "total": 0},
            "watches": {"watchCount": 3},
        },
    },
]


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_sources(root: Path) -> dict[str, Path]:
    """One directory per group, holding exactly the files its reader reads."""
    dirs = {name: root / name for name in groups_module.GROUPS}
    _write_json(
        dirs["cric"] / "sites.json",
        {
            "T2_US_MIT": {
                "tier_level": 2,
                "country": "United States",
                "facility": "MIT",
                "state": "ACTIVE",
                "sitedb_title": "MIT Bates",
                "computeunits": {"MIT-CE1": {}},
                "contact_email": PLANTED["cric"],
            }
        },
    )
    _write_json(
        dirs["cric"] / "facilities.json",
        {
            "MIT": {
                "country": "United States",
                "fullname": f"MIT (admin {PLANTED['cric']})",
                "cmssites": [{"name": "T2_US_MIT"}],
            }
        },
    )
    _write_json(
        dirs["cric"] / "storage_units.json",
        {"MIT-SE": {"type": "DISK", "pledged-CMS": 4200.0, "site": {"name": "T2_US_MIT"}}},
    )
    _write_json(
        dirs["cric"] / "compute_units.json",
        {"MIT-CE1": {"corepower": 11.0, "potential_max": 20000.0, "state": "ACTIVE"}},
    )
    _write_json(
        dirs["cric"] / "responsibilities.json",
        {
            "desc": {"columns": ["username", "site_name", "role"]},
            "result": [
                ["adalove", "MIT Bates", "Site Executive"],
                ["ghopper", None, "Site Admin"],  # a null site, as in real exports
            ],
        },
    )
    _write_json(
        dirs["cric-core"] / "services.json",
        {
            "mit-ce.mit.edu": {
                "type": "CE",
                "endpoint": "https://mit-ce.mit.edu:9619",
                "rcsite": "MIT",
                "contact": PLANTED["cric-core"],
            }
        },
    )
    _write_json(
        dirs["cric-core"] / "rcsites.json",
        {"MIT": {"country": "US", "sites": [{"name": "T2_US_MIT", "vo_name": "cms"}]}},
    )
    _write_json(
        dirs["cric-core"] / "federations.json",
        {
            "US-MIT": {
                "accounting_name": f"US MIT ({PLANTED['cric-core']})",
                "vos": ["cms"],
                "rcsites": ["MIT"],
                "pledges": {},
            }
        },
    )
    releases = dirs["cmssw-releases"]
    releases.mkdir(parents=True)
    (releases / "releases.map").write_text(
        "architecture=el8_amd64_gcc12;label=CMSSW_14_0_1;type=Production;state=Announced;prodarch=1;\n"
        "architecture=el8_amd64_gcc12;label=CMSSW_14_0_2;type=Production;state=Announced;prodarch=1;\n",
        encoding="utf-8",
    )
    _write_json(dirs["jira"] / "records.json", JIRA_RECORDS)
    _write_json(
        dirs["jira"] / "meta.json",
        {"record_count": 2, "server": "https://its.cern.ch/jira", "fetched_at": 1.0},
    )
    _write_json(
        dirs["indico"] / "records.json",
        [
            {
                "id": "1234",
                "title": "Computing operations weekly",
                "type": "meeting",
                "category": "CMS Computing",
                "categoryId": 42,
                "startDate": {"date": "2026-06-12", "time": "15:00:00", "tz": "Europe/Zurich"},
                "endDate": {"date": "2026-06-12", "time": "16:00:00", "tz": "Europe/Zurich"},
                "description": f"<p>Questions to {PLANTED['indico']}</p>",
                "chairs": [
                    {
                        "fullName": "Doe, Jane",
                        "email": PLANTED["indico"],
                        "affiliation": "CERN",
                        "id": "77",
                    }
                ],
                "folders": [
                    {
                        "title": "Slides",
                        "attachments": [
                            {"download_url": "https://indico.cern.ch/event/1234/a.pdf", "size": 10}
                        ],
                    }
                ],
                "_contributions_text": f"Site report ({PLANTED['indico']})",
                "_pdf_texts": [
                    {
                        "title": "Slides",
                        "text": f"Contact {PLANTED['indico']} for access.",
                        "url": "https://indico.cern.ch/event/1234/a.pdf",
                    }
                ],
            }
        ],
    )
    _write_json(
        dirs["dqm"] / "records.json",
        [
            {
                "cert_name": "Cert_Collisions2024_378981_386951_Golden.json",
                "filename": f"golden.json (owner {PLANTED['dqm']})",
                "run_range": [378981, 386951],
                "datasets": ["/Muon0/Run2024C-PromptReco-v1/MINIAOD"],
                "num_lumi_sections": 1000,
            }
        ],
    )
    _write_json(
        dirs["gocdb-downtimes"] / "records.json",
        [
            {
                "downtime_id": 35001,
                "primary_key": "35001G0",
                "hostname": "mit-ce.mit.edu",
                "hosted_by": "MIT",
                "service_type": "CE",
                "severity": "OUTAGE",
                "classification": "SCHEDULED",
                "description": f"Kernel upgrade; contact {PLANTED['gocdb-downtimes']}",
                "start_date": "2026-08-01 08:00",
                "end_date": "2026-08-01 12:00",
            }
        ],
    )
    for group, repo in (("gitlab-docs", "cmsdmops/Documentation"), ("docsite", "")):
        _write_json(
            dirs[group] / "records.json",
            [
                {
                    "title": "Operations guide",
                    "url": f"https://example.cern.ch/{group}/guide",
                    "body": f"Page owner: {PLANTED[group]}. Restart the agent first.",
                    "site_name": "example.cern.ch",
                    "repo": repo,
                    "file_path": "guide.md",
                    "last_editor_email": PLANTED[group],
                }
            ],
        )
    twiki = dirs["twiki-eos"]
    (twiki / "Sub").mkdir(parents=True)
    (twiki / "CompOpsGuide.txt").write_text(
        '%META:TOPICINFO{author="JaneDoe" date="1700000000"}%\n'
        "---+ Comp ops guide\n"
        f"Ask {PLANTED['twiki-eos']} before draining a site. See [[SubTopicPage]].\n",
        encoding="utf-8",
    )
    (twiki / "Sub" / "SubTopicPage.txt").write_text(
        "---+ Sub topic\nDrain procedure details.\n", encoding="utf-8"
    )
    (twiki / "Sub" / "diagram.png").write_bytes(b"\x89PNG not read by the reader")
    _write_json(
        dirs["conddb-global-tags"] / "records.json",
        [
            {
                "name": "140X_dataRun3_Prompt_v3",
                "release": "CMSSW_14_0_2",
                "scenario": "prompt",
                "description": f"Prompt GT; owner {PLANTED['conddb-global-tags']}",
                "snapshot_time": "2026-05-01",
                "insertion_author": PLANTED["conddb-global-tags"],
            }
        ],
    )
    _write_json(
        dirs["wmstats"] / "records.json",
        [
            {
                "RequestName": "pdmvserv_task_TOP-Run3Summer23-00001",
                "RequestType": "TaskChain",
                "RequestStatus": "running-open",
                "Campaign": f"Run3Summer23 ({PLANTED['wmstats']})",
                "PrepID": "TOP-Run3Summer23-00001",
                "RequestPriority": 85000,
                "CMSSWVersion": "CMSSW_13_0_13",
                "InputDataset": "/TTto2L2Nu/Run3Summer23-v1/GEN-SIM",
                "OutputDatasets": ["/TTto2L2Nu/Run3Summer23-v1/AODSIM"],
                "RequestDate": "2026-01-15",
                "Requestor": "pdmvserv",
                "RequestorDN": DN,
                "RequestTransition": [{"Status": "new", "DN": DN}],
            }
        ],
    )
    _write_json(
        dirs["cmssst-site-status"] / "records.json",
        [
            {
                "site": "T2_US_MIT",
                "time": 1791383941,
                "alert": "",
                "msg": f"Questions to {PLANTED['cmssst-site-status']}",
                "ggus": [[1003759, 1788530148]],
                "metrics": {
                    "LifeStatus": {"today": ["o" * 48, "o" * 48], "pmonth": ["o" * 60]},
                    "SiteReadiness": {"today": ["w" * 48, "u" * 48]},
                },
                "url": "http://cmssst.web.cern.ch/siteStatus/",
                "elements": [{"host": "se.mit.edu", "metrics": {}}],
            }
        ],
    )
    _write_json(
        dirs["dbs"] / "records.json",
        [
            {
                "dataset": "/TTto2L2Nu/Run3Summer23-v1/AODSIM",
                "data_tier_name": "AODSIM",
                "physics_group_name": f"Top ({PLANTED['dbs']})",
                "nevents": 100,
                "create_by": PLANTED["dbs"],
            }
        ],
    )
    # The cluster-manager caches, from the recorded shapes the pack tests use
    # (each writer puts its cache under <root>/<group>/, as dirs expects).
    # This directory has no __init__.py, so tests/ is not on the path by itself.
    import sys

    tests_dir = str(Path(__file__).resolve().parents[1])
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    from packs.cluster_fixtures import (
        write_condor_cache,
        write_node_health_cache,
        write_servicenow_cache,
        write_slurm_cache,
        write_tickets_cache,
    )

    for write in (write_tickets_cache, write_servicenow_cache, write_node_health_cache,
                  write_slurm_cache, write_condor_cache):
        write(root)
    return dirs


# Aug 31 for five groups and a June range for five others, as in the first
# snapshot Jason chose; the three optional groups carry their own dates.
COLLECTED = {
    "cric": "2026-08-31",
    "cric-core": "2026-08-31",
    "jira": "2026-08-31",
    "gitlab-docs": "2026-08-31",
    "twiki-eos": "2026-08-31",
    "indico": "2026-06-12..2026-06-16",
    "dqm": "2026-06-12..2026-06-16",
    "gocdb-downtimes": "2026-06-12..2026-06-16",
    "cmssw-releases": "2026-06-12..2026-06-16",
    "conddb-global-tags": "2026-06-12..2026-06-16",
    "docsite": "2026-08-31",
    "wmstats": "2026-08-31",
    "dbs": "2026-08-31",
    "cmssst-site-status": "2026-10-07",
    "tickets": "2026-10-20",
    "servicenow": "2026-10-20",
    "node-health": "2026-10-20",
    "slurm": "2026-10-20",
    "htcondor": "2026-10-20",
}


def write_config(tmp_path: Path, dirs: dict[str, Path], only=None, extra=None) -> Path:
    names = only or sorted(dirs)
    groups = {name: {"path": str(dirs[name]), "collected": COLLECTED[name]} for name in names}
    for name, fields in (extra or {}).items():
        groups[name].update(fields)
    config = tmp_path / "snapshot.yaml"
    config.write_text(
        yaml.safe_dump({"snapshot": "cms-cache-test", "groups": groups}), encoding="utf-8"
    )
    return config


@pytest.fixture
def sources(tmp_path):
    return write_sources(tmp_path / "raw")


@pytest.fixture
def built(tmp_path, sources):
    out = tmp_path / "snapshot"
    lock = build(load_config(write_config(tmp_path, sources)), out, built_by="test")
    return out, lock


def _all_bytes(out: Path, group: str) -> bytes:
    members = read_archive(out / f"{group}.tar.zst")
    return b"\n".join(members[name] for name in sorted(members))


# --- build, lock, verify -----------------------------------------------------


def test_build_writes_every_group_with_lock_rows(built):
    out, lock = built
    assert sorted(lock["groups"]) == sorted(groups_module.GROUPS)
    on_disk = yaml.safe_load((out / LOCK_NAME).read_text(encoding="utf-8"))
    assert on_disk == lock
    for name, row in lock["groups"].items():
        assert row["collected"] == COLLECTED[name]
        assert (out / row["archive"]).stat().st_size == row["bytes"]
        assert len(row["sha256"]) == 64
        assert row["record_count"] > 0 and row["file_count"] > 0
    # Records as each reader counts them: 2 JIRA issues, 2 TWiki pages, one
    # of everything else in these fixtures.
    assert lock["groups"]["jira"]["record_count"] == 2
    assert lock["groups"]["jira"]["file_count"] == 2
    assert lock["groups"]["twiki-eos"]["record_count"] == 2
    assert lock["groups"]["twiki-eos"]["file_count"] == 2  # the .png is not read
    assert lock["groups"]["twiki-eos"]["empty_topic_files"] == 0
    assert lock["groups"]["twiki-eos"]["empty_topic_names"] == []
    assert "empty_topic_files" not in lock["groups"]["jira"]
    assert lock["archive_format"]["compressor"] == "zstd"
    assert verify(out / LOCK_NAME, out).ok


def test_archives_unpack_to_the_reader_layout(built):
    out, _ = built
    members = read_archive(out / "jira.tar.zst")
    assert sorted(members) == ["data/jira/meta.json", "data/jira/records.json"]
    twiki = read_archive(out / "twiki-eos.tar.zst")
    assert sorted(twiki) == [
        "data/twiki-eos/CompOpsGuide.txt",
        "data/twiki-eos/Sub/SubTopicPage.txt",
    ]


@pytest.mark.parametrize("group", sorted(PLANTED))
def test_planted_address_is_removed_from_every_text_group(built, sources, group):
    out, _ = built
    raw = b"".join(p.read_bytes() for p in sorted(sources[group].rglob("*")) if p.is_file())
    assert PLANTED[group].encode() in raw  # the fixture really plants it
    data = _all_bytes(out, group)
    assert PLANTED[group].encode() not in data
    assert b"@" not in data.replace(b"@2x", b"")


def test_text_around_a_removed_address_is_kept(built):
    out, _ = built
    records = json.loads(read_archive(out / "jira.tar.zst")["data/jira/records.json"])
    assert records[0]["description"] == "Stuck transfers; mail  or see CMSPROD-100."
    twiki = read_archive(out / "twiki-eos.tar.zst")["data/twiki-eos/CompOpsGuide.txt"]
    assert b"Ask  before draining a site." in twiki


def test_fields_the_readers_never_read_are_dropped(built):
    out, _ = built
    jira = json.loads(read_archive(out / "jira.tar.zst")["data/jira/records.json"])
    assert "duedate" not in jira[0]
    assert set(jira[1]["fields"]["assignee"]) == {"displayName"}
    assert "watches" not in jira[1]["fields"]
    indico = json.loads(read_archive(out / "indico.tar.zst")["data/indico/records.json"])
    assert indico[0]["chairs"] == [{"fullName": "Doe, Jane"}]
    assert indico[0]["startDate"] == {"date": "2026-06-12"}
    assert indico[0]["folders"] == [
        {"attachments": [{"download_url": "https://indico.cern.ch/event/1234/a.pdf"}]}
    ]
    meta = json.loads(read_archive(out / "jira.tar.zst")["data/jira/meta.json"])
    assert meta == {"record_count": 2}
    docs = json.loads(read_archive(out / "docsite.tar.zst")["data/docsite/records.json"])
    assert "last_editor_email" not in docs[0]
    cric = read_archive(out / "cric.tar.zst")
    assert "contact_email" not in json.loads(cric["data/cric/sites.json"])["T2_US_MIT"]
    assert json.loads(cric["data/cric/responsibilities.json"]) == {
        "result": [["adalove", "MIT Bates", "Site Executive"], ["ghopper", None, "Site Admin"]]
    }
    cmssst = json.loads(
        read_archive(out / "cmssst-site-status.tar.zst")[
            "data/cmssst-site-status/records.json"
        ]
    )
    assert set(cmssst[0]) == {"site", "time", "alert", "msg", "ggus", "metrics"}
    assert cmssst[0]["metrics"]["LifeStatus"]["pmonth"] == ["o" * 60]


def test_wmstats_requestor_dn_never_appears_and_requestor_is_kept(built, sources):
    out, lock = built
    assert DN.encode() in (sources["wmstats"] / "records.json").read_bytes()
    data = _all_bytes(out, "wmstats")
    assert DN.encode() not in data
    assert b"RequestorDN" not in data and b"CN=" not in data
    record = json.loads(read_archive(out / "wmstats.tar.zst")["data/wmstats-workflows/records.json"])[0]
    assert record["Requestor"] == "pdmvserv"
    assert "RequestTransition" not in record
    assert lock["groups"]["wmstats"]["dropped_keys_at_any_depth"] == ["DN", "RequestorDN"]
    assert lock["groups"]["wmstats"]["kept_extra_fields"] == ["Requestor"]


def test_wmstats_dn_drop_cannot_be_turned_off_by_config(tmp_path, sources):
    config = write_config(
        tmp_path, sources, only=["wmstats"], extra={"wmstats": {"keep_fields": ["RequestorDN"]}}
    )
    out = tmp_path / "out"
    build(load_config(config), out)
    assert DN.encode() not in _all_bytes(out, "wmstats")


def test_configured_drop_of_an_unread_field(tmp_path, sources):
    config = write_config(
        tmp_path, sources, only=["wmstats"], extra={"wmstats": {"drop_fields": ["Requestor"]}}
    )
    out = tmp_path / "out"
    lock = build(load_config(config), out)
    record = json.loads(read_archive(out / "wmstats.tar.zst")["data/wmstats-workflows/records.json"])[0]
    assert "Requestor" not in record
    assert lock["groups"]["wmstats"]["dropped_fields"] == ["Requestor"]


def test_configured_drop_of_a_field_the_reader_reads_is_refused(tmp_path, sources):
    config = write_config(
        tmp_path, sources, only=["jira"], extra={"jira": {"drop_fields": ["description"]}}
    )
    with pytest.raises(BuildRefused, match="changed what the reader emits"):
        build(load_config(config), tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_a_schema_missing_a_read_field_refuses_the_group(tmp_path, sources, monkeypatch):
    spec = groups_module.GROUPS["dqm"]
    records = spec.json_files[0]
    schema = {k: v for k, v in records.schema.items() if k != "datasets"}
    broken = dataclasses.replace(
        spec, json_files=(dataclasses.replace(records, schema=schema),)
    )
    monkeypatch.setitem(groups_module.GROUPS, "dqm", broken)
    config = write_config(tmp_path, sources, only=["dqm"])
    with pytest.raises(BuildRefused) as info:
        build(load_config(config), tmp_path / "out")
    assert [r.group for r in info.value.refusals] == ["dqm"]
    assert "misses a field the reader reads" in str(info.value)


# --- the JIRA reader sees the same issues --------------------------------------


def _jira_facts(root: Path) -> list[str]:
    from archi.sources.jira import JiraIssueSource

    source = JiraIssueSource(
        records_path="data/jira/records.json", meta_path="data/jira/meta.json", base=str(root)
    )
    return [normalize_fact(f) for f in source.run("r", mode="scope_complete").facts]


def test_jira_facts_from_raw_and_redacted_caches_are_equal(built, sources, tmp_path):
    out, _ = built
    raw_root = tmp_path / "raw-root"
    (raw_root / "data").mkdir(parents=True)
    (raw_root / "data" / "jira").symlink_to(sources["jira"], target_is_directory=True)
    unpacked = tmp_path / "unpacked"
    for name, data in read_archive(out / "jira.tar.zst").items():
        (unpacked / name).parent.mkdir(parents=True, exist_ok=True)
        (unpacked / name).write_bytes(data)
    raw_facts = _jira_facts(raw_root)
    redacted_facts = _jira_facts(unpacked)
    assert raw_facts and raw_facts == redacted_facts
    # The reader removes the address itself at ingest, so neither side has it.
    assert not any(PLANTED["jira"] in fact for fact in raw_facts)
    # Guard against a vacuous pass: the raw cache does hold the address.
    assert PLANTED["jira"].encode() in (sources["jira"] / "records.json").read_bytes()


# --- malformed input refuses its group ----------------------------------------

MALFORMED = {
    "jira": ("records.json", lambda p: p + [{"summary": "no key"}], "record 2 is malformed"),
    "indico": ("records.json", lambda p: p + ["not an object"], "record 1 is malformed"),
    "dqm": ("records.json", lambda p: p + [{"filename": "x"}], "no cert_name"),
    "gocdb-downtimes": (
        "records.json",
        lambda p: p + [{"downtime_id": "abc"}],
        "downtime_id is not numeric",
    ),
    "gitlab-docs": ("records.json", lambda p: p + [{"title": "no url"}], "no url"),
    "docsite": ("records.json", lambda p: p + [{"url": ""}], "no url"),
    "conddb-global-tags": ("records.json", lambda p: p + [{"release": "x"}], "no name / tag_name"),
    "wmstats": ("records.json", lambda p: p + [{"Campaign": "x"}], "RequestName"),
    "dbs": ("records.json", lambda p: p + [{"nevents": 1}], "dataset_name / dataset"),
    "cmssst-site-status": (
        "records.json",
        lambda p: p + [{"site": "not-a-site", "metrics": {}}],
        "site is missing or not a CMS site name",
    ),
    "cric": ("sites.json", lambda p: {**p, "T2_XX_Bad": "not an object"}, "entry 'T2_XX_Bad'"),
    "cric-core": ("services.json", lambda p: {**p, "bad": []}, "entry 'bad'"),
}


@pytest.mark.parametrize("group", sorted(MALFORMED))
def test_a_malformed_record_refuses_its_group_and_writes_nothing(tmp_path, sources, group):
    file_name, mutate, expected = MALFORMED[group]
    path = sources[group] / file_name
    path.write_text(json.dumps(mutate(json.loads(path.read_text()))), encoding="utf-8")
    config = write_config(tmp_path, sources)  # every group, one of them broken
    out = tmp_path / "out"
    with pytest.raises(BuildRefused) as info:
        build(load_config(config), out)
    assert [r.group for r in info.value.refusals] == [group]
    assert expected in info.value.refusals[0].reason
    assert not out.exists()
    assert not list(tmp_path.glob(".out.*"))


def test_an_address_only_jira_key_refuses_the_group(tmp_path, sources):
    # Refused before redaction: an address is not an issue key.
    path = sources["jira"] / "records.json"
    records = json.loads(path.read_text()) + [{"key": "bob@cern.ch", "summary": "x"}]
    path.write_text(json.dumps(records))
    with pytest.raises(BuildRefused, match="only 2 of 3 records carry a key / issue_key matching"):
        build(load_config(write_config(tmp_path, sources, only=["jira"])), tmp_path / "out")


def test_an_identity_that_is_only_an_address_is_refused_after_redaction(tmp_path, sources):
    # Passes the first check (an id is present), then redaction empties it:
    # the reader would skip that event, so the group is refused.
    path = sources["indico"] / "records.json"
    records = json.loads(path.read_text()) + [
        {"id": "bob@cern.ch", "title": "x", "_contributions_text": ""}
    ]
    path.write_text(json.dumps(records))
    with pytest.raises(BuildRefused, match="record 1 is malformed: record has no id / event_id"):
        build(load_config(write_config(tmp_path, sources, only=["indico"])), tmp_path / "out")


@pytest.mark.parametrize(
    "group, file_name, content, expected",
    [
        ("jira", "records.json", "{not json", "not valid JSON"),
        ("jira", "records.json", '{"key": "CMSPROD-1"}', "expected a JSON list"),
        ("cric", "responsibilities.json", '{"status": "error"}', "expected a 'result' list"),
        ("cmssw-releases", "releases.map", "label=NOT_A_RELEASE;type=x;\n", "did not accept"),
    ],
)
def test_unreadable_files_refuse_their_group(tmp_path, sources, group, file_name, content, expected):
    (sources[group] / file_name).write_text(content, encoding="utf-8")
    with pytest.raises(BuildRefused) as info:
        build(load_config(write_config(tmp_path, sources, only=[group])), tmp_path / "out")
    assert expected in info.value.refusals[0].reason


def test_jira_meta_count_mismatch_refuses_the_group(tmp_path, sources):
    _write_json(sources["jira"] / "meta.json", {"record_count": 5})
    with pytest.raises(BuildRefused, match="reports record_count=5"):
        build(load_config(write_config(tmp_path, sources, only=["jira"])), tmp_path / "out")


def test_a_missing_required_file_refuses_the_group(tmp_path, sources):
    (sources["cric"] / "facilities.json").unlink()
    with pytest.raises(BuildRefused, match="facilities.json is missing"):
        build(load_config(write_config(tmp_path, sources, only=["cric"])), tmp_path / "out")


@pytest.mark.parametrize(
    "content",
    [
        "jdoe@cern.ch wrote this".encode("utf-16-le"),  # NUL between letters
        b"---+ Page\nText\x00with a NUL.\n",
        b"---+ Caf\xe9 page\x00\n",  # Latin-1 does not excuse a NUL
    ],
    ids=["utf16", "nul", "latin1-with-nul"],
)
def test_text_with_nul_bytes_refuses_the_group(tmp_path, sources, content):
    (sources["twiki-eos"] / "Odd.txt").write_bytes(content)
    with pytest.raises(BuildRefused) as info:
        build(load_config(write_config(tmp_path, sources, only=["twiki-eos"])), tmp_path / "out")
    reason = info.value.refusals[0].reason
    assert reason == "Odd.txt contains NUL bytes (not UTF-8 text)"


def _add_twiki_topics(directory: Path, count: int) -> None:
    for index in range(count):
        (directory / f"Filler{index:03d}.txt").write_text(
            f"---+ Filler {index}\nBody {index}.\n", encoding="utf-8"
        )


def test_two_empty_twiki_topics_are_left_out_and_listed(tmp_path, sources):
    # The real snapshot holds two topics empty upstream. They used to
    # refuse the group (the reader claimed no scope), so every release
    # snapshot needed a hand-staged copy without them.
    twiki = sources["twiki-eos"]
    _add_twiki_topics(twiki, 40)
    (twiki / "EmptyUpstream.txt").write_bytes(b"")
    (twiki / "Sub" / "NulOnly.txt").write_bytes(b"\x00" * 64 + b"\n \t")
    out = tmp_path / "out"
    lock = build(load_config(write_config(tmp_path, sources, only=["twiki-eos"])), out)
    row = lock["groups"]["twiki-eos"]
    assert row["empty_topic_files"] == 2
    assert row["empty_topic_names"] == ["EmptyUpstream.txt", "Sub/NulOnly.txt"]
    assert row["record_count"] == 42
    assert row["file_count"] == 42
    members = read_archive(out / "twiki-eos.tar.zst")
    assert "data/twiki-eos/EmptyUpstream.txt" not in members
    assert "data/twiki-eos/Sub/NulOnly.txt" not in members
    on_disk = yaml.safe_load((out / LOCK_NAME).read_text(encoding="utf-8"))
    assert on_disk["groups"]["twiki-eos"]["empty_topic_names"] == row["empty_topic_names"]
    result = verify(out / LOCK_NAME, out)
    assert result.ok, result.lines


def test_many_empty_twiki_topics_refuse_the_group(tmp_path, sources):
    # 2 real + 11 empty = 13 text files: max(10, 0.1%) allows 10.
    twiki = sources["twiki-eos"]
    for index in range(11):
        (twiki / f"Empty{index:02d}.txt").write_bytes(b"")
    with pytest.raises(BuildRefused) as info:
        build(load_config(write_config(tmp_path, sources, only=["twiki-eos"])), tmp_path / "out")
    reason = info.value.refusals[0].reason
    assert reason.startswith(
        "11 of 13 text files are zero bytes or only whitespace/NUL bytes, "
        "more than the 10 allowed (an interrupted sync), e.g. Empty00.txt"
    )
    assert not (tmp_path / "out").exists()


def test_ten_empty_twiki_topics_are_within_the_limit(tmp_path, sources):
    twiki = sources["twiki-eos"]
    for index in range(10):
        (twiki / f"Empty{index:02d}.txt").write_bytes(b"  \n")
    lock = build(
        load_config(write_config(tmp_path, sources, only=["twiki-eos"])), tmp_path / "out"
    )
    assert lock["groups"]["twiki-eos"]["empty_topic_files"] == 10


def test_an_all_empty_twiki_tree_refuses_the_group(tmp_path, sources):
    twiki = sources["twiki-eos"]
    for path in twiki.rglob("*.txt"):
        path.write_bytes(b"")
    with pytest.raises(BuildRefused) as info:
        build(load_config(write_config(tmp_path, sources, only=["twiki-eos"])), tmp_path / "out")
    assert info.value.refusals[0].reason.startswith(
        "all 2 text files are zero bytes or only whitespace/NUL bytes"
    )


def test_verify_catches_an_empty_topic_list_that_disagrees(tmp_path, sources):
    (sources["twiki-eos"] / "EmptyUpstream.txt").write_bytes(b"")
    out = tmp_path / "out"
    build(load_config(write_config(tmp_path, sources, only=["twiki-eos"])), out)
    lock_path = out / LOCK_NAME
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    lock["groups"]["twiki-eos"]["empty_topic_files"] = 3
    lock_path.write_text(yaml.safe_dump(lock, sort_keys=False), encoding="utf-8")
    result = verify(lock_path, out)
    assert not result.ok
    assert "lock says 3 empty topic files" in result.lines[0]
    lock["groups"]["twiki-eos"]["empty_topic_files"] = 1
    lock["groups"]["twiki-eos"]["empty_topic_names"] = ["CompOpsGuide.txt"]
    lock_path.write_text(yaml.safe_dump(lock, sort_keys=False), encoding="utf-8")
    result = verify(lock_path, out)
    assert not result.ok
    assert "empty topic file CompOpsGuide.txt is in the archive" in result.lines[0]


def test_non_utf8_text_without_a_fallback_refuses_the_group(tmp_path, sources):
    path = sources["cmssw-releases"] / "releases.map"
    path.write_bytes(path.read_bytes() + b"architecture=x;label=CMSSW_14_0_3;type=Caf\xe9;\n")
    with pytest.raises(BuildRefused, match="releases.map is not UTF-8"):
        build(
            load_config(write_config(tmp_path, sources, only=["cmssw-releases"])),
            tmp_path / "out",
        )


#: Plain text that keeps one control character under the 1% binary limit.
PADDING = b"Ordinary page text. " * 10 + b"\n"

REVIEWER_MIXED_PAGE = (
    "---+ Contacts\nCaf\u00e9 team: j\u00e9r\u00f4me.dupont@cern.ch, jdoe@c\u00e9rn.ch "
    "and jdoe\uff20cern\uff0ech.\n"
).encode("utf-8") + b"A stray Windows quote: \x93quoted.\n"


def test_a_stray_byte_does_not_garble_the_valid_utf8_around_it(tmp_path, sources):
    # The second review's page: mostly UTF-8, three addresses, one stray
    # cp1252 byte. Decoding the whole page as cp1252 leaked "j\u00c3\u00a9r\u00c3\u00b4".
    (sources["twiki-eos"] / "Mixed.txt").write_bytes(REVIEWER_MIXED_PAGE)
    out = tmp_path / "out"
    lock = build(load_config(write_config(tmp_path, sources, only=["twiki-eos"])), out)
    page = read_archive(out / "twiki-eos.tar.zst")["data/twiki-eos/Mixed.txt"].decode("utf-8")
    assert page == (
        "---+ Contacts\nCaf\u00e9 team: ,  and .\n"
        "A stray Windows quote: \u201cquoted.\n"
    )
    for fragment in ("dupont", "j\u00e9r\u00f4me", "jdoe", "c\u00e9rn", "\u00c3"):
        assert fragment not in page
    row = lock["groups"]["twiki-eos"]
    assert row["text_pages"] == 3
    assert row["pages_valid_utf8"] == 2
    assert row["pages_with_fallback_runs"] == 1
    assert row["fallback_bytes"] == {"cp1252": 1, "latin-1": 0}
    # Two here and one planted elsewhere; the full-width one is removed whole.
    assert row["addresses_removed"] == 3
    assert row["normalized_addresses_removed"] == 1


def test_cp1252_and_latin1_runs_are_decoded_and_counted(tmp_path, sources):
    twiki = sources["twiki-eos"]
    # A whole Latin-1 page: its address has an accented local part.
    (twiki / "LatinPage.txt").write_bytes(
        b"---+ Contacts\nWrite to j\xe9r\xf4me.dupont@cern.ch today.\n"
    )
    # cp1252 smart quotes and an en dash, and one byte cp1252 leaves undefined.
    (twiki / "QuotePage.txt").write_bytes(
        b"---+ Quotes\nThe \x93golden\x94 JSON \x96 see notes \x81here.\n" + PADDING
    )
    out = tmp_path / "out"
    lock = build(load_config(write_config(tmp_path, sources, only=["twiki-eos"])), out)
    members = read_archive(out / "twiki-eos.tar.zst")
    assert members["data/twiki-eos/LatinPage.txt"].decode("utf-8") == (
        "---+ Contacts\nWrite to  today.\n"
    )
    assert members["data/twiki-eos/QuotePage.txt"].decode("utf-8") == (
        "---+ Quotes\nThe \u201cgolden\u201d JSON \u2013 see notes \x81here.\n"
        + PADDING.decode("ascii")
    )
    row = lock["groups"]["twiki-eos"]
    assert row["pages_with_fallback_runs"] == 2
    assert row["fallback_bytes"] == {"cp1252": 5, "latin-1": 1}


def _write_bytes_name(folder: Path, name: bytes, data: bytes) -> None:
    """Write a file whose name is raw bytes (not always valid UTF-8)."""
    target = os.path.join(os.fsencode(folder), name)
    try:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "wb") as handle:
            handle.write(data)
    except OSError as exc:  # APFS and some others refuse non-UTF-8 names
        pytest.skip(f"this filesystem refuses the name {name!r}: {exc}")


def test_a_latin1_file_name_is_decoded_and_listed(tmp_path, sources):
    # A real TWiki dump: "DsTotauTo" + 0xB5 x3 (Latin-1 micro signs). Python
    # reads the name with surrogates, which the reader could not re-encode.
    twiki = sources["twiki-eos"]
    _write_bytes_name(
        twiki,
        b"BPH-DsTotauTo\xb5\xb5\xb5.txt",
        b"---+ Decay\nTau to three muons.\n" + PADDING,
    )
    _write_bytes_name(twiki, "Sub\u00b5/Empty".encode("utf-8") + b"\xb5.txt", b"")
    out = tmp_path / "out"
    lock = build(load_config(write_config(tmp_path, sources, only=["twiki-eos"])), out)
    members = read_archive(out / "twiki-eos.tar.zst")
    page = members["data/twiki-eos/BPH-DsTotauToµµµ.txt"].decode("utf-8")
    assert page.startswith("---+ Decay\nTau to three muons.\n")
    row = lock["groups"]["twiki-eos"]
    assert row["record_count"] == 3
    assert row["renamed_files"] == [
        {
            "source_name": "BPH-DsTotauTo\\xb5\\xb5\\xb5.txt",
            "archive_name": "BPH-DsTotauToµµµ.txt",
            "archived": True,
        },
        {
            "source_name": "Subµ/Empty\\xb5.txt",
            "archive_name": "Subµ/Emptyµ.txt",
            "archived": False,
        },
    ]
    assert row["empty_topic_names"] == ["Subµ/Emptyµ.txt"]
    assert verify(out / LOCK_NAME, out).ok


def test_a_latin1_folder_name_is_decoded(tmp_path, sources):
    twiki = sources["twiki-eos"]
    _write_bytes_name(twiki, b"Web\xe9/Page.txt", b"---+ Page\n" + PADDING)
    out = tmp_path / "out"
    lock = build(load_config(write_config(tmp_path, sources, only=["twiki-eos"])), out)
    assert "data/twiki-eos/Webé/Page.txt" in read_archive(out / "twiki-eos.tar.zst")
    assert [
        r["archive_name"] for r in lock["groups"]["twiki-eos"]["renamed_files"]
    ] == ["Webé/Page.txt"]


def test_file_names_that_decode_to_the_same_name_refuse_the_group(tmp_path, sources):
    twiki = sources["twiki-eos"]
    _write_bytes_name(twiki, b"Mu\xb5.txt", b"---+ Latin-1 name\n" + PADDING)
    _write_bytes_name(twiki, "Muµ.txt".encode("utf-8"), b"---+ UTF-8 name\n" + PADDING)
    with pytest.raises(BuildRefused, match=r"Muµ\.txt and Mu\\xb5\.txt collide as"):
        build(
            load_config(write_config(tmp_path, sources, only=["twiki-eos"])),
            tmp_path / "out",
        )


def test_a_file_name_that_decodes_to_a_folder_name_refuses_the_group(tmp_path, sources):
    twiki = sources["twiki-eos"]
    _write_bytes_name(twiki, "Aµ.txt".encode("utf-8"), b"---+ File\n" + PADDING)
    _write_bytes_name(twiki, b"A\xb5.txt/x.txt", b"---+ Inside\n" + PADDING)
    with pytest.raises(BuildRefused, match=r"collide as Aµ\.txt/x\.txt"):
        build(
            load_config(write_config(tmp_path, sources, only=["twiki-eos"])),
            tmp_path / "out",
        )


@pytest.mark.parametrize("byte", [b"\x81", b"\x9d", b"\xa0", b"\xad"])
def test_a_name_byte_that_becomes_a_hidden_character_refuses_the_group(
    tmp_path, sources, byte
):
    _write_bytes_name(
        sources["twiki-eos"], b"Odd" + byte + b"Name.txt", b"---+ P\n" + PADDING
    )
    with pytest.raises(BuildRefused, match=r"would become the hidden character U\+"):
        build(
            load_config(write_config(tmp_path, sources, only=["twiki-eos"])),
            tmp_path / "out",
        )


def test_an_address_hidden_by_a_fallback_byte_in_a_name_refuses_the_group(
    tmp_path, sources
):
    # Decoded, 0x93 is a left quote, which ends the address for the redactor.
    _write_bytes_name(
        sources["twiki-eos"], b"jdoe\x93@cern.ch.txt", b"---+ P\n" + PADDING
    )
    with pytest.raises(BuildRefused, match=r"file name jdoe\\x93@cern\.ch\.txt"):
        build(
            load_config(write_config(tmp_path, sources, only=["twiki-eos"])),
            tmp_path / "out",
        )


def test_an_address_split_by_a_c1_fallback_byte_is_removed(tmp_path, sources):
    # 0x81 is undefined in cp1252; as Latin-1 it is the C1 control U+0081,
    # which would hide a.b@cd.ch. The page falls back to dropping the byte.
    (sources["twiki-eos"] / "Split.txt").write_bytes(
        b"---+ Page\nMail a.b@c\x81d.ch now.\n" + PADDING
    )
    out = tmp_path / "out"
    lock = build(load_config(write_config(tmp_path, sources, only=["twiki-eos"])), out)
    page = read_archive(out / "twiki-eos.tar.zst")["data/twiki-eos/Split.txt"].decode("utf-8")
    assert page.startswith("---+ Page\nMail  now.\n")
    assert lock["groups"]["twiki-eos"]["pages_bytes_dropped"] == 1


def test_an_address_split_by_a_control_in_json_refuses_the_group(tmp_path, sources):
    path = sources["dqm"] / "records.json"
    records = json.loads(path.read_text())
    records[0]["filename"] = "owner a.b@c\u0001d.ch"
    path.write_text(json.dumps(records))
    with pytest.raises(BuildRefused, match=r"span rule"):
        build(load_config(write_config(tmp_path, sources, only=["dqm"])), tmp_path / "out")


@pytest.mark.parametrize(
    "content",
    [b"---+ Page\n" + b"\x01\x02 data " * 20, b"---+ Page\n" + b"\x81\x8d\x8f" * 10],
    ids=["c0-controls", "c1-from-fallback"],
)
def test_a_page_that_is_mostly_control_characters_is_refused_as_binary(
    tmp_path, sources, content
):
    (sources["twiki-eos"] / "Blob.txt").write_bytes(content)
    with pytest.raises(BuildRefused, match=r"Blob.txt looks binary: \d+ of \d+ characters"):
        build(load_config(write_config(tmp_path, sources, only=["twiki-eos"])), tmp_path / "out")


def test_fewer_than_16_control_characters_are_not_binary(tmp_path, sources):
    # 15 controls in a 40-character page: 37%, but too few to call it binary.
    (sources["twiki-eos"] / "Short.txt").write_bytes(b"---+ Page\n" + b"\x01" * 15 + b" short page text.\n")
    build(load_config(write_config(tmp_path, sources, only=["twiki-eos"])), tmp_path / "out")


def test_16_controls_over_1_percent_are_binary(tmp_path, sources):
    (sources["twiki-eos"] / "Blob.txt").write_bytes(b"---+ Page\n" + b"\x01" * 16 + b" short page text.\n")
    with pytest.raises(BuildRefused, match="Blob.txt looks binary: 16 of"):
        build(load_config(write_config(tmp_path, sources, only=["twiki-eos"])), tmp_path / "out")


def test_form_feed_and_vertical_tab_are_not_control_characters_here(tmp_path, sources):
    # Twenty \f and \v in a short page: not binary. \f before an @ does not
    # count as splitting an address; the redactor sees no local part there
    # (a form feed cannot be in one), so nothing is removed or refused.
    (sources["twiki-eos"] / "Pages.txt").write_bytes(
        b"---+ Pages\n" + b"\x0c\x0b" * 10 + b"\nfoo\x0c@cern.ch and bar@cern.ch\n"
    )
    out = tmp_path / "out"
    build(load_config(write_config(tmp_path, sources, only=["twiki-eos"])), out)
    page = read_archive(out / "twiki-eos.tar.zst")["data/twiki-eos/Pages.txt"].decode("utf-8")
    assert page.endswith("\nfoo\x0c@cern.ch and \n")


def test_a_few_control_characters_are_allowed(tmp_path, sources):
    (sources["twiki-eos"] / "Tabs.txt").write_bytes(
        b"---+ Page\n" + b"x" * 200 + b"\x0c\n\ttab\r\n"
    )
    build(load_config(write_config(tmp_path, sources, only=["twiki-eos"])), tmp_path / "out")


def test_utf8_only_twiki_reports_no_fallback(built):
    _, lock = built
    row = lock["groups"]["twiki-eos"]
    assert row["pages_valid_utf8"] == row["text_pages"] == 2
    assert row["pages_with_fallback_runs"] == 0
    assert row["fallback_bytes"] == {"cp1252": 0, "latin-1": 0}


def test_cmssw_group_accepts_records_json_when_the_map_is_absent(tmp_path, sources):
    group = sources["cmssw-releases"]
    (group / "releases.map").unlink()
    _write_json(
        group / "records.json",
        [
            {"label": "CMSSW_14_0_1", "type": "Production", "state": "Announced",
             "architecture": ["el8_amd64_gcc12"], "release_notes": "Notes by rel.manager@cern.ch",
             "download_count": 7},
            {"label": "CMSSW_14_0_2", "type": "Production", "state": "Announced"},
        ],
    )
    (group / "meta.json").write_text('{"record_count": 2}')
    out = tmp_path / "out"
    lock = build(
        load_config(write_config(tmp_path, sources, only=["cmssw-releases"])), out
    )
    row = lock["groups"]["cmssw-releases"]
    assert row["input_file"] == "records.json"
    assert row["record_count"] == 2 and row["addresses_removed"] == 1
    members = read_archive(out / "cmssw-releases.tar.zst")
    assert sorted(members) == ["data/cmssw-releases/records.json"]
    records = json.loads(members["data/cmssw-releases/records.json"])
    assert "download_count" not in records[0]
    assert verify(out / LOCK_NAME, out).ok


def test_cmssw_group_prefers_the_map_when_both_exist(tmp_path, sources):
    _write_json(sources["cmssw-releases"] / "records.json", [{"label": "CMSSW_1_0_0"}])
    out = tmp_path / "out"
    lock = build(
        load_config(write_config(tmp_path, sources, only=["cmssw-releases"])), out
    )
    assert lock["groups"]["cmssw-releases"]["input_file"] == "releases.map"
    assert sorted(read_archive(out / "cmssw-releases.tar.zst")) == [
        "data/cmssw-releases/releases.map"
    ]


def test_cmssw_group_with_neither_input_is_refused(tmp_path, sources):
    (sources["cmssw-releases"] / "releases.map").unlink()
    with pytest.raises(BuildRefused, match="neither releases.map nor records.json"):
        build(
            load_config(write_config(tmp_path, sources, only=["cmssw-releases"])),
            tmp_path / "out",
        )


def test_malformed_cmssw_records_refuse_the_group(tmp_path, sources):
    (sources["cmssw-releases"] / "releases.map").unlink()
    _write_json(sources["cmssw-releases"] / "records.json", [{"type": "Production"}])
    with pytest.raises(BuildRefused, match="record 0 is malformed: record has no label"):
        build(
            load_config(write_config(tmp_path, sources, only=["cmssw-releases"])),
            tmp_path / "out",
        )


def test_wmstats_dn_is_removed_at_any_depth_even_when_its_field_is_kept(tmp_path, sources):
    config = write_config(
        tmp_path,
        sources,
        only=["wmstats"],
        extra={"wmstats": {"keep_fields": ["RequestTransition", "DN", "RequestorDN"]}},
    )
    out = tmp_path / "out"
    build(load_config(config), out)
    record = json.loads(
        read_archive(out / "wmstats.tar.zst")["data/wmstats-workflows/records.json"]
    )[0]
    assert record["RequestTransition"] == [{"Status": "new"}]
    assert "RequestorDN" not in record
    assert DN.encode() not in _all_bytes(out, "wmstats")


# --- what the builder will not read ---------------------------------------------


def test_a_symlinked_text_file_refuses_the_group(tmp_path, sources):
    outside = tmp_path / "outside.txt"
    outside.write_text("secret page\n")
    (sources["twiki-eos"] / "link.txt").symlink_to(outside)
    with pytest.raises(BuildRefused, match="link.txt: symbolic link"):
        build(load_config(write_config(tmp_path, sources, only=["twiki-eos"])), tmp_path / "out")


def test_a_symlinked_directory_refuses_the_group(tmp_path, sources):
    other = tmp_path / "elsewhere"
    other.mkdir()
    (other / "Page.txt").write_text("page\n")
    (sources["twiki-eos"] / "Linked").symlink_to(other, target_is_directory=True)
    with pytest.raises(BuildRefused, match="Linked: symbolic link"):
        build(load_config(write_config(tmp_path, sources, only=["twiki-eos"])), tmp_path / "out")


def test_a_symlink_that_is_not_text_still_refuses_the_group(tmp_path, sources):
    (sources["twiki-eos"] / "image.png").symlink_to(tmp_path / "missing.png")
    with pytest.raises(BuildRefused, match="image.png: symbolic link"):
        build(load_config(write_config(tmp_path, sources, only=["twiki-eos"])), tmp_path / "out")


@pytest.mark.parametrize(
    "rel", [".cookies/hypernews.txt", ".Hidden.txt", "Sub/.git/HEAD"]
)
def test_a_dot_path_refuses_the_group(tmp_path, sources, rel):
    path = sources["twiki-eos"] / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# Netscape HTTP Cookie File\n")
    with pytest.raises(BuildRefused, match="hidden path"):
        build(load_config(write_config(tmp_path, sources, only=["twiki-eos"])), tmp_path / "out")


def test_a_symlinked_json_file_refuses_the_group(tmp_path, sources):
    real = tmp_path / "real-records.json"
    (sources["dqm"] / "records.json").rename(real)
    (sources["dqm"] / "records.json").symlink_to(real)
    with pytest.raises(BuildRefused, match="records.json: symbolic link"):
        build(load_config(write_config(tmp_path, sources, only=["dqm"])), tmp_path / "out")


def test_a_dangling_json_symlink_is_refused_not_called_missing(tmp_path, sources):
    (sources["jira"] / "meta.json").unlink()
    (sources["jira"] / "meta.json").symlink_to(tmp_path / "nowhere.json")
    with pytest.raises(BuildRefused, match="meta.json: symbolic link"):
        build(load_config(write_config(tmp_path, sources, only=["jira"])), tmp_path / "out")


def test_a_symlinked_single_text_file_refuses_the_group(tmp_path, sources):
    real = tmp_path / "releases.map"
    (sources["cmssw-releases"] / "releases.map").rename(real)
    (sources["cmssw-releases"] / "releases.map").symlink_to(real)
    with pytest.raises(BuildRefused, match="releases.map: symbolic link"):
        build(
            load_config(write_config(tmp_path, sources, only=["cmssw-releases"])),
            tmp_path / "out",
        )


def test_the_configured_directory_itself_may_be_a_link(tmp_path, sources):
    alias = tmp_path / "dqm-alias"
    alias.symlink_to(sources["dqm"], target_is_directory=True)
    config = tmp_path / "c.yaml"
    config.write_text(
        yaml.safe_dump(
            {"snapshot": "s", "groups": {"dqm": {"path": str(alias), "collected": "2026-06-12"}}}
        )
    )
    build(load_config(config), tmp_path / "out")


def test_an_address_in_a_file_name_refuses_the_group(tmp_path, sources):
    (sources["twiki-eos"] / "jdoe@cern.ch.txt").write_text("---+ Page\nText.\n")
    with pytest.raises(BuildRefused, match="a file path contains an email address"):
        build(load_config(write_config(tmp_path, sources, only=["twiki-eos"])), tmp_path / "out")


def test_an_unreadable_file_is_a_named_refusal_and_other_groups_are_reported(
    tmp_path, sources, capsys
):
    records = sources["dqm"] / "records.json"
    records.chmod(0)
    try:
        if os.access(records, os.R_OK):
            pytest.skip("running with privileges that ignore file modes")
        _write_json(sources["indico"] / "records.json", [{"title": "no id"}])
        config = write_config(tmp_path, sources, only=["dqm", "indico", "jira"])
        code = cli_main(["build", "--config", str(config), "--out", str(tmp_path / "o")])
    finally:
        records.chmod(0o644)
    err = capsys.readouterr().err
    assert code == 2
    assert "2 group(s) refused" in err
    assert "group 'dqm' refused: cannot read its cache: PermissionError" in err
    assert "group 'indico' refused" in err
    assert "Traceback" not in err


def test_json_nested_too_deeply_is_a_named_refusal(tmp_path, sources):
    depth = 100_000
    (sources["dqm"] / "records.json").write_text("[" * depth + "]" * depth)
    with pytest.raises(BuildRefused, match="nested too deeply"):
        build(load_config(write_config(tmp_path, sources, only=["dqm"])), tmp_path / "out")


def test_lock_counts_the_addresses_removed(built):
    _, lock = built
    counts = {name: row["addresses_removed"] for name, row in lock["groups"].items()}
    # jira: 2 planted + 1 emailAddress field; indico: chair email + 3 texts;
    # docs: body + last_editor_email; cric: facility name + contact_email.
    assert counts["jira"] == 3
    assert counts["indico"] == 4
    assert counts["gitlab-docs"] == counts["docsite"] == 2
    assert counts["cric"] == 2
    assert counts["twiki-eos"] == 1
    assert counts["cmssw-releases"] == 0
    assert all(isinstance(v, int) for v in counts.values())


# --- determinism and verification ---------------------------------------------


def test_archives_are_byte_identical_across_two_runs(tmp_path, sources):
    config = load_config(write_config(tmp_path, sources))
    first = build(config, tmp_path / "one")
    second = build(config, tmp_path / "two")
    for name in groups_module.GROUPS:
        a = (tmp_path / "one" / f"{name}.tar.zst").read_bytes()
        b = (tmp_path / "two" / f"{name}.tar.zst").read_bytes()
        assert a == b, name
    assert first["groups"] == second["groups"]


def test_archive_members_have_fixed_metadata(built):
    out, _ = built
    raw = subprocess.run(
        ["zstd", "-d", "-q", "-c", str(out / "jira.tar.zst")], capture_output=True, check=True
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
        infos = tar.getmembers()
    assert [i.name for i in infos] == sorted(i.name for i in infos)
    assert {(i.mtime, i.uid, i.gid, i.uname, i.gname, i.mode) for i in infos} == {
        (946684800, 0, 0, "", "", 0o644)
    }


def test_verify_fails_on_one_flipped_byte(built):
    out, _ = built
    archive = out / "dqm.tar.zst"
    data = bytearray(archive.read_bytes())
    data[len(data) // 2] ^= 0x01
    archive.write_bytes(bytes(data))
    result = verify(out / LOCK_NAME, out)
    assert not result.ok
    failed = [line for line in result.lines if line.startswith("FAIL")]
    assert len(failed) == 1 and failed[0].startswith("FAIL dqm:") and "SHA-256" in failed[0]


def test_verify_reports_missing_and_resized_archives(built):
    out, _ = built
    (out / "dbs.tar.zst").unlink()
    with (out / "jira.tar.zst").open("ab") as handle:
        handle.write(b"\0")
    result = verify(out / LOCK_NAME, out)
    assert not result.ok
    assert "FAIL dbs: dbs.tar.zst is missing" in result.lines
    assert any(line.startswith("FAIL jira:") and "bytes" in line for line in result.lines)


def test_verify_checks_unpacked_contents_against_the_lock(built):
    out, lock = built
    lock["groups"]["indico"]["contents_sha256"] = "0" * 64
    (out / LOCK_NAME).write_text(yaml.safe_dump(lock), encoding="utf-8")
    result = verify(out / LOCK_NAME, out)
    assert "FAIL indico: unpacked contents do not match the lock's contents_sha256" in result.lines


def test_build_refuses_a_non_empty_output_directory(tmp_path, sources):
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("x")
    with pytest.raises(SnapshotError, match="not empty"):
        build(load_config(write_config(tmp_path, sources, only=["dqm"])), out)


# --- config -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "groups, expected",
    [
        ({"jira": {"path": "x"}}, "'collected' (the collection date) is required"),
        ({"jira": {"path": "x", "collected": "Aug 31"}}, "must be YYYY-MM-DD"),
        ({"jira": {"path": "x", "collected": "2026-06-16..2026-06-12"}}, "ends before"),
        ({"jira": {"path": "x", "collected": "2026-02-30"}}, "day is out of range"),
        ({"hypernews": {"path": "x", "collected": "2026-08-31"}}, "unknown group"),
        ({"jira": {"path": "x", "collected": "2026-08-31", "dropfields": []}}, "unknown keys"),
        ({"jira": {"collected": "2026-08-31"}}, "'path' is required"),
        ({"jira": {"path": "x", "collected": "2026-08-31", "drop_fields": "a"}}, "list of field"),
        ({}, "at least one group"),
    ],
)
def test_config_errors_are_named(groups, expected):
    with pytest.raises(SnapshotError, match=expected.replace("(", r"\(").replace(")", r"\)")):
        parse_config({"snapshot": "s", "groups": groups}, base=Path("/tmp"))


def test_config_accepts_yaml_dates_and_relative_paths(tmp_path):
    config = tmp_path / "c.yaml"
    config.write_text(
        "snapshot: cms-cache-20260928\n"
        "groups:\n"
        "  jira: {path: raw/jira, collected: 2026-08-31}\n"
        "  indico: {path: /abs/indico, collected: '2026-06-12..2026-06-16'}\n",
        encoding="utf-8",
    )
    parsed = load_config(config)
    rows = {g.name: g for g in parsed.groups}
    assert rows["jira"].collected == "2026-08-31"
    assert rows["jira"].path == tmp_path / "raw" / "jira"
    assert rows["indico"].collected == "2026-06-12..2026-06-16"


# --- command line -------------------------------------------------------------


def test_cli_build_then_verify_then_tamper(tmp_path, sources, capsys):
    config = write_config(tmp_path, sources, only=["jira", "indico"])
    out = tmp_path / "cli-out"
    assert cli_main(["build", "--config", str(config), "--out", str(out)]) == 0
    assert "jira: 2 records, 2 files, collected 2026-08-31" in capsys.readouterr().out
    assert cli_main(["verify", "--lock", str(out / LOCK_NAME), "--archives", str(out)]) == 0
    archive = out / "indico.tar.zst"
    data = bytearray(archive.read_bytes())
    data[-1] ^= 0xFF
    archive.write_bytes(bytes(data))
    assert cli_main(["verify", "--lock", str(out / LOCK_NAME), "--archives", str(out)]) == 1
    assert "FAIL indico" in capsys.readouterr().out


def test_cli_build_refusal_exits_2_and_names_the_group(tmp_path, sources, capsys):
    _write_json(sources["dqm"] / "records.json", [{"filename": "x"}])
    config = write_config(tmp_path, sources, only=["dqm", "jira"])
    assert cli_main(["build", "--config", str(config), "--out", str(tmp_path / "o")]) == 2
    err = capsys.readouterr().err
    assert "group 'dqm' refused" in err and "no snapshot written" in err


def test_the_counting_redactor_matches_the_plain_one():
    from archi.enrichment.anonymizer import (
        redact_email_addresses,
        redact_email_addresses_with_count,
    )

    for text, count in (
        ("no address here, AT&amp;T", 0),
        ("mail a@cern.ch or b@fnal.gov", 2),
        ("|bob@cern.ch|alice@fnal.gov|", 2),
        ("numpy@1.26.4 stays", 0),
    ):
        clean, removed = redact_email_addresses_with_count(text)
        assert clean == redact_email_addresses(text)
        assert removed == count


# --- a valid-looking file that is really another export --------------------------

SITES_COMPAT = {
    "desc": {"columns": ["site_name", "tier_level", "tier", "country", "usage"]},
    "result": [["T2_US_MIT", 2, "T2", "US", "production"]],
}


def test_a_sites_compat_export_as_responsibilities_refuses_cric(tmp_path, sources):
    _write_json(sources["cric"] / "responsibilities.json", SITES_COMPAT)
    with pytest.raises(BuildRefused) as info:
        build(load_config(write_config(tmp_path, sources, only=["cric"])), tmp_path / "out")
    [refusal] = info.value.refusals
    assert refusal.group == "cric"
    assert refusal.reason.startswith("responsibilities.json: columns are ['site_name'")
    assert "sites-compat" in refusal.reason


@pytest.mark.parametrize(
    "rows, expected",
    [
        ([["T2_US_MIT", 2, "T2", "US", "production"]], "row 0 has 5 fields, expected 3"),
        ([["adalove", "MIT Bates"]], "row 0 has 2 fields"),
        ([{"username": "adalove"}], "row 0 has dict fields"),
        ([["adalove", 7, "Site Admin"]], "site title must be a string or null"),
        ([[None, "MIT Bates", "Site Admin"]], "username and role must be strings"),
    ],
)
def test_responsibility_rows_of_the_wrong_shape_refuse_cric(tmp_path, sources, rows, expected):
    # No header, so only the rows can show the file is wrong.
    _write_json(sources["cric"] / "responsibilities.json", {"result": rows})
    with pytest.raises(BuildRefused, match=expected):
        build(load_config(write_config(tmp_path, sources, only=["cric"])), tmp_path / "out")


CRIC_EXPORTS = {
    ("cric", "sites.json"): "sitedb_title",
    ("cric", "storage_units.json"): "pledged-CMS",
    ("cric", "compute_units.json"): "potential_max",
    ("cric", "facilities.json"): "cmssites",
    ("cric-core", "services.json"): "rcsite",
    ("cric-core", "rcsites.json"): "sites",
    ("cric-core", "federations.json"): "accounting_name",
}
SIBLING_SWAPS = [
    (target, donor) for target in CRIC_EXPORTS for donor in CRIC_EXPORTS if donor != target
]


@pytest.mark.parametrize(
    "target, donor", SIBLING_SWAPS, ids=[f"{t[1]}<-{d[1]}" for t, d in SIBLING_SWAPS]
)
def test_every_cric_file_swapped_for_any_sibling_export_is_refused(
    tmp_path, sources, target, donor
):
    group, name = target
    (sources[group] / name).write_bytes((sources[donor[0]] / donor[1]).read_bytes())
    key = CRIC_EXPORTS[target]
    with pytest.raises(BuildRefused) as info:
        build(load_config(write_config(tmp_path, sources, only=[group])), tmp_path / "out")
    assert info.value.refusals[0].reason.startswith(
        f"{name}: only 0 of 1 records carry a {key} key (95% needed)"
    )


def _services(count_with, count_without):
    services = {
        f"ce{i}.example.org": {"type": "CE", "endpoint": f"ce{i}.example.org", "rcsite": "MIT"}
        for i in range(count_with)
    }
    services.update(
        {f"odd{i}.example.org": {"type": "CE", "endpoint": "x"} for i in range(count_without)}
    )
    return services


def test_a_few_records_without_the_signature_key_are_accepted(tmp_path, sources):
    _write_json(sources["cric-core"] / "services.json", _services(19, 1))  # 95%
    build(load_config(write_config(tmp_path, sources, only=["cric-core"])), tmp_path / "out")


def test_too_many_records_without_the_signature_key_are_refused(tmp_path, sources):
    _write_json(sources["cric-core"] / "services.json", _services(18, 2))  # 90%
    with pytest.raises(BuildRefused, match="only 18 of 20 records carry a rcsite key"):
        build(load_config(write_config(tmp_path, sources, only=["cric-core"])), tmp_path / "out")


def test_an_empty_export_is_refused(tmp_path, sources):
    _write_json(sources["cric-core"] / "services.json", {})
    with pytest.raises(BuildRefused, match="services.json holds no records"):
        build(load_config(write_config(tmp_path, sources, only=["cric-core"])), tmp_path / "out")


@pytest.mark.parametrize(
    "group, record, expected",
    [
        ("jira", {"key": "not an issue key", "summary": "x"}, "only 2 of 3 records carry a key"),
        ("indico", {"id": "9", "title": "no derived text"}, "_contributions_text / _pdf_texts"),
        ("conddb-global-tags", {"name": "GT", "description": "x"}, "release / scenario"),
        ("dbs", {"dataset": "not-a-dataset-path"}, "dataset_name / dataset matching"),
    ],
)
def test_a_record_from_another_export_is_refused(tmp_path, sources, group, record, expected):
    path = sources[group] / "records.json"
    path.write_text(json.dumps(json.loads(path.read_text()) + [record]))
    if group == "jira":
        _write_json(sources["jira"] / "meta.json", {"record_count": 3})
    with pytest.raises(BuildRefused, match=expected):
        build(load_config(write_config(tmp_path, sources, only=[group])), tmp_path / "out")


def test_cmssw_records_with_non_release_labels_are_refused(tmp_path, sources):
    (sources["cmssw-releases"] / "releases.map").unlink()
    _write_json(sources["cmssw-releases"] / "records.json", [{"label": "ECALTBH4_0_2_2"}])
    with pytest.raises(BuildRefused, match=r"only 0 of 1 records carry a label matching CMSSW_"):
        build(
            load_config(write_config(tmp_path, sources, only=["cmssw-releases"])),
            tmp_path / "out",
        )


# --- temporary copies -------------------------------------------------------------


@pytest.fixture
def system_tmp(tmp_path, monkeypatch):
    """A stand-in for the system temp dir, to prove nothing lands there."""
    sentinel = tmp_path / "system-tmp"
    sentinel.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(sentinel))
    return sentinel


def test_reader_check_copies_go_to_tmp_dir_and_are_removed(tmp_path, sources, system_tmp):
    work = tmp_path / "work-tmp"
    work.mkdir()
    seen = []
    real = tempfile.TemporaryDirectory

    def spy(*args, **kwargs):
        handle = real(*args, **kwargs)
        seen.append(Path(handle.name))
        return handle

    import archi.snapshot.builder as builder_module

    original = builder_module.tempfile.TemporaryDirectory
    builder_module.tempfile.TemporaryDirectory = spy
    try:
        build(load_config(write_config(tmp_path, sources)), tmp_path / "out", tmp_dir=work)
    finally:
        builder_module.tempfile.TemporaryDirectory = original
    assert len(seen) == len(groups_module.GROUPS)
    assert all(path.parent == work for path in seen)
    assert list(work.iterdir()) == []
    assert list(system_tmp.iterdir()) == []


def test_tmp_dir_is_emptied_on_refusal(tmp_path, sources, system_tmp):
    work = tmp_path / "work-tmp"
    work.mkdir()
    config = write_config(
        tmp_path, sources, only=["jira", "dqm"], extra={"jira": {"drop_fields": ["description"]}}
    )
    with pytest.raises(BuildRefused, match="changed what the reader emits"):
        build(load_config(config), tmp_path / "out", tmp_dir=work)
    assert list(work.iterdir()) == []
    assert list(system_tmp.iterdir()) == []
    assert not (tmp_path / "out").exists()


def _spy_tempdirs(monkeypatch):
    import archi.snapshot.builder as builder_module

    seen = []
    real = tempfile.TemporaryDirectory

    def spy(*args, **kwargs):
        handle = real(*args, **kwargs)
        seen.append(Path(handle.name))
        return handle

    monkeypatch.setattr(builder_module.tempfile, "TemporaryDirectory", spy)
    return seen


def test_tmp_dir_defaults_to_the_system_temp_dir(tmp_path, sources, system_tmp, monkeypatch):
    seen = _spy_tempdirs(monkeypatch)
    build(load_config(write_config(tmp_path, sources, only=["dqm", "jira"])), tmp_path / "out")
    assert len(seen) == 2
    assert all(path.parent == system_tmp for path in seen)
    assert list(system_tmp.iterdir()) == []


def test_a_full_tmp_dir_is_reported_as_a_write_failure(tmp_path, sources, monkeypatch):
    import errno as errno_module

    import archi.snapshot.builder as builder_module

    def full(root, files):
        raise OSError(errno_module.ENOSPC, "No space left on device")

    monkeypatch.setattr(builder_module, "_stage", full)
    with pytest.raises(BuildRefused) as info:
        build(load_config(write_config(tmp_path, sources, only=["dqm"])), tmp_path / "out")
    reason = info.value.refusals[0].reason
    assert reason.startswith("cannot write the reader-check copy in ")
    assert "(disk full?)" in reason and "cannot read its cache" not in reason


def _sigterm_during(monkeypatch, attribute):
    import archi.snapshot.builder as builder_module

    real = getattr(builder_module, attribute)
    fired = []

    def wrapper(*args, **kwargs):
        if not fired:
            fired.append(True)
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(1)  # the handler raises before this returns
        return real(*args, **kwargs)

    monkeypatch.setattr(builder_module, attribute, wrapper)


@pytest.mark.parametrize("attribute", ["_run_reader", "write_archive"])
def test_sigterm_cleans_tmp_dir_and_output_staging(
    tmp_path, sources, system_tmp, monkeypatch, attribute
):
    work = tmp_path / "work-tmp"
    work.mkdir()
    _sigterm_during(monkeypatch, attribute)
    config = write_config(tmp_path, sources, only=["dqm", "jira"])
    before = signal.getsignal(signal.SIGTERM)
    with pytest.raises(SystemExit) as info:
        cli_main(["build", "--config", str(config), "--out", str(tmp_path / "o"),
                  "--tmp-dir", str(work)])
    assert info.value.code == 128 + signal.SIGTERM
    assert list(work.iterdir()) == []
    assert list(system_tmp.iterdir()) == []
    assert not (tmp_path / "o").exists()
    assert not list(tmp_path.glob(".o.*"))  # the output staging directory
    assert signal.getsignal(signal.SIGTERM) is before


def test_cli_tmp_dir_must_exist(tmp_path, sources, capsys):
    config = write_config(tmp_path, sources, only=["dqm"])
    code = cli_main(
        ["build", "--config", str(config), "--out", str(tmp_path / "o"),
         "--tmp-dir", str(tmp_path / "no-such-dir")]
    )
    assert code == 2
    assert "is not a directory" in capsys.readouterr().err


# --- per-file dates ---------------------------------------------------------------


def test_file_dates_are_carried_into_the_lock(tmp_path, sources):
    # Jason, 2026-09-28: the June CRIC set, files dated 06-12, facilities
    # fetched 04-09; the lock records both.
    config = write_config(
        tmp_path,
        sources,
        only=["cric"],
        extra={"cric": {"collected": "2026-06-12", "file_dates": {"facilities.json": "2026-04-09"}}},
    )
    lock = build(load_config(config), tmp_path / "out")
    row = lock["groups"]["cric"]
    assert row["collected"] == "2026-06-12"
    assert row["file_dates"] == {"facilities.json": "2026-04-09"}
    on_disk = yaml.safe_load((tmp_path / "out" / LOCK_NAME).read_text())
    assert on_disk["groups"]["cric"]["file_dates"] == {"facilities.json": "2026-04-09"}


def test_file_dates_must_name_a_file_the_group_archives(tmp_path, sources):
    config = write_config(
        tmp_path, sources, only=["cric"], extra={"cric": {"file_dates": {"meta.json": "2026-04-09"}}}
    )
    with pytest.raises(BuildRefused, match="file_dates names meta.json"):
        build(load_config(config), tmp_path / "out")


@pytest.mark.parametrize("value", [{"facilities.json": "April 9"}, [], {}])
def test_file_dates_must_be_dates(value):
    with pytest.raises(SnapshotError, match="file_dates|YYYY-MM-DD"):
        parse_config(
            {"snapshot": "s", "groups": {"cric": {"path": "x", "collected": "2026-06-12",
                                                    "file_dates": value}}},
            base=Path("/tmp"),
        )


# --- group notes ---------------------------------------------------------------------

CRIC_NOTE = "files written 2026-06-12; fetched 2026-04-09"


def test_a_group_note_is_carried_into_the_lock(tmp_path, sources):
    # Lead decision, 2026-09-28: the June CRIC set is dated by its fetch, and the
    # lock itself says when the files were written.
    config = write_config(
        tmp_path, sources, only=["cric"], extra={"cric": {"collected": "2026-04-09", "note": CRIC_NOTE}}
    )
    out = tmp_path / "out"
    build(load_config(config), out)
    row = yaml.safe_load((out / LOCK_NAME).read_text())["groups"]["cric"]
    assert row["collected"] == "2026-04-09"
    assert row["note"] == CRIC_NOTE
    assert "file_dates" not in row
    assert verify(out / LOCK_NAME, out).ok


def test_a_group_without_a_note_has_none_in_the_lock(built):
    _, lock = built
    assert all("note" not in row for row in lock["groups"].values())


@pytest.mark.parametrize(
    "note, expected",
    [("x" * 301, "'note' is 301 characters; the limit is 300"), ("   ", "non-empty text"), (7, "non-empty text")],
)
def test_note_must_be_short_text(note, expected):
    with pytest.raises(SnapshotError, match=expected):
        parse_config(
            {"snapshot": "s", "groups": {"cric": {"path": "x", "collected": "2026-04-09", "note": note}}},
            base=Path("/tmp"),
        )


@pytest.mark.parametrize(
    "note, expected",
    [
        ("fetched by jdoe@cern.ch", "note: an address survived redaction"),
        # The redactor reads the C1 control as part of the address (#14).
        ("fetched by a.b@c\u0081d.ch", "note: an address survived redaction"),
    ],
)
def test_an_address_in_a_note_refuses_the_group(tmp_path, sources, note, expected):
    config = write_config(tmp_path, sources, only=["cric"], extra={"cric": {"note": note}})
    with pytest.raises(BuildRefused, match=expected):
        build(load_config(config), tmp_path / "out")
    assert not (tmp_path / "out").exists()


# --- a printable fallback character must not hide an address -----------------------

@pytest.mark.parametrize(
    "raw",
    [
        b"jean.dupont\x93@cern.ch",      # a smart quote before the @
        b"jean.dupont@ce\x96rn.ch",      # an en dash inside the domain
        b"jean.dupont@ce\xa0rn.ch",      # a no-break space inside the domain
        b"jean.dupont\xc0\x80@cern.ch",  # an overlong NUL (two invalid bytes)
    ],
    ids=["quote-before-at", "dash-in-domain", "nbsp-in-domain", "overlong-nul"],
)
def test_the_third_reviews_examples_lose_the_whole_address(tmp_path, sources, raw):
    (sources["twiki-eos"] / "Hidden.txt").write_bytes(b"---+ Page\nWrite to " + raw + b" today.\n" + PADDING)
    out = tmp_path / "out"
    lock = build(load_config(write_config(tmp_path, sources, only=["twiki-eos"])), out)
    page = read_archive(out / "twiki-eos.tar.zst")["data/twiki-eos/Hidden.txt"].decode("utf-8")
    assert "jean" not in page and "dupont" not in page and "cern" not in page and "@" not in page
    assert "Write to  today." in page
    assert lock["groups"]["twiki-eos"]["pages_bytes_dropped"] == 1


def test_a_stray_byte_away_from_addresses_keeps_its_fallback_character(tmp_path, sources):
    (sources["twiki-eos"] / "Quote.txt").write_bytes(
        b"---+ Page\nThe \x93golden\x94 JSON; mail jean.dupont@cern.ch.\n" + PADDING
    )
    out = tmp_path / "out"
    lock = build(load_config(write_config(tmp_path, sources, only=["twiki-eos"])), out)
    page = read_archive(out / "twiki-eos.tar.zst")["data/twiki-eos/Quote.txt"].decode("utf-8")
    assert "\u201cgolden\u201d" in page and "dupont" not in page
    assert lock["groups"]["twiki-eos"]["pages_bytes_dropped"] == 0


def test_fuzz_no_address_from_the_dropped_view_survives():
    """2,000 random pages, each with an address and 1-3 stray bytes placed
    anywhere, including inside and next to the address."""
    import random

    from archi.enrichment.anonymizer import email_address_spans
    from archi.snapshot.builder import _decode_with_fallback

    rng = random.Random(20260928)
    # The address is written only with letters and digits that the words
    # around it never use, so any of them left in the output is a leak.
    address_letters = "qxzkjvy"
    word_letters = "abdefgilmnoprstuw"
    marks = set(address_letters + "0123456789@")
    checked = 0
    for case in range(2000):
        local = "".join(rng.choice(address_letters + "0123456789") for _ in range(rng.randint(3, 12)))
        if rng.random() < 0.3:
            local = local[:2] + "." + local[2:]
        domain = "".join(rng.choice(address_letters) for _ in range(rng.randint(2, 8)))
        address = f"{local}@{domain}.{rng.choice(['qz', 'kj', 'yx'])}".encode()
        chunks = [bytes([b]) for b in address]
        for _ in range(rng.randint(1, 3)):
            stray = bytes([rng.randint(0x80, 0xFF)])
            chunks.insert(rng.randint(0, len(chunks)), stray)
        words = [
            "".join(rng.choice(word_letters) for _ in range(rng.randint(1, 8)))
            for _ in range(rng.randint(0, 6))
        ]
        before = " ".join(words[: len(words) // 2]).encode()
        after = " ".join(words[len(words) // 2 :]).encode()
        sep = rng.choice([b" ", b"", b"(", b"<"])
        data = before + sep + b"".join(chunks) + sep + after + b"\n"
        page = _decode_with_fallback(data)
        output = redact(page.text)
        dropped = data.decode("utf-8", errors="ignore")
        found = [dropped[a:b] for a, b in email_address_spans(dropped)]
        for address in found:
            local, _, domain = address.partition("@")
            assert address not in output, (case, data, output)
            assert local not in output and domain not in output, (case, data, output)
        checked += len(found)
        # Stray bytes that do not pair into a valid UTF-8 character leave
        # none of the address's letters behind at all.
        if not any(0xC0 <= b for b in data if b >= 0x80) or data.decode("utf-8", "ignore").isascii():
            assert not marks & set(output), (case, data, output)
    assert checked >= 1000


def redact(text):
    from archi.enrichment.anonymizer import redact_email_addresses

    return redact_email_addresses(text)


# --- ANSI colour codes ----------------------------------------------------------------

def test_ansi_colour_codes_away_from_an_address_are_stripped(tmp_path, sources):
    (sources["twiki-eos"] / "Log.txt").write_bytes(
        b"---+ Log\n$ make done \x1b[1;31mERROR\x1b[0m\nmail jdoe@example.org\n" + PADDING
    )
    out = tmp_path / "out"
    lock = build(load_config(write_config(tmp_path, sources, only=["twiki-eos"])), out)
    page = read_archive(out / "twiki-eos.tar.zst")["data/twiki-eos/Log.txt"].decode("utf-8")
    assert page.startswith("---+ Log\n$ make done ERROR\nmail \n")
    assert lock["groups"]["twiki-eos"]["ansi_sequences_stripped"] == 2


def test_a_colour_code_on_the_same_line_as_an_address_is_stripped(tmp_path, sources):
    # The colour codes do not touch the address: they are stripped and the
    # address is redacted.
    (sources["twiki-eos"] / "Log.txt").write_bytes(
        b"---+ Log\n$ make done \x1b[1;31mERROR\x1b[0m, mail jdoe@example.org\n" + PADDING
    )
    out = tmp_path / "out"
    lock = build(load_config(write_config(tmp_path, sources, only=["twiki-eos"])), out)
    page = read_archive(out / "twiki-eos.tar.zst")["data/twiki-eos/Log.txt"].decode("utf-8")
    assert page.startswith("---+ Log\n$ make done ERROR, mail \n")
    assert lock["groups"]["twiki-eos"]["ansi_sequences_stripped"] == 2


def test_ansi_colour_codes_inside_an_address_refuse_the_group(tmp_path, sources):
    # Detection runs before the strip: a colour code in or next to an
    # address refuses rather than being stripped and re-matched.
    (sources["twiki-eos"] / "Log.txt").write_bytes(
        b"---+ Log\n$ \x1b[32mjdoe\x1b[0m@laptop.example.org done\n" + PADDING
    )
    with pytest.raises(BuildRefused, match=r"Log.txt, line 2: span rule"):
        build(load_config(write_config(tmp_path, sources, only=["twiki-eos"])), tmp_path / "out")


# --- signals -----------------------------------------------------------------------------

def test_sigterm_handler_restores_the_default_when_none_was_set(monkeypatch, tmp_path, sources):
    import archi.snapshot.__main__ as main_module

    calls = []
    real_signal = signal.signal

    def fake_signal(signum, handler):
        calls.append(handler)
        if len(calls) == 1:
            real_signal(signum, handler)
            return None  # as for a handler not installed from Python
        return real_signal(signum, handler)

    monkeypatch.setattr(main_module.signal, "signal", fake_signal)
    before = signal.getsignal(signal.SIGTERM)
    try:
        cli_main(["verify", "--lock", str(tmp_path / "missing.yaml"), "--archives", str(tmp_path)])
    finally:
        real_signal(signal.SIGTERM, before)
    assert calls[-1] is signal.SIG_DFL


def test_a_second_sigterm_during_cleanup_is_ignored(tmp_path, sources, system_tmp, monkeypatch):
    import archi.snapshot.builder as builder_module

    work = tmp_path / "work-tmp"
    work.mkdir()
    real = builder_module._run_reader
    fired = []

    def wrapper(*args, **kwargs):
        if not fired:
            fired.append(True)
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(builder_module, "_run_reader", wrapper)
    real_rmtree = builder_module.shutil.rmtree

    def slow_cleanup(*args, **kwargs):  # a second SIGTERM arrives mid-cleanup
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(0.2)
        return real_rmtree(*args, **kwargs)

    monkeypatch.setattr(builder_module.shutil, "rmtree", slow_cleanup)
    import shutil as shutil_module

    monkeypatch.setattr(shutil_module, "rmtree", slow_cleanup)
    config = write_config(tmp_path, sources, only=["dqm"])
    before = signal.getsignal(signal.SIGTERM)
    with pytest.raises(SystemExit) as info:
        cli_main(["build", "--config", str(config), "--out", str(tmp_path / "o"),
                  "--tmp-dir", str(work)])
    assert info.value.code == 128 + signal.SIGTERM
    assert list(work.iterdir()) == []
    assert signal.getsignal(signal.SIGTERM) is before


def test_file_dates_errors_name_the_file_date_not_collected():
    with pytest.raises(SnapshotError) as info:
        parse_config(
            {"snapshot": "s", "groups": {"cric": {"path": "x", "collected": "2026-06-12",
                                                    "file_dates": {"facilities.json": "April 9"}}}},
            base=Path("/tmp"),
        )
    message = str(info.value)
    assert "\"file_dates['facilities.json']\" must be YYYY-MM-DD" in message
    assert "'collected'" not in message


# --- git@ exemption and spelled-out addresses (archi-okg #13) -----------------------

GIT_REMOTE = "git@gitlab.cern.ch:cmsdmops/Documentation.git"
ADJACENT = "git@github.com:org/tools.git by jdoe@cern.ch"


def test_jira_keeps_a_git_remote_through_the_snapshot(tmp_path, sources):
    path = sources["jira"] / "records.json"
    records = json.loads(path.read_text())
    records[0]["description"] = f"Clone {GIT_REMOTE} first. Pushed {ADJACENT}."
    path.write_text(json.dumps(records))
    out = tmp_path / "out"
    build(load_config(write_config(tmp_path, sources, only=["jira"])), out)
    stored = json.loads(read_archive(out / "jira.tar.zst")["data/jira/records.json"])
    assert stored[0]["description"] == (
        f"Clone {GIT_REMOTE} first. Pushed git@github.com:org/tools.git by ."
    )


def test_twiki_keeps_a_git_remote_and_loses_the_address_after_it(tmp_path, sources):
    (sources["twiki-eos"] / "Git.txt").write_bytes(
        f"---+ Git\nClone {GIT_REMOTE}. Pushed {ADJACENT} today.\n".encode() + PADDING
    )
    out = tmp_path / "out"
    build(load_config(write_config(tmp_path, sources, only=["twiki-eos"])), out)
    page = read_archive(out / "twiki-eos.tar.zst")["data/twiki-eos/Git.txt"].decode()
    assert f"Clone {GIT_REMOTE}. Pushed git@github.com:org/tools.git by  today." in page
    assert "jdoe" not in page


OBFUSCATED = {
    "cric": ("facilities.json", ["MIT", "fullname"]),
    "cric-core": ("federations.json", ["US-MIT", "accounting_name"]),
    "jira": ("records.json", [0, "description"]),
    "indico": ("records.json", [0, "_contributions_text"]),
    "dqm": ("records.json", [0, "filename"]),
    "gocdb-downtimes": ("records.json", [0, "description"]),
    "gitlab-docs": ("records.json", [0, "body"]),
    "docsite": ("records.json", [0, "body"]),
    "conddb-global-tags": ("records.json", [0, "description"]),
    "wmstats": ("records.json", [0, "Campaign"]),
    "dbs": ("records.json", [0, "physics_group_name"]),
    "twiki-eos": ("CompOpsGuide.txt", None),
}


@pytest.mark.parametrize("group", sorted(OBFUSCATED))
def test_a_spelled_out_address_is_removed_in_every_text_group(tmp_path, sources, group):
    file_name, field = OBFUSCATED[group]
    spelled = "ob.person[AT]cern(dot)ch"
    path = sources[group] / file_name
    if field is None:
        path.write_text(path.read_text() + f"Ask {spelled} or john.doe at cern.ch.\n")
    else:
        payload = json.loads(path.read_text())
        holder = payload
        for key in field[:-1]:
            holder = holder[key]
        holder[field[-1]] = f"{holder[field[-1]]} ask {spelled} or john.doe at cern.ch"
        path.write_text(json.dumps(payload))
    out = tmp_path / "out"
    lock = build(load_config(write_config(tmp_path, sources, only=[group])), out)
    data = _all_bytes(out, group)
    assert b"ob.person" not in data and b"[AT]" not in data
    # Free prose is kept (operator rule, 2026-09-28).
    assert b"john.doe at cern.ch" in data
    assert lock["groups"][group]["obfuscated_addresses_removed"] == 1


def test_a_nospam_token_and_a_glued_form_are_removed(tmp_path, sources):
    path = sources["jira"] / "records.json"
    records = json.loads(path.read_text())
    records[0]["description"] = "Mail jdoe_at_cern.ch or jdoeNOSPAM@cernNOSPAM.ch today."
    path.write_text(json.dumps(records))
    out = tmp_path / "out"
    lock = build(load_config(write_config(tmp_path, sources, only=["jira"])), out)
    stored = json.loads(read_archive(out / "jira.tar.zst")["data/jira/records.json"])
    assert "jdoe" not in stored[0]["description"]
    assert lock["groups"]["jira"]["obfuscated_addresses_removed"] >= 1


def test_a_spelled_out_address_in_a_note_refuses_the_group(tmp_path, sources):
    config = write_config(
        tmp_path, sources, only=["dqm"], extra={"dqm": {"note": "ask ob.person[at]cern.ch"}}
    )
    with pytest.raises(BuildRefused, match="note: a spelled-out address survived redaction"):
        build(load_config(config), tmp_path / "out")


def test_obfuscated_counting_matches_the_plain_function():
    from archi.enrichment.anonymizer import (
        redact_obfuscated_email_addresses,
        redact_obfuscated_email_addresses_with_count,
    )

    for text, count in (
        ("plain text, no address", 0),
        ("ob.person[AT]cern(dot)ch and a.b(at)fnal.gov", 2),
        ("john.doe at cern.ch stays", 0),
        ("jdoe_at_cern.ch", 1),
        ("jdoeNOSPAM@cernNOSPAM.ch", 1),
        ("NoSpamFilter stays", 0),
    ):
        clean, removed = redact_obfuscated_email_addresses_with_count(text)
        assert clean == redact_obfuscated_email_addresses(text)
        assert removed == count, text


# --- final review: spelled-out forms get the same defences as @ addresses -----------


@pytest.mark.parametrize("value", ["jdoe\x7f[at]cern.ch ok", "jdoe[at]cern\x7f.ch"])
def test_a_control_character_inside_a_spelled_out_address_refuses_the_group(
    tmp_path, sources, value
):
    path = sources["dqm"] / "records.json"
    records = json.loads(path.read_text())
    records[0]["filename"] = value
    path.write_text(json.dumps(records))
    with pytest.raises(BuildRefused, match=r"span rule"):
        build(load_config(write_config(tmp_path, sources, only=["dqm"])), tmp_path / "out")


def test_a_control_character_inside_a_spelled_out_address_in_twiki_refuses(tmp_path, sources):
    (sources["twiki-eos"] / "Del.txt").write_bytes(b"---+ Page\nAsk jdoe\x7f[at]cern.ch ok.\n" + PADDING)
    with pytest.raises(BuildRefused, match=r"Del.txt, line 2: span rule"):
        build(load_config(write_config(tmp_path, sources, only=["twiki-eos"])), tmp_path / "out")


@pytest.mark.parametrize(
    "raw",
    [
        b"jdoe\x93[at]cern.ch",
        b"jdoe[at]\x93cern.ch",
        b"jdoe\x93(at)cern(dot)ch",
        b"john.doe\x93_at_cern.ch",
    ],
    ids=["quote-before-at", "quote-after-at", "quote-before-parenthesised", "quote-before-glued"],
)
def test_a_stray_byte_does_not_hide_a_spelled_out_address(tmp_path, sources, raw):
    (sources["twiki-eos"] / "Spelled.txt").write_bytes(
        b"---+ Page\nAsk " + raw + b" today.\n" + PADDING
    )
    out = tmp_path / "out"
    lock = build(load_config(write_config(tmp_path, sources, only=["twiki-eos"])), out)
    page = read_archive(out / "twiki-eos.tar.zst")["data/twiki-eos/Spelled.txt"].decode("utf-8")
    assert "Ask  today." in page
    for fragment in ("jdoe", "john", "cern", "at]", "(at)", "_at_"):
        assert fragment not in page
    row = lock["groups"]["twiki-eos"]
    assert row["pages_bytes_dropped"] == 1 and row["obfuscated_addresses_removed"] >= 1


def test_fuzz_no_spelled_out_address_from_the_dropped_view_survives():
    """2,000 random pages with a spelled-out address and 1-3 stray bytes."""
    import random

    from archi.snapshot.builder import _decode_with_fallback, _redaction_mask

    rng = random.Random(1409)
    address_letters = "qxzkjvy"
    word_letters = "bfilmnoprsuw"  # no a, t, d, c, h: they spell the separators
    separators = ["[at]", "[AT]", "(at)", "{at}", "<at>", "_at_"]
    dots = [".", "(dot)", "[DOT]"]
    checked = 0
    for case in range(2000):
        local = "".join(rng.choice(address_letters) for _ in range(rng.randint(3, 10)))
        sep = rng.choice(separators)
        if sep == "_at_":
            domain = rng.choice(["cern.ch", "fnal.gov", "gmail.com"])
        else:
            domain = (
                "".join(rng.choice(address_letters) for _ in range(rng.randint(2, 6)))
                + rng.choice(dots)
                + rng.choice(["qz", "kj", "yx"])
            )
        chunks = [bytes([b]) for b in f"{local}{sep}{domain}".encode()]
        for _ in range(rng.randint(1, 3)):
            chunks.insert(rng.randint(0, len(chunks)), bytes([rng.randint(0x80, 0xFF)]))
        words = [
            "".join(rng.choice(word_letters) for _ in range(rng.randint(1, 8)))
            for _ in range(rng.randint(0, 6))
        ]
        data = (
            " ".join(words[: len(words) // 2]).encode()
            + b" " + b"".join(chunks) + b" "
            + " ".join(words[len(words) // 2 :]).encode() + b"\n"
        )
        page = _decode_with_fallback(data)
        output = redact(page.text)
        from archi.enrichment.anonymizer import redact_obfuscated_email_addresses

        output = redact_obfuscated_email_addresses(output)
        dropped = data.decode("utf-8", errors="ignore")
        mask = _redaction_mask(dropped)
        if not any(mask):
            continue
        checked += 1
        removed_text = "".join(c for c, m in zip(dropped, mask, strict=True) if m)
        assert local not in output, (case, data, output)
        # Stray bytes that cannot pair into a valid UTF-8 character leave no
        # letter of the address behind.
        if dropped.isascii():
            assert not set(address_letters) & set(output), (case, data, output, removed_text)
    assert checked >= 1000


def test_ansi_codes_are_stripped_from_json_strings(tmp_path, sources):
    path = sources["jira"] / "records.json"
    records = json.loads(path.read_text())
    records[0]["description"] = "Logged by jdoe@example.org in \x1b[1mbold\x1b[0m."
    path.write_text(json.dumps(records))
    out = tmp_path / "out"
    lock = build(load_config(write_config(tmp_path, sources, only=["jira"])), out)
    stored = json.loads(read_archive(out / "jira.tar.zst")["data/jira/records.json"])
    assert stored[0]["description"] == "Logged by  in bold."
    assert lock["groups"]["jira"]["ansi_sequences_stripped"] == 2


# --- fifth and sixth reviews: hidden characters in or next to an address --------
#
# Until 51bb241 the control-character check ran on the redacted output, so a
# control character inside the local part left the name before it stored.
# d45838f checked the input, but after stripping terminal sequences, and its
# wider strip could delete an address's own "." or "@". Now each value is
# checked on the raw input first: matched with every hidden character (C0/C1
# controls and format characters) removed, a match that contains or touches
# a hidden character, or a colour sequence, refuses the group.

HIDDEN_IN_ADDRESS = [
    # fifth review
    "jdoe\x7fx@example.org",
    "jdoe\x01.x@example.org",
    "jdoe\x7fx[at]example.org",
    "j\x7fdoe.x(at)example(dot)org",
    "\x1b[1mjdoe\x1b(B\x1b[m@example.org",
    "jdoe\x9b32m@example.org",
    # sixth review
    "jdoe@example\x1b.org",
    "jdoe@example\x9b.org",
    "jdoe\x9b@example.org",
    "\u2068jdoe\u2069@example.org",
    "jdoe\u200ex@example.org",
    "\x1b[31mjdoe@example.org\x1b[0m",
]


@pytest.mark.parametrize("value", HIDDEN_IN_ADDRESS)
def test_a_hidden_character_in_an_address_in_a_json_string_refuses(
    tmp_path, sources, value
):
    path = sources["jira"] / "records.json"
    records = json.loads(path.read_text())
    records[0]["description"] = f"Logged by {value} today."
    path.write_text(json.dumps(records))
    with pytest.raises(BuildRefused, match=r"span rule"):
        build(load_config(write_config(tmp_path, sources, only=["jira"])), tmp_path / "out")


@pytest.mark.parametrize("value", HIDDEN_IN_ADDRESS)
def test_a_hidden_character_in_an_address_on_a_text_page_refuses(
    tmp_path, sources, value
):
    (sources["twiki-eos"] / "Local.txt").write_bytes(
        f"---+ Page\nAsk {value} today.\n".encode("utf-8") + PADDING
    )
    with pytest.raises(
        BuildRefused, match=r"Local.txt, line 2: span rule"
    ):
        build(load_config(write_config(tmp_path, sources, only=["twiki-eos"])), tmp_path / "out")


@pytest.mark.parametrize("key", ["jdoe\x7fx@example.org", "jdoe@example\u200e.org"])
def test_a_hidden_character_in_an_address_in_a_json_key_refuses(tmp_path, sources, key):
    path = sources["jira"] / "records.json"
    records = json.loads(path.read_text())
    records[0][key] = "owner"
    path.write_text(json.dumps(records))
    with pytest.raises(BuildRefused, match=r"span rule"):
        build(load_config(write_config(tmp_path, sources, only=["jira"])), tmp_path / "out")


#: Format characters (Unicode category Cf) the redactor does not read as part
#: of an address; each kept "jdoe" out of jdoe<char>x@example.org.
FORMAT_CHARACTERS = [
    "\ufeff", "\u200e", "\u200f", "\u202a", "\u202b", "\u202c", "\u202d", "\u202e",
    "\u2066", "\u2067", "\u2068", "\u2069", "\u2061", "\u2062", "\u2063", "\u2064",
    "\u180e", "\u0600", "\u0601", "\u0602", "\u0603", "\u0604", "\u0605",
]


@pytest.mark.parametrize("char", FORMAT_CHARACTERS, ids=lambda c: f"U+{ord(c):04X}")
def test_a_format_character_in_the_local_part_refuses(char):
    with pytest.raises(GroupRefused, match=r"span rule"):
        builder_module._clean(
            "g", "f", f"Ask jdoe{char}x@example.org today.", builder_module._Counter()
        )


@pytest.mark.parametrize("char", ["\u00ad", "\u200b", "\u200c", "\u200d", "\u2060"])
def test_invisible_characters_the_redactor_reads_are_removed_with_the_address(
    tmp_path, sources, char
):
    path = sources["jira"] / "records.json"
    records = json.loads(path.read_text())
    records[0]["description"] = f"Logged by jdoe{char}x@example.org today."
    path.write_text(json.dumps(records))
    out = tmp_path / "out"
    build(load_config(write_config(tmp_path, sources, only=["jira"])), out)
    stored = json.loads(read_archive(out / "jira.tar.zst")["data/jira/records.json"])
    assert stored[0]["description"] == "Logged by  today."


@pytest.mark.parametrize("char", ["\u00ad", "\u200b"])
def test_invisible_characters_in_words_without_an_at_are_kept(tmp_path, sources, char):
    path = sources["jira"] / "records.json"
    records = json.loads(path.read_text())
    records[0]["description"] = f"hyphen{char}ated text, mail jdoe@example.org"
    path.write_text(json.dumps(records))
    out = tmp_path / "out"
    build(load_config(write_config(tmp_path, sources, only=["jira"])), out)
    stored = json.loads(read_archive(out / "jira.tar.zst")["data/jira/records.json"])
    assert stored[0]["description"] == f"hyphen{char}ated text, mail "


#: A lone ESC or U+009B that starts no CSI sequence is left in place: d45838f
#: stripped "ESC 5" and "U+009B P" along with them.
LONE_CONTROLS = ["price \x1b 5 x", "Home \x9b Projects", "\x9b2023 jdoe"]


@pytest.mark.parametrize("value", LONE_CONTROLS)
def test_a_lone_escape_in_a_json_string_is_kept_with_its_text(tmp_path, sources, value):
    path = sources["jira"] / "records.json"
    records = json.loads(path.read_text())
    records[0]["description"] = value
    path.write_text(json.dumps(records))
    out = tmp_path / "out"
    lock = build(load_config(write_config(tmp_path, sources, only=["jira"])), out)
    stored = json.loads(read_archive(out / "jira.tar.zst")["data/jira/records.json"])
    assert stored[0]["description"] == value
    assert lock["groups"]["jira"]["ansi_sequences_stripped"] == 0


@pytest.mark.parametrize("value", LONE_CONTROLS)
def test_a_lone_escape_on_a_text_page_is_kept_with_its_text(tmp_path, sources, value):
    (sources["twiki-eos"] / "Lone.txt").write_bytes(
        f"---+ Page\n{value}\n".encode("utf-8") + PADDING
    )
    out = tmp_path / "out"
    build(load_config(write_config(tmp_path, sources, only=["twiki-eos"])), out)
    page = read_archive(out / "twiki-eos.tar.zst")["data/twiki-eos/Lone.txt"].decode("utf-8")
    assert page.startswith(f"---+ Page\n{value}\n")


# --- seventh review: short escapes, string sequences, paths ---------------------

SHORT_AND_STRING_ESCAPES = [
    "jdoe\x1b\\@example.org",
    "jdoe@example\x1b\\.org",
    "jdoe\x1bPq\x1b\\@example.org",
    "jdoe\x1b>@example.org",
]
#: Round 8 refused these by its word rule. Round 9 refuses only what the
#: redactors read as an address in some reading of the value, and none of
#: these is one: each is stored as its plain-text equivalent would be.
NOT_AN_ADDRESS_IN_ANY_READING = [
    # terminal: "jdoeexample.org"; ignoring controls: "jdoe[@example.org"
    ("jdoe\x1b[@example.org", "jdoeexample.org"),
    # both readings: "jdoe(\u00e9@example.org", whose address starts after "("
    ("jdoe\x1b(\u00e9@example.org", "jdoe\x1b("),
    # both readings: "jdoe[at]laptop", a host with no dot
    ("jdoe[\u200eat]laptop", "jdoe[\u200eat]laptop"),
]


@pytest.mark.parametrize(("value", "stored"), NOT_AN_ADDRESS_IN_ANY_READING)
def test_a_value_no_reading_shows_as_an_address_is_stored(
    tmp_path, sources, value, stored
):
    path = sources["jira"] / "records.json"
    records = json.loads(path.read_text())
    records[0]["description"] = f"Logged by {value} today."
    path.write_text(json.dumps(records))
    out = tmp_path / "out"
    build(load_config(write_config(tmp_path, sources, only=["jira"])), out)
    result = json.loads(read_archive(out / "jira.tar.zst")["data/jira/records.json"])
    assert result[0]["description"] == f"Logged by {stored} today."


@pytest.mark.parametrize("value", SHORT_AND_STRING_ESCAPES)
def test_a_short_or_string_escape_in_an_address_in_json_refuses(tmp_path, sources, value):
    path = sources["jira"] / "records.json"
    records = json.loads(path.read_text())
    records[0]["description"] = f"Logged by {value} today."
    path.write_text(json.dumps(records))
    with pytest.raises(BuildRefused, match=r"span rule"):
        build(load_config(write_config(tmp_path, sources, only=["jira"])), tmp_path / "out")


@pytest.mark.parametrize("value", SHORT_AND_STRING_ESCAPES)
def test_a_short_or_string_escape_in_an_address_on_a_text_page_refuses(
    tmp_path, sources, value
):
    (sources["twiki-eos"] / "Short.txt").write_bytes(
        f"---+ Page\nAsk {value} today.\n".encode("utf-8") + PADDING
    )
    with pytest.raises(
        BuildRefused, match=r"Short.txt, line 2: span rule"
    ):
        build(load_config(write_config(tmp_path, sources, only=["twiki-eos"])), tmp_path / "out")


def test_a_hidden_character_in_an_address_in_a_file_path_refuses(tmp_path, sources):
    (sources["twiki-eos"] / "jdoe\u200e@example.org.txt").write_bytes(
        b"---+ Page\nPlain text.\n" + PADDING
    )
    with pytest.raises(BuildRefused, match=r"file path .*: path rule"):
        build(load_config(write_config(tmp_path, sources, only=["twiki-eos"])), tmp_path / "out")


def test_a_hidden_character_in_a_word_without_a_separator_is_kept_or_stripped(
    tmp_path, sources
):
    # Not new behaviour (it passes on 6adb358 too): a hidden character away
    # from any address is kept, and a colour code is stripped.
    path = sources["jira"] / "records.json"
    records = json.loads(path.read_text())
    records[0]["description"] = (
        "See \x1b[1mbold\x1b[0m, price\x1b 5, mail jdoe@example.org (at) noon."
    )
    path.write_text(json.dumps(records))
    out = tmp_path / "out"
    lock = build(load_config(write_config(tmp_path, sources, only=["jira"])), out)
    stored = json.loads(read_archive(out / "jira.tar.zst")["data/jira/records.json"])
    assert stored[0]["description"] == "See bold, price\x1b 5, mail  (at) noon."
    assert lock["groups"]["jira"]["ansi_sequences_stripped"] == 2


#: Issue #17: invisible characters the redactor reads, and full-width forms.
INVISIBLE_AND_FULL_WIDTH = [
    "jdoe\u200b[at]example.org",
    "jdoe(\u200bat)example(dot)org",
    "jdoe\uff3bat\uff3dexample.org",
    "jdoe\uff20example.org",
    "jdoe(at)example\uff08dot\uff09org",
]


def test_accented_and_decomposed_addresses_are_still_redacted(tmp_path, sources):
    # NFKC is compared with NFC, so a decomposed accent is not "full width".
    path = sources["jira"] / "records.json"
    records = json.loads(path.read_text())
    records[0]["description"] = "By je\u0301ro\u0302me@example.org and ren\u00e9@example.org."
    path.write_text(json.dumps(records))
    out = tmp_path / "out"
    build(load_config(write_config(tmp_path, sources, only=["jira"])), out)
    stored = json.loads(read_archive(out / "jira.tar.zst")["data/jira/records.json"])
    assert stored[0]["description"] == "By  and ."


# --- eighth review: one rule over the real redactors ----------------------------
#
# Round 8's word rule saw a separator only in the same word as the hidden
# character and knew only some separators. Now every value is read with
# terminal sequences and hidden characters (every Cf character included)
# removed and NFKC applied, and both real redactors match that reading.

#: (The redactor reads ``_at_`` only before some top-level domains, such as
#: ``.edu`` and ``.ch``, so those cases use the reserved example.edu.)
EIGHTH_REVIEW_LEAKS = [
    "jdoe\u200b [at] example.org",
    "jdoe [at] example\u00ad.org",
    "john.doe\u200d_at_example.edu",
    "jdoe\u2060 at example(dot)org",
    "jdoe [at] example\uff0eorg",
    "jdoe\u200b at example(dot)org",
    "jdoe [at] example\u200b.org",
    "john.doe_at_example\u200b.edu",
    "jdoe\u200b_at_example.edu",
    "john.doe\uff3fat\uff3fexample.edu",
    "jdoe\uff05\uff14\uff10example.org",
    "jdoe\u200b\u00a0[at]\u00a0example.org",
    "jdoe\u200b <at> example.org",
    "jdoe\u200b<at>example.org",
    "jdoe\u200b%40example.org",
    "jdoe\u200b&#64;example.org",
    "jdoe&#64;example\u200b&#46;org",
]


#: Everything the redactors match once invisible characters are removed and
#: NFKC applied, and that no control character touches, is removed whole.
REMOVED_WHOLE = INVISIBLE_AND_FULL_WIDTH + EIGHTH_REVIEW_LEAKS


def _letters_left(stored: str) -> set[str]:
    return set("jdoehn") & set(unicodedata.normalize("NFKC", stored))


@pytest.mark.parametrize("value", REMOVED_WHOLE)
def test_a_normalized_address_in_json_is_removed_whole(tmp_path, sources, value):
    path = sources["jira"] / "records.json"
    records = json.loads(path.read_text())
    records[0]["description"] = f"Logged by {value} now."
    path.write_text(json.dumps(records))
    out = tmp_path / "out"
    lock = build(load_config(write_config(tmp_path, sources, only=["jira"])), out)
    stored = json.loads(read_archive(out / "jira.tar.zst")["data/jira/records.json"])
    assert stored[0]["description"] == "Logged by  now."
    assert lock["groups"]["jira"]["normalized_addresses_removed"] >= 1


@pytest.mark.parametrize("value", REMOVED_WHOLE)
def test_a_normalized_address_on_a_text_page_is_removed_whole(tmp_path, sources, value):
    (sources["twiki-eos"] / "Leak.txt").write_bytes(
        f"---+ Page\nAsk {value} now.\n".encode("utf-8") + PADDING
    )
    out = tmp_path / "out"
    lock = build(load_config(write_config(tmp_path, sources, only=["twiki-eos"])), out)
    page = read_archive(out / "twiki-eos.tar.zst")["data/twiki-eos/Leak.txt"].decode("utf-8")
    first = page.split("\n")[1]
    assert first == "Ask  now."
    assert not _letters_left(first.replace("Ask", "").replace("now", ""))
    assert lock["groups"]["twiki-eos"]["normalized_addresses_removed"] >= 1


@pytest.mark.parametrize(
    "value",
    [
        "\x1b[32mjdoe@laptop.example.org\x1b[0m:~$",
        "jdoe\u200b\x7fx@example.org",
        "jdoe\uff20exam\x01ple.org",
    ],
)
def test_a_control_character_still_refuses_next_to_a_normalized_address(
    tmp_path, sources, value
):
    path = sources["jira"] / "records.json"
    records = json.loads(path.read_text())
    records[0]["description"] = f"Logged by {value} now."
    path.write_text(json.dumps(records))
    with pytest.raises(BuildRefused, match=r"span rule"):
        build(load_config(write_config(tmp_path, sources, only=["jira"])), tmp_path / "out")


#: Round 8 refused these; a character NFKC changes that only touches an
#: address, or is in text that is not an address, no longer refuses.
NO_LONGER_REFUSED = [
    ("write jdoe@example.org\u2026 soon", "write \u2026 soon"),
    ("\u90ae\u7bb1\uff1ajdoe@example.org", "\u90ae\u7bb1\uff1a"),
    ("follow @example\u2122", "follow @example\u2122"),
    ("see user@host\u00b2 note", "see user@host\u00b2 note"),
]


@pytest.mark.parametrize(("value", "stored"), NO_LONGER_REFUSED)
def test_a_compatibility_character_outside_an_address_is_stored(
    tmp_path, sources, value, stored
):
    path = sources["jira"] / "records.json"
    records = json.loads(path.read_text())
    records[0]["description"] = value
    path.write_text(json.dumps(records))
    out = tmp_path / "out"
    build(load_config(write_config(tmp_path, sources, only=["jira"])), out)
    result = json.loads(read_archive(out / "jira.tar.zst")["data/jira/records.json"])
    assert result[0]["description"] == stored


def test_an_address_in_a_window_title_is_redacted():
    # Read with only hidden characters removed, the title's text is checked
    # and redacted like any other text. (The ESC ] opener is stripped as a
    # two-character escape.)
    clean, counter = builder_module._clean, builder_module._Counter
    stored = clean("g", "f", "\x1b]0;jdoe@example.org: ~\x07$ ls", counter())
    assert stored == "0;: ~\x07$ ls"
    with pytest.raises(GroupRefused, match="span rule"):
        clean("g", "f", "\x1b]0;jdoe\x7fx@example.org\x07", counter())


@pytest.mark.parametrize("opener", ["\x90", "\x9d", "\x1b]", "\x1bP"])
def test_unterminated_string_sequences_are_checked_in_linear_time(opener):
    # Round 8 scanned to the end of the text from each opener (quadratic).
    text = (opener + "ab ") * 40_000 + "jdoe@example.org"
    started = time.perf_counter()
    builder_module._hidden_address_spans("g", "f", text)
    assert time.perf_counter() - started < 10


# --- ninth review: what a screen shows differs from the bytes -------------------

NINTH_REVIEW_REFUSED = ["jdoe&lrm;@example.org"]
#: Display tricks: a screen can show an address that no reading of the bytes
#: holds (right-to-left override, backspace, cursor save and restore, a lone
#: carriage return). Adversarial-only known limit: no reading has an
#: address, so they are stored (ESC 7 and ESC 8 stripped), not refused.
DISPLAY_TRICKS_STORED = [
    ("\u202egro.elpmaxe@eodj\u202c", "\u202egro.elpmaxe@eodj\u202c"),
    # The backspace is a control character, which the redactor reads as a
    # local part (#14), so "\x08@example.org" goes and "jdoe(" stays.
    ("jdoe(\x08@example.org", "jdoe("),
    ("\x1b7    @example.org\x1b8jdoe", "    @example.orgjdoe"),
    ("    @example.org\rjdoe", "    @example.org\rjdoe"),
    ("XXXXXXXXXXXX.org\rjdoe@example", "XXXXXXXXXXXX.org\rjdoe@example"),
]
#: HTML character references a browser shows as an invisible character or a
#: full-width form, and ideographic full stops: removed whole.
NINTH_REVIEW_REMOVED = [
    "jdoe&#8203;@example.org",
    "jdoe&#x200B;@example.org",
    "jdoe&shy;@example.org",
    "jdoe&#65312;example.org",
    "jdoe&#8203;&#64;example.org",
    "jdoe@example\u3002org",
    "jdoe@example\uff61org",
]


@pytest.mark.parametrize("value", NINTH_REVIEW_REFUSED)
def test_a_ninth_review_display_trick_in_json_refuses(tmp_path, sources, value):
    path = sources["jira"] / "records.json"
    records = json.loads(path.read_text())
    records[0]["description"] = f"Logged by {value} now"
    path.write_text(json.dumps(records))
    with pytest.raises(BuildRefused, match=r"span rule"):
        build(load_config(write_config(tmp_path, sources, only=["jira"])), tmp_path / "out")


@pytest.mark.parametrize("value", NINTH_REVIEW_REFUSED)
def test_a_ninth_review_display_trick_on_a_text_page_refuses(tmp_path, sources, value):
    (sources["twiki-eos"] / "Show.txt").write_bytes(
        f"---+ Page\nAsk {value} now\n".encode("utf-8") + PADDING
    )
    with pytest.raises(BuildRefused, match=r"Show.txt, line 2: span rule"):
        build(load_config(write_config(tmp_path, sources, only=["twiki-eos"])), tmp_path / "out")


@pytest.mark.parametrize(("value", "stored"), DISPLAY_TRICKS_STORED)
def test_a_display_trick_in_json_is_stored_as_a_known_limit(
    tmp_path, sources, value, stored
):
    path = sources["jira"] / "records.json"
    records = json.loads(path.read_text())
    records[0]["description"] = f"Logged by {value} now"
    path.write_text(json.dumps(records))
    out = tmp_path / "out"
    build(load_config(write_config(tmp_path, sources, only=["jira"])), out)
    result = json.loads(read_archive(out / "jira.tar.zst")["data/jira/records.json"])
    assert result[0]["description"] == f"Logged by {stored} now"


@pytest.mark.parametrize(("value", "stored"), DISPLAY_TRICKS_STORED)
def test_a_display_trick_on_a_text_page_is_stored_as_a_known_limit(
    tmp_path, sources, value, stored
):
    (sources["twiki-eos"] / "Show.txt").write_bytes(
        f"---+ Page\nAsk {value} now\n".encode("utf-8") + PADDING
    )
    out = tmp_path / "out"
    build(load_config(write_config(tmp_path, sources, only=["twiki-eos"])), out)
    page = read_archive(out / "twiki-eos.tar.zst")["data/twiki-eos/Show.txt"].decode("utf-8")
    assert page.startswith(f"---+ Page\nAsk {stored} now\n")


@pytest.mark.parametrize("value", NINTH_REVIEW_REMOVED)
def test_a_ninth_review_entity_or_ideographic_address_in_json_is_removed(
    tmp_path, sources, value
):
    path = sources["jira"] / "records.json"
    records = json.loads(path.read_text())
    records[0]["description"] = f"Logged by {value} now."
    path.write_text(json.dumps(records))
    out = tmp_path / "out"
    lock = build(load_config(write_config(tmp_path, sources, only=["jira"])), out)
    stored = json.loads(read_archive(out / "jira.tar.zst")["data/jira/records.json"])
    assert stored[0]["description"] == "Logged by  now."
    assert lock["groups"]["jira"]["normalized_addresses_removed"] >= 1


@pytest.mark.parametrize("value", NINTH_REVIEW_REMOVED)
def test_a_ninth_review_entity_or_ideographic_address_on_a_page_is_removed(
    tmp_path, sources, value
):
    (sources["twiki-eos"] / "Show.txt").write_bytes(
        f"---+ Page\nAsk {value} now.\n".encode("utf-8") + PADDING
    )
    out = tmp_path / "out"
    build(load_config(write_config(tmp_path, sources, only=["twiki-eos"])), out)
    page = read_archive(out / "twiki-eos.tar.zst")["data/twiki-eos/Show.txt"].decode("utf-8")
    assert page.startswith("---+ Page\nAsk  now.\n")


def test_save_and_restore_cursor_escapes_are_stripped(tmp_path, sources):
    path = sources["jira"] / "records.json"
    records = json.loads(path.read_text())
    records[0]["description"] = "\x1b7saved\x1b8 text"
    path.write_text(json.dumps(records))
    out = tmp_path / "out"
    lock = build(load_config(write_config(tmp_path, sources, only=["jira"])), out)
    stored = json.loads(read_archive(out / "jira.tar.zst")["data/jira/records.json"])
    assert stored[0]["description"] == "saved text"
    assert lock["groups"]["jira"]["ansi_sequences_stripped"] == 2


def test_crlf_line_ends_do_not_count_as_a_carriage_return(tmp_path, sources):
    (sources["twiki-eos"] / "Dos.txt").write_bytes(
        b"---+ Page\r\nMail jdoe@example.org now.\r\n" + PADDING
    )
    out = tmp_path / "out"
    build(load_config(write_config(tmp_path, sources, only=["twiki-eos"])), out)
    page = read_archive(out / "twiki-eos.tar.zst")["data/twiki-eos/Dos.txt"].decode("utf-8")
    assert page.startswith("---+ Page\r\nMail  now.\r\n")


def test_a_cr_cr_lf_line_end_is_stored_and_the_address_redacted(tmp_path, sources):
    (sources["twiki-eos"] / "Dos.txt").write_bytes(
        b"---+ Page\r\nMail jdoe@example.org now.\r\r\nnext\n" + PADDING
    )
    out = tmp_path / "out"
    build(load_config(write_config(tmp_path, sources, only=["twiki-eos"])), out)
    page = read_archive(out / "twiki-eos.tar.zst")["data/twiki-eos/Dos.txt"].decode("utf-8")
    assert page.startswith("---+ Page\r\nMail  now.\r\r\nnext\n")


def test_a_lone_surrogate_in_json_refuses_cleanly(tmp_path, sources):
    path = sources["jira"] / "records.json"
    text = path.read_text()
    records = json.loads(text)
    records[0]["description"] = "SURROGATE"
    path.write_text(json.dumps(records).replace("SURROGATE", "bad \\ud800 text"))
    with pytest.raises(BuildRefused, match=r"records.json \[0\]\.description, line 1: surrogate rule"):
        build(load_config(write_config(tmp_path, sources, only=["jira"])), tmp_path / "out")


# --- tenth review: backslash-escaped @, the separator list, messages ------------

#: Perl, Doxygen and shell text escape @ as \@ (TWiki is written in Perl).
BACKSLASH_ESCAPED = [
    ('my $to = "jdoe\\@example.org";', 'my $to = "";'),
    ("\\author J. Doe (jdoe\\@example.org)", "\\author J. Doe ()"),
    ("mail jdoe\\@example.org now", "mail  now"),
    ("/^jdoe\\@example\\.org$/", "/$/"),
]


@pytest.mark.parametrize(("value", "stored"), BACKSLASH_ESCAPED)
def test_a_backslash_escaped_address_in_json_is_removed(tmp_path, sources, value, stored):
    path = sources["jira"] / "records.json"
    records = json.loads(path.read_text())
    records[0]["description"] = value
    path.write_text(json.dumps(records))
    out = tmp_path / "out"
    lock = build(load_config(write_config(tmp_path, sources, only=["jira"])), out)
    result = json.loads(read_archive(out / "jira.tar.zst")["data/jira/records.json"])
    assert result[0]["description"] == stored
    assert "jdoe" not in result[0]["description"]
    assert lock["groups"]["jira"]["addresses_removed"] >= 2  # this one and the planted one


@pytest.mark.parametrize(("value", "stored"), BACKSLASH_ESCAPED)
def test_a_backslash_escaped_address_on_a_text_page_is_removed(
    tmp_path, sources, value, stored
):
    (sources["twiki-eos"] / "Perl.txt").write_bytes(
        f"---+ Page\n{value}\n".encode("utf-8") + PADDING
    )
    out = tmp_path / "out"
    build(load_config(write_config(tmp_path, sources, only=["twiki-eos"])), out)
    page = read_archive(out / "twiki-eos.tar.zst")["data/twiki-eos/Perl.txt"].decode("utf-8")
    assert page.startswith(f"---+ Page\n{stored}\n")


def test_a_coloured_line_with_a_bare_hash_number_is_stored(tmp_path, sources):
    # "#642" is not "&#64;" (that needs its "&"): no address, so the colour
    # codes are stripped and the text is stored.
    path = sources["jira"] / "records.json"
    records = json.loads(path.read_text())
    records[0]["description"] = "\x1b[32mfixed\x1b[0m in PR #642, see commat; and #x40"
    path.write_text(json.dumps(records))
    out = tmp_path / "out"
    lock = build(load_config(write_config(tmp_path, sources, only=["jira"])), out)
    result = json.loads(read_archive(out / "jira.tar.zst")["data/jira/records.json"])
    assert result[0]["description"] == "fixed in PR #642, see commat; and #x40"
    assert lock["groups"]["jira"]["ansi_sequences_stripped"] == 2


def test_a_refusal_names_the_file_line_and_rule(tmp_path, sources):
    (sources["twiki-eos"] / "Show.txt").write_bytes(
        b"---+ Page\nfirst line\nAsk jdoe\x7fx@example.org now\n" + PADDING
    )
    with pytest.raises(BuildRefused) as caught:
        build(load_config(write_config(tmp_path, sources, only=["twiki-eos"])), tmp_path / "out")
    message = str(caught.value)
    assert "Show.txt, line 3: span rule: an address contains or touches a DEL" in message
    assert "an address is split" not in message
    assert "jdoe" not in message


def test_a_json_refusal_names_the_record_and_field(tmp_path, sources):
    path = sources["jira"] / "records.json"
    records = json.loads(path.read_text())
    records[0]["description"] = "one\ntwo jdoe\x7fx@example.org"
    path.write_text(json.dumps(records))
    with pytest.raises(BuildRefused) as caught:
        build(load_config(write_config(tmp_path, sources, only=["jira"])), tmp_path / "out")
    assert "records.json [0].description, line 2: span rule" in str(caught.value)


# --- wmstats-collection-shape ---------------------------------------------
# The Aug 31 collector writes records.json as {cutoff_utc, status_payloads:
# {status: {"result": [{name: record}]}}} next to an authority.json with its
# record_count. Fake data in that structure.


def _collection(sources, *, count=None, extra_status=None):
    """Rewrite the wmstats fixture into the collection shape."""
    path = sources["wmstats"] / "records.json"
    (record,) = json.loads(path.read_text())
    second = {
        **record,
        "RequestName": "fakeuser_ReReco_Run2026A_FakePD_000002",
        "RequestStatus": "assigned",
    }
    payload = {
        "cutoff_utc": "2026-01-01T00:00:00Z",
        "status_payloads": {
            "new": {"result": []},
            "assigned": {"result": [{second["RequestName"]: second}]},
            "running-open": {"result": [{record["RequestName"]: record}]},
            **(extra_status or {}),
        },
    }
    path.write_text(json.dumps(payload))
    if count is not None:
        (sources["wmstats"] / "authority.json").write_text(
            json.dumps({"record_count": count, "complete": True, "scope": "fake"})
        )


def test_wmstats_collection_shape_builds_and_checks_the_authority_count(tmp_path, sources):
    _collection(sources, count=2)
    out = tmp_path / "out"
    lock = build(load_config(write_config(tmp_path, sources, only=["wmstats"])), out)
    assert lock["groups"]["wmstats"]["record_count"] == 2
    members = read_archive(out / "wmstats.tar.zst")
    assert json.loads(members["data/wmstats-workflows/authority.json"]) == {"record_count": 2}
    records = json.loads(members["data/wmstats-workflows/records.json"])
    assert set(records) == {"status_payloads"}  # cutoff_utc is not read
    (entry,) = records["status_payloads"]["running-open"]["result"]
    (record,) = entry.values()
    assert record["Requestor"] == "pdmvserv"
    assert "RequestorDN" not in record and "RequestTransition" not in record
    assert records["status_payloads"]["new"] == {"result": []}
    data = _all_bytes(out, "wmstats")
    assert DN.encode() not in data and PLANTED["wmstats"].encode() not in data
    assert verify(out / LOCK_NAME, out).ok


@pytest.mark.parametrize(
    "case, expected",
    [
        ("count-mismatch", "counts 3"),
        ("no-workflow", "no workflow under status_payloads"),
        ("malformed-record", "status 'failed' result 0 entry 'x' is malformed"),
        ("no-result", "status 'failed' has no result list"),
    ],
)
def test_wmstats_collection_shape_refuses_a_broken_export(tmp_path, sources, case, expected):
    if case == "count-mismatch":
        _collection(sources, count=3)
    elif case == "no-workflow":
        (sources["wmstats"] / "records.json").write_text(
            json.dumps({"cutoff_utc": "2026-01-01T00:00:00Z", "status_payloads": {}})
        )
    elif case == "malformed-record":
        _collection(sources, extra_status={"failed": {"result": [{"x": {"Campaign": "x"}}]}})
    else:
        _collection(sources, extra_status={"failed": {"error": "timeout"}})
    with pytest.raises(BuildRefused) as info:
        build(load_config(write_config(tmp_path, sources, only=["wmstats"])), tmp_path / "out")
    assert [r.group for r in info.value.refusals] == ["wmstats"]
    assert expected in info.value.refusals[0].reason
    assert not (tmp_path / "out").exists()


def test_cmssw_group_builds_with_production_architecture_in_records_json(
    tmp_path, sources
):
    group = sources["cmssw-releases"]
    (group / "releases.map").unlink()
    _write_json(
        group / "records.json",
        [
            {"label": "CMSSW_14_0_1", "type": "Production", "state": "Announced",
             "architecture": ["el8_amd64_gcc12"],
             "production_architecture": "el8_amd64_gcc12"},
        ],
    )
    (group / "meta.json").write_text('{"record_count": 1}')
    out = tmp_path / "out"
    lock = build(
        load_config(write_config(tmp_path, sources, only=["cmssw-releases"])), out
    )
    assert lock["groups"]["cmssw-releases"]["record_count"] == 1
    members = read_archive(out / "cmssw-releases.tar.zst")
    records = json.loads(members["data/cmssw-releases/records.json"])
    assert records[0]["production_architecture"] == "el8_amd64_gcc12"
    assert verify(out / LOCK_NAME, out).ok
