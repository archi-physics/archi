"""Helpers shared by the pack tests (imported by module name).

`FIXTURES` gives every shipped context pack the inputs it reads, so the pack
can be installed alone and publish (acc.r3-packs-split.matrix). A fixture is
a function `(data_root, server) -> answers`: it writes the caches the pack's
sources read under `data_root`, registers any HTTP responses a live source
fetches on `server`, and returns the pack's own init-question answers by
their local ids.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlsplit

import yaml

REPO = Path(__file__).resolve().parents[3]
PACKS_DIR = REPO / "bundles" / "packs"
BUNDLES_DIR = REPO / "bundles"
SKILLS_DIR = REPO / "skills"
TEST_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def shipped_catalog():
    """The packs and bundles this repository ships, with no problems."""
    from archi.bundles import discovery

    catalog = discovery.from_roots([PACKS_DIR], [BUNDLES_DIR])
    assert catalog.problems == [], [str(p) for p in catalog.problems]
    return catalog


def solo_catalog(tmp_path: Path, pack: str):
    """The shipped packs plus a bundle `solo-<pack>` whose only default is `pack`.

    The bundle is added without `validate_bundle`: a pack that needs a
    credential may not be a real bundle's only default, but alone on the base
    is exactly what the isolation tests install.
    """
    from archi.bundles import discovery
    from archi.bundles.manifest import read_bundle

    bdir = tmp_path / "bundles" / f"solo-{pack}"
    bdir.mkdir(parents=True)
    (bdir / "bundle.yaml").write_text(
        yaml.safe_dump(
            {
                "name": f"solo-{pack}",
                "version": "0.1.0",
                "description": f"{pack} alone on the base.",
                "packs": {"default": [pack]},
                "okg": ">=0.2.0,<0.3",
            }
        )
    )
    catalog = discovery.from_roots([PACKS_DIR], [])
    assert catalog.problems == [], [str(p) for p in catalog.problems]
    bundle, problems = read_bundle(bdir)
    assert problems == [], problems
    catalog.bundles[bundle.name] = bundle
    return catalog


def render_alone(tmp_path: Path, pack: str):
    """Render `pack` alone into `tmp_path/instance`; return the `Rendered`."""
    from archi.bundles.render import render
    from archi.bundles.resolve import resolve

    catalog = solo_catalog(tmp_path, pack)
    return render(resolve(catalog, [f"solo-{pack}"]), tmp_path / "instance", name=f"solo-{pack}")


def okg_plan(rendered, tmp_path: Path, answers: dict):
    """okg's own install plan for a rendered profile (no database touched)."""
    from okg.substrate.deployment_bootstrap.profile_init import build_plan, discover_profile

    answers = {"deployment_name": "qt", "postgres_dsn": "postgresql://localhost/qt", **answers}
    return build_plan(
        profile=discover_profile(rendered.name, profiles_root=rendered.profile_dir.parent),
        deployment_name="qt",
        deployments_root=tmp_path / "deployments",
        raw_answers=answers,
    )


def pack_dirs() -> list[Path]:
    return sorted(p.parent for p in PACKS_DIR.glob("*/pack.yaml"))


def manifest(pack: str) -> dict:
    return yaml.safe_load((PACKS_DIR / pack / "pack.yaml").read_text(encoding="utf-8"))


def context_packs() -> list[str]:
    """Packs that bring sources. (ops-record is a context pack with none: its
    records arrive through okg admission, so it has nothing to ingest alone.)"""
    return [
        p.name for p in pack_dirs()
        if manifest(p.name)["kind"] == "context" and "sources" in manifest(p.name)
    ]


def credential_names() -> set[str]:
    """Every credential variable a shipped pack reads, aliases included."""
    names: set[str] = set()
    for pack in context_packs():
        names |= set(manifest(pack).get("credentials", []))
        for entry in templates(pack).values():
            names |= set(entry.get("credential_refs") or [])
            for aliases in (entry.get("credential_aliases") or {}).values():
                names |= set(aliases)
            names |= set((entry.get("params") or {}).get("aliases") or [])
    return names | {"GITLAB_CERN_TOKEN", "ARCHI_INDICO_COOKIES"}


def templates(pack: str) -> dict[str, dict]:
    """source name -> registry entry, for one pack."""
    out = {}
    for path in sorted((PACKS_DIR / pack / "sources").glob("*.yaml")):
        body = yaml.safe_load(path.read_text(encoding="utf-8"))
        out[path.stem] = body[path.stem]
    return out


def all_templates() -> list[tuple[str, str, Path, dict]]:
    """(pack, source, file, entry) for every source every pack ships."""
    out = []
    for pack in context_packs():
        for source, entry in templates(pack).items():
            out.append((pack, source, PACKS_DIR / pack / "sources" / f"{source}.yaml", entry))
    return out


def invariants(pack: str) -> list[dict]:
    path = PACKS_DIR / pack / "invariants.yaml"
    if not path.is_file():
        return []
    return yaml.safe_load(path.read_text(encoding="utf-8"))["invariants"]


_REQUIRED = re.compile(r"required\(subtype\)\s+AS\s+\(\s*VALUES\s+(.*?)\n\s*\)\s*,", re.S)


def required_subtypes(invariant: dict) -> list[str]:
    """The subtypes a coverage invariant requires (its `required` VALUES list)."""
    match = _REQUIRED.search(invariant["sql"])
    if match is None:
        return []
    return re.findall(r"'([a-z_]+)'", match.group(1))


def emitted_subtypes(pack: str) -> set[str]:
    out = set()
    for entry in templates(pack).values():
        for node in entry["admission_policy"]["output_signature"]["nodes"]:
            out.add(node["subtype"])
    return out


# ---------------------------------------------------------------------------
# A local HTTP server for the sources that fetch live
# ---------------------------------------------------------------------------


class FixtureServer:
    """Serve registered responses on 127.0.0.1 so a source run by okg in a
    subprocess fetches fixtures instead of the network.

    `add(path, body)` answers GET <path> (any query) with `body`;
    `add(path, body, query={"k": "v"})` answers only when the query carries
    those values, and is tried before the query-free route.
    """

    def __init__(self) -> None:
        self.routes: list[tuple[str, dict[str, str], bytes, str]] = []
        self.requests: list[str] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                server.requests.append(self.path)
                parts = urlsplit(self.path)
                query = {k: v[0] for k, v in parse_qs(parts.query).items()}
                for path, want, body, ctype in sorted(
                    server.routes, key=lambda r: -len(r[1])
                ):
                    if path == parts.path and all(query.get(k) == v for k, v in want.items()):
                        self.send_response(200)
                        self.send_header("Content-Type", ctype)
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                        return
                self.send_response(404)
                self.end_headers()

            def log_message(self, *args):
                pass

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._httpd.server_address[1]}"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def add(self, path: str, body, *, query: dict | None = None, ctype: str = "") -> None:
        if not isinstance(body, (bytes, str)):
            body = json.dumps(body)
            ctype = ctype or "application/json"
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.routes.append((path, dict(query or {}), body, ctype or "text/plain"))

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


# ---------------------------------------------------------------------------
# Fixture inputs, one function per context pack
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _docsite(data: Path, server: FixtureServer) -> dict:
    _write_json(
        data / "docsite" / "records.json",
        [
            {
                "title": "Getting started",
                "url": "https://docs.example.org/start.html",
                "body": "Install the client, then run the setup script. " * 20,
                "site_name": "example-docs",
                "source_repo": "example/docs",
                "path": "start.md",
            }
        ],
    )
    return {}


def _indico(data: Path, server: FixtureServer) -> dict:
    event = (TEST_FIXTURES / "indico" / "event_1716863_contributions.json").read_bytes()
    server.add("/export/event/1716863.json", event, ctype="application/json")
    return {"base_url": server.url, "event_ids": '["1716863"]'}


def _twiki(data: Path, server: FixtureServer) -> dict:
    root = data / "twiki-snapshot"
    (root / "Sub").mkdir(parents=True)
    (root / "TopicOne.txt").write_text(
        '%META:TOPICINFO{author="jdoe" date="1700000000" version="12"}%\n'
        '%META:TOPICPARENT{name="TopicTwo"}%\n'
        "---++ Topic One\nSee [[TopicTwo]] for the transfer procedure.\n",
        encoding="utf-8",
    )
    (root / "TopicTwo.txt").write_text(
        "---+ Topic Two\nBody two with [[Sub.SubTopicPage]].\n", encoding="utf-8"
    )
    (root / "Sub" / "SubTopicPage.txt").write_text(
        "---+ Sub Topic\nGrandchild body.\n", encoding="utf-8"
    )
    return {"eos_root": str(root)}


def _cds(data: Path, server: FixtureServer) -> dict:
    page1 = (TEST_FIXTURES / "cds_page1.xml").read_bytes()
    page2 = (TEST_FIXTURES / "cds_page2.xml").read_bytes()
    server.add("/oai2d", page1, ctype="text/xml")
    server.add("/oai2d", page2, query={"resumptionToken": "TOKEN-PAGE-2"}, ctype="text/xml")
    return {"base_url": server.url + "/oai2d", "sets": '["cerncds:cms-pas"]'}


def _jira(data: Path, server: FixtureServer) -> dict:
    records = [
        {
            "key": "DEMO-1",
            "summary": "Transfer failures at the test site",
            "description": "Stuck transfers since the upgrade; see DEMO-2.",
            "project": "DEMO",
            "status": "Open",
            "priority": "Critical",
            "issue_type": "Bug",
            "assignee": "Ada Lovelace",
            "reporter": "Grace Hopper",
            "created": "2026-01-05T10:00:00.000+0000",
            "updated": "2026-02-01T09:30:00.000+0000",
        },
        {"key": "DEMO-2", "summary": "Upgrade the storage servers", "project": "DEMO"},
    ]
    _write_json(data / "jira" / "records.json", records)
    _write_json(data / "jira" / "meta.json", {"record_count": len(records)})
    return {}


def _analysis_notes(data: Path, server: FixtureServer) -> dict:
    from packs.physics_fixtures import write_note_mirror

    return {"mirror_root": str(write_note_mirror(data / "analysis-notes"))}


#: An invented password, assembled from pieces so no complete secret-shaped
#: string sits in the repository. git-code's fixture plants it in a file.
PLANTED_SECRET = "Pk7w" + "Rq2Zt9Lx"
PLANTED_SECRET_PATH = "deploy/db-values.yaml"


def _git_code(data: Path, server: FixtureServer) -> dict:
    upstream = data.parent / "upstream" / "demo"
    if not (upstream / ".git").is_dir():
        (upstream / "pkg").mkdir(parents=True)
        (upstream / "pkg" / "transfer.py").write_text(
            '"""Transfer helpers."""\n\n\n'
            "def retry(job, attempts=3):\n"
            '    """Retry a job a few times."""\n'
            "    for _ in range(attempts):\n"
            "        if submit(job):\n"
            "            return True\n"
            "    return False\n\n\n"
            "def submit(job):\n"
            "    return bool(job)\n",
            encoding="utf-8",
        )
        (upstream / "README.md").write_text("# demo\n\nA test repository.\n", encoding="utf-8")
        # A planted, invented secret: code-text's screen must keep this file's
        # text out of the graph (test_pack_isolation checks after publish).
        (upstream / "deploy").mkdir()
        (upstream / "deploy" / "db-values.yaml").write_text(
            "db:\n  user: demo\n  password: " + PLANTED_SECRET + "\n", encoding="utf-8"
        )
        git(["init", "-q", "-b", "main"], upstream)
        git(["add", "-A"], upstream)
        git(["commit", "-q", "-m", "first commit"], upstream)
        git(["tag", "v0.1.0"], upstream)
    return {"repo_name": "demo", "repo_url": str(upstream)}


def _cmssw_releases(data: Path, server: FixtureServer) -> dict:
    server.add("/releases.map", (TEST_FIXTURES / "cmssw" / "releases_15_0.map").read_bytes())
    return {"releases_map_url": server.url + "/releases.map"}


def _compops_cache(source: str, key: str):
    """The small cache test_bundle_source_adapters runs each reader on."""
    from test_bundle_source_adapters import _COMPOPS_CACHES

    return _COMPOPS_CACHES[source][key]


def _cms_catalogs(data: Path, server: FixtureServer) -> dict:
    files = {
        "cric/sites.json": ("cric", "sites_path"),
        "cric/storage_units.json": ("cric", "storage_units_path"),
        "cric/compute_units.json": ("cric", "compute_units_path"),
        "cric/facilities.json": ("cric", "facilities_path"),
        "cric/responsibilities.json": ("cric", "responsibilities_path"),
        "cric-core/services.json": ("cric_core", "services_path"),
        "cric-core/rcsites.json": ("cric_core", "rcsites_path"),
        "cric-core/federations.json": ("cric_core", "federations_path"),
        "dbs-datasets/records.json": ("dbs_datasets", "records_path"),
        "wmstats-workflows/records.json": ("wmstats_workflows", "records_path"),
        "gocdb-downtimes/records.json": ("gocdb_downtimes", "records_path"),
        "conddb-global-tags/records.json": ("conddb_global_tags", "records_path"),
        "cmssw-releases/records.json": ("conddb_global_tags", "cmssw_records_path"),
    }
    for rel, (source, key) in files.items():
        _write_json(data / rel, _compops_cache(source, key))
    # CRIC core links services and federations to their CRIC sites, and okg
    # refuses a link to a site no source emits: add the site the cric_core
    # cache names (T0_CH_CERN) to the cric one.
    sites = dict(_compops_cache("cric", "sites_path"))
    sites["T0_CH_CERN"] = {"tier_level": 0, "facility": "CERN", "sitedb_title": "CERN"}
    _write_json(data / "cric" / "sites.json", sites)
    # SITECONF reads GitLab live: one site project with a storage.json.
    server.add("/api/v4/user", {"username": "pack-test"})
    # The preflight lists without a page; the run pages until an empty one.
    server.add("/api/v4/groups/4099/projects", [{"id": 11, "path": "T2_US_MIT"}])
    server.add("/api/v4/groups/4099/projects", [], query={"page": "2"})
    server.add(
        "/api/v4/projects/11/repository/files/storage.json/raw",
        json.dumps([{"protocol": "XRootD", "se": "xrootd.mit.example.org"}]),
    )
    return {"gitlab_url": server.url}


def _cmssst(data: Path, server: FixtureServer) -> dict:
    sites = [
        json.loads((TEST_FIXTURES / "cmssst" / f"{site}.json").read_text(encoding="utf-8"))
        for site in ("T1_US_FNAL", "T2_US_MIT", "T2_CH_CERN")
    ]
    _write_json(data / "cmssst-site-status" / "records.json", sites)
    return {}


def _dqm(data: Path, server: FixtureServer) -> dict:
    _write_json(data / "dqm" / "records.json", _compops_cache("dqm", "records_path"))
    return {}


def _hypernews(data: Path, server: FixtureServer) -> dict:
    from sources.test_hypernews import RECORDS

    _write_json(data / "hypernews" / "records.json", RECORDS)
    return {}


def _monit_live(data: Path, server: FixtureServer) -> dict:
    from sources import test_monit

    _write_json(data / "monit-sam" / "records.json", test_monit.SITEMON_RESPONSE)
    _write_json(data / "monit-condor" / "records.json", test_monit.CONDOR_RESPONSE)
    _write_json(data / "monit-rucio-transfer" / "records.json", test_monit.TRANSFER_RESPONSE)
    _write_json(data / "monit-rucio-datasets" / "records.json", test_monit.DATASET_RESPONSE)
    # Every MONIT edge starts at a CRIC site or storage endpoint, and okg
    # refuses a source whose link names a node no source emits. Add the
    # sites and endpoints the MONIT fixtures name to cms-catalogs' caches
    # (written first: the closure runs cms-catalogs' fixture before this).
    sites_path = data / "cric" / "sites.json"
    sites = json.loads(sites_path.read_text(encoding="utf-8"))
    for site in ("T1_DE_KIT", "T3_US_Baylor"):
        sites.setdefault(site, {"tier_level": int(site[1]), "facility": site.split("_")[2]})
    _write_json(sites_path, sites)
    units_path = data / "cric" / "storage_units.json"
    units = json.loads(units_path.read_text(encoding="utf-8"))
    units.setdefault("T1_DE_KIT_Disk", {"type": "DISK", "site": {"name": "T1_DE_KIT"}})
    units.setdefault("T2_US_MIT", {"type": "DISK", "site": {"name": "T2_US_MIT"}})
    _write_json(units_path, units)
    return {}


def _servicenow(data: Path, server: FixtureServer) -> dict:
    from packs.cluster_fixtures import write_servicenow_cache

    write_servicenow_cache(data)
    return {"instance_url": "https://example.service-now.com"}


def _tickets(data: Path, server: FixtureServer) -> dict:
    from packs.cluster_fixtures import write_tickets_cache

    write_tickets_cache(data)
    return {"base_url": "https://redmine.example.org"}


def _slurm(data: Path, server: FixtureServer) -> dict:
    from packs.cluster_fixtures import write_slurm_cache

    # The cache the exporter writes from recorded command output (a fake runner; no Slurm command runs).
    write_slurm_cache(data)
    return {"cluster_name": "submit"}


def _htcondor(data: Path, server: FixtureServer) -> dict:
    from packs.cluster_fixtures import write_condor_cache

    write_condor_cache(data)
    return {"pool_name": "submit-pool"}


def _node_health(data: Path, server: FixtureServer) -> dict:
    from packs.cluster_fixtures import write_node_health_cache

    write_node_health_cache(data)
    return {"cluster_name": "submit"}


FIXTURES: dict[str, Callable[[Path, FixtureServer], dict]] = {
    "docsite": _docsite,
    "indico": _indico,
    "twiki": _twiki,
    "cds": _cds,
    "analysis-notes": _analysis_notes,
    # Its inputs are twiki's dump and the notes mirror, written by the fixtures
    # of the packs it requires; the cross-link rules read what those publish.
    "cms-analyses": lambda data, server: {},
    "jira": _jira,
    "git-code": _git_code,
    # Both read git-code's checkout; its fixture (run for the closure) is all
    # they need.
    "code-text": lambda data, server: {},
    "code-symbols": lambda data, server: {},
    "cmssw-releases": _cmssw_releases,
    "cms-catalogs": _cms_catalogs,
    "cmssst": _cmssst,
    "dqm": _dqm,
    "hypernews": _hypernews,
    "monit-live": _monit_live,
    "servicenow": _servicenow,
    "tickets": _tickets,
    "slurm": _slurm,
    "htcondor": _htcondor,
    "node-health": _node_health,
}

#: Credential variables a pack's sources read, set to dummy values so a
#: fixture run gets past the credential check. Never real values.
FIXTURE_ENV: dict[str, dict[str, str]] = {
    "cms-catalogs": {"CERN_GITLAB_TOKEN": "fixture-token"},
    "monit-live": {"MONIT_GRAFANA_TOKEN": "fixture-token"},
}


# ---------------------------------------------------------------------------
# Running okg
# ---------------------------------------------------------------------------


def okg_bin() -> str:
    import sys

    return str(Path(sys.executable).with_name("okg"))


def run_okg(args: list[str], *, cwd: Path, env: dict, timeout: int = 900):
    return subprocess.run(
        [okg_bin(), *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout
    )


def git(args: list[str], cwd: Path) -> None:
    subprocess.run(
        ["git", "-c", "user.name=pack-test", "-c", "user.email=pack-test@example.org", *args],
        cwd=cwd, check=True, capture_output=True,
    )


def copy_tree(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, symlinks=False)
