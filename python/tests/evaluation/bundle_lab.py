"""Install a bundle on its synthetic fixture corpus and publish it.

The lab setup behind `bundles/<bundle>/eval/smoke.yaml` for the bundles whose
smoke set runs on fixtures (cms-physics, cluster-manager): every
smoke question is answerable from what this script feeds the bundle. Nothing
here is real.

- cms-physics: the analysis notes, TWiki pages and CDS records are the
  invented ones in `packs/physics_fixtures.py` (note ids in the 2099 and 99
  year ranges).
- cluster-manager: documentation pages, Redmine tickets and ops records about
  an invented cluster (`fixtures/cluster/`). None of it is CERN- or
  CMS-specific, because the bundle is for any cluster operator.

Every other pack gets the input the pack-isolation tests give it. Run it on a
host with the AKMON Postgres image (the isolation tests' setup), then serve
MCP over the result:

    python bundle_lab.py --bundle cluster-manager --dsn postgresql://postgres:okg@127.0.0.1:55640/evals --work /path/to/scratch

It recreates the database named in the DSN, installs the bundle with okg,
ingests from a local fixture server, publishes, admits the bundle's ops
records (when it has the ops-record pack) and publishes again, sets the read
role's password, and writes `<work>/serve.sh`, which starts `okg mcp-serve`
on a port with a bearer token. `archi eval --bundle <bundle> --mcp-url
http://127.0.0.1:<port>/mcp` then runs the smoke set against it.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import shlex
import subprocess
import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TESTS))

from packs import pack_support as ps  # noqa: E402
from packs.cluster_fixtures import CLUSTER  # noqa: E402
from packs.physics_fixtures import (  # noqa: E402
    synthetic_cds_records,
    write_twiki_dump,
)

BUNDLES = ("cms-physics", "cluster-manager")
DEPLOYMENT = "px"

#: The ops records each bundle's lab admits (JSONL, one record per line).
OPS_RECORDS = {
    "cluster-manager": CLUSTER / "ops_records.jsonl",
}

_OAI = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><responseDate>2099-01-02T00:00:00Z</responseDate>'
    '<request verb="ListRecords" set="cerncds:cms-pas" metadataPrefix="oai_dc">http://cds.cern.ch/oai2d</request>'
    "<ListRecords>{records}</ListRecords></OAI-PMH>\n"
)
_RECORD = (
    "<record><header><identifier>{oai}</identifier><datestamp>{datestamp}</datestamp>"
    "<setSpec>cerncds:cms-pas</setSpec></header><metadata>"
    '<oai_dc:dc xmlns:dc="http://purl.org/dc/elements/1.1/" '
    'xmlns:oai_dc="http://www.openarchives.org/OAI/2.0/oai_dc/">'
    "<dc:identifier>{url}</dc:identifier><dc:creator>{creator}</dc:creator><dc:title>{title}</dc:title>"
    "<dc:subject>{subject}</dc:subject><dc:description>{description}</dc:description>"
    "<dc:identifier>{report}</dc:identifier><dc:identifier>{oai}</dc:identifier>"
    "<dc:date>{date}</dc:date></oai_dc:dc></metadata></record>"
)


def synthetic_oai_page() -> bytes:
    """The two invented CDS records (`synthetic_cds_records`) as one OAI-PMH page."""
    rows = []
    for record in synthetic_cds_records():
        rows.append(
            _RECORD.format(
                oai=record.oai_id, datestamp=record.datestamp, url=record.url, creator=record.creators[0],
                title=record.title, subject=record.subjects[0], description=record.description,
                report=record.report_numbers[0].raw, date=record.date,
            )
        )
    return _OAI.format(records="".join(rows)).encode("utf-8")


def _cluster_docsite(data: Path) -> dict:
    """The invented cluster's documentation pages, in place of the isolation fixture's one page."""
    out = data / "docsite" / "records.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text((CLUSTER / "docsite_records.json").read_text(encoding="utf-8"), encoding="utf-8")
    return {}


def fixture_answers(bundle: str, pack_names, data: Path, server) -> dict:
    """The install answers, from the isolation fixtures with the bundle's own inputs swapped in."""
    answers = {}
    for pack in pack_names:
        if bundle == "cms-physics" and pack == "twiki":
            local = {"eos_root": str(write_twiki_dump(data / "twiki-snapshot"))}
        elif bundle == "cms-physics" and pack == "cds":
            server.add("/oai2d", synthetic_oai_page(), ctype="text/xml")
            local = {"base_url": server.url + "/oai2d", "sets": '["cerncds:cms-pas"]'}
        elif bundle == "cluster-manager" and pack == "docsite":
            local = _cluster_docsite(data)
        else:
            local = ps.FIXTURES.get(pack, lambda d, s: {})(data, server)
        for key, value in local.items():
            answers[pack.replace("-", "_") + "_" + key] = value
    if bundle == "cms-physics":
        # The bundle fixes a note floor calibrated on a real mirror build (2500);
        # the fixture mirror holds two invented notes.
        answers["analysis_notes_min_notes"] = 1
    return answers


def _flag(qid: str) -> str:
    return "--" + qid.replace("_", "-")


def _must(result, what: str) -> None:
    if result.returncode != 0:
        raise SystemExit(f"{what} failed (exit {result.returncode})\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}")


def build(bundle: str, dsn: str, work: Path, port: int) -> dict:
    from archi.bundles.render import render
    from archi.bundles.resolve import resolve

    import psycopg
    from psycopg import sql

    work.mkdir(parents=True, exist_ok=True)
    parts = psycopg.conninfo.conninfo_to_dict(dsn)
    admin = psycopg.conninfo.make_conninfo(dsn, dbname="postgres")
    with psycopg.connect(admin, autocommit=True) as conn:
        conn.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(parts["dbname"])))
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(parts["dbname"])))

    instance = work / "instance"
    deployments = instance / "deployments"
    data = work / "data"
    data.mkdir(exist_ok=True)
    deployments.mkdir(parents=True, exist_ok=True)
    ps.git(["init", "-q"], deployments)
    ps.git(["commit", "-q", "--allow-empty", "-m", "baseline"], deployments)

    catalog = ps.shipped_catalog()
    resolution = resolve(catalog, [bundle])
    rendered = render(resolution, instance, name=bundle)
    server = ps.FixtureServer()
    try:
        answers = {"postgres_dsn": dsn, "archi_data_root": str(data)}
        answers.update(fixture_answers(bundle, resolution.pack_names, data, server))
        known = {q["id"] for q in rendered.composition.questions}
        unknown = sorted(set(answers) - known)
        if unknown:
            raise SystemExit(f"answers the bundle does not ask: {unknown}")
        env = {k: v for k, v in os.environ.items() if k != "OKG_PROFILES_DIR" and k not in ps.credential_names()}
        env.update(
            HOME=str(work), OKG_DEPLOYMENTS_DIR=str(deployments), OKG_DSN=dsn, ARCHI_DATA_ROOT=str(data),
            OKG_ROLE_SECRETS_FILE=str(work / "role-secrets.json"),
            GIT_AUTHOR_NAME="evals", GIT_AUTHOR_EMAIL="evals@example.org",
            GIT_COMMITTER_NAME="evals", GIT_COMMITTER_EMAIL="evals@example.org",
        )
        for pack in resolution.pack_names:
            env.update(ps.FIXTURE_ENV.get(pack, {}))
        flags = [x for qid, value in answers.items() for x in (_flag(qid), str(value))]

        def okg(*args: str):
            return ps.run_okg(list(args), cwd=instance, env=env)

        install = okg("install", "--profile", rendered.name, "--deployment-name", DEPLOYMENT,
                      "--deployments-root", str(deployments), "--non-interactive", "--json", *flags)
        # The first install stops before publishing (okg will not publish from
        # configuration that is not committed yet); it is done when it wrote the registry.
        if not (deployments / DEPLOYMENT / "source_registry.yaml").is_file():
            _must(install, "okg install")
        ps.git(["add", "-A"], deployments)
        ps.git(["commit", "-q", "-m", "px deployment config"], deployments)
        _must(okg("ingest", "--deployment", DEPLOYMENT, "--json"), "okg ingest")
        _must(okg("run", "--once", "--deployment", DEPLOYMENT, "--apply", "--json"), "okg run (publish)")
        records = OPS_RECORDS.get(bundle)
        if records and "ops-record" in resolution.pack_names:
            # Admitted ops records join the graph in the next published generation.
            submit = subprocess.run(
                [sys.executable, "-m", "archi.ops.record", "submit", "--deployment", DEPLOYMENT,
                 "--records", str(records)],
                cwd=instance, env=env, capture_output=True, text=True, timeout=900,
            )
            _must(submit, "archi.ops.record submit")
            _must(okg("run", "--once", "--deployment", DEPLOYMENT, "--apply", "--json"), "okg run (publish ops records)")
        status = okg("status", "--deployment", DEPLOYMENT, "--json")
        _must(status, "okg status")
        published = json.loads(status.stdout)["publishing"]["latest_published_status"]
        if published != "published":
            raise SystemExit(f"the generation is {published!r}, not published")
        _must(okg("db", "role-passwords", "--deployment", DEPLOYMENT, "--apply"), "okg db role-passwords")
    finally:
        server.close()

    roles = json.loads((work / "role-secrets.json").read_text())["roles"]
    token = secrets.token_hex(24)
    # okg wrote the read role's URL (with its password) into the secrets file.
    ro_dsn = roles["okg_mcp"]
    (work / "mcp.env").write_text(
        f"OKG_DEPLOYMENTS_DIR={deployments}\nOKG_DSN={dsn}\nOKG_MCP_RO_DSN={ro_dsn}\nOKG_CHAT_MCP_TOKEN={token}\n"
    )
    (work / "mcp.env").chmod(0o600)
    serve = work / "serve.sh"
    serve.write_text(
        "#!/bin/bash\n# Serve MCP for the fixture deployment; the token and DSNs are in mcp.env (mode 0600).\n"
        "set -euo pipefail\n"
        f"cd {shlex.quote(str(instance))}\n"
        "set -a; . " + shlex.quote(str(work / "mcp.env")) + "; set +a\n"
        f"exec {shlex.quote(ps.okg_bin())} mcp-serve --deployment {DEPLOYMENT} --transport streamable-http "
        f"--host 127.0.0.1 --port {port} --auth-token-env OKG_CHAT_MCP_TOKEN "
        "--allowed-host '127.0.0.1:*' --allowed-host 'localhost:*'\n"
    )
    serve.chmod(0o755)
    return {"instance": str(instance),
            "chat_prompt": str(deployments / DEPLOYMENT / "skills" / "chat-system-prompt.md"),
            "serve": str(serve), "mcp_url": f"http://127.0.0.1:{port}/mcp",
            "packs": list(resolution.pack_names)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", choices=BUNDLES, default="cms-physics", help="the bundle to install (default cms-physics)")
    parser.add_argument("--dsn", required=True, help="superuser DSN of a scratch AKMON Postgres; its database is recreated")
    parser.add_argument("--work", type=Path, required=True, help="scratch directory for the instance, data and serve.sh")
    parser.add_argument("--port", type=int, default=11601, help="port serve.sh gives okg mcp-serve")
    args = parser.parse_args(argv)
    print(json.dumps(build(args.bundle, args.dsn, args.work, args.port), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
