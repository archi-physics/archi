"""Every context pack, installed alone on the base, publishes (acc.r3-packs-split.matrix).

For each pack this renders a profile holding only that pack (and the packs it
requires), installs it with okg into a fresh database, feeds it the fixture
inputs in `pack_support.FIXTURES`, and runs the publish. It passes when okg
publishes a generation, the deployment lints with zero blockers, and the
pack's own coverage invariant holds on the published graph. The same run
installs the cern-team bundle with every optional pack, the proof that
cern-team installs from packs.

Needs a real Postgres with okg's extensions (AKMON's image, see
docs/cern-team-quickstart.md step 2), named by ARCHI_PACK_ISOLATION_DSN:

    ARCHI_PACK_ISOLATION_DSN=postgresql://postgres:okg@127.0.0.1:5491/packtest

The database in the DSN is dropped and re-created before each pack, so give
the test a server of its own whose `cron.database_name` is that database.
Without the variable the install tests are skipped; the fixture-coverage
test below always runs.
"""
import json
import os
from pathlib import Path

import pytest
import yaml

from archi.bundles.render import render
from archi.bundles.resolve import resolve
from packs.pack_support import (
    BUNDLES_DIR,
    FIXTURE_ENV,
    FIXTURES,
    PACKS_DIR,
    FixtureServer,
    context_packs,
    credential_names,
    git,
    invariants,
    manifest,
    run_okg,
    solo_catalog,
)

DSN = os.environ.get("ARCHI_PACK_ISOLATION_DSN")
needs_db = pytest.mark.skipif(not DSN, reason="ARCHI_PACK_ISOLATION_DSN is not set")


def test_every_pack_with_sources_has_isolation_fixtures():
    missing = sorted(set(context_packs()) - set(FIXTURES))
    assert not missing, f"packs with no fixture inputs in pack_support.FIXTURES: {missing}"


def _recreate_database(dsn: str) -> None:
    import psycopg
    from psycopg import sql

    parts = psycopg.conninfo.conninfo_to_dict(dsn)
    name = parts["dbname"]
    admin = psycopg.conninfo.make_conninfo(dsn, dbname="postgres")
    with psycopg.connect(admin, autocommit=True) as conn:
        conn.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name)))
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))


def _closure(pack: str) -> list[str]:
    out: list[str] = []

    def visit(name):
        for dep in manifest(name).get("requires", {}).get("packs", []):
            visit(dep)
        if name not in out:
            out.append(name)

    visit(pack)
    return out


def _flag(qid: str) -> str:
    return "--" + qid.replace("_", "-")


def _report(result) -> str:
    return f"exit {result.returncode}\nstdout:\n{result.stdout[-4000:]}\nstderr:\n{result.stderr[-4000:]}"


@needs_db
@pytest.mark.parametrize("pack", sorted(FIXTURES))
def test_pack_installs_alone_and_publishes(tmp_path, pack):
    catalog = solo_catalog(tmp_path, pack)
    resolution = resolve(catalog, [f"solo-{pack}"])
    assert resolution.pack_names == tuple(_closure(pack))
    _install_and_publish(tmp_path, resolution, f"solo-{pack}")


@needs_db
def test_cern_team_installs_from_packs_and_publishes(tmp_path):
    """The cern-team bundle, every optional pack selected, from fixture inputs."""
    from packs.pack_support import shipped_catalog

    catalog = shipped_catalog()
    optional = list(catalog.bundles["cern-team"].optional_packs)
    resolution = resolve(catalog, ["cern-team"], with_packs=optional)
    assert set(resolution.pack_names) == set(catalog.bundles["cern-team"].default_packs) | set(optional)
    _install_and_publish(tmp_path, resolution, "cern-team")


@needs_db
@pytest.mark.parametrize("with_optional", [False, True], ids=["defaults", "every-optional-pack"])
def test_cluster_manager_installs_from_packs_and_publishes(tmp_path, with_optional):
    """The cluster-manager bundle (no CERN pack): every default, then every default and optional pack."""
    from packs.pack_support import shipped_catalog

    catalog = shipped_catalog()
    bundle = catalog.bundles["cluster-manager"]
    optional = list(bundle.optional_packs) if with_optional else []
    resolution = resolve(catalog, ["cluster-manager"], with_packs=optional)
    assert set(resolution.pack_names) >= set(bundle.default_packs) | set(optional)
    _install_and_publish(tmp_path, resolution, "cluster-manager")


@needs_db
def test_cms_comp_ops_installs_from_packs_and_publishes(tmp_path):
    """The cms-comp-ops bundle (PACT r3-bundle-cms-comp-ops): every default and
    optional pack selected, from fixture inputs."""
    from packs.pack_support import shipped_catalog

    catalog = shipped_catalog()
    bundle = catalog.bundles["cms-comp-ops"]
    optional = list(bundle.optional_packs)
    resolution = resolve(catalog, ["cms-comp-ops"], with_packs=optional)
    expected = set(bundle.default_packs) | set(optional)
    assert expected <= set(resolution.pack_names)
    _install_and_publish(tmp_path, resolution, "cms-comp-ops")


@needs_db
def test_cms_physics_installs_from_packs_and_publishes(tmp_path):
    """The cms-physics bundle (PACT r3-bundle-cms-physics): its default packs,
    from fixture inputs, with the bundle's physics filter on.

    Two inputs differ from the per-pack fixtures. The bundle fixes a note
    floor of 2500 for a real mirror; the fixture mirror holds a handful, so
    the floor is lowered the way an operator overrides it, with the flag on
    `okg install`. And the generic TWiki fixture has no physics topic, so
    the filter would leave the source empty; the TWiki root points at the
    physics dump the cms-analyses tests use instead."""
    from packs.pack_support import shipped_catalog
    from packs.physics_fixtures import write_twiki_dump

    catalog = shipped_catalog()
    bundle = catalog.bundles["cms-physics"]
    resolution = resolve(catalog, ["cms-physics"])
    assert set(bundle.default_packs) <= set(resolution.pack_names)
    physics_twiki = write_twiki_dump(tmp_path / "physics-twiki")
    _install_and_publish(tmp_path, resolution, "cms-physics", overrides={
        "analysis_notes_min_notes": "1",
        "twiki_eos_root": str(physics_twiki),
    })


def _install_and_publish(tmp_path: Path, resolution, name: str, overrides: dict | None = None) -> None:
    """Render, install with okg, commit, ingest and publish; then check lint
    and that each selected pack's invariants were installed."""
    _recreate_database(DSN)
    instance = tmp_path / "instance"
    deployments = instance / "deployments"
    data = tmp_path / "data"
    data.mkdir()
    deployments.mkdir(parents=True)
    git(["init", "-q"], deployments)
    git(["commit", "-q", "--allow-empty", "-m", "baseline"], deployments)

    rendered = render(resolution, instance, name=name)

    server = FixtureServer()
    try:
        answers = {"postgres_dsn": DSN, "archi_data_root": str(data)}
        for pack in resolution.pack_names:
            for local, value in FIXTURES.get(pack, lambda d, s: {})(data, server).items():
                answers[pack.replace("-", "_") + "_" + local] = value
        answers.update(overrides or {})
        known = {q["id"] for q in rendered.composition.questions}
        assert set(answers) <= known, sorted(set(answers) - known)

        # No credential reaches a pack unless its fixture sets a dummy one, so
        # a pack that says needs_credentials: false proves it here.
        env = {
            k: v for k, v in os.environ.items()
            if k != "OKG_PROFILES_DIR" and k not in credential_names()
        }
        env.update(
            HOME=str(tmp_path),
            OKG_DEPLOYMENTS_DIR=str(deployments),
            OKG_DSN=DSN,
            ARCHI_DATA_ROOT=str(data),
            GIT_AUTHOR_NAME="pack-test", GIT_AUTHOR_EMAIL="pack-test@example.org",
            GIT_COMMITTER_NAME="pack-test", GIT_COMMITTER_EMAIL="pack-test@example.org",
        )
        env.update(_fixture_env(resolution.pack_names, tmp_path))
        flags = []
        for qid, value in answers.items():
            flags += [_flag(qid), value]
        install = run_okg(
            ["install", "--profile", rendered.name, "--deployment-name", "px",
             "--deployments-root", str(deployments), "--non-interactive", "--json", *flags],
            cwd=instance, env=env,
        )
        (tmp_path / "install.json").write_text(install.stdout + "\n" + install.stderr)
        # The first install stops before publishing: okg will not publish from
        # configuration that is not committed yet (quickstart step 3).
        deployment = deployments / "px"
        assert (deployment / "source_registry.yaml").is_file(), _report(install)
        git(["add", "-A"], deployments)
        git(["commit", "-q", "-m", "px deployment config"], deployments)

        ingest = run_okg(["ingest", "--deployment", "px", "--json"], cwd=instance, env=env)
        (tmp_path / "ingest.json").write_text(ingest.stdout)
        assert ingest.returncode == 0, _report(ingest)
        publish = run_okg(["run", "--once", "--deployment", "px", "--apply", "--json"],
                          cwd=instance, env=env)
        assert publish.returncode == 0, _report(publish)

        status = run_okg(["status", "--deployment", "px", "--json"], cwd=instance, env=env)
        assert status.returncode == 0, _report(status)
        published = json.loads(status.stdout)["publishing"]
        assert published["latest_published_status"] == "published", published

        lint = run_okg(["deployment", "lint", str(deployment), "--json"], cwd=instance, env=env)
        findings = json.loads(lint.stdout)["findings"]
        blockers = [f for f in findings if f["severity"] in ("blocker", "error")]
        assert blockers == [], blockers

        _assert_coverage_invariants_hold(resolution.pack_names)
        if "code-text" in resolution.pack_names:
            _assert_planted_secret_stays_out(instance, env)
    finally:
        server.close()


def _assert_planted_secret_stays_out(instance: Path, env: dict) -> None:
    """code-text's secret screen held through install, ingest and publish.

    git-code's fixture repository holds a clean README and a file with a
    planted, invented password. In the published graph the README has a
    code-text document and the planted file has none, and a literal search
    for the password finds nothing anywhere in the graph. This fails if the
    pack stops naming the screened source or the screen stops running.
    """
    from packs.pack_support import PLANTED_SECRET, PLANTED_SECRET_PATH

    # The read commands connect as okg's least-privilege read role, which a
    # fresh database creates without a login; give it one (local test DB).
    roles = run_okg(["db", "role-passwords", "--apply", "--deployment", "px"], cwd=instance, env=env)
    assert roles.returncode == 0, _report(roles)
    docs = run_okg(["filter", "--deployment", "px", "--type", "document", "--ids", "--all"],
                   cwd=instance, env=env)
    assert docs.returncode == 0, _report(docs)
    ids = set(docs.stdout.split())
    assert "doc:demo:README.md" in ids, sorted(ids)[:20]
    assert f"doc:demo:{PLANTED_SECRET_PATH}" not in ids, sorted(ids)
    def grep(text):
        found = run_okg(["search", "--deployment", "px", "--method", "grep", "--ids", "--all", text],
                        cwd=instance, env=env)
        assert found.returncode == 0, _report(found)
        return found.stdout.split()

    # Positive control: the same search finds the README's own text, so an
    # empty answer for the secret means it is absent, not that grep saw nothing.
    assert grep("A test repository"), "grep found nothing at all; the absence check would be vacuous"
    assert grep(PLANTED_SECRET) == []


def _fixture_env(pack_names, tmp_path: Path) -> dict:
    """Environment the fixtures need (credential env vars hold dummies)."""
    env = {}
    for name in pack_names:
        env.update(FIXTURE_ENV.get(name, {}))
    return env


def _assert_coverage_invariants_hold(pack_names) -> None:
    """Each selected pack's invariants were installed into okg.

    okg evaluates every installed error-severity invariant before a publish
    and blocks the publish when one fails (`cross_source_error_invariant_failed`;
    `_build_invariant_summary` in okg's publisher), so with the generation
    published they held. Reading the graph to recount is not possible here:
    bare SQL on the graph is refused outside a pinned checkout, and
    `okg status` lists only the largest subtypes.
    """
    import psycopg

    with psycopg.connect(DSN) as conn:
        installed = {row[0] for row in conn.execute("SELECT name FROM okg.invariants")}
    for name in pack_names:
        for inv in invariants(name):
            assert inv["name"] in installed, f"{name}: invariant {inv['name']} was not installed"
