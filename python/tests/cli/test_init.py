"""req.r3-golden-path.init: `archi init` (archi.cli.init).

The published round trip on a real AKMON Postgres is the executed test
recorded in the PACT (python/tests/cli/golden_path_run.sh). These tests cover
what init decides on its own, and the okg command sequence with a fake okg:

- acc.r3-golden-path.secrets: .env is git-ignored and no secret appears in
  the committed tree.
"""
import json
import stat
import subprocess
from pathlib import Path

import pytest
import yaml

from archi.bundles import discovery
from archi.cli import init as init_mod
from archi.cli.init import InitError, absolute_answers, init
from archi.stack import compose as stack_compose
from archi.stack.instance import SECRET_KEYS, Instance, read_env

from bundles.bundle_support import build_cern_team_catalog


def _git(directory, *args):
    return subprocess.run(["git", "-C", str(directory), *args], capture_output=True, text=True)


@pytest.fixture
def shipped():
    """The packs and bundles installed with archi (cern-team as shipped)."""
    return discovery.discover([], [])


@pytest.fixture(autouse=True)
def _git_identity(monkeypatch):
    for key, value in (("GIT_AUTHOR_NAME", "t"), ("GIT_AUTHOR_EMAIL", "t@example.org"),
                       ("GIT_COMMITTER_NAME", "t"), ("GIT_COMMITTER_EMAIL", "t@example.org")):
        monkeypatch.setenv(key, value)


def _committed_tree(directory: Path) -> str:
    """Every byte of every file in HEAD."""
    files = _git(directory, "ls-tree", "-r", "--name-only", "HEAD").stdout.split()
    return "".join(_git(directory, "show", f"HEAD:{f}").stdout for f in files)


# ---------------------------------------------------------------------------
# answers and the data root
# ---------------------------------------------------------------------------

QUESTIONS = [
    {"id": "deployment_name", "type": "slug"},
    {"id": "postgres_dsn", "type": "string"},
    {"id": "archi_data_root", "type": "string", "default": "data"},
    {"id": "twiki_eos_root", "type": "string", "default": "data/twiki-snapshot"},
    {"id": "cmssw_map_path", "type": "path", "default": ""},
    {"id": "code_extractors_dir", "type": "string", "default": "${deployment_dir}"},
    {"id": "cache_dir", "type": "string", "default": "caches/x"},
    {"id": "chat_model", "type": "string", "default": "llama3.1:8b"},
]


def test_relative_path_defaults_are_anchored_on_the_data_root(tmp_path):
    root = tmp_path / "data-root"
    answers = absolute_answers(QUESTIONS, {}, root, tmp_path / "cwd")
    assert answers["archi_data_root"] == str(root)
    # "data/..." in a bundle means "under the data root", not under okg's cwd.
    assert answers["twiki_eos_root"] == str(root / "twiki-snapshot")
    assert answers["cache_dir"] == str(root / "caches" / "x")
    # Empty and ${...} defaults are okg's to expand; non-path defaults are not passed.
    assert "cmssw_map_path" not in answers
    assert "code_extractors_dir" not in answers
    assert "chat_model" not in answers
    assert "deployment_name" not in answers and "postgres_dsn" not in answers


def test_given_relative_paths_resolve_against_where_archi_ran(tmp_path):
    answers = absolute_answers(
        QUESTIONS, {"twiki_eos_root": "dumps/twiki", "chat_model": "qwen3:4b"},
        tmp_path / "root", tmp_path / "cwd",
    )
    assert answers["twiki_eos_root"] == str(tmp_path / "cwd" / "dumps" / "twiki")
    assert answers["chat_model"] == "qwen3:4b"


def test_unknown_answers_and_data_root_answers_are_refused(tmp_path):
    with pytest.raises(InitError, match="no question"):
        absolute_answers(QUESTIONS, {"nope": "1"}, tmp_path, tmp_path)
    with pytest.raises(InitError, match="--data-root"):
        absolute_answers(QUESTIONS, {"archi_data_root": "x"}, tmp_path, tmp_path)


def test_data_root_inside_the_instance_is_refused(tmp_path, shipped):
    with pytest.raises(InitError, match="inside the instance"):
        init("myteam", bundles=["cern-team"], catalog=shipped, directory=tmp_path / "myteam",
             data_root=tmp_path / "myteam" / "data", run_okg=False, echo=lambda _: None)


def test_a_non_empty_directory_and_a_bad_name_are_refused(tmp_path, shipped):
    (tmp_path / "busy").mkdir()
    (tmp_path / "busy" / "file").write_text("x")
    with pytest.raises(InitError, match="not empty"):
        init("busy", bundles=["cern-team"], catalog=shipped, directory=tmp_path / "busy",
             data_root=tmp_path / "d", run_okg=False, echo=lambda _: None)
    with pytest.raises(InitError, match="not a valid instance name"):
        init("My Team", bundles=["cern-team"], catalog=shipped, directory=tmp_path / "x",
             data_root=tmp_path / "d", run_okg=False, echo=lambda _: None)


def test_an_underscore_name_is_refused_before_anything_is_created(tmp_path, shipped):
    # okg 0.4.0 `chat sync` refuses "_" in a deployment name, so `archi up` would fail.
    with pytest.raises(InitError, match="no _"):
        init("my_team", bundles=["cern-team"], catalog=shipped, directory=tmp_path / "x",
             data_root=tmp_path / "d", run_okg=False, echo=lambda _: None)
    assert not (tmp_path / "x").exists()


# ---------------------------------------------------------------------------
# the instance directory (no okg, no containers)
# ---------------------------------------------------------------------------


def test_init_writes_a_git_instance_with_secrets_kept_out(tmp_path, shipped):
    inst = init("myteam", bundles=["cern-team"], catalog=shipped, directory=tmp_path / "myteam",
                data_root=tmp_path / "data", run_okg=False, echo=lambda _: None)
    d = inst.dir
    log = _git(d, "log", "--format=%s").stdout.splitlines()
    assert log[-1] == "archi init: baseline"
    assert len(log) == 2
    assert _git(d, "status", "--porcelain").stdout == ""

    # acc.r3-golden-path.secrets
    env = read_env(d / ".env")
    assert set(SECRET_KEYS) <= set(env) and all(env[k] for k in SECRET_KEYS)
    assert stat.S_IMODE((d / ".env").stat().st_mode) == 0o600
    assert _git(d, "check-ignore", "-q", ".env").returncode == 0
    assert _git(d, "check-ignore", "-q", ".secrets/role-secrets.json").returncode == 0
    assert ".env" not in _git(d, "ls-files").stdout.split()
    tree = _committed_tree(d)
    for key in SECRET_KEYS:
        assert env[key] not in tree, f"{key} leaked into the committed tree"

    # Settings, the rendered profile and the stack are committed.
    loaded = Instance.load(d)
    assert loaded.data_root == (tmp_path / "data").resolve()
    assert loaded.profile == "cern-team"
    assert (d / "profiles" / "cern-team" / "profile.yaml").is_file()
    compose = yaml.safe_load((d / "compose.yaml").read_text())
    assert compose["name"] == "archi-myteam"
    assert set(compose["services"]) == {"graph-db", "chat-db", "mcp", "webui"}
    assert "--build" not in (d / "compose.yaml").read_text()


def test_init_renders_a_composed_bundle_with_its_lock(tmp_path):
    root = build_cern_team_catalog(tmp_path / "catalog")
    catalog = discovery.from_roots([root / "packs"], [root])
    inst = init("myteam", bundles=["cern-team"], catalog=catalog, directory=tmp_path / "myteam",
                data_root=tmp_path / "data", run_okg=False, echo=lambda _: None)
    lock = yaml.safe_load((inst.dir / "archi.lock").read_text())
    assert lock["selection"]["bundles"] == ["cern-team"]
    assert lock["deployment"] == str(inst.dir / "deployments" / "myteam")
    assert ".env" not in _git(inst.dir, "ls-files").stdout.split()


# ---------------------------------------------------------------------------
# the okg sequence, with a fake okg and a fake compose
# ---------------------------------------------------------------------------


class FakeOkg:
    """Records okg calls; writes what install and role-passwords would write."""

    def __init__(self):
        self.calls = []
        self.committed_at_provision = None

    def __call__(self, cmd, cwd=None, env=None, **kw):
        args = cmd[1:]
        self.calls.append((args, dict(env or {})))
        instance = Path(cwd)
        if args[0] == "install":
            dep = Path(args[args.index("--deployments-root") + 1]) / args[args.index("--deployment-name") + 1]
            dep.mkdir(parents=True)
            (dep / "deployment.yaml").write_text(yaml.safe_dump({
                "name": dep.name,
                "postgres": {"dsn": "${OKG_DSN}"},
                "chat": {"ui": {"site_name": "Team chat"}, "mcp": {"port": 8765}},
            }))
            (dep / "source_registry.yaml").write_text("sources: {}\n")
        elif args[0] == "provision":
            self.committed_at_provision = _git(instance, "status", "--porcelain").stdout == ""
        elif args[:2] == ["db", "role-passwords"]:
            path = Path(env["OKG_ROLE_SECRETS_FILE"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"roles": {
                "okg_mcp": "postgresql://okg_mcp:rolepw-123@127.0.0.1:5/x",
                "app_rw": "postgresql://app_rw:rolepw-456@127.0.0.1:5/x",
            }}))
        return subprocess.CompletedProcess(cmd, 0, "{}", "")


def _no_download(cmd, **kw):
    """The prepare runner unit tests inject: nothing is fetched."""
    return subprocess.CompletedProcess(cmd, 0, "", "")


def test_init_installs_commits_then_publishes(tmp_path, shipped, monkeypatch):
    monkeypatch.setattr(stack_compose, "detect_runtime", lambda preferred, runner: "docker")
    monkeypatch.setattr(init_mod, "okg_version", lambda: "0.2.0")
    monkeypatch.setenv("OKG_PROFILES_DIR", "/somewhere/else")
    compose_calls = []

    def fake_compose(cmd, **kw):
        compose_calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    okg = FakeOkg()
    inst = init(
        "myteam", bundles=["cern-team"], catalog=shipped, directory=tmp_path / "myteam",
        with_packs=["twiki"],
        data_root=tmp_path / "data",
        answers={"chat_model": "qwen3:4b", "twiki_eos_root": str(tmp_path / "dump")},
        okg_runner=okg, compose_runner=fake_compose, wait_db=lambda dsn: None,
        prepare_runner=_no_download, echo=lambda _: None,
    )
    verbs = [args[0] if args[0] != "db" else "db " + args[1] for args, _ in okg.calls]
    assert verbs == ["install", "provision", "db role-passwords"]

    install, env = okg.calls[0]
    assert "--no-publish" in install and "--non-interactive" in install
    # The DSN is stored as a reference; the password only travels in the environment.
    assert install[install.index("--postgres-dsn") + 1] == "${OKG_DSN}"
    secrets = read_env(inst.env_path)
    assert secrets["ARCHI_GRAPH_DB_PASSWORD"] in env["OKG_DSN"]
    assert not any(secrets["ARCHI_GRAPH_DB_PASSWORD"] in a for a in install)
    # okg runs from the instance and finds ./profiles/, never OKG_PROFILES_DIR.
    assert "OKG_PROFILES_DIR" not in env
    assert env["OKG_ROLE_SECRETS_FILE"] == str(inst.dir / ".secrets" / "role-secrets.json")
    # Every path answer is absolute, the data root among them.
    assert install[install.index("--archi-data-root") + 1] == str((tmp_path / "data").resolve())
    assert install[install.index("--twiki-eos-root") + 1] == str((tmp_path / "dump").resolve())
    assert install[install.index("--chat-model") + 1] == "qwen3:4b"

    # The install's files were committed before the publish (okg refuses otherwise).
    assert okg.committed_at_provision is True
    assert _git(inst.dir, "log", "-1", "--format=%s").stdout.strip() == "archi init: okg deployment myteam"

    # The MCP role password reached .env, and only .env.
    assert secrets["ARCHI_OKG_MCP_PASSWORD"] == "rolepw-123"
    assert "rolepw-123" not in _committed_tree(inst.dir)
    assert _git(inst.dir, "status", "--porcelain").stdout == ""

    # Only the graph database was started; `archi up` starts the rest.
    assert compose_calls[-1][-3:] == ["up", "-d", "graph-db"]


def test_provision_capacity_warning_is_not_a_failure(tmp_path, shipped, monkeypatch):
    monkeypatch.setattr(stack_compose, "detect_runtime", lambda preferred, runner: "docker")
    monkeypatch.setattr(init_mod, "okg_version", lambda: "0.2.0")
    okg = FakeOkg()

    def okg8(cmd, **kw):
        result = okg(cmd, **kw)
        return subprocess.CompletedProcess(cmd, 8 if cmd[1] == "provision" else 0, result.stdout, "")

    notes = []
    init("myteam", bundles=["cern-team"], catalog=shipped, directory=tmp_path / "myteam",
         data_root=tmp_path / "data", okg_runner=okg8,
         compose_runner=lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "", ""),
         wait_db=lambda dsn: None, prepare_runner=_no_download, echo=notes.append)
    assert any("capacity_warning" in n for n in notes)
