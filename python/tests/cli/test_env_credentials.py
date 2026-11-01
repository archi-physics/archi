"""A source credential written into the instance's `.env` reaches ingestion.

The pack pages tell users to put a source's credential (`CERN_JIRA_TOKEN`,
`REDMINE_API_KEY`, ...) in the instance's `.env`. A source reads it from the
environment of the okg process that ingests. Before this was fixed, archi passed
only its own generated secrets from `.env` and dropped the rest, so a token
placed there was ignored and the source ingested public data only, with no
error.

These tests run a stand-in `okg` executable as a real subprocess. It writes the
credential it sees in its own environment, and its arguments, to a file.
"""
import argparse
import stat
import subprocess

import pytest
import yaml

from archi.stack import okg as okg_mod
from archi.stack.instance import Instance, Ports, generate_secrets, host_env, read_env, write_env
from archi.stack.okg import Okg

TOKEN = "CERN_JIRA_TOKEN"


@pytest.fixture
def instance(tmp_path) -> Instance:
    inst = Instance(
        dir=tmp_path / "myteam", name="my-team", bundles=["cern-team"],
        data_root=tmp_path / "data", runtime="docker",
        ports=Ports(graph_db=15432, mcp=18765, webui=18080), profile="cern-team",
        okg_version="0.2.0", admin_email="admin@my-team.archi.invalid",
    )
    inst.deployment_dir.mkdir(parents=True)
    (inst.deployment_dir / "deployment.yaml").write_text(yaml.safe_dump({"chat": {"mcp": {"port": 8765}}}))
    secrets = generate_secrets()
    secrets["ARCHI_OKG_MCP_PASSWORD"] = "role-pw"
    write_env(inst.env_path, secrets)
    inst.save()
    return inst


def add_to_env(instance: Instance, **values: str) -> None:
    """What the docs tell a user to do: append KEY=VALUE lines to `.env`."""
    with open(instance.env_path, "a", encoding="utf-8") as fh:
        for key, value in values.items():
            fh.write(f"{key}={value}\n")


@pytest.fixture
def fake_okg(tmp_path, monkeypatch):
    """A stand-in okg that records the token in its environment and its argv."""
    seen = tmp_path / "okg-saw.txt"
    script = tmp_path / "bin" / "okg"
    script.parent.mkdir()
    script.write_text(
        "#!/bin/sh\n"
        f'printf "token=%s\\n" "${{{TOKEN}:-<unset>}}" > "{seen}"\n'
        f'printf "args=%s\\n" "$*" >> "{seen}"\n'
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(okg_mod, "executable", lambda: str(script))
    monkeypatch.delenv(TOKEN, raising=False)
    return seen


def saw(path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text().splitlines())


def test_a_credential_in_the_instance_env_reaches_the_okg_ingest_process(instance, fake_okg):
    add_to_env(instance, **{TOKEN: "from-dotenv"})
    Okg(instance, read_env(instance.env_path), echo=lambda _: None).run("ingest", "--deployment", instance.name)
    assert saw(fake_okg)["token"] == "from-dotenv"


def test_archi_ingest_runs_okg_ingest_with_the_instance_env(instance, fake_okg, capsys):
    from archi.cli import ingest as ingest_mod

    add_to_env(instance, **{TOKEN: "from-dotenv"})
    args = argparse.Namespace(instance=instance.dir, include="jira.jira")
    assert ingest_mod.cmd_ingest(args) == 0
    got = saw(fake_okg)
    assert got["token"] == "from-dotenv"
    assert got["args"] == "ingest --deployment my-team --include jira.jira"
    assert "from-dotenv" not in capsys.readouterr().out, "the value is never printed"


def test_a_value_exported_in_the_shell_wins_over_the_env_file(instance, fake_okg, monkeypatch):
    add_to_env(instance, **{TOKEN: "from-dotenv"})
    monkeypatch.setenv(TOKEN, "from-shell")
    Okg(instance, read_env(instance.env_path), echo=lambda _: None).run("ingest", "--deployment", instance.name)
    assert saw(fake_okg)["token"] == "from-shell"


def test_no_credential_anywhere_stays_unset(instance, fake_okg):
    Okg(instance, read_env(instance.env_path), echo=lambda _: None).run("ingest", "--deployment", instance.name)
    assert saw(fake_okg)["token"] == "<unset>"


def test_the_instance_settings_still_come_from_the_instance(instance):
    """A stray OKG_DSN or OKG_PROFILES_DIR in `.env` or the shell cannot redirect okg."""
    add_to_env(instance, OKG_DSN="postgresql://elsewhere/x", OKG_PROFILES_DIR="/elsewhere")
    env = host_env(instance, read_env(instance.env_path), base={"OKG_DSN": "postgresql://shell/x"})
    assert env["OKG_DSN"].endswith(f"127.0.0.1:{instance.ports.graph_db}/{instance.database}")
    assert "OKG_PROFILES_DIR" not in env
    assert env["OKG_DEPLOYMENTS_DIR"] == str(instance.dir / "deployments")


def test_archi_ingest_outside_an_instance_exits_2(tmp_path, capsys):
    from archi.cli import ingest as ingest_mod

    assert ingest_mod.cmd_ingest(argparse.Namespace(instance=tmp_path, include=None)) == 2
    assert "error:" in capsys.readouterr().err


def test_a_failed_okg_ingest_returns_its_exit_code(instance, monkeypatch, capsys):
    from archi.cli import ingest as ingest_mod

    def runner(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 3, "", "boom")

    monkeypatch.setattr(okg_mod, "executable", lambda: "okg")
    assert ingest_mod.cmd_ingest(argparse.Namespace(instance=instance.dir, include=None), runner=runner) == 3
    assert "boom" in capsys.readouterr().err


def test_archi_ingest_is_a_command():
    from archi.cli import ingest as ingest_mod
    from archi.bundles.cli import build_parser

    args = build_parser().parse_args(["ingest", "--include", "a.b"])
    assert args.func is ingest_mod.cmd_ingest and args.include == "a.b"
