"""`archi restart -c` and ``data_manager.torch_threads``.

Restart recreates only the named service, and refuses a config that changes the
``data_manager`` section because the data-manager would not pick it up. The torch
thread count is a data-manager setting rendered into compose for both the
data-manager and the chatbot, so restart must refuse a change to it, and must
refuse an invalid value before it rewrites any deployment file.
"""

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from src.cli import cli_main
from src.cli.managers.config_manager import ConfigurationManager
from src.cli.source_registry import source_registry
from src.cli.utils.helpers import _render_config_for_compare

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG = REPO_ROOT / "examples" / "deployments" / "basic-openai" / "config.yaml"
DEPLOYMENT_NAME = "my_archi"

# Halts `restart` right after the config checks, before the compose build.
SENTINEL = "stopped-before-compose-build"


@pytest.fixture
def archi_home(tmp_path, monkeypatch):
    """A throwaway ARCHI_DIR; cli_main freezes it at import, so patch the attribute."""
    home = tmp_path / "archi-home"
    monkeypatch.setenv("ARCHI_DIR", str(home))
    monkeypatch.setattr(cli_main, "ARCHI_DIR", str(home))
    return home


@pytest.fixture(autouse=True)
def docker_available(monkeypatch):
    monkeypatch.setattr(cli_main, "check_docker_available", lambda: True)


@pytest.fixture(autouse=True)
def stop_before_compose_build(monkeypatch):
    def _raise(*args, **kwargs):
        raise RuntimeError(SENTINEL)

    monkeypatch.setattr(
        cli_main.ServiceBuilder, "build_compose_config", staticmethod(_raise)
    )


def _write_config(tmp_path, torch_threads):
    config = yaml.safe_load(EXAMPLE_CONFIG.read_text())
    if torch_threads is not None:
        config.setdefault("data_manager", {})["torch_threads"] = torch_threads
    path = tmp_path / f"config-{torch_threads}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _render_deployed_config(config_path):
    """Render the deployed config the way `create` does: load, normalize, render."""
    config_manager = ConfigurationManager([str(config_path)], cli_main.env)
    enabled_sources = list(
        dict.fromkeys(["links"] + config_manager.get_enabled_sources())
    )
    disabled_sources = config_manager.get_disabled_sources()
    enabled_sources = [src for src in enabled_sources if src not in disabled_sources]
    config_manager.set_sources_enabled(
        source_registry.resolve_dependencies(enabled_sources)
    )
    return _render_config_for_compare(config_manager.config, False, 3, cli_main.env)


def _deploy(archi_home, config_path):
    deployment_dir = archi_home / f"archi-{DEPLOYMENT_NAME}"
    (deployment_dir / "configs").mkdir(parents=True)
    compose = {"services": {"chatbot": {}, "postgres": {}}}
    (deployment_dir / "compose.yaml").write_text(yaml.safe_dump(compose))
    deployed = deployment_dir / "configs" / f"{DEPLOYMENT_NAME}.yaml"
    deployed.write_text(yaml.safe_dump(_render_deployed_config(config_path)))
    return deployed


def _restart(config_path):
    env_file = config_path.parent / "secrets.env"
    env_file.write_text("PG_PASSWORD=x\nOPENAI_API_KEY=x\n")
    return CliRunner().invoke(
        cli_main.restart,
        [
            "--name",
            DEPLOYMENT_NAME,
            "--config",
            str(config_path),
            "--env-file",
            str(env_file),
        ],
    )


@pytest.mark.parametrize(("deployed", "requested"), [(None, 4), (4, 2), (4, None)])
def test_a_torch_threads_change_is_refused(archi_home, tmp_path, deployed, requested):
    _deploy(archi_home, _write_config(tmp_path, deployed))

    result = _restart(_write_config(tmp_path, requested))

    assert "Restart config changes are restricted" in result.output, (
        f"exit_code={result.exit_code}\noutput:\n{result.output}\n"
        f"exception:\n{result.exception}\n"
    )
    assert "data_manager" in result.output


@pytest.mark.parametrize("threads", [None, 4])
def test_an_unchanged_torch_threads_is_not_refused(archi_home, tmp_path, threads):
    config_path = _write_config(tmp_path, threads)
    _deploy(archi_home, config_path)

    result = _restart(config_path)

    assert result.exception is not None and SENTINEL in str(result.exception), (
        f"exit_code={result.exit_code}\noutput:\n{result.output}\n"
        f"exception:\n{result.exception}\n"
    )


@pytest.mark.parametrize("value", [0, "8"])
def test_an_invalid_value_is_refused_before_any_file_is_written(
    archi_home, tmp_path, value
):
    deployed = _deploy(archi_home, _write_config(tmp_path, None))
    before = deployed.read_bytes()

    result = _restart(_write_config(tmp_path, value))

    message = f"{result.output}\n{result.exception}"
    assert "data_manager.torch_threads" in message, message
    assert SENTINEL not in message
    assert deployed.read_bytes() == before
