"""The two embedding containers, and only those, get a bounded torch thread count.

The data-manager embeds every chunk at ingest and the chatbot embeds every search
query. Both run torch on CPU, which without ``OMP_NUM_THREADS`` starts one thread
per host core. Other services do not embed and are left alone.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml
from jinja2 import ChainableUndefined, Environment, FileSystemLoader

from src.cli.managers import templates_manager
from src.cli.managers.secrets_manager import SecretsManager
from src.cli.managers.templates_manager import TemplateContext, TemplateManager
from src.cli.utils import torch_threads
from src.cli.utils.service_builder import ServiceBuilder

CAPPED_BLOCKS = ("data-manager", "chatbot")
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS")
UNCAPPED_BLOCKS = ("grader", "piazza", "mattermost", "redmine", "mailbox", "benchmark")


def _render_compose(tmp_path, **extra_vars):
    repository = Path(__file__).resolve().parents[2]
    environment = Environment(
        loader=FileSystemLoader(str(repository / "src/cli/templates")),
        undefined=ChainableUndefined,
    )
    plan = ServiceBuilder.build_compose_config(
        name="demo",
        verbosity=3,
        base_dir=tmp_path,
        enabled_services=[
            "chatbot",
            "data-manager",
            "grader",
            "piazza",
            "mattermost",
            "redmine-mailer",
            "benchmarking",
        ],
        secrets={"PG_PASSWORD"},
        tag="dev",
    )
    template_vars = plan.to_template_vars()
    template_vars.update(
        app_version="test",
        postgres_port=5432,
        data_manager_port_host=7871,
        data_manager_port_container=7871,
        chatbot_port_host=7861,
        chatbot_port_container=7861,
        grader_port_host=7862,
        grader_port_container=7862,
        prompt_files=[],
        rubrics=[],
        **extra_vars,
    )
    rendered = environment.get_template("base-compose.yaml").render(**template_vars)
    return yaml.safe_load(rendered)["services"]


def _env(services, block):
    assert block in services, f"{block} is missing from the rendered compose file"
    return services[block].get("environment") or {}


def _thread_vars(services, block):
    """The torch thread variables compose sets for ``block``, absent ones omitted."""
    environment = _env(services, block)
    return {name: environment[name] for name in THREAD_VARS if name in environment}


@pytest.mark.parametrize("threads", [8, 4])
@pytest.mark.parametrize("block", CAPPED_BLOCKS)
def test_embedding_containers_get_the_thread_count(tmp_path, block, threads):
    """MKL_NUM_THREADS takes precedence over OMP_NUM_THREADS in torch, so set both."""
    services = _render_compose(tmp_path, torch_threads=threads)

    assert _thread_vars(services, block) == dict.fromkeys(THREAD_VARS, str(threads))


@pytest.mark.parametrize("block", UNCAPPED_BLOCKS)
def test_other_services_get_no_thread_count(tmp_path, block):
    services = _render_compose(tmp_path, torch_threads=8)
    if block not in services:
        pytest.skip(f"{block} is not rendered in this configuration")

    assert _thread_vars(services, block) == {}


@pytest.mark.parametrize("block", CAPPED_BLOCKS)
def test_no_value_renders_no_variable(tmp_path, block):
    """A caller that renders without the value must not ship OMP_NUM_THREADS: ''."""
    services = _render_compose(tmp_path)

    assert _thread_vars(services, block) == {}


def _render_through_manager(tmp_path, monkeypatch, data_manager, cpus=32):
    """Drive the real _render_compose_file with a config carrying ``data_manager``.

    ``cpus`` is the CPU count the default is capped at; the real one varies by host.
    """
    monkeypatch.setattr(torch_threads, "visible_cpu_count", lambda: cpus)
    repository = Path(__file__).resolve().parents[2]
    manager = TemplateManager(
        jinja_env=Environment(
            loader=FileSystemLoader(str(repository / "src/cli/templates")),
            undefined=ChainableUndefined,
        ),
        verbosity=0,
    )
    monkeypatch.setattr(manager, "_check_ports_available", lambda *a, **k: None)
    monkeypatch.setattr(templates_manager, "get_git_version", lambda: "test")
    config = {"services": {}, "data_manager": data_manager}
    plan = ServiceBuilder.build_compose_config(
        name="demo",
        verbosity=0,
        base_dir=tmp_path,
        enabled_services=["chatbot", "data-manager"],
        secrets={"PG_PASSWORD"},
        tag="dev",
    )
    context = TemplateContext(
        plan=plan,
        config_manager=SimpleNamespace(config=config, get_configs=lambda: [config]),
        secrets_manager=MagicMock(),
        options={},
    )
    manager._render_compose_file(context)
    return yaml.safe_load((tmp_path / "compose.yaml").read_text())["services"]


@pytest.mark.parametrize(
    ("data_manager", "expected"), [({}, "8"), ({"torch_threads": 4}, "4")]
)
def test_render_compose_file_writes_the_resolved_count(
    tmp_path, monkeypatch, data_manager, expected
):
    services = _render_through_manager(tmp_path, monkeypatch, data_manager)

    for block in CAPPED_BLOCKS:
        assert _thread_vars(services, block) == dict.fromkeys(THREAD_VARS, expected)


def test_render_compose_file_caps_the_default_at_the_visible_cpus(
    tmp_path, monkeypatch
):
    services = _render_through_manager(tmp_path, monkeypatch, {}, cpus=4)

    for block in CAPPED_BLOCKS:
        assert _thread_vars(services, block) == dict.fromkeys(THREAD_VARS, "4")


@pytest.mark.parametrize("name", THREAD_VARS)
def test_render_compose_file_keeps_the_env_file_value(tmp_path, monkeypatch, name):
    """No ``environment`` entry, so the deployment .env value reaches the container."""
    (tmp_path / ".env").write_text(f"PG_PASSWORD=x\n{name}=4\n")

    services = _render_through_manager(tmp_path, monkeypatch, {})

    for block in CAPPED_BLOCKS:
        assert _thread_vars(services, block) == {}
        assert ".env" in services[block]["env_file"]


@pytest.mark.parametrize("name", THREAD_VARS)
def test_an_env_file_value_survives_the_secrets_manager(tmp_path, monkeypatch, name):
    """``archi create`` copies every --env-file key into the deployment .env.

    The env-file rule above reads that .env, so a thread variable must survive
    the copy, not only a .env written by hand.
    """
    env_file = tmp_path / "operator.env"
    env_file.write_text(f"PG_PASSWORD=x\n{name}=4\n")
    deployment_dir = tmp_path / "deployment"
    deployment_dir.mkdir()
    secrets_manager = SecretsManager(str(env_file), MagicMock())
    _, all_secrets = secrets_manager.get_secrets(set(), set())
    secrets_manager.write_secrets_to_files(deployment_dir, all_secrets)

    services = _render_through_manager(deployment_dir, monkeypatch, {})

    for block in CAPPED_BLOCKS:
        assert _thread_vars(services, block) == {}


@pytest.mark.parametrize("name", THREAD_VARS)
def test_render_compose_file_lets_the_key_win_over_the_env_file(
    tmp_path, monkeypatch, name
):
    (tmp_path / ".env").write_text(f"PG_PASSWORD=x\n{name}=12\n")

    services = _render_through_manager(tmp_path, monkeypatch, {"torch_threads": 2})

    for block in CAPPED_BLOCKS:
        assert _thread_vars(services, block) == dict.fromkeys(THREAD_VARS, "2")


def test_render_compose_file_refuses_an_invalid_count(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match=r"data_manager\.torch_threads"):
        _render_through_manager(tmp_path, monkeypatch, {"torch_threads": 0})

    assert not (tmp_path / "compose.yaml").exists()
