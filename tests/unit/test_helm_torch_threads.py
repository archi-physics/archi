"""The Helm chart gives the two embedding deployments a bounded torch thread count.

Kubernetes CPU limits are a CFS quota, which torch ignores when it sizes its
thread pool, so a pod still starts one thread per node core unless
``OMP_NUM_THREADS`` is set. Torch reads ``MKL_NUM_THREADS`` ahead of it, so both
are set. ``values.yaml`` carries the resolved ``torch_threads``; the data-manager
and chatbot deployments read it.

The pods run on cluster nodes, not on the machine running ``archi create``, so
the chart's count ignores that machine's CPU count and deployment ``.env``.
"""

import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml
from jinja2 import ChainableUndefined, Environment, FileSystemLoader

from src.cli.managers import templates_manager
from src.cli.managers.templates_manager import TemplateContext, TemplateManager
from src.cli.utils import torch_threads
from src.cli.utils.service_builder import ServiceBuilder

TEMPLATES = Path(__file__).resolve().parents[2] / "src/cli/templates"
CAPPED = ("data-manager", "chatbot")
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS")
GUARDED_ENTRY = re.compile(
    r"\{\{- if \.Values\.torch_threads \}\}\s*\n"
    r"\s*- name: OMP_NUM_THREADS\s*\n"
    r'\s*value: "\{\{ \.Values\.torch_threads \}\}"\s*\n'
    r"\s*- name: MKL_NUM_THREADS\s*\n"
    r'\s*value: "\{\{ \.Values\.torch_threads \}\}"\s*\n'
    r"\s*\{\{- end \}\}"
)


def _env():
    return Environment(
        loader=FileSystemLoader(str(TEMPLATES)), undefined=ChainableUndefined
    )


def _render_deployment(service):
    return (
        _env()
        .get_template(f"helm/templates/{service}/deployment.yaml")
        .render(name="demo", selenium_scraper=False)
    )


@pytest.mark.parametrize("service", CAPPED)
def test_embedding_deployments_read_the_value(service):
    assert GUARDED_ENTRY.search(_render_deployment(service))


def test_other_deployments_do_not_set_it():
    for path in sorted((TEMPLATES / "helm/templates").glob("*/deployment.yaml")):
        if path.parent.name in CAPPED:
            continue
        for name in THREAD_VARS:
            assert name not in path.read_text(), (path, name)


def _stage_values(tmp_path, monkeypatch, data_manager, cpus=32):
    monkeypatch.setattr(torch_threads, "visible_cpu_count", lambda: cpus)
    manager = TemplateManager(jinja_env=_env(), verbosity=0)
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
    manager._stage_values(context)
    return yaml.safe_load((tmp_path / "values.yaml").read_text())


@pytest.mark.parametrize(
    ("data_manager", "expected"), [({}, 8), ({"torch_threads": 4}, 4)]
)
def test_stage_values_writes_the_resolved_count(
    tmp_path, monkeypatch, data_manager, expected
):
    assert (
        _stage_values(tmp_path, monkeypatch, data_manager)["torch_threads"] == expected
    )


def test_stage_values_refuses_an_invalid_count(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match=r"data_manager\.torch_threads"):
        _stage_values(tmp_path, monkeypatch, {"torch_threads": 0})

    assert not (tmp_path / "values.yaml").exists()


@pytest.mark.parametrize("cpus", [1, 4])
def test_stage_values_does_not_cap_at_this_machines_cpus(tmp_path, monkeypatch, cpus):
    """A 4-core laptop running ``archi create`` says nothing about a cluster node."""
    assert _stage_values(tmp_path, monkeypatch, {}, cpus=cpus)["torch_threads"] == 8


@pytest.mark.parametrize("name", THREAD_VARS)
def test_stage_values_ignores_a_thread_value_in_the_local_env_file(
    tmp_path, monkeypatch, name
):
    """The pods read the cluster Secret, not this machine's deployment .env."""
    (tmp_path / ".env").write_text(f"PG_PASSWORD=x\n{name}=4\n")

    assert _stage_values(tmp_path, monkeypatch, {})["torch_threads"] == 8


def test_values_template_renders_no_count_as_null():
    """Jinja would write None, a non-empty string Helm's ``if`` treats as set."""
    rendered = _env().get_template("helm/values.yaml").render(torch_threads=None)

    assert yaml.safe_load(rendered)["torch_threads"] is None
