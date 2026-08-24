from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml
from flask import Flask, render_template
from jinja2 import ChainableUndefined, Environment, FileSystemLoader

from src.interfaces.chat_app import app as chat_app_module


def _template_env():
    repository = Path(__file__).resolve().parents[2]
    return Environment(
        loader=FileSystemLoader(str(repository / "src/cli/templates")),
        undefined=ChainableUndefined,
    )


@pytest.mark.parametrize(
    ("chat_app_config", "expected_enabled"),
    [
        ({}, False),
        ({"evaluations": {}}, False),
        ({"evaluations": {"enabled": False}}, False),
        ({"evaluations": {"enabled": True}}, True),
        ({"evaluations": {"enabled": 1}}, False),
        ({"evaluations": {"enabled": "true"}}, False),
    ],
)
def test_generated_evaluation_console_requires_explicit_enablement(
    chat_app_config, expected_enabled
):
    template = _template_env().get_template("base-config.yaml")

    rendered_config = yaml.safe_load(
        template.render(services={"chat_app": chat_app_config})
    )

    assert rendered_config["services"]["chat_app"]["evaluations"] == {
        "enabled": expected_enabled,
        "root": "/root/archi/evaluations",
        "agent_config_path": "/root/archi/configs/config.yaml",
        "mcp_config_path": None,
    }


@pytest.mark.parametrize(
    ("chat_app_config", "expected_enabled"),
    [
        ({}, False),
        ({"evaluations": {}}, False),
        ({"evaluations": {"enabled": False}}, False),
        ({"evaluations": {"enabled": True}}, True),
        ({"evaluations": {"enabled": 1}}, False),
        ({"evaluations": {"enabled": "true"}}, False),
    ],
)
def test_evaluation_console_routes_require_explicit_enablement(
    monkeypatch, tmp_path, chat_app_config, expected_enabled
):
    full_config = {
        "name": "test",
        "global": {
            "DATA_PATH": str(tmp_path),
            "ACCOUNTS_PATH": str(tmp_path / "accounts"),
        },
        "services": {
            "chat_app": chat_app_config,
            "data_manager": {},
            "postgres": {},
        },
    }
    evaluation_service = Mock(name="evaluation_service")
    evaluation_service_factory = Mock(return_value=evaluation_service)

    monkeypatch.setattr(
        chat_app_module.FlaskAppWrapper,
        "configs",
        lambda self, **configs: None,
    )
    monkeypatch.setattr(chat_app_module, "get_full_config", lambda: full_config)
    monkeypatch.setattr(chat_app_module, "read_secret", lambda _name: "")
    monkeypatch.setattr(
        chat_app_module,
        "read_or_create_persistent_secret",
        lambda _name, _data_path: "test-secret",
    )
    monkeypatch.setattr(chat_app_module, "ConfigService", Mock)
    monkeypatch.setattr(chat_app_module, "get_registry", Mock())
    monkeypatch.setattr(chat_app_module, "ChatWrapper", Mock)
    monkeypatch.setattr(chat_app_module, "CORS", Mock())
    monkeypatch.setattr(chat_app_module, "register_playbooks", Mock())
    monkeypatch.setattr(chat_app_module, "register_service_alerts", Mock())
    monkeypatch.setattr(
        chat_app_module, "EvaluationConsoleService", evaluation_service_factory
    )
    monkeypatch.setattr(
        chat_app_module.FlaskAppWrapper,
        "add_endpoint",
        lambda self, *args, **kwargs: None,
    )
    monkeypatch.setattr(
        chat_app_module.FlaskAppWrapper,
        "require_auth",
        lambda self, view: view,
    )
    monkeypatch.setattr(
        chat_app_module.FlaskAppWrapper,
        "require_perm",
        lambda self, _permission: lambda view: view,
    )
    monkeypatch.setattr(
        chat_app_module.FlaskAppWrapper,
        "authorize_request",
        lambda self, _permission: None,
    )

    repository = Path(__file__).resolve().parents[2]
    flask_app = Flask(
        __name__,
        template_folder=str(repository / "src/interfaces/chat_app/templates"),
        static_folder=str(repository / "src/interfaces/chat_app/static"),
    )
    wrapper = chat_app_module.FlaskAppWrapper(flask_app)
    registered_routes = {rule.rule for rule in flask_app.url_map.iter_rules()}
    page_response = flask_app.test_client().get("/evaluations")

    assert wrapper.evaluations_enabled is expected_enabled
    assert evaluation_service_factory.call_count == int(expected_enabled)
    if expected_enabled:
        assert evaluation_service_factory.call_args.kwargs["mcp_config_path"] is None
    assert ("/evaluations" in registered_routes) is expected_enabled
    assert ("/api/evaluations/catalog" in registered_routes) is expected_enabled
    assert page_response.status_code == (200 if expected_enabled else 404)


@pytest.mark.parametrize(
    (
        "evaluations_enabled",
        "auth_enabled",
        "has_view_permission",
        "expected_visible",
    ),
    [
        (False, False, True, False),
        (False, True, True, False),
        (True, False, False, True),
        (True, True, False, False),
        (True, True, True, True),
    ],
)
def test_evaluation_navigation_visibility_matches_route_access(
    evaluations_enabled,
    auth_enabled,
    has_view_permission,
    expected_visible,
):
    wrapper = object.__new__(chat_app_module.FlaskAppWrapper)
    wrapper.evaluations_enabled = evaluations_enabled
    wrapper.auth_enabled = auth_enabled

    with Flask(__name__).test_request_context():
        with patch.object(
            chat_app_module,
            "has_permission",
            return_value=has_view_permission,
        ) as has_permission:
            visible = wrapper._can_view_evaluations()

    assert visible is expected_visible
    assert has_permission.call_count == int(evaluations_enabled and auth_enabled)


def test_chat_index_passes_evaluation_visibility_to_template():
    wrapper = object.__new__(chat_app_module.FlaskAppWrapper)
    wrapper._can_view_evaluations = Mock(return_value=True)

    with patch.object(
        chat_app_module,
        "render_template",
        return_value="rendered",
    ) as render_template_mock:
        assert wrapper.index() == "rendered"

    render_template_mock.assert_called_once_with(
        "index.html",
        can_view_evaluations=True,
    )


@pytest.mark.parametrize("can_view_evaluations", [False, True])
def test_chat_template_shows_evaluation_tab_only_when_allowed(
    can_view_evaluations,
):
    repository = Path(__file__).resolve().parents[2]
    flask_app = Flask(
        __name__,
        template_folder=str(repository / "src/interfaces/chat_app/templates"),
        static_folder=str(repository / "src/interfaces/chat_app/static"),
    )

    with flask_app.test_request_context():
        rendered = render_template(
            "index.html",
            can_view_evaluations=can_view_evaluations,
        )

    evaluation_label = ">Evaluation</a>"
    assert (evaluation_label in rendered) is can_view_evaluations
    if can_view_evaluations:
        assert rendered.index(">Data</button>") < rendered.index(evaluation_label)
        assert rendered.index(evaluation_label) < rendered.index(">Status</a>")


def test_chatbot_deployments_persist_the_evaluation_root():
    repository = Path(__file__).resolve().parents[2]
    compose = (repository / "src/cli/templates/base-compose.yaml").read_text()
    helm = (
        repository / "src/cli/templates/helm/templates/chatbot/deployment.yaml"
    ).read_text()

    assert "./data/evaluations:/root/archi/evaluations" in compose
    assert "mountPath: /root/archi/evaluations" in helm
    assert "subPath: evaluations" in helm


def test_helm_mounts_dedicated_evaluation_config_only_when_configured():
    repository = Path(__file__).resolve().parents[2]
    template = _template_env().get_template("helm/templates/chatbot/deployment.yaml")

    configured = template.render(
        name="demo",
        evaluation_mcp_configured=True,
    )
    omitted = template.render(
        name="demo",
        evaluation_mcp_configured=False,
    )

    assert "mountPath: /root/archi/evaluation_config" in configured
    assert "name: demo-evaluation-config" in configured
    assert "mountPath: /root/archi/evaluation_config" not in omitted
    assert "name: demo-evaluation-config" not in omitted
