from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml
from jinja2 import ChainableUndefined, Environment, FileSystemLoader

from src.bin import service_jira
from src.cli.managers.config_manager import ConfigurationManager
from src.cli.managers.secrets_manager import SecretsManager
from src.cli.utils.service_builder import ServiceBuilder


class _FakeConfigManager:
    def __init__(self, config):
        self.config = config

    def get_models_configs(self):
        return []

    def get_configs(self):
        return [self.config]


def _template_env():
    repo_root = Path(__file__).resolve().parents[2]
    return Environment(
        loader=FileSystemLoader(str(repo_root / "src/cli/templates")),
        undefined=ChainableUndefined,
    )


def _write_config(tmp_path, services):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"name": "demo", "services": services}))
    return config_path


def _valid_services_config():
    return {
        "jira_ticket_responder": {
            "url": "https://jira.example/",
            "projects": ["CMSTZ"],
            "visible_to_role": "Developers",
        },
        "chat_app": {
            "default_provider": "openai",
            "default_model": "gpt-5",
        },
    }


class TestResolveJiraPat:
    def test_uses_service_pat(self, monkeypatch):
        secrets = {
            "JIRA_TICKET_RESPONDER_PAT": "service-token",
        }
        monkeypatch.setattr(
            service_jira, "read_secret", lambda name: secrets.get(name, "")
        )

        assert service_jira.resolve_jira_pat() == "service-token"

    def test_fails_when_service_pat_is_missing(self, monkeypatch):
        monkeypatch.setattr(service_jira, "read_secret", lambda name: "")

        with pytest.raises(ValueError, match="JIRA_TICKET_RESPONDER_PAT"):
            service_jira.resolve_jira_pat()


class TestJiraServiceStartup:
    def test_main_ensures_trigger_schema_before_polling(self, monkeypatch):
        order = []
        factory = SimpleNamespace(connection_pool=SimpleNamespace())
        agent_config = SimpleNamespace()
        trigger_store_instances = []

        class FakeTriggerStore:
            def __init__(self, postgres_factory):
                assert postgres_factory is factory
                trigger_store_instances.append(self)

            def ensure_schema(self):
                order.append("ensure_schema")

        class FakeResponderService:
            def __init__(self, **kwargs):
                assert kwargs["postgres_factory"] is factory
                assert kwargs["trigger_store"] is trigger_store_instances[0]
                assert kwargs["archi_instance"] == "archi"
                assert kwargs["agent_config"] is agent_config

            def poll_once(self):
                order.append("poll_once")
                raise StopIteration

        monkeypatch.setattr(service_jira, "read_secret", lambda name: "pg-password")
        monkeypatch.setattr(
            service_jira.PostgresServiceFactory,
            "from_env",
            Mock(return_value=factory),
        )
        monkeypatch.setattr(
            service_jira.PostgresServiceFactory,
            "set_instance",
            Mock(),
        )
        monkeypatch.setattr(
            service_jira, "get_services_config", lambda: _valid_services_config()
        )
        monkeypatch.setattr(service_jira, "resolve_jira_pat", Mock(return_value="pat"))
        monkeypatch.setattr(
            service_jira.responder_config,
            "resolve_jira_agent_config",
            Mock(return_value=agent_config),
        )
        monkeypatch.setattr(
            service_jira.archi_factory,
            "build_archi_for_jira",
            Mock(return_value="archi"),
        )
        monkeypatch.setattr(
            service_jira, "JiraIssueClient", Mock(return_value="issue-client")
        )
        monkeypatch.setattr(
            service_jira.responder_store, "JiraTriggerStore", FakeTriggerStore
        )
        monkeypatch.setattr(
            service_jira.responder_service,
            "JiraTicketResponderService",
            FakeResponderService,
        )

        with pytest.raises(StopIteration):
            service_jira.main()

        assert order == ["ensure_schema", "poll_once"]
        service_jira.responder_config.resolve_jira_agent_config.assert_called_once_with(
            _valid_services_config()
        )
        service_jira.archi_factory.build_archi_for_jira.assert_called_once_with(
            agent_config
        )

    def test_main_fails_before_polling_when_schema_ensure_fails(self, monkeypatch):
        order = []
        factory = SimpleNamespace(connection_pool=SimpleNamespace())
        issue_client_cls = Mock(return_value="issue-client")
        responder_cls = Mock()

        class FailingTriggerStore:
            def __init__(self, postgres_factory):
                assert postgres_factory is factory

            def ensure_schema(self):
                order.append("ensure_schema")
                raise RuntimeError("schema failed")

        monkeypatch.setattr(service_jira, "read_secret", lambda name: "pg-password")
        monkeypatch.setattr(
            service_jira.PostgresServiceFactory,
            "from_env",
            Mock(return_value=factory),
        )
        monkeypatch.setattr(
            service_jira.PostgresServiceFactory,
            "set_instance",
            Mock(),
        )
        monkeypatch.setattr(
            service_jira, "get_services_config", lambda: _valid_services_config()
        )
        monkeypatch.setattr(service_jira, "resolve_jira_pat", Mock(return_value="pat"))
        monkeypatch.setattr(
            service_jira.responder_store, "JiraTriggerStore", FailingTriggerStore
        )
        monkeypatch.setattr(service_jira, "JiraIssueClient", issue_client_cls)
        monkeypatch.setattr(
            service_jira.responder_service, "JiraTicketResponderService", responder_cls
        )

        with pytest.raises(RuntimeError, match="schema failed"):
            service_jira.main()

        assert order == ["ensure_schema"]
        issue_client_cls.assert_not_called()
        responder_cls.assert_not_called()


class TestJiraServiceBuilder:
    def test_build_compose_config_enables_jira_and_dependencies(self, tmp_path):
        plan = ServiceBuilder.build_compose_config(
            name="demo",
            verbosity=3,
            base_dir=tmp_path,
            enabled_services=["jira_ticket_responder"],
            secrets={"PG_PASSWORD", "JIRA_TICKET_RESPONDER_PAT"},
            tag="dev",
        )

        assert set(plan.get_enabled_services()) == {
            "data-manager",
            "postgres",
            "jira_ticket_responder",
        }
        assert (
            plan.get_service("jira_ticket_responder").container_name
            == "jira_ticket_responder-demo"
        )
        assert (
            plan.get_service("jira_ticket_responder").image_name
            == "jira_ticket_responder-demo"
        )
        assert plan.get_service("jira_ticket_responder").required_secrets == [
            "JIRA_TICKET_RESPONDER_PAT"
        ]


class TestJiraConfigValidation:
    def test_base_config_renders_jira_agent_and_model_defaults(self):
        rendered = (
            _template_env()
            .get_template("base-config.yaml")
            .render(
                services={
                    "jira_ticket_responder": {
                        "url": "https://jira.example/",
                        "projects": ["CMSTZ", "CMSDM"],
                        "visible_to_role": "Developers",
                    },
                    "chat_app": {
                        "default_provider": "openai",
                        "default_model": "gpt-5",
                    },
                }
            )
        )

        config = yaml.safe_load(rendered)

        assert "jira" not in config["services"]
        jira_config = config["services"]["jira_ticket_responder"]
        assert jira_config["agent_class"] == "CMSCompOpsAgent"
        assert jira_config["default_provider"] == "openai"
        assert jira_config["default_model"] == "gpt-5"
        assert jira_config["poll_interval_minutes"] == 1
        assert jira_config["lookback_days"] == 7
        assert jira_config["respond_to_mentions"] is False
        assert jira_config["mention_allowed_roles"] == []
        assert jira_config["eligible_statuses"] == ["Open", "In Progress"]
        assert jira_config["projects"] == ["CMSTZ", "CMSDM"]

    def test_base_config_renders_explicit_respond_to_mentions(self):
        rendered = (
            _template_env()
            .get_template("base-config.yaml")
            .render(
                services={
                    "jira_ticket_responder": {
                        "url": "https://jira.example/",
                        "projects": ["CMSTZ"],
                        "visible_to_role": "Developers",
                        "respond_to_mentions": True,
                        "mention_allowed_roles": ["Developers", "Administrators"],
                    },
                }
            )
        )

        config = yaml.safe_load(rendered)

        assert (
            config["services"]["jira_ticket_responder"]["respond_to_mentions"] is True
        )
        assert config["services"]["jira_ticket_responder"]["mention_allowed_roles"] == [
            "Developers",
            "Administrators",
        ]

    def test_base_config_renders_public_visibility_when_role_is_omitted(self):
        rendered = (
            _template_env()
            .get_template("base-config.yaml")
            .render(
                services={
                    "jira_ticket_responder": {
                        "url": "https://jira.example/",
                        "projects": ["CMSTZ"],
                    },
                }
            )
        )

        config = yaml.safe_load(rendered)

        assert config["services"]["jira_ticket_responder"]["visible_to_role"] == ""

    def test_base_config_renders_explicit_jira_agent_class(self):
        rendered = (
            _template_env()
            .get_template("base-config.yaml")
            .render(
                services={
                    "jira_ticket_responder": {
                        "url": "https://jira.example/",
                        "projects": ["CMSTZ"],
                        "visible_to_role": "Developers",
                        "poll_interval_minutes": 1,
                        "agent_class": "CustomPipeline",
                    },
                }
            )
        )

        config = yaml.safe_load(rendered)

        assert (
            config["services"]["jira_ticket_responder"]["agent_class"]
            == "CustomPipeline"
        )
        assert "pipeline" not in config["services"]["jira_ticket_responder"]

    def test_validate_jira_config_accepts_required_fields(self, tmp_path):
        config_path = _write_config(
            tmp_path,
            {
                "jira_ticket_responder": {
                    "url": "https://jira.example/",
                    "projects": ["CMSTZ", "CMSDM"],
                    "visible_to_role": "Developers",
                    "poll_interval_minutes": 1,
                }
            },
        )
        manager = ConfigurationManager([str(config_path)], _template_env())

        manager.validate_configs(["jira_ticket_responder"], [])

    def test_validate_jira_config_accepts_missing_poll_interval(self, tmp_path):
        config_path = _write_config(
            tmp_path,
            {
                "jira_ticket_responder": {
                    "url": "https://jira.example/",
                    "projects": ["CMSTZ"],
                    "visible_to_role": "Developers",
                }
            },
        )
        manager = ConfigurationManager([str(config_path)], _template_env())

        manager.validate_configs(["jira_ticket_responder"], [])

    def test_validate_jira_config_accepts_explicit_lookback_days(self, tmp_path):
        config_path = _write_config(
            tmp_path,
            {
                "jira_ticket_responder": {
                    "url": "https://jira.example/",
                    "projects": ["CMSTZ"],
                    "visible_to_role": "Developers",
                    "lookback_days": 14,
                }
            },
        )
        manager = ConfigurationManager([str(config_path)], _template_env())

        manager.validate_configs(["jira_ticket_responder"], [])

    def test_validate_jira_config_accepts_explicit_eligible_statuses(self, tmp_path):
        config_path = _write_config(
            tmp_path,
            {
                "jira_ticket_responder": {
                    "url": "https://jira.example/",
                    "projects": ["CMSTZ"],
                    "visible_to_role": "Developers",
                    "eligible_statuses": ["Open", "Triaged"],
                }
            },
        )
        manager = ConfigurationManager([str(config_path)], _template_env())

        manager.validate_configs(["jira_ticket_responder"], [])

    def test_validate_jira_config_accepts_explicit_respond_to_mentions(self, tmp_path):
        config_path = _write_config(
            tmp_path,
            {
                "jira_ticket_responder": {
                    "url": "https://jira.example/",
                    "projects": ["CMSTZ"],
                    "visible_to_role": "Developers",
                    "respond_to_mentions": True,
                }
            },
        )
        manager = ConfigurationManager([str(config_path)], _template_env())

        manager.validate_configs(["jira_ticket_responder"], [])

    def test_validate_jira_config_accepts_optional_visibility_and_mention_roles(
        self, tmp_path
    ):
        config_path = _write_config(
            tmp_path,
            {
                "jira_ticket_responder": {
                    "url": "https://jira.example/",
                    "projects": ["CMSTZ"],
                    "respond_to_mentions": True,
                    "mention_allowed_roles": ["Developers"],
                }
            },
        )
        manager = ConfigurationManager([str(config_path)], _template_env())

        manager.validate_configs(["jira_ticket_responder"], [])

    @pytest.mark.parametrize(
        "value",
        ["Developers", [""], ["Developers", 7], [None]],
    )
    def test_validate_jira_config_rejects_invalid_mention_allowed_roles(
        self, tmp_path, value
    ):
        config_path = _write_config(
            tmp_path,
            {
                "jira_ticket_responder": {
                    "url": "https://jira.example/",
                    "projects": ["CMSTZ"],
                    "mention_allowed_roles": value,
                }
            },
        )
        manager = ConfigurationManager([str(config_path)], _template_env())

        with pytest.raises(
            ValueError,
            match="services.jira_ticket_responder.mention_allowed_roles",
        ):
            manager.validate_configs(["jira_ticket_responder"], [])

    @pytest.mark.parametrize(
        "projects",
        [
            [],
            "CMSTZ",
            ["CMSTZ", ""],
            ["CMSTZ", 7],
            ["CMSTZ, CMSDM"],
            ["cms"],
            ["2013PROJECT"],
            ["PRODUCT-2012"],
        ],
    )
    def test_validate_jira_config_rejects_invalid_projects(self, tmp_path, projects):
        config_path = _write_config(
            tmp_path,
            {
                "jira_ticket_responder": {
                    "url": "https://jira.example/",
                    "projects": projects,
                    "visible_to_role": "Developers",
                }
            },
        )
        manager = ConfigurationManager([str(config_path)], _template_env())

        with pytest.raises(ValueError, match="services.jira_ticket_responder.projects"):
            manager.validate_configs(["jira_ticket_responder"], [])

    def test_validate_jira_config_rejects_invalid_poll_interval(self, tmp_path):
        config_path = _write_config(
            tmp_path,
            {
                "jira_ticket_responder": {
                    "url": "https://jira.example/",
                    "projects": ["CMSTZ"],
                    "visible_to_role": "Developers",
                    "poll_interval_minutes": 0,
                }
            },
        )
        manager = ConfigurationManager([str(config_path)], _template_env())

        with pytest.raises(
            ValueError, match="services.jira_ticket_responder.poll_interval_minutes"
        ):
            manager.validate_configs(["jira_ticket_responder"], [])

    @pytest.mark.parametrize("value", [0, -1, "x", True])
    def test_validate_jira_config_rejects_invalid_lookback_days(self, tmp_path, value):
        config_path = _write_config(
            tmp_path,
            {
                "jira_ticket_responder": {
                    "url": "https://jira.example/",
                    "projects": ["CMSTZ"],
                    "visible_to_role": "Developers",
                    "lookback_days": value,
                }
            },
        )
        manager = ConfigurationManager([str(config_path)], _template_env())

        with pytest.raises(
            ValueError, match="services.jira_ticket_responder.lookback_days"
        ):
            manager.validate_configs(["jira_ticket_responder"], [])

    @pytest.mark.parametrize("value", ["true", "false", 1, 0, None])
    def test_validate_jira_config_rejects_non_boolean_respond_to_mentions(
        self, tmp_path, value
    ):
        config_path = _write_config(
            tmp_path,
            {
                "jira_ticket_responder": {
                    "url": "https://jira.example/",
                    "projects": ["CMSTZ"],
                    "visible_to_role": "Developers",
                    "respond_to_mentions": value,
                }
            },
        )
        manager = ConfigurationManager([str(config_path)], _template_env())

        with pytest.raises(
            ValueError, match="services.jira_ticket_responder.respond_to_mentions"
        ):
            manager.validate_configs(["jira_ticket_responder"], [])


class TestJiraSecrets:
    def test_jira_service_requires_service_pat_only_at_deployment(self, tmp_path):
        env_path = tmp_path / ".env"
        env_path.write_text(
            "PG_PASSWORD=postgres\n" "JIRA_TICKET_RESPONDER_PAT=service-token\n"
        )
        config_manager = _FakeConfigManager(
            {
                "services": {
                    "jira_ticket_responder": {},
                    "chat_app": {"default_provider": "openai"},
                }
            }
        )
        manager = SecretsManager(str(env_path), config_manager)

        required = manager.get_required_secrets_for_services({"jira_ticket_responder"})

        assert required == {"PG_PASSWORD", "JIRA_TICKET_RESPONDER_PAT"}
        manager.validate_secrets(required)
        deployment_dir = tmp_path / "deployment"
        deployment_dir.mkdir()
        manager.write_secrets_to_files(deployment_dir, required)
        assert (
            tmp_path / "deployment/secrets/jira_ticket_responder_pat.txt"
        ).read_text() == "service-token"

    def test_jira_service_auth_fails_when_service_pat_is_missing(self, tmp_path):
        env_path = tmp_path / ".env"
        env_path.write_text("PG_PASSWORD=postgres\nJIRA_PAT=source-token\n")
        config_manager = _FakeConfigManager(
            {
                "services": {
                    "jira_ticket_responder": {},
                    "chat_app": {"default_provider": "local"},
                }
            }
        )
        manager = SecretsManager(str(env_path), config_manager)

        required = manager.get_required_secrets_for_services({"jira_ticket_responder"})

        with pytest.raises(ValueError, match="JIRA_TICKET_RESPONDER_PAT"):
            manager.validate_secrets(required)

    def test_jira_provider_api_secret_is_not_required_by_deployment_validation(
        self, tmp_path
    ):
        env_path = tmp_path / ".env"
        env_path.write_text(
            "PG_PASSWORD=postgres\nJIRA_TICKET_RESPONDER_PAT=service-token\n"
        )
        config_manager = _FakeConfigManager(
            {
                "services": {
                    "jira_ticket_responder": {},
                    "chat_app": {"default_provider": "openai"},
                }
            }
        )
        manager = SecretsManager(str(env_path), config_manager)

        required = manager.get_required_secrets_for_services({"jira_ticket_responder"})

        assert required == {"PG_PASSWORD", "JIRA_TICKET_RESPONDER_PAT"}
        manager.validate_secrets(required)


class TestJiraComposeTemplate:
    def test_compose_template_renders_jira_service(self, tmp_path):
        plan = ServiceBuilder.build_compose_config(
            name="demo",
            verbosity=3,
            base_dir=tmp_path,
            enabled_services=["chatbot", "jira_ticket_responder"],
            secrets={"PG_PASSWORD", "JIRA_TICKET_RESPONDER_PAT", "OPENAI_API_KEY"},
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
            prompt_files=[],
            rubrics=[],
            mcp_servers={},
            host_file_mounts=[],
        )

        rendered = (
            _template_env().get_template("base-compose.yaml").render(**template_vars)
        )
        compose = yaml.safe_load(rendered)

        jira_service = compose["services"]["jira_ticket_responder"]
        assert jira_service["container_name"] == "jira_ticket_responder-demo"
        assert (
            jira_service["build"]["dockerfile"]
            == "archi_code/cli/templates/dockerfiles/Dockerfile-jira"
        )
        assert jira_service["depends_on"]["postgres"]["condition"] == "service_healthy"
        assert (
            jira_service["depends_on"]["config-seed"]["condition"]
            == "service_completed_successfully"
        )
        assert (
            jira_service["environment"]["JIRA_TICKET_RESPONDER_PAT_FILE"]
            == "/run/secrets/jira_ticket_responder_pat"
        )
