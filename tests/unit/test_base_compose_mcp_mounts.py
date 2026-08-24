from pathlib import Path

import yaml
from jinja2 import ChainableUndefined, Environment, FileSystemLoader

from src.cli.utils.service_builder import ServiceBuilder


def _render_compose(tmp_path, host_file_mounts, *, evaluation_mcp_configured=False):
    repository = Path(__file__).resolve().parents[2]
    environment = Environment(
        loader=FileSystemLoader(str(repository / "src/cli/templates")),
        undefined=ChainableUndefined,
    )
    plan = ServiceBuilder.build_compose_config(
        name="demo",
        verbosity=3,
        base_dir=tmp_path,
        enabled_services=["chatbot"],
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
        prompt_files=[],
        rubrics=[],
        mcp_servers={
            "example": {
                "transport": "stdio",
                "host_file_mounts": host_file_mounts,
            }
        },
        evaluation_mcp_configured=evaluation_mcp_configured,
    )

    rendered = environment.get_template("base-compose.yaml").render(**template_vars)
    return yaml.safe_load(rendered)


class TestChatbotMcpHostFileMounts:
    def test_renders_string_and_structured_mounts(self, tmp_path):
        compose = _render_compose(
            tmp_path,
            [
                "/host/same-path",
                {
                    "src": "/host/writable-source",
                    "dest": "/container/writable-destination",
                    "read_only": False,
                },
                {
                    "src": "/host/read-only-source",
                    "dest": "/container/read-only-destination",
                    "read_only": True,
                },
                {
                    "src": "/host/default-source",
                    "dest": "/container/default-destination",
                },
            ],
        )

        assert compose["services"]["chatbot"]["volumes"][-4:] == [
            "/host/same-path:/host/same-path:ro",
            "/host/writable-source:/container/writable-destination",
            "/host/read-only-source:/container/read-only-destination:ro",
            "/host/default-source:/container/default-destination:ro",
        ]

    def test_mounts_staged_evaluation_registry_only_when_configured(self, tmp_path):
        configured = _render_compose(
            tmp_path,
            [],
            evaluation_mcp_configured=True,
        )
        omitted = _render_compose(tmp_path, [])

        mount = "./evaluation_config:/root/archi/evaluation_config:ro"
        assert mount in configured["services"]["chatbot"]["volumes"]
        assert mount not in omitted["services"]["chatbot"]["volumes"]
