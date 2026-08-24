from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from jinja2 import ChainableUndefined, Environment, FileSystemLoader

from src.cli.managers.templates_manager import TemplateManager

REGISTRY_YAML = """schema_version: qa-evaluation-mcp-v1
servers:
  dbs:
    transport: streamable_http
    url: http://localhost:8013/mcp
    authentication: {mode: none}
"""


class _FakeConfigManager:
    def __init__(self, config):
        self.config = config

    def get_configs(self):
        return [self.config]


def _template_manager():
    repository = Path(__file__).resolve().parents[2]
    environment = Environment(
        loader=FileSystemLoader(str(repository / "src/cli/templates")),
        undefined=ChainableUndefined,
    )
    return TemplateManager(environment, verbosity=0)


def _context(tmp_path, config, *, helm=False):
    return SimpleNamespace(
        base_dir=tmp_path / "deployment",
        config_manager=_FakeConfigManager(config),
        plan=SimpleNamespace(host_mode=False, verbosity=0, name="demo"),
        benchmarking=False,
        helm=helm,
    )


def _config(config_path, mcp_config_path):
    return {
        "_config_path": str(config_path),
        "name": "demo",
        "global": {"LOGGING": {}},
        "utils": {},
        "services": {
            "chat_app": {
                "evaluations": {
                    "enabled": True,
                    "mcp_config_path": mcp_config_path,
                }
            }
        },
    }


class TestEvaluationMCPConfigStaging:
    def test_stages_relative_source_and_renders_container_path(self, tmp_path):
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        config_path = source_dir / "archi.yaml"
        registry_path = source_dir / "evaluator.yaml"
        registry_path.write_text(REGISTRY_YAML, encoding="utf-8")
        context = _context(tmp_path, _config(config_path, "evaluator.yaml"))
        manager = _template_manager()

        manager._stage_evaluation_config(context)
        manager._render_config_files(context)

        staged_path = context.base_dir / "evaluation_config" / "qa_evaluation_mcp.yaml"
        assert staged_path.read_text(encoding="utf-8") == REGISTRY_YAML
        assert context.evaluation_mcp_configured is True
        rendered = yaml.safe_load(
            (context.base_dir / "configs" / "config.yaml").read_text(encoding="utf-8")
        )
        assert (
            rendered["services"]["chat_app"]["evaluations"]["mcp_config_path"]
            == "/root/archi/evaluation_config/qa_evaluation_mcp.yaml"
        )

    def test_explicit_missing_source_fails_instead_of_staging_empty_registry(
        self, tmp_path
    ):
        config_path = tmp_path / "source" / "archi.yaml"
        context = _context(tmp_path, _config(config_path, "missing.yaml"))

        with pytest.raises(
            ValueError,
            match="Evaluator MCP configuration file not found",
        ):
            _template_manager()._stage_evaluation_config(context)

        assert not (context.base_dir / "evaluation_config").exists()

    def test_invalid_source_fails_before_it_is_copied(self, tmp_path):
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        config_path = source_dir / "archi.yaml"
        registry_path = source_dir / "invalid.yaml"
        registry_path.write_text("servers: {}\n", encoding="utf-8")
        context = _context(tmp_path, _config(config_path, "invalid.yaml"))

        with pytest.raises(ValueError, match="missing: schema_version"):
            _template_manager()._stage_evaluation_config(context)

        assert not (context.base_dir / "evaluation_config").exists()

    def test_omitted_source_removes_managed_stale_snapshot(self, tmp_path):
        config_path = tmp_path / "source" / "archi.yaml"
        config = _config(config_path, None)
        context = _context(tmp_path, config)
        staged_dir = context.base_dir / "evaluation_config"
        staged_dir.mkdir(parents=True)
        staged_path = staged_dir / "qa_evaluation_mcp.yaml"
        staged_path.write_text(REGISTRY_YAML, encoding="utf-8")

        _template_manager()._stage_evaluation_config(context)

        assert not staged_path.exists()
        assert context.evaluation_mcp_configured is False

    def test_helm_stages_registry_in_dedicated_configmap(self, tmp_path):
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        config_path = source_dir / "archi.yaml"
        registry_path = source_dir / "evaluator.yaml"
        registry_path.write_text(REGISTRY_YAML, encoding="utf-8")
        context = _context(
            tmp_path,
            _config(config_path, str(registry_path)),
            helm=True,
        )

        _template_manager()._stage_evaluation_config(context)

        configmap_path = (
            context.base_dir / "templates" / "chatbot-evaluation-config-configmap.yaml"
        )
        configmap = yaml.safe_load(configmap_path.read_text(encoding="utf-8"))
        assert configmap["metadata"]["name"] == "demo-evaluation-config"
        assert configmap["data"] == {"qa_evaluation_mcp.yaml": REGISTRY_YAML}
        assert context.evaluation_mcp_configured is True
        assert not (context.base_dir / "evaluation_config").exists()

    def test_staging_runs_before_runtime_config_rendering(self, tmp_path):
        context = _context(
            tmp_path,
            _config(tmp_path / "source" / "archi.yaml", None),
        )

        stages = _template_manager()._build_workflow(context)
        stage_names = [stage.__name__ for stage in stages]

        assert stage_names.index("_stage_evaluation_config") < stage_names.index(
            "_stage_configs"
        )
