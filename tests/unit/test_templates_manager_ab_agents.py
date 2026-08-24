from pathlib import Path
from types import SimpleNamespace

from jinja2 import ChainableUndefined, Environment, FileSystemLoader

from src.cli.managers.templates_manager import TemplateManager


class _FakeConfigManager:
    def __init__(self, config):
        self.config = config

    def get_configs(self):
        return [self.config]


def _template_manager():
    repo_root = Path(__file__).resolve().parents[2]
    env = Environment(
        loader=FileSystemLoader(str(repo_root / "src/cli/templates")),
        undefined=ChainableUndefined,
    )
    return TemplateManager(env, verbosity=0)


def test_stage_agents_copies_configured_ab_agents_dir(tmp_path):
    agents_src = tmp_path / "agents-src"
    ab_agents_src = tmp_path / "ab-agents-src"
    agents_src.mkdir()
    ab_agents_src.mkdir()
    (agents_src / "main.md").write_text("---\nname: Main\n---\nPrompt\n")
    (ab_agents_src / "variant.md").write_text("---\nname: Variant\nab_only: true\n---\nPrompt\n")

    config = {
        "services": {
            "chat_app": {
                "agents_dir": str(agents_src),
                "ab_testing": {
                    "enabled": True,
                    "ab_agents_dir": str(ab_agents_src),
                },
            }
        }
    }
    context = SimpleNamespace(
        config_manager=_FakeConfigManager(config),
        base_dir=tmp_path / "deployment",
        benchmarking=False,
        helm = False
    )

    _template_manager()._stage_agents(context)

    assert (context.base_dir / "data" / "agents" / "main.md").exists()
    assert (context.base_dir / "data" / "ab_agents" / "variant.md").exists()


def test_render_config_files_rewrites_ab_agents_dir_to_runtime_path(tmp_path):
    config = {
        "name": "ab-demo",
        "global": {"LOGGING": {}},
        "utils": {},
        "services": {
            "chat_app": {
                "agents_dir": str(tmp_path / "agents-src"),
                "ab_testing": {
                    "enabled": True,
                    "ab_agents_dir": str(tmp_path / "ab-agents-src"),
                },
            }
        },
    }
    context = SimpleNamespace(
        base_dir=tmp_path / "deployment",
        config_manager=_FakeConfigManager(config),
        plan=SimpleNamespace(host_mode=False, verbosity=0),
        evaluation_mcp_configured=False,
        helm = False,
    )

    _template_manager()._render_config_files(context)

    rendered = (context.base_dir / "configs" / "config.yaml").read_text()
    assert 'agents_dir: "/root/archi/agents"' in rendered
    assert 'ab_agents_dir: "/root/archi/ab_agents"' in rendered


def test_render_config_files_preserves_benchmarking_rewrite_with_explicit_flag(tmp_path):
    benchmark_agent = tmp_path / "bench-agent.md"
    benchmark_agent.write_text("---\nname: Bench\n---\nPrompt\n")

    config = {
        "name": "bench-demo",
        "global": {"LOGGING": {}},
        "utils": {},
        "services": {
            "benchmarking": {
                "agent_md_file": str(benchmark_agent),
            },
            "chat_app": {
                "agents_dir": str(tmp_path / "agents-src"),
            },
        },
    }
    context = SimpleNamespace(
        base_dir=tmp_path / "deployment",
        config_manager=_FakeConfigManager(config),
        plan=SimpleNamespace(host_mode=False, verbosity=0),
        benchmarking=True,
        evaluation_mcp_configured=False,
        helm = False,
    )

    _template_manager()._render_config_files(context)

    rendered = (context.base_dir / "configs" / "config.yaml").read_text()
    assert 'agent_md_file: "/root/archi/agents/bench-agent.md"' in rendered
