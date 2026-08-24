import copy
import os
import shutil
import socket
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from jinja2 import Environment

from src.cli.service_registry import service_registry
from src.cli.utils.service_builder import DeploymentPlan
from src.cli.utils.grafana_styling import assign_feedback_palette
from src.cli.utils.helpers import HELM_PREFIX
from src.utils.ab_testing import DEFAULT_AB_AGENTS_DIR
from src.utils.logging import get_logger

logger = get_logger(__name__)


# Template file constants
BASE_CONFIG_TEMPLATE = "base-config.yaml"
BASE_COMPOSE_TEMPLATE = "base-compose.yaml"
BASE_INIT_SQL_TEMPLATE = "init.sql"  # PostgreSQL + pgvector schema
BASE_GRAFANA_DATASOURCES_TEMPLATE = "grafana/datasources.yaml"
BASE_GRAFANA_DASHBOARDS_TEMPLATE = "grafana/dashboards.yaml"
BASE_GRAFANA_ARCHI_DEFAULT_DASHBOARDS_TEMPLATE = "grafana/archi-default-dashboard.json"
BASE_GRAFANA_CONFIG_TEMPLATE = "grafana/grafana.ini"
DEPLOYMENT_AGENTS_DIR = "/root/archi/agents"
EVALUATION_CONFIG_DIR = "evaluation_config"
EVALUATION_MCP_CONFIG_FILENAME = "qa_evaluation_mcp.yaml"
EVALUATION_MCP_RUNTIME_PATH = (
    f"/root/archi/{EVALUATION_CONFIG_DIR}/{EVALUATION_MCP_CONFIG_FILENAME}"
)

HELM_CHAT_CONFIGMAP = "helm/templates/chatbot/configmap.yaml"
HELM_EVALUATION_CONFIGMAP = "helm/templates/chatbot/evaluation-configmap.yaml"
HELM_DM_CONFIGMAP = "helm/templates/data-manager/configmap.yaml"
HELM_POSTGRES_CONFIGMAP = "helm/templates/postgres/configmap.yaml"
HELM_GRAFANA_CONFIGMAP = "helm/templates/grafana/configmap.yaml"
HELM_CONFIG_SEED = "helm/templates/config-seed.yaml"
HELM_CHART_YAML_TEMPLATE = "helm/Chart.yaml"
HELM_VALUES_YAML_TEMPLATE = "helm/values.yaml"


def get_git_information() -> Dict[str, str]:

    meta_data: Dict[str, str] = {}
    wd = Path(__file__).parent

    if (
        subprocess.call(
            ["git", "branch"],
            cwd=wd,
            stderr=subprocess.STDOUT,
            stdout=open(os.devnull, "w"),
        )
        != 0
    ):
        meta_data["git_info"] = {
            "hash": "Not a git repository!",
            "diff": "Not a git repository",
        }
    else:
        meta_data["last_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=wd, encoding="UTF-8"
        )
        diff_comm = ["git", "diff"]
        meta_data["git_diff"] = subprocess.check_output(
            diff_comm, encoding="UTF-8", cwd=wd
        )
    return meta_data


def get_git_version() -> str:
    """Get the current git version using 'git describe --tags --always --dirty'."""
    
    try:
        version = subprocess.check_output(
            ["git", "describe", "--tags", "--always", "--dirty"],
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).parent
        ).strip().decode("utf-8")
        return version
    except Exception:
        return "unknown"


@dataclass
class TemplateContext:
    plan: DeploymentPlan
    config_manager: Any
    secrets_manager: Any
    options: Dict[str, Any]
    base_dir: Path = field(init=False)
    prompt_mappings: Dict[str, Dict[str, str]] = field(default_factory=dict)
    evaluation_mcp_configured: bool = False

    def __post_init__(self) -> None:
        self.base_dir = self.plan.base_dir

    def pop_option(self, key: str, default: Any = None) -> Any:
        return self.options.pop(key, default)

    def get_option(self, key: str, default: Any = None) -> Any:
        return self.options.get(key, default)

    @property
    def benchmarking(self) -> bool:
        return bool(self.options.get("benchmarking"))
    
    @property
    def helm(self) -> bool:
        return bool(self.options.get("helm"))
    


class TemplateManager:
    """Manages template rendering and file preparation using service registry"""

    def __init__(self, jinja_env: Environment, verbosity: int, helm: bool = False):
        self.env = jinja_env
        self.global_verbosity = verbosity
        self.registry = service_registry
        self._service_hooks: Dict[str, Callable[[TemplateContext], None]] = {
            "grafana": self._render_grafana_assets,
            "grader": self._copy_grader_assets,
        }
        self.helm = helm

    def prepare_deployment_files(
        self,
        plan: DeploymentPlan,
        config_manager,
        secrets_manager,
        **options,
    ) -> TemplateContext:
        context = TemplateContext(
            plan=plan,
            config_manager=config_manager,
            secrets_manager=secrets_manager,
            options=dict(options),
        )

        logger.info(
            f"Preparing deployment artifacts for `{plan.name}` in {str(context.base_dir)}"
        )

        for stage in self._build_workflow(context):
            logger.debug(f"Starting template stage {stage.__name__}")
            stage(context)
            logger.debug(f"Completed template stage {stage.__name__}")

        logger.info(f"Finished preparing deployment artifacts for {plan.name}")
        return context

    # workflow construction
    def _build_workflow(self, context: TemplateContext) -> List[Callable[[TemplateContext], None]]:
        stages: List[Callable[[TemplateContext], None]] = [
            self._stage_prompts,
            self._stage_agents, 
            self._stage_skills,
            self._stage_mcp_copy,
            self._stage_evaluation_config,
            self._stage_configs,
            self._stage_service_artifacts,
            self._stage_postgres_init,
            self._stage_compose,
            self._stage_web_lists,
            self._stage_source_copy,
        ]

        if context.benchmarking:
            stages.append(self._stage_benchmarking)

        if context.helm:
            stages.append(self._stage_chart)
            stages.append(self._stage_values)
            stages.append(self._stage_config_seed)
            stages.append(self._stage_tools)
            stages.remove(self._stage_compose) #Not needed for Helm deployments
            stages.remove(self._stage_source_copy)
            stages.remove(self._stage_mcp_copy) #Ignore for now, pending MCP sidecar implementation

        return stages

    # individual stages
    def _stage_prompts(self, context: TemplateContext) -> None:
        # Copy default prompt templates (condense/, chat/, system/ structure)
        if context.helm:
            self._helm_render_default_prompts(context)
        else:
            self._copy_default_prompts(context)
        context.prompt_mappings = {}

    def _stage_agents(self, context: TemplateContext) -> None:
        config = context.config_manager.config or {}
        dst_dir = context.base_dir / "data" / "agents"
        ab_dst_dir = context.base_dir / "data" / "ab_agents"
        services_cfg = config.get("services", {}) or {}
        agents_data = {}

        if context.benchmarking:
            benchmark_cfg = services_cfg.get("benchmarking", {}) or {}
            agent_md_file = benchmark_cfg.get("agent_md_file")
            if not agent_md_file:
                raise ValueError("Missing required services.benchmarking.agent_md_file in config.")
            source_path = Path(str(agent_md_file)).expanduser()
            config_path = Path(str(config.get("_config_path", ""))).expanduser()
            if not source_path.is_absolute() and config_path:
                candidate = (config_path.parent / source_path).resolve()
                if candidate.exists():
                    source_path = candidate
            if not source_path.exists() or not source_path.is_file():
                raise ValueError(f"Benchmark agent file not found: {source_path}")
            if source_path.suffix.lower() != ".md":
                raise ValueError(f"Benchmark agent file must be a .md file: {source_path}")

            if context.helm:
                with open(source_path, 'r') as f:
                    agents_data[source_path.name] = f.read()
                self._helm_render_agents(context, agents_data)
            else:
                dst_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_path, dst_dir / source_path.name)
            return

        agents_dir = (services_cfg.get("chat_app") or {}).get("agents_dir")
        if not agents_dir:
            if dst_dir.exists() and any(p.suffix.lower() == ".md" for p in dst_dir.iterdir()):
                return
            raise ValueError("Missing required services.chat_app.agents_dir in config.")
        src_dir = self._resolve_directory_path(str(agents_dir), config)

        if context.helm:
            for agent_file in list(src_dir.glob("*.md"))+list(src_dir.glob("*.py")):
                with open(agent_file, 'r') as f:
                    file_name = os.path.basename(agent_file)
                    agents_data[file_name] = f.read()
        else:
            self._copy_markdown_directory(
                src_dir,
                dst_dir,
                missing_message=f"Agents directory not found: {src_dir}",
                empty_message=f"No agent markdown files found in {src_dir}",
                required=True,
            )

        if not context.helm:
            ab_dst_dir.mkdir(parents=True, exist_ok=True)
        ab_cfg = ((services_cfg.get("chat_app") or {}).get("ab_testing") or {})
        ab_agents_dir = ab_cfg.get("ab_agents_dir")
        if not ab_agents_dir:
            if context.helm:
                #Saves the agents_data as fall back
                self._helm_render_agents(context, agents_data) 
            return
        ab_src_dir = self._resolve_directory_path(str(ab_agents_dir), config)
        
        if context.helm:
            for ab_agent_file in ab_src_dir.glob("*.md"):
                with open(ab_agent_file, 'r') as f:
                    file_name = os.path.basename(ab_agent_file)
                    agents_data[file_name] = f.read()
            self._helm_render_agents(context, agents_data)
        else:
            self._copy_markdown_directory(
                ab_src_dir,
                ab_dst_dir,
                missing_message=f"A/B agents directory not found: {ab_src_dir}",
                empty_message=f"No A/B agent markdown files found in {ab_src_dir}",
                required=False,
            )

        
    def _helm_render_agents(self, context: TemplateContext, agents_data: dict) -> None:
        chart_dir = context.base_dir
        tmpl = self.env.get_template(HELM_CHAT_CONFIGMAP)  
        helm_config = tmpl.render(agents=agents_data, archi_name=context.plan.name) 
        file_path = chart_dir / "templates/chatbot-agents-configmap.yaml"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path,"w") as f:
            f.write(helm_config)

    def _helm_render_skills(self, context: TemplateContext, skills_data: dict) -> None:
        chart_dir = context.base_dir
        tmpl = self.env.get_template(HELM_CHAT_CONFIGMAP)  
        helm_config = tmpl.render(skills=skills_data, archi_name=context.plan.name) 
        file_path = chart_dir / "templates/chatbot-skills-configmap.yaml"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path,"w") as f:
            f.write(helm_config)

    def _stage_chart(self, context: TemplateContext) -> None:
        chart_dir = context.base_dir
        tmpl = self.env.get_template(HELM_CHART_YAML_TEMPLATE)  
        rendered = tmpl.render(  
            name=context.plan.name,  
            app_version=get_git_version(),   
        )
        with open(chart_dir / "Chart.yaml","w") as f:
            f.write(rendered)

    def _stage_values(self, context: TemplateContext) -> None:

        template_vars = context.plan.to_template_vars()
        port_config = self._extract_port_config(context)
        allow_port_reuse = context.get_option("allow_port_reuse", False)
        self._check_ports_available(context, port_config, allow_port_reuse=allow_port_reuse)
        template_vars.update(port_config)
        template_vars.setdefault("postgres_port", context.config_manager.config.get("services", {}).get("postgres", {}).get("port", 5432))
        template_vars.setdefault("verbosity", self.global_verbosity)

        template_vars["app_version"] = get_git_version()

        # Compose template still expects optional lists
        template_vars.setdefault("prompt_files", [])
        template_vars.setdefault("rubrics", [])

        if context.plan.get_service("grader").enabled:
            template_vars["rubrics"] = self._get_grader_rubrics(context.config_manager)

        # Pass MCP server configs so compose can volume-mount stdio packages
        # and emit sidecar services for servers with build_context/image.
        mcp_servers = context.config_manager.config.get("mcp_servers", {}) or {}
        template_vars["mcp_servers"] = mcp_servers

        chart_dir = context.base_dir
        tmpl = self.env.get_template(HELM_VALUES_YAML_TEMPLATE)  

        rendered = tmpl.render(archi_name=context.plan.name,**template_vars)
        with open(chart_dir / "values.yaml","w") as f:
            f.write(rendered)

    def _stage_config_seed(self, context: TemplateContext) -> None:

        chart_dir = context.base_dir
        tmpl = self.env.get_template(HELM_CONFIG_SEED)  

        rendered = tmpl.render(name=context.plan.name)
        with open(chart_dir / "templates/config-seed.yaml","w") as f:
            f.write(rendered)

    @staticmethod
    def _resolve_directory_path(raw_path: str, config: Dict[str, Any]) -> Path:
        source_path = Path(str(raw_path)).expanduser()
        config_path_raw = config.get("_config_path", "")
        config_path = Path(str(config_path_raw)).expanduser() if config_path_raw else None
        if source_path.is_absolute() or not config_path:
            return source_path
        candidate = (config_path.parent / source_path).resolve()
        if candidate.exists():
            return candidate
        return source_path

    @staticmethod
    def _copy_markdown_directory(
        source_dir: Path,
        destination_dir: Path,
        *,
        missing_message: str,
        empty_message: str,
        required: bool,
    ) -> None:
        if not source_dir.exists() or not source_dir.is_dir():
            if required:
                raise ValueError(missing_message)
            logger.warning(missing_message)
            return
        destination_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        for source_file in sorted(source_dir.iterdir()):
            if source_file.is_file() and source_file.suffix.lower() == ".md":
                shutil.copyfile(source_file, destination_dir / source_file.name)
                copied += 1
        if copied == 0:
            if required:
                raise ValueError(empty_message)
            logger.warning(empty_message)

    def _stage_skills(self, context: TemplateContext) -> None:
        config = context.config_manager.config or {}
        services_cfg = config.get("services", {}) or {}
        skills_dir = (services_cfg.get("chat_app") or {}).get("skills_dir")
        skills_data = {}
        if not skills_dir:
            logger.debug("No skills_dir configured; skipping skills copy")
            return

        src_dir = Path(skills_dir).expanduser()
        if not src_dir.exists() or not src_dir.is_dir():
            logger.warning("Skills directory not found: %s", src_dir)
            return

        dst_dir = context.base_dir / "data" / "skills"
        dst_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        for skill_file in sorted(src_dir.iterdir()):
            if skill_file.is_file() and skill_file.suffix.lower() == ".md":
                if context.helm:
                    with open(skill_file, 'r') as f:
                        skills_data[skill_file.name] = f.read()
                else:
                    shutil.copyfile(skill_file, dst_dir / skill_file.name)
                copied += 1
        if copied:
            logger.info("Copied %d skill file(s) from %s", copied, src_dir)
        else:
            logger.warning("No skill markdown files found in %s", src_dir)

        if context.helm:
            self._helm_render_skills(context, skills_data)

    def _stage_tools(self, context: TemplateContext) -> None:
        #Only required for helm deployments for now 
        config = context.config_manager.config or {}
        services_cfg = config.get("services", {}) or {}
        tools_dir = (services_cfg.get("chat_app") or {}).get("tools_dir")
        tools_data = {}

        if not tools_dir:
            logger.debug("No tools_dir configured; skipping tools copy")
            return

        src_dir = Path(tools_dir).expanduser()
        if not src_dir.exists() or not src_dir.is_dir():
            logger.warning("Tools directory not found: %s", src_dir)
            return
        
        copied = 0
        for tool_file in sorted(src_dir.iterdir()):
            if tool_file.is_file() and tool_file.suffix.lower() == ".py":
                with open(tool_file, 'r') as f:
                    tools_data[tool_file.name] = f.read()
                copied += 1
        if copied:
            logger.info("Copied %d tool file(s) from %s", copied, src_dir)
        else:
            logger.warning("No tool python files found in %s", src_dir)

        chart_dir = context.base_dir
        tmpl = self.env.get_template(HELM_CHAT_CONFIGMAP)  
        helm_config = tmpl.render(tools=tools_data, archi_name=context.plan.name) 
        file_path = chart_dir / "templates/chatbot-tools-configmap.yaml"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path,"w") as f:
            f.write(helm_config)
    def _stage_mcp_copy(self, context: TemplateContext) -> None:
        """Make MCP sidecar build contexts available inside the deployment dir.

        The compose template renders ``build: <build_context>`` verbatim and
        ``docker compose build`` resolves that relative to the deployment dir,
        so any source-built sidecar needs its build context present there. For
        each ``mcp_server`` that builds from source (has ``build_context``), the
        value can take three forms:

        * **Git source** (a mapping ``{repo, ref?, subdir?}``): the repo is
          cloned fresh on every ``archi create`` (default ``ref: main``) and the
          chosen ``subdir`` is copied into ``<base_dir>/mcp_build/<name>``. This
          is *not* pinned/vendored — configuring the sidecar pulls the latest.
        * **A local path inside the shipped source tree** (``archi_code/...``):
          left as-is — :meth:`copy_source_code` already ships ``src`` as
          ``archi_code``.
        * **Any other local path** (absolute or arbitrary dir): resolved at
          create time and copied into ``<base_dir>/mcp_build/<name>``.

        In the two copy/clone cases ``build_context`` is rewritten to the
        deployment-relative ``./mcp_build/<name>`` so the deployment is portable.
        Image-based sidecars (no ``build_context``) are untouched.
        """
        config = context.config_manager.config or {}
        mcp_servers = config.get("mcp_servers") or {}
        if not mcp_servers:
            return

        dest_root = context.base_dir / "mcp_build"
        rewrites: Dict[str, str] = {}

        for name, server_cfg in mcp_servers.items():
            if not isinstance(server_cfg, dict):
                continue
            build_context = server_cfg.get("build_context")
            if not build_context:
                # Image-based sidecar (or no build at all); nothing to fetch.
                continue

            dest_dir = dest_root / name

            if isinstance(build_context, dict):
                # Git source: cloned fresh on every create (not pinned/vendored).
                self._clone_mcp_build_context(name, build_context, dest_dir)
                rewrites[name] = f"./mcp_build/{name}"
                continue

            # Local path already shipped inside archi_code via copy_source_code.
            normalized = str(build_context).lstrip("./")
            if normalized == "archi_code" or normalized.startswith("archi_code/"):
                logger.debug(
                    f"MCP sidecar '{name}' builds from shipped source "
                    f"({build_context}); no copy needed"
                )
                continue

            src_dir = self._resolve_directory_path(str(build_context), config)
            if not src_dir.exists() or not src_dir.is_dir():
                raise ValueError(
                    f"MCP sidecar '{name}' build_context not found: {src_dir} "
                    f"(from build_context: {build_context})"
                )

            dest_root.mkdir(parents=True, exist_ok=True)
            if dest_dir.exists():
                shutil.rmtree(dest_dir)
            shutil.copytree(src_dir, dest_dir)

            rewrites[name] = f"./mcp_build/{name}"
            logger.info(
                f"Copied MCP sidecar '{name}' build context from {src_dir} to "
                f"{dest_dir} (build_context -> {rewrites[name]})"
            )

        # Propagate rewrites to every loaded config so the rendered compose.yaml
        # (which reads configs[0]) and each config.yaml agree on the path.
        for cfg in context.config_manager.get_configs():
            servers = cfg.get("mcp_servers") or {}
            for name, rewritten in rewrites.items():
                if isinstance(servers.get(name), dict):
                    servers[name]["build_context"] = rewritten

    def _stage_evaluation_config(self, context: TemplateContext) -> None:
        """Validate and stage the evaluator-owned MCP registry.

        The deployment configuration contains a host path. Runtime configuration
        always receives the fixed path where this stage mounts the validated
        snapshot into the chatbot container.
        """
        config = context.config_manager.config or {}
        services = config.get("services", {}) or {}
        chat_app = services.get("chat_app", {}) or {}
        evaluations = chat_app.get("evaluations", {}) or {}
        raw_path = evaluations.get("mcp_config_path")

        staged_path = (
            context.base_dir
            / EVALUATION_CONFIG_DIR
            / EVALUATION_MCP_CONFIG_FILENAME
        )
        helm_path = (
            context.base_dir
            / "templates"
            / "chatbot-evaluation-config-configmap.yaml"
        )

        if raw_path is None:
            context.evaluation_mcp_configured = False
            for managed_path in (staged_path, helm_path):
                if managed_path.exists() or managed_path.is_symlink():
                    managed_path.unlink()
            return
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(
                "services.chat_app.evaluations.mcp_config_path must be a "
                "non-empty string"
            )

        source_path = Path(raw_path).expanduser()
        if not source_path.is_absolute():
            config_path_raw = config.get("_config_path")
            if not config_path_raw:
                raise ValueError(
                    "Cannot resolve relative evaluator MCP configuration path "
                    "without the deployment configuration file path"
                )
            config_path = Path(str(config_path_raw)).expanduser()
            source_path = (config_path.parent / source_path).resolve()

        try:
            if not source_path.exists():
                raise ValueError(
                    f"Evaluator MCP configuration file not found: {source_path}"
                )
            if not source_path.is_file():
                raise ValueError(
                    f"Evaluator MCP configuration must be a file: {source_path}"
                )

            # Import locally to keep CLI module loading independent of the MCP
            # client until an evaluator registry is actually configured.
            from src.evaluation.qa.oracle_config import EvaluatorMCPRegistry

            EvaluatorMCPRegistry.load(source_path)
        except PermissionError:
            raise ValueError(
                f"Evaluator MCP configuration is not readable: {source_path}"
            ) from None

        context.evaluation_mcp_configured = True
        if context.helm:
            content = source_path.read_text(encoding="utf-8")
            if staged_path.exists() or staged_path.is_symlink():
                staged_path.unlink()
            template = self.env.get_template(HELM_EVALUATION_CONFIGMAP)
            rendered = template.render(
                archi_name=context.plan.name,
                content=content,
            )
            helm_path.parent.mkdir(parents=True, exist_ok=True)
            helm_path.write_text(rendered, encoding="utf-8")
        else:
            if helm_path.exists() or helm_path.is_symlink():
                helm_path.unlink()
            staged_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, staged_path)
            logger.info(
                "Staged evaluator MCP configuration from %s to %s",
                source_path,
                staged_path,
            )

    def _clone_mcp_build_context(self, name: str, spec: Dict[str, Any], dest_dir: Path) -> None:
        """Clone a git-sourced MCP build context into ``dest_dir``.

        ``spec`` is ``{repo, ref?, subdir?}``. ``ref`` defaults to ``main`` and
        may be a branch, tag, or commit SHA; ``subdir`` selects a path within the
        repo (the whole repo is used if omitted). The clone is shallow when the
        ref is a branch/tag and falls back to a full clone + checkout for a SHA.
        The source's ``.git`` directory is never copied into the build context.
        """
        repo = spec.get("repo")
        if not repo:
            raise ValueError(f"MCP sidecar '{name}' build_context is missing 'repo'")
        ref = str(spec.get("ref") or "main")
        subdir = spec.get("subdir")

        dest_dir.parent.mkdir(parents=True, exist_ok=True)
        if dest_dir.exists():
            shutil.rmtree(dest_dir)

        tmp_dir = Path(tempfile.mkdtemp(prefix=f"mcp-{name}-"))
        try:
            try:
                # Fast path: branch/tag refs allow a shallow single-branch clone.
                subprocess.run(
                    ["git", "clone", "--depth", "1", "--branch", ref, str(repo), str(tmp_dir)],
                    check=True, capture_output=True, text=True,
                )
            except subprocess.CalledProcessError:
                # Fall back for commit SHAs (not valid for --branch).
                shutil.rmtree(tmp_dir, ignore_errors=True)
                tmp_dir.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    ["git", "clone", str(repo), str(tmp_dir)],
                    check=True, capture_output=True, text=True,
                )
                subprocess.run(
                    ["git", "-C", str(tmp_dir), "checkout", ref],
                    check=True, capture_output=True, text=True,
                )

            src = tmp_dir / subdir if subdir else tmp_dir
            if not src.is_dir():
                raise ValueError(
                    f"MCP sidecar '{name}': subdir '{subdir}' not found in {repo}@{ref}"
                )
            shutil.copytree(src, dest_dir, ignore=shutil.ignore_patterns(".git"))
            logger.info(
                f"Cloned MCP sidecar '{name}' build context from {repo}@{ref}"
                + (f" (subdir {subdir})" if subdir else "")
                + f" -> {dest_dir}"
            )
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or "").strip() or str(exc)
            raise ValueError(
                f"MCP sidecar '{name}': failed to clone {repo}@{ref}: {detail}"
            ) from exc
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _copy_default_prompts(self, context: TemplateContext) -> None:
        """Copy default prompt templates to deployment for PromptService."""
        # Source from examples/defaults/prompts/ (not source code)
        repo_root = Path(__file__).parent.parent.parent.parent
        defaults_prompts_dir = repo_root / "examples" / "defaults" / "prompts"
        # Deploy to data/prompts/ (admin-editable location)
        deployment_prompts_dir = context.base_dir / "data" / "prompts"
        
        if not defaults_prompts_dir.exists():
            logger.warning(f"Default prompts directory not found: {defaults_prompts_dir}")
            return
        
        # Copy the entire prompts directory structure (condense/, chat/, system/)
        for prompt_type in ["condense", "chat", "system"]:
            src_dir = defaults_prompts_dir / prompt_type
            dst_dir = deployment_prompts_dir / prompt_type
            
            if src_dir.exists():
                dst_dir.mkdir(parents=True, exist_ok=True)
                for prompt_file in src_dir.glob("*.prompt"):
                    dst_file = dst_dir / prompt_file.name
                    if not dst_file.exists():  # Don't overwrite existing prompts
                        shutil.copyfile(prompt_file, dst_file)
                        logger.debug(f"Copied default prompt: {prompt_type}/{prompt_file.name}")

    def _helm_render_default_prompts(self, context: TemplateContext) -> None:
        # Source from examples/defaults/prompts/ (not source code)
        repo_root = Path(__file__).parent.parent.parent.parent
        defaults_prompts_dir = repo_root / "examples" / "defaults" / "prompts"

        dict_prompts = {
            "condense": {},
            "chat": {},
            "system": {}
        }
        for prompt_type in ["condense", "chat", "system"]:
            src_dir = defaults_prompts_dir / prompt_type
            
            if src_dir.exists():
                for prompt_file in src_dir.glob("*.prompt"):
                    with open(prompt_file, 'r') as f:
                        file_name = os.path.basename(prompt_file)
                        dict_prompts[prompt_type][file_name] = f.read()

        chart_dir = context.base_dir
        tmpl = self.env.get_template(HELM_CHAT_CONFIGMAP)  
        helm_config = tmpl.render(condense_prompts=dict_prompts["condense"],
                                  chat_prompts=dict_prompts["chat"],
                                  system_prompts=dict_prompts["system"],
                                  archi_name=context.plan.name) 
        file_path = chart_dir / "templates/chatbot-prompts-configmap.yaml"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path,"w") as f:
            f.write(helm_config)


    def _stage_configs(self, context: TemplateContext) -> None:
        self._render_config_files(context)

    def _stage_service_artifacts(self, context: TemplateContext) -> None:
        helm = context.helm
        if helm:
            enabled_services = context.plan.get_enabled_services()
            for service in enabled_services:
                chart_dir = context.base_dir / "templates" / f"{service}-service.yaml"
                tmpl = self.env.get_template(str(HELM_PREFIX / service / "service.yaml"))  
                helm_config = tmpl.render(name=context.plan.name) 
                with open(chart_dir,"w") as f:
                    f.write(helm_config)
        
        else:
            for name, hook in self._service_hooks.items():
                if context.plan.get_service(name).enabled:
                    logger.info(f"Rendering supplemental assets for service {name}")
                    hook(context)

    def _stage_postgres_init(self, context: TemplateContext) -> None:
        self._render_postgres_init(context)

    def _stage_compose(self, context: TemplateContext) -> None:
        self._render_compose_file(context)

    def _stage_web_lists(self, context: TemplateContext) -> None:
        if context.helm:
            self._helm_render_web_input_lists(context)
        else:
            self._copy_web_input_lists(context)

    def _stage_source_copy(self, context: TemplateContext) -> None:
        self.copy_source_code(context.base_dir)

    def _stage_benchmarking(self, context: TemplateContext) -> None:
        query_file = context.pop_option("query_file")
        if not query_file:
            logger.warning("Benchmarking requested but no query file provided; skipping copy")
        else:
            query_file_dest = context.base_dir / "queries.txt"
            shutil.copyfile(query_file, query_file_dest)

        git_info = get_git_information()
        git_info_path = context.base_dir / "git_info.yaml"

        import yaml

        with open(git_info_path, "w") as f:
            yaml.dump(git_info, f)

    # prompt preparation
    def _collect_prompt_mappings(self, context: TemplateContext) -> Dict[str, Dict[str, str]]:
        return {}

    def _copy_pipeline_prompts(
        self,
        base_dir: Path,
        prompts_config: Dict[str, Any],
        *,
        config_dir: Optional[Path] = None,
    ) -> Dict[str, str]:
        prompt_mappings: Dict[str, str] = {}

        for _, section_prompts in prompts_config.items():
            if not isinstance(section_prompts, dict):
                continue

            for prompt_key, prompt_path in section_prompts.items():
                if not prompt_path or prompt_path == "null":
                    continue

                source_path = Path(prompt_path).expanduser()
                if not source_path.is_absolute() and config_dir:
                    # Prefer config-relative paths but fall back to CWD if it already exists.
                    if not source_path.exists():
                        source_path = (config_dir / source_path).resolve()
                if not source_path.exists():
                    logger.warning(f"Prompt file not found: {prompt_path}")
                    continue

                target_path = base_dir / "data" / "prompts" / source_path.name
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_path, target_path)

                prompt_mappings[prompt_key] = f"/root/archi/data/prompts/{source_path.name}"
                logger.debug(f"Copied prompt {prompt_key} to {target_path}")

        return prompt_mappings

    # config rendering
    def _render_config_files(self, context: TemplateContext) -> None:
        configs_path = context.base_dir / "configs"
        benchmarking_enabled = bool(getattr(context, "benchmarking", False))
        helm = bool(getattr(context, "helm", False))
        if not helm:
            configs_path.mkdir(parents=True, exist_ok=True)

        archi_configs = context.config_manager.get_configs()
        single_mode = len(archi_configs) == 1
        config_data = {}
        for archi_config in archi_configs:
            name = archi_config["name"]
            updated_config = copy.deepcopy(archi_config)

            if context.plan.host_mode:
                updated_config["host_mode"] = context.plan.host_mode
                self._apply_host_mode_port_overrides(updated_config)

            services_cfg = updated_config.get("services", {})
            for service_name in ("chat_app", "redmine_mailbox", "piazza", "jira_ticket_responder"):
                service_cfg = services_cfg.get(service_name)
                if isinstance(service_cfg, dict):
                    service_cfg["agents_dir"] = DEPLOYMENT_AGENTS_DIR
                    if service_cfg.get("skills_dir"):
                        service_cfg["skills_dir"] = "/root/archi/skills"
                    if service_name == "chat_app":
                        evaluations_cfg = service_cfg.get("evaluations")
                        if isinstance(evaluations_cfg, dict):
                            evaluations_cfg["mcp_config_path"] = (
                                EVALUATION_MCP_RUNTIME_PATH
                                if context.evaluation_mcp_configured
                                else None
                            )
                        ab_cfg = service_cfg.get("ab_testing")
                        if isinstance(ab_cfg, dict) and ab_cfg.get("ab_agents_dir"):
                            ab_cfg["ab_agents_dir"] = DEFAULT_AB_AGENTS_DIR
            if benchmarking_enabled:
                benchmark_cfg = services_cfg.get("benchmarking")
                if isinstance(benchmark_cfg, dict):
                    agent_md_file = benchmark_cfg.get("agent_md_file")
                    if agent_md_file:
                        benchmark_cfg["agent_md_file"] = f"{DEPLOYMENT_AGENTS_DIR}/{Path(str(agent_md_file)).name}"
            if helm:
                updated_config.setdefault("utils", {}).setdefault("postgres", {})["host"] = "{{ .Values.archi.name }}-postgres"

            config_template = self.env.get_template(BASE_CONFIG_TEMPLATE)
            config_rendered = config_template.render(verbosity=context.plan.verbosity, **updated_config)

            target_name = "config.yaml" if single_mode else f"{name}.yaml"
            config_data[target_name] = config_rendered
            if not helm:
                with open(configs_path / target_name, "w") as f:
                    f.write(config_rendered)
                logger.info(f"Rendered  configuration file {configs_path / target_name}")

        if helm:
            chart_dir = context.base_dir
            tmpl = self.env.get_template(HELM_CHAT_CONFIGMAP)  
            helm_config = tmpl.render(configs=config_data, archi_name=context.plan.name) 
            file_path = chart_dir / "templates/chatbot-configmap.yaml"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path,"w") as f:
                f.write(helm_config)

    # service-specific assets
    def _render_grafana_assets(self, context: TemplateContext) -> None:
        base_dir = context.base_dir
        grafana_dir = base_dir / "grafana"
        grafana_dir.mkdir(exist_ok=True)

        grafana_pg_password = context.secrets_manager.get_secret("GRAFANA_PG_PASSWORD")
        postgres_port = context.config_manager.config.get("services", {}).get("postgres", {}).get("port", 5432)

        datasources_template = self.env.get_template(BASE_GRAFANA_DATASOURCES_TEMPLATE)
        datasources = datasources_template.render(
            grafana_pg_password=grafana_pg_password,
            host_mode=context.plan.host_mode,
            postgres_port=postgres_port,
        )
        dashboards_template = self.env.get_template(BASE_GRAFANA_DASHBOARDS_TEMPLATE)
        dashboards = dashboards_template.render()

        configs = context.config_manager.get_configs()
        palette = assign_feedback_palette(configs)

        dashboard_template = self.env.get_template(BASE_GRAFANA_ARCHI_DEFAULT_DASHBOARDS_TEMPLATE)
        dashboard = dashboard_template.render(
            feedback_palette=palette,
        )
        config_template = self.env.get_template(BASE_GRAFANA_CONFIG_TEMPLATE)
        grafana_config = config_template.render()

        if context.helm:
            grafana_dict = {}
            grafana_dict["datasources.yaml"] = datasources
            grafana_dict["dashboards.yaml"] = dashboards
            grafana_dict["archi-default-dashboard.json"] = dashboard
            grafana_dict["grafana.ini"] = grafana_config

            chart_dir = context.base_dir
            tmpl = self.env.get_template(HELM_GRAFANA_CONFIGMAP)  
            helm_config = tmpl.render(grafana_dict=grafana_dict, archi_name=context.plan.name) 
            file_path = chart_dir / "templates/grafana-configmap.yaml"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path,"w") as f:
                f.write(helm_config)
        
        else:
            with open(grafana_dir / "datasources.yaml", "w") as f:
                f.write(datasources)
            with open(grafana_dir / "dashboards.yaml", "w") as f:
                f.write(dashboards)
            with open(grafana_dir / "archi-default-dashboard.json", "w") as f:
                f.write(dashboard)
            with open(grafana_dir / "grafana.ini", "w") as f:
                f.write(grafana_config)

    def _copy_grader_assets(self, context: TemplateContext) -> None:
        archi_config = context.config_manager.get_configs()[0]
        grader_config = archi_config.get("services", {}).get("grader_app", {})

        users_csv_dir = grader_config.get("local_users_csv_dir")
        if users_csv_dir:
            users_csv_path = Path(users_csv_dir).expanduser() / "users.csv"
            if users_csv_path.exists():
                shutil.copyfile(users_csv_path, context.base_dir / "users.csv")

        rubric_dir = grader_config.get("local_rubric_dir")
        num_problems = grader_config.get("num_problems", 1)

        if rubric_dir:
            for problem in range(1, num_problems + 1):
                rubric_path = Path(rubric_dir).expanduser() / f"solution_with_rubric_{problem}.txt"
                if rubric_path.exists():
                    target_path = context.base_dir / f"solution_with_rubric_{problem}.txt"
                    shutil.copyfile(rubric_path, target_path)

    # postgres + compose rendering
    def _render_postgres_init(self, context: TemplateContext) -> None:
        helm = bool(getattr(context, "helm", False))
        grafana_enabled = context.plan.get_service("grafana").enabled
        grafana_pg_password = (
            context.secrets_manager.get_secret("GRAFANA_PG_PASSWORD") if grafana_enabled else ""
        )
        
        # PostgreSQL + pgvector schema
        init_sql_template = self.env.get_template(BASE_INIT_SQL_TEMPLATE)
        
        # Get embedding dimensions from data_manager config
        data_manager_config = context.config_manager.config.get("data_manager", {})
        embedding_class_map = data_manager_config.get("embedding_class_map", {})
        embedding_name = data_manager_config.get("embedding_name", "all-MiniLM-L6-v2")
        
        # Default dimensions based on common embedding models
        default_dimensions = {
            "all-MiniLM-L6-v2": 384,
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }
        embedding_dimensions = default_dimensions.get(embedding_name, 384)
        
        # Allow override from config
        if embedding_name in embedding_class_map:
            embedding_dimensions = embedding_class_map[embedding_name].get(
                "dimensions", embedding_dimensions
            )
        
        init_sql = init_sql_template.render(
            use_grafana=grafana_enabled,
            grafana_pg_password=grafana_pg_password,
            embedding_dimensions=embedding_dimensions,
            # Vector index settings (optional overrides)
            vector_index_type=data_manager_config.get("vector_index_type", "hnsw"),
            vector_index_hnsw_m=data_manager_config.get("vector_index_hnsw_m", 16),
            vector_index_hnsw_ef=data_manager_config.get("vector_index_hnsw_ef", 64),
        )
        dest = context.base_dir / "init.sql"

        if not helm:
            with open(dest, "w") as f:
                f.write(init_sql)
            logger.debug(f"Wrote PostgreSQL init script to {dest}")

        if helm:
            chart_dir = context.base_dir
            tmpl = self.env.get_template(HELM_POSTGRES_CONFIGMAP)  
            helm_config = tmpl.render(init_sql=init_sql,archi_name=context.plan.name) 
            with open(chart_dir / "templates/postgres-init-configmap.yaml","w") as f:
                f.write(helm_config)



    def _render_compose_file(self, context: TemplateContext) -> None:
        template_vars = context.plan.to_template_vars()
        port_config = self._extract_port_config(context)
        allow_port_reuse = context.get_option("allow_port_reuse", False)
        self._check_ports_available(context, port_config, allow_port_reuse=allow_port_reuse)
        template_vars.update(port_config)
        template_vars.setdefault("postgres_port", context.config_manager.config.get("services", {}).get("postgres", {}).get("port", 5432))
        template_vars.setdefault("verbosity", self.global_verbosity)

        template_vars["app_version"] = get_git_version()

        # Compose template still expects optional lists
        template_vars.setdefault("prompt_files", [])
        template_vars.setdefault("rubrics", [])

        if context.plan.get_service("grader").enabled:
            template_vars["rubrics"] = self._get_grader_rubrics(context.config_manager)

        # Pass MCP server configs so compose can volume-mount stdio packages
        # and emit sidecar services for servers with build_context/image.
        mcp_servers = context.config_manager.config.get("mcp_servers", {}) or {}
        template_vars["mcp_servers"] = mcp_servers
        template_vars["evaluation_mcp_configured"] = context.evaluation_mcp_configured

        compose_template = self.env.get_template(BASE_COMPOSE_TEMPLATE)
        compose_rendered = compose_template.render(**template_vars)

        dest = context.base_dir / "compose.yaml"
        with open(dest, "w") as f:
            f.write(compose_rendered)
        logger.info(f"Rendered compose file {dest}")

    def _extract_port_config(self, context: TemplateContext) -> Dict[str, Any]:
        port_config: Dict[str, Any] = {}
        host_mode = context.plan.host_mode
        base_config = (context.config_manager.get_configs() or [{}])[0]

        for service_name, service_def in self.registry.get_all_services().items():
            key_prefix = service_name.replace("-", "_")
            host_port = service_def.default_host_port
            container_port = service_def.default_container_port

            if service_def.port_config_path:
                try:
                    config_value: Any = base_config
                    for key in service_def.port_config_path.split('.'):
                        config_value = config_value[key]

                    host_port, container_port = self._resolve_ports_from_config(
                        config_value,
                        host_mode=host_mode,
                        host_default=host_port,
                        container_default=container_port,
                    )
                except (KeyError, TypeError):
                    pass

            if host_port:
                port_config[f"{key_prefix}_port_host"] = host_port
            if container_port:
                port_config[f"{key_prefix}_port_container"] = container_port

        return port_config

    def _check_ports_available(self, context: TemplateContext, port_config: Dict[str, Any], *, allow_port_reuse: bool = False) -> None:
        host_mode = context.plan.host_mode
        enabled_services = context.plan.get_enabled_services()
        base_config = (context.config_manager.get_configs() or [{}])[0]
        services_cfg = base_config.get("services", {}) if isinstance(base_config, dict) else {}

        port_usages: List[tuple[int, str, Optional[str]]] = []
        for service_name in enabled_services:
            if service_name not in self.registry.get_all_services():
                continue
            key_prefix = service_name.replace("-", "_")
            host_port = port_config.get(f"{key_prefix}_port_host")
            if host_port is None:
                continue
            service_def = self.registry.get_service(service_name)
            config_hint = self._service_port_config_hint(service_def, host_mode)
            port_usages.append(
                (self._normalize_port(host_port, service_name, config_hint), service_name, config_hint)
            )

        if host_mode and context.plan.get_service("postgres").enabled:
            postgres_port = services_cfg.get("postgres", {}).get("port", 5432)
            port_usages.append(
                (self._normalize_port(postgres_port, "postgres", "services.postgres.port"), "postgres", "services.postgres.port")
            )

        if not port_usages:
            return

        port_to_services: Dict[int, List[tuple[str, Optional[str]]]] = {}
        for port, service_name, config_hint in port_usages:
            port_to_services.setdefault(port, []).append((service_name, config_hint))

        errors: List[str] = []
        for port, services in sorted(port_to_services.items()):
            if len(services) > 1:
                details = ", ".join(
                    f"{service} ({hint})" if hint else service for service, hint in services
                )
                errors.append(f"Port {port} is assigned to multiple services: {details}")

        if not allow_port_reuse:
            for port, services in sorted(port_to_services.items()):
                error = self._probe_port(port)
                if error:
                    details = ", ".join(
                        f"{service} ({hint})" if hint else service for service, hint in services
                    )
                    errors.append(f"Port {port} is already in use ({details}): {error}")

        if errors:
            raise ValueError("Port check failed:\n" + "\n".join(errors))

    def _service_port_config_hint(self, service_def, host_mode: bool) -> Optional[str]:
        if not service_def.port_config_path:
            return None
        suffix = "port" if host_mode else "external_port"
        return f"{service_def.port_config_path}.{suffix}"

    def _normalize_port(self, port: Any, service_name: str, config_hint: Optional[str]) -> int:
        try:
            port_value = int(port)
        except (TypeError, ValueError):
            location = f" ({config_hint})" if config_hint else ""
            raise ValueError(f"Invalid port value '{port}' for {service_name}{location}")

        if port_value < 1 or port_value > 65535:
            location = f" ({config_hint})" if config_hint else ""
            raise ValueError(f"Port out of range for {service_name}{location}: {port_value}")

        return port_value

    def _probe_port(self, port: int) -> Optional[str]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("0.0.0.0", port))
            except OSError as exc:
                return str(exc)
        return None

    def _get_grader_rubrics(self, config_manager) -> List[str]:
        archi_config = config_manager.get_configs()[0]
        grader_config = archi_config.get('services', {}).get('grader_app', {})
        num_problems = grader_config.get('num_problems', 1)
        return [f"solution_with_rubric_{i}" for i in range(1, num_problems + 1)]

    def _apply_host_mode_port_overrides(self, config: Dict[str, Any]) -> None:
        """Normalize service ports in host mode using port/external_port only."""
        services_cfg = config.get("services", {})
        if not isinstance(services_cfg, dict):
            return

        for service_cfg in services_cfg.values():
            if not isinstance(service_cfg, dict):
                continue

            external = service_cfg.get("external_port")
            if external is not None:
                service_cfg["port"] = external

    def _resolve_ports_from_config(
        self,
        config_value: Any,
        *,
        host_mode: bool,
        host_default: Optional[int],
        container_default: Optional[int],
    ) -> tuple[Optional[int], Optional[int]]:
        """Extract host/container ports using the standardized keys."""
        host_port = host_default
        container_port = container_default

        if isinstance(config_value, dict):
            container_port = config_value.get("port", container_port)
            host_port = container_port if host_mode else config_value.get("external_port", host_port)
        else:
            host_port = config_value

        return host_port, container_port

    # input list / source copying helpers
    def _copy_web_input_lists(self, context: TemplateContext) -> None:
        # Always create weblists directory (required by Dockerfiles, even if empty)
        weblists_path = context.base_dir / "weblists"
        weblists_path.mkdir(exist_ok=True)
        logger.debug(f"Created weblists directory at {weblists_path}")
        
        input_lists = context.config_manager.get_input_lists()
        if not input_lists:
            return
        

        for input_list in input_lists:
            if os.path.exists(input_list):
                shutil.copyfile(input_list, weblists_path / os.path.basename(input_list))
                logger.debug(f"Copied input list {input_list}")
            else:
                logger.warning(f"Configured input list {input_list} not found; skipping")
    

    def _helm_render_web_input_lists(self, context: TemplateContext) -> None:
        input_lists = context.config_manager.get_input_lists()
        if not input_lists:
            return
        
        dict_input_lists = {}
        for input_list in input_lists:
            with open(input_list, 'r') as f:
                file_name = os.path.basename(input_list)
                dict_input_lists[file_name] = f.read()

        chart_dir = context.base_dir
        tmpl = self.env.get_template(HELM_DM_CONFIGMAP)  
        helm_config = tmpl.render(input_lists=dict_input_lists, archi_name=context.plan.name) 
        file_path = chart_dir / "templates/data-manager-configmap.yaml"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path,"w") as f:
            f.write(helm_config)



    def copy_source_code(self, base_dir: Path) -> None:
        # Try to locate the repository root in a robust way. Prefer CWD when
        # it contains expected marker files (pyproject.toml, LICENSE, .git)
        # — this is what the template/preview code typically uses. If CWD
        # doesn't look like the repo root, fall back to walking up from this
        # file's location. Avoid assuming a fixed number of parent hops which
        # breaks in PR-preview, installed-package, or temporary test layouts.

        try:
            import src.cli.utils._repository_info
            repo_root = Path(src.cli.utils._repository_info.REPO_PATH)
        except Exception as e:
            logger.warning(f"Could not import repository path information. {str(e)}",
                            "Falling back to current working directory.")
            repo_root = Path(__file__).resolve()

        source_files = [
            ("src", "archi_code"),
            ("pyproject.toml", "pyproject.toml"),
            ("LICENSE", "LICENSE"),
        ]

        for src, dst in source_files:
            src_path = repo_root / src
            dst_path = base_dir / dst
            logger.debug(f"Copying source from {src_path} to {dst_path}")
            if src_path.is_dir():
                if dst_path.exists():
                    shutil.rmtree(dst_path)
                shutil.copytree(src_path, dst_path)
            elif src_path.exists():
                shutil.copyfile(src_path, dst_path)
            else:
                raise FileNotFoundError(f"Source path {src_path} does not exist. Something went wrong in the repo structure.")
