import copy
import os
import shutil
import socket
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from jinja2 import Environment

from src.cli.service_registry import service_registry
from src.cli.utils.service_builder import DeploymentPlan
from src.cli.utils.grafana_styling import assign_feedback_palette
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

    def __post_init__(self) -> None:
        self.base_dir = self.plan.base_dir

    def pop_option(self, key: str, default: Any = None) -> Any:
        return self.options.pop(key, default)

    def get_option(self, key: str, default: Any = None) -> Any:
        return self.options.get(key, default)

    @property
    def benchmarking(self) -> bool:
        return bool(self.options.get("benchmarking"))


class TemplateManager:
    """Manages template rendering and file preparation using service registry"""

    def __init__(self, jinja_env: Environment, verbosity: int):
        self.env = jinja_env
        self.global_verbosity = verbosity
        self.registry = service_registry
        self._service_hooks: Dict[str, Callable[[TemplateContext], None]] = {
            "grafana": self._render_grafana_assets,
            "grader": self._copy_grader_assets,
        }

    def prepare_deployment_files(
        self,
        plan: DeploymentPlan,
        config_manager,
        secrets_manager,
        **options,
    ) -> None:
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

    # workflow construction
    def _build_workflow(self, context: TemplateContext) -> List[Callable[[TemplateContext], None]]:
        stages: List[Callable[[TemplateContext], None]] = [
            self._stage_prompts,
            self._stage_agents,
            self._stage_configs,
            self._stage_service_artifacts,
            self._stage_postgres_init,
            self._stage_compose,
            self._stage_web_lists,
            self._stage_source_copy,
        ]

        if context.benchmarking:
            stages.append(self._stage_benchmarking)

        return stages

    # individual stages
    def _stage_prompts(self, context: TemplateContext) -> None:
        # Copy default prompt templates (condense/, chat/, system/ structure)
        self._copy_default_prompts(context)
        context.prompt_mappings = {}

    def _stage_agents(self, context: TemplateContext) -> None:
        config = context.config_manager.config or {}
        dst_dir = context.base_dir / "data" / "agents"
        services_cfg = config.get("services", {}) or {}

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
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, dst_dir / source_path.name)
            return

        agents_dir = (services_cfg.get("chat_app") or {}).get("agents_dir")
        if not agents_dir:
            if dst_dir.exists() and any(p.suffix.lower() == ".md" for p in dst_dir.iterdir()):
                return
            raise ValueError("Missing required services.chat_app.agents_dir in config.")
        src_dir = Path(agents_dir).expanduser()
        if not src_dir.exists() or not src_dir.is_dir():
            raise ValueError(f"Agents directory not found: {src_dir}")
        dst_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        for agent_file in sorted(src_dir.iterdir()):
            if agent_file.is_file() and agent_file.suffix.lower() == ".md":
                shutil.copyfile(agent_file, dst_dir / agent_file.name)
                copied += 1
        if copied == 0:
            raise ValueError(f"No agent markdown files found in {src_dir}")

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

    def _stage_configs(self, context: TemplateContext) -> None:
        self._render_config_files(context)

    def _stage_service_artifacts(self, context: TemplateContext) -> None:
        for name, hook in self._service_hooks.items():
            if context.plan.get_service(name).enabled:
                logger.info(f"Rendering supplemental assets for service {name}")
                hook(context)

    def _stage_postgres_init(self, context: TemplateContext) -> None:
        self._render_postgres_init(context)

    def _stage_compose(self, context: TemplateContext) -> None:
        self._render_compose_file(context)

    def _stage_web_lists(self, context: TemplateContext) -> None:
        self._copy_web_input_lists(context)

    def _stage_source_copy(self, context: TemplateContext) -> None:
        self._copy_source_code(context.base_dir)

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
        configs_path.mkdir(exist_ok=True)

        archi_configs = context.config_manager.get_configs()
        single_mode = len(archi_configs) == 1
        for archi_config in archi_configs:
            name = archi_config["name"]
            updated_config = copy.deepcopy(archi_config)

            if context.plan.host_mode:
                updated_config["host_mode"] = context.plan.host_mode
                self._apply_host_mode_port_overrides(updated_config)

            services_cfg = updated_config.get("services", {})
            for service_name in ("chat_app", "redmine_mailbox", "piazza"):
                service_cfg = services_cfg.get(service_name)
                if isinstance(service_cfg, dict):
                    service_cfg["agents_dir"] = "/root/archi/agents"
            if context.benchmarking:
                benchmark_cfg = services_cfg.get("benchmarking")
                if isinstance(benchmark_cfg, dict):
                    agent_md_file = benchmark_cfg.get("agent_md_file")
                    if agent_md_file:
                        benchmark_cfg["agent_md_file"] = f"/root/archi/agents/{Path(str(agent_md_file)).name}"

            config_template = self.env.get_template(BASE_CONFIG_TEMPLATE)
            config_rendered = config_template.render(verbosity=context.plan.verbosity, **updated_config)

            target_name = "config.yaml" if single_mode else f"{name}.yaml"
            with open(configs_path / target_name, "w") as f:
                f.write(config_rendered)
            logger.info(f"Rendered configuration file {configs_path / target_name}")

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
        with open(grafana_dir / "datasources.yaml", "w") as f:
            f.write(datasources)

        dashboards_template = self.env.get_template(BASE_GRAFANA_DASHBOARDS_TEMPLATE)
        dashboards = dashboards_template.render()
        with open(grafana_dir / "dashboards.yaml", "w") as f:
            f.write(dashboards)

        configs = context.config_manager.get_configs()
        palette = assign_feedback_palette(configs)

        dashboard_template = self.env.get_template(BASE_GRAFANA_ARCHI_DEFAULT_DASHBOARDS_TEMPLATE)
        dashboard = dashboard_template.render(
            feedback_palette=palette,
        )
        with open(grafana_dir / "archi-default-dashboard.json", "w") as f:
            f.write(dashboard)

        config_template = self.env.get_template(BASE_GRAFANA_CONFIG_TEMPLATE)
        grafana_config = config_template.render()
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

        with open(dest, "w") as f:
            f.write(init_sql)
        logger.debug(f"Wrote PostgreSQL init script to {dest}")


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

    def _copy_source_code(self, base_dir: Path) -> None:
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
