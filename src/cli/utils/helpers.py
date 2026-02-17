import copy
from importlib import resources
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import click
import yaml

from src.cli.service_registry import service_registry
from src.cli.utils.service_builder import ServiceBuilder
from src.utils.logging import get_logger

logger = get_logger(__name__)

TEMPLATE_COMPARISON_PATHS = (
    "base-config.yaml",
    "base-compose.yaml",
    "init.sql",  # PostgreSQL + pgvector schema
    "grafana/datasources.yaml",
    "grafana/dashboards.yaml",
    "grafana/archi-default-dashboard.json",
    "grafana/grafana.ini",
)

def check_docker_available() -> bool:
    """Check if Docker is available and not just Podman emulation."""
    if not shutil.which("docker"):
        return False
    
    try:
        # Run 'docker --version' to check if it's actually Docker
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        # If stderr contains podman message, it's actually podman emulation
        if result.returncode == 0 and "podman" not in result.stderr.lower():
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    
    return False

def parse_gpu_ids_option(ctx, param, value):
    """Parse GPU IDs option - 'all' or comma-separated integers"""
    if value is None:
        return None
    if value.lower() == "all":
        return "all"
    try:
        return [int(x.strip()) for x in value.split(",")]
    except ValueError:
        raise click.BadParameter('--gpu-ids option must be "all" or comma-separated integers')

def parse_services_option(ctx, param, value):
    """Parse comma-separated services list using service registry"""
    if not value:
        return []
    
    # Get available services from registry
    available_services = list(ServiceBuilder.get_available_services().keys())
    services = [s.strip() for s in value.split(',')]
    
    invalid_services = [s for s in services if s not in available_services]
    if invalid_services:
        raise click.BadParameter(
            f'Invalid services: {", ".join(invalid_services)}. '
            f'Available: {", ".join(available_services)}'
        )
    
    return services

def _read_installed_template(pkg_root, rel_path: str) -> Optional[bytes]:
    try:
        target = pkg_root.joinpath(rel_path)
        if not target.is_file():
            return None
        return target.read_bytes()
    except Exception:
        return None

def _read_repo_template(repo_templates_dir: Path, rel_path: str) -> Optional[bytes]:
    target = repo_templates_dir / rel_path
    try:
        with open(target, "rb") as f:
            return f.read()
    except FileNotFoundError:
        return None

def _get_template_mismatches() -> List[str]:
    try:
        from src.cli.utils import _repository_info
        repo_root = Path(_repository_info.REPO_PATH)
    except Exception:
        return []

    repo_templates_dir = repo_root / "src" / "cli" / "templates"
    if not repo_templates_dir.exists():
        return []

    pkg_root = resources.files("src.cli.templates")
    mismatches: List[str] = []

    for rel_path in TEMPLATE_COMPARISON_PATHS:
        installed_bytes = _read_installed_template(pkg_root, rel_path)
        repo_bytes = _read_repo_template(repo_templates_dir, rel_path)
        if installed_bytes != repo_bytes:
            mismatches.append(rel_path)

    return mismatches

def warn_if_template_mismatch() -> None:
    mismatches = _get_template_mismatches()
    if not mismatches:
        return

    details = "\n  ".join(mismatches)
    logger.warning(
        "Detected template changes in the working tree that are not in the installed package."
    )
    message = (
        "Template files differ from the installed package:\n"
        f"  {details}\n"
        "Re-run `pip install .` (or `pip install -e .` to avoid this in future) to pick up changes.\n"
        "Continue anyway?"
    )
    if not click.confirm(message, default=False):
        raise click.ClickException("Aborted due to template mismatch.")

def _infer_host_mode_from_compose(compose_data: Dict[str, Any]) -> bool:
    services = compose_data.get("services", {}) or {}
    for svc in services.values():
        if isinstance(svc, dict) and svc.get("network_mode") == "host":
            return True
    return False

def _infer_gpu_ids_from_compose(compose_data: Dict[str, Any]) -> Optional[object]:
    services = compose_data.get("services", {}) or {}
    gpu_ids: List[int] = []

    for svc in services.values():
        if not isinstance(svc, dict):
            continue
        deploy_devices = (
            svc.get("deploy", {})
            .get("resources", {})
            .get("reservations", {})
            .get("devices")
        )
        if deploy_devices:
            return "all"

        for entry in svc.get("devices", []) or []:
            if not isinstance(entry, str):
                continue
            if "nvidia.com/gpu=all" in entry:
                return "all"
            if "nvidia.com/gpu=" in entry:
                try:
                    gpu_ids.append(int(entry.split("=", 1)[1]))
                except ValueError:
                    continue

        for volume in svc.get("volumes", []) or []:
            if isinstance(volume, str) and volume.startswith("archi-models"):
                return "all"

    return sorted(set(gpu_ids)) if gpu_ids else None

def _infer_tag_from_compose(compose_data: Dict[str, Any]) -> str:
    services = compose_data.get("services", {}) or {}
    for svc in services.values():
        if not isinstance(svc, dict):
            continue
        image = svc.get("image")
        if isinstance(image, str) and ":" in image:
            return image.rsplit(":", 1)[1]
    return "2000"

def _render_config_for_compare(
    config: Dict[str, Any],
    host_mode: bool,
    verbosity: int,
    env,
) -> Dict[str, Any]:
    from src.cli.managers.templates_manager import TemplateManager

    updated_config = copy.deepcopy(config)
    if host_mode:
        updated_config["host_mode"] = True
        TemplateManager(env)._apply_host_mode_port_overrides(updated_config)

    config_template = env.get_template("base-config.yaml")
    rendered = config_template.render(verbosity=verbosity, **updated_config)
    return yaml.safe_load(rendered)

def _load_rendered_configs(configs_dir: Path) -> Dict[str, Dict[str, Any]]:
    rendered: Dict[str, Dict[str, Any]] = {}
    for config_path in configs_dir.glob("*.yaml"):
        with open(config_path, "r") as f:
            data = yaml.safe_load(f) or {}
        name = data.get("name") or config_path.stem
        rendered[name] = data
    return rendered

def _get_nested(config: Dict[str, Any], path: Tuple[str, ...]) -> Any:
    value: Any = config
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value

def _validate_non_chatbot_sections(
    current_configs: Dict[str, Dict[str, Any]],
    new_configs: List[Dict[str, Any]],
    *,
    host_mode: bool,
    verbosity: int,
    env,
) -> None:
    restricted_paths = (
        ("data_manager",),
        ("services", "data_manager"),
        ("services", "postgres"),
    )
    differences: List[str] = []

    for config in new_configs:
        name = config.get("name")
        if not name or name not in current_configs:
            raise click.ClickException(
                f"Config '{name or 'unknown'}' does not match an existing deployment config."
            )

        rendered_new = _render_config_for_compare(config, host_mode, verbosity, env)
        rendered_current = current_configs[name]

        for path in restricted_paths:
            if _get_nested(rendered_new, path) != _get_nested(rendered_current, path):
                differences.append(f"{name}: {'.'.join(path)}")

    if differences:
        details = "\n  ".join(differences)
        raise click.ClickException(
            "Restart config changes are restricted to chatbot settings only. "
            "The following sections changed:\n"
            f"  {details}"
        )

def validate_services_selection(services: List[str]) -> None:
    """Validate that at least one service is selected, raise ClickException if not"""
    if not services:
        from src.cli.utils.service_builder import ServiceBuilder
        available_services = ServiceBuilder.get_available_services()
        service_list = '\n'.join([f"  {name}: {desc}" for name, desc in available_services.items()])
        raise click.ClickException(
            f"No services selected. Please specify at least one service using --services.\n"
            f"Available services:\n{service_list}\n"
            f"Example: --services chatbot,grafana"
        )


def log_dependency_resolution(services: List[str], enabled_services: List[str]) -> None:
    """Log which dependencies were auto-enabled"""
    resolved_services = service_registry.resolve_dependencies(enabled_services)
    service_only_resolved = [s for s in resolved_services if s in service_registry.get_all_services()]
    
    if set(service_only_resolved) != set(services):
        added_services = set(service_only_resolved) - set(services)
        if added_services:
            logger.info(f"Auto-enabling dependencies: {', '.join(added_services)}")


def handle_existing_deployment(base_dir: Path, name: str, force: bool, dry: bool, 
                              use_podman: bool) -> None:
    """Handle existing deployment - either remove it or raise error"""
    if base_dir.exists():
        if force:
            if not dry:
                logger.info(f"Removing existing deployment at {base_dir}")
                from src.cli.managers.deployment_manager import \
                    DeploymentManager
                deployment_manager = DeploymentManager(use_podman)
                try:
                    deployment_manager.delete_deployment(
                        deployment_name=name,
                        remove_images=False,
                        remove_volumes=False,
                        remove_files=True
                    )
                except Exception as e:
                    logger.info(f"Warning: Could not clean up existing deployment: {e}")
            else:
                logger.info(f"[DRY RUN] Would remove existing deployment at {base_dir}")
        else:
            raise click.ClickException(
                f"Deployment '{name}' already exists at {base_dir}.\n"
                f"Use --force to overwrite, or delete it first with: archi delete --name {name}"
            )


def print_dry_run_summary(name: str, services: List[str], service_only_resolved: List[str], 
                         sources: List[str], required_secrets: Set[str], 
                         compose_config, other_flags: Dict[str, Any], base_dir: Path) -> None:
    """Print comprehensive dry run summary"""
    logger.info(f"[DRY RUN] Deployment summary:\n")
    click.echo(f"\tName: {name}")
    click.echo(f"\tRequested services: {', '.join(services)}")
    click.echo(f"\tAll services (with dependencies): {', '.join(service_only_resolved)}")
    
    if sources:
        click.echo(f"\tData sources: {', '.join(sources)}")
    
    click.echo(f"\tRequired secrets: {', '.join(sorted(required_secrets))}")
    click.echo(f"\tRequired volumes: {', '.join(compose_config.get_required_volumes())}")
    click.echo(f"\tContainer tool: {'Podman' if other_flags['podman'] else 'Docker'}")
    
    if other_flags.get('gpu_ids'):
        click.echo(f"\tGPU configuration: {other_flags['gpu_ids']}")
    
    click.echo(f"\tDeployment directory: {base_dir}\n")
    logger.info(f"[DRY RUN] Configuration and secrets are valid. Run without --dry to deploy.\n")


def show_service_urls(services: List[str], archi_config: Dict[str, Any], host_mode: bool) -> None:
    """Show service URLs using registry configuration"""
    for service_name in services:
        if service_name not in service_registry.get_all_services():
            continue
            
        service_def = service_registry.get_service(service_name)
        if not service_def.port_config_path:
            continue
            
        try:
            # Navigate config path to get port
            config_value = archi_config
            for key in service_def.port_config_path.split('.'):
                config_value = config_value[key]
                
            if isinstance(config_value, dict):
                if host_mode:
                    port = config_value.get('port', service_def.default_container_port)
                else:
                    port = config_value.get('external_port', service_def.default_host_port)
            else:
                port = config_value
                
            if port:
                logger.info(f"{service_name.title()}: http://localhost:{port}")
                
        except (KeyError, TypeError):
            # Use default port if config navigation fails
            fallback_port = (
                service_def.default_container_port if host_mode else service_def.default_host_port
            )
            if fallback_port:
                logger.info(f"{service_name.title()}: http://localhost:{fallback_port}")


def log_deployment_start(name: str, services: List[str], sources: List[str], dry: bool) -> None:
    """Log deployment start information"""
    logger.info(f"{'[DRY RUN] ' if dry else ''}Creating deployment '{name}' with services: {', '.join(services)}")
    if sources:
        logger.info(f"Data sources: {', '.join(sources)}")


def log_deployment_success(name: str, service_only_resolved: List[str], services: List[str], 
                          config_manager, host_mode) -> None:
    """Log successful deployment and show service URLs"""
    print(f"\nARCHI deployment '{name}' created successfully!")
    print(f"Services running: {', '.join(service_only_resolved)}")
    #All services are part of static configuration and equal for all configs
    archi_config = config_manager.get_configs()[0]
    show_service_urls(service_only_resolved, archi_config, host_mode=host_mode)
