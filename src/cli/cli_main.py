import os
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import yaml
from jinja2 import (ChainableUndefined, Environment, PackageLoader,
                    select_autoescape)

from src.cli.managers.config_manager import ConfigurationManager
from src.cli.managers.deployment_manager import DeploymentManager
from src.cli.managers.secrets_manager import SecretsManager
from src.cli.managers.templates_manager import TemplateManager
from src.cli.managers.volume_manager import VolumeManager
from src.cli.service_registry import service_registry
from src.cli.source_registry import source_registry
from src.cli.utils.helpers import *
from src.cli.utils.helpers import (
    _infer_gpu_ids_from_compose,
    _infer_host_mode_from_compose,
    _infer_tag_from_compose,
    _load_rendered_configs,
    _validate_non_chatbot_sections,
)
from src.cli.utils.service_builder import ServiceBuilder
from src.utils.logging import get_logger, setup_cli_logging
from src.cli.tools.config_seed import seed_entry
import subprocess

# DEFINITIONS
env = Environment(
    loader=PackageLoader("src.cli"),
    autoescape=select_autoescape(),
    undefined=ChainableUndefined,
)
ARCHI_DIR = os.environ.get('ARCHI_DIR',os.path.join(os.path.expanduser('~'), ".archi"))

@click.group()
def cli():
    pass

@click.command()
@click.option('--name', '-n', type=str, required=True, help="Name of the archi deployment")
@click.option('--config', '-c', 'config_files', type=str, multiple=True, help="Path to .yaml archi configuration")
@click.option('--config-dir', '-cd', 'config_dir', type=str, help="Path to configs directory")
@click.option('--env-file', '-e', type=str, required=False, help="Path to .env file with secrets")
@click.option('--services', '-s', callback=parse_services_option, 
              help="Comma-separated list of services")
@click.option('--podman', '-p', is_flag=True, help="Use Podman instead of Docker")
@click.option('--gpu-ids', callback=parse_gpu_ids_option, help='GPU configuration: "all" or comma-separated IDs')
@click.option('--tag', '-t', type=str, default="2000", help="Image tag for built containers")
@click.option('--hostmode', 'host_mode', is_flag=True, help="Use host network mode")
@click.option('--verbosity', '-v', type=int, default=3, help="Logging verbosity level (0-4)")
@click.option('--force', '-f', is_flag=True, help="Force deployment creation, overwriting existing deployment")
@click.option('--dry', '--dry-run', is_flag=True, help="Validate configuration and show what would be created without actually deploying")
def create(name: str, config_files: list, config_dir: str, env_file: str, services: list,
           force: bool, dry: bool, verbosity: int, **other_flags):
    """Create an ARCHI deployment with selected services and data sources."""

    if not (bool(config_files) ^ bool(config_dir)): 
        raise click.ClickException(f"Must specify only one of config files or config dir")
    if config_dir: 
        config_path = Path(config_dir)
        config_files = [item for item in config_path.iterdir() if item.is_file()]
    if len(config_files) != 1:
        raise click.ClickException("Exactly one config file is supported; please provide a single -c file.")

    print("Starting ARCHI deployment process...")
    setup_cli_logging(verbosity=verbosity)
    logger = get_logger(__name__)

    warn_if_template_mismatch()
    
    # Check if Docker is available when --podman is not specified
    if not other_flags.get('podman', False) and not check_docker_available():
        raise click.ClickException(
            "Docker is not available on this system. "
            "Please install Docker or use the '--podman' option to use Podman instead.\n"
            "Example: archi create --name mybot --podman ..."
        )
    
    try:
        # Validate inputs
        validate_services_selection(services)
        
        # Combine services and data sources for processing
        enabled_services = services.copy()
        # Handle existing deployment
        base_dir = Path(ARCHI_DIR) / f"archi-{name}"
        handle_existing_deployment(base_dir, name, force, dry, other_flags.get('podman', False))
        
        # Initialize managers
        config_manager = ConfigurationManager(config_files,env)
        secrets_manager = SecretsManager(env_file, config_manager)

        # Resolve enabled sources from config (no CLI source overrides).
        # Keep links enabled by default.
        config_defined_sources = config_manager.get_enabled_sources()
        config_disabled_sources = config_manager.get_disabled_sources()
        enabled_sources = list(dict.fromkeys(["links"] + config_defined_sources))
        enabled_sources = [src for src in enabled_sources if src not in config_disabled_sources]
        enabled_sources = source_registry.resolve_dependencies(enabled_sources)

        disabled_conflicts = sorted(set(enabled_sources) & set(config_disabled_sources))
        if disabled_conflicts:
            raise click.ClickException(
                f"Cannot enable sources due to disabled dependencies in config: {', '.join(disabled_conflicts)}"
            )

        # Log deployment info and dependency resolution
        log_deployment_start(name, services, enabled_sources, dry)
        log_dependency_resolution(services, enabled_services)

        # Validate configuration and secrets
        config_manager.validate_configs(enabled_services, enabled_sources)
        logger.info("Configurations validated successfully")

        required_secrets, all_secrets = secrets_manager.get_secrets(set(enabled_services), set(enabled_sources))
        secrets_manager.validate_secrets(required_secrets)
        logger.info(f"Required secrets validated: {', '.join(sorted(required_secrets))}")
        extra = all_secrets - required_secrets
        if extra:
            logger.info(f"Also passing additional secrets found: {', '.join(sorted(extra))}")

        config_manager.set_sources_enabled(enabled_sources)
        
        # Build compose configuration
        compose_config = ServiceBuilder.build_compose_config(
            name=name, verbosity=verbosity, base_dir=base_dir,
            enabled_services=enabled_services, enabled_sources=enabled_sources, secrets=all_secrets,
            **other_flags
        )
        
        # Handle dry run
        if dry:
            service_only_resolved = [s for s in service_registry.resolve_dependencies(enabled_services) 
                                   if s in service_registry.get_all_services()]
            print_dry_run_summary(name, services, service_only_resolved, enabled_sources, 
                                 required_secrets, compose_config, other_flags, base_dir)
            return
        
        # Actual deployment
        template_manager = TemplateManager(env, verbosity)
        base_dir.mkdir(parents=True, exist_ok=True)
        
        secrets_manager.write_secrets_to_files(base_dir, all_secrets)
        
        volume_manager = VolumeManager(compose_config.use_podman)
        volume_manager.create_required_volumes(compose_config, config_manager.config)

        template_manager.prepare_deployment_files(
            compose_config,
            config_manager,
            secrets_manager,
            **other_flags,
        )

        # Host-side seeding removed; container config-seed handles schema + ingestion before services start.
        
        deployment_manager = DeploymentManager(compose_config.use_podman)
        deployment_manager.start_deployment(base_dir)
        
        # Log success
        service_only_resolved = [s for s in service_registry.resolve_dependencies(enabled_services) 
                               if s in service_registry.get_all_services()]
        log_deployment_success(name, service_only_resolved, services, config_manager, host_mode=other_flags.get('host_mode', False))
        
    except Exception as e:
        if verbosity >= 4:
            traceback.print_exc()
        else:
            raise click.ClickException(str(e))
    

@click.command()
@click.option('--name', '-n', type=str, help="Name of the archi deployment to delete")
@click.option('--rmi', is_flag=True, help="Remove images (--rmi all)")
@click.option('--rmv', is_flag=True, help="Remove volumes (--volumes)")
@click.option('--keep-files', is_flag=True, help="Keep deployment files (don't remove directory)")
@click.option('--list', 'list_deployments', is_flag=True, help="List all available deployments")
@click.option('--verbosity', '-v', type=int, default=3, help="Logging verbosity level (0-4)")
@click.option('--podman', '-p', is_flag=True, default=False, help="specify if podman is being used")
def delete(name: str, rmi: bool, rmv: bool, keep_files: bool, list_deployments: bool, verbosity: int, podman: bool):
    """
    Delete an ARCHI deployment with the specified name.
    
    This command stops containers and optionally removes images, volumes, and files.
    
    Examples:
    
    # List available deployments
    archi delete --list
    
    # Delete deployment (keep images and volumes)
    archi delete --name mybot
    
    # Delete deployment and remove images
    archi delete --name mybot --rmi
    
    # Complete cleanup (remove everything)
    archi delete --name mybot --rmi --rmv
    
    # Stop deployment but keep files for debugging
    archi delete --name mybot --keep-files
    """
    
    setup_cli_logging(verbosity=verbosity)
    logger = get_logger(__name__)

    try:
        # We don't know which tool was used to create it, so try to detect from files
        deployment_manager = DeploymentManager(use_podman=podman)  # Will try both tools
        
        # Handle list option
        if list_deployments:
            deployments = deployment_manager.list_deployments()
            if deployments:
                logger.info("Available deployments:")
                for deployment in deployments:
                    logger.info(f"  - {deployment}")
            else:
                logger.info("No deployments found")
            return
        
        # Validate name is provided
        if not name:
            available = deployment_manager.list_deployments()
            if available:
                available_str = ", ".join(available)
                raise click.ClickException(
                    f"Please provide a deployment name using --name.\n"
                    f"Available deployments: {available_str}\n"
                    f"Use 'archi delete --list' to see all deployments."
                )
            else:
                raise click.ClickException(
                    "Please provide a deployment name using --name.\n"
                    "No deployments found. Use 'archi create' to create one."
                )
        
        # Clean the name
        name = name.strip()
        
        # Confirm deletion if removing volumes
        if rmv:
            click.confirm(
                f"This will permanently delete volumes for deployment '{name}'. Continue?",
                abort=True
            )
        
        # Perform deletion using DeploymentManager
        deployment_manager.delete_deployment(
            deployment_name=name,
            remove_images=rmi,
            remove_volumes=rmv,
            remove_files=not keep_files
        )
        
    except Exception as e:
        traceback.print_exc()
        raise click.ClickException(str(e))

@click.command()
@click.option('--name', '-n', type=str, required=True, help="Name of the archi deployment")
@click.option('--service', '-s', type=str, default="chatbot", help="Service to restart (default: chatbot)")
@click.option('--config', '-c', 'config_files', type=str, multiple=True, help="Path to .yaml archi configuration")
@click.option('--config-dir', '-cd', 'config_dir', type=str, help="Path to configs directory")
@click.option('--env-file', '-e', type=str, required=False, help="Path to .env file with secrets")
@click.option('--no-build', is_flag=True, help="Restart without rebuilding the image")
@click.option('--with-deps', is_flag=True, help="Also restart dependent services")
@click.option('--podman', '-p', is_flag=True, default=False, help="specify if podman is being used")
@click.option('--verbosity', '-v', type=int, default=3, help="Logging verbosity level (0-4)")
def restart(
    name: str,
    service: str,
    config_files: tuple,
    config_dir: Optional[str],
    env_file: Optional[str],
    no_build: bool,
    with_deps: bool,
    podman: bool,
    verbosity: int,
):
    """Restart a specific service in an existing deployment while reusing its configured ports."""
    setup_cli_logging(verbosity=verbosity)

    if not podman and not check_docker_available():
        raise click.ClickException(
            "Docker is not available on this system. "
            "Please install Docker or use the '--podman' option to use Podman instead.\n"
            "Example: archi restart --name mybot --podman ..."
        )

    deployment_dir = Path(ARCHI_DIR) / f"archi-{name}"
    compose_file = deployment_dir / "compose.yaml"
    if not compose_file.exists():
        raise click.ClickException(
            f"Deployment '{name}' not found at {deployment_dir}. "
            "Use 'archi list-deployments' to see available deployments."
        )

    try:
        with open(compose_file, 'r') as f:
            compose_data = yaml.safe_load(f) or {}
        services = compose_data.get("services", {})
    except Exception as e:
        raise click.ClickException(f"Failed to read compose file: {e}")

    if service not in services:
        available = ", ".join(sorted(services.keys()))
        raise click.ClickException(
            f"Service '{service}' not found in deployment '{name}'. "
            f"Available services: {available}"
        )

    if config_files or config_dir:
        if not (bool(config_files) ^ bool(config_dir)):
            raise click.ClickException("Must specify only one of config files or config dir")

        if config_dir:
            config_path = Path(config_dir)
            config_files = tuple(item for item in config_path.iterdir() if item.is_file())

        configs_dir = deployment_dir / "configs"
        current_configs = _load_rendered_configs(configs_dir)
        if not current_configs:
            raise click.ClickException(f"No rendered configs found at {configs_dir}")

        enabled_services = [
            name for name in services.keys() if name in service_registry.get_all_services()
        ]
        host_mode = _infer_host_mode_from_compose(compose_data)
        gpu_ids = _infer_gpu_ids_from_compose(compose_data)
        tag = _infer_tag_from_compose(compose_data)
        existing_secrets = set((compose_data.get("secrets") or {}).keys())

        config_manager = ConfigurationManager(list(config_files), env)
        
        enabled_sources = config_manager.get_enabled_sources()
        config_disabled_sources = config_manager.get_disabled_sources()
        enabled_sources = [src for src in enabled_sources if src not in config_disabled_sources]
        enabled_sources = source_registry.resolve_dependencies(enabled_sources)

        config_manager.validate_configs(enabled_services, enabled_sources)

        _validate_non_chatbot_sections(
            current_configs,
            config_manager.get_configs(),
            host_mode=host_mode,
            verbosity=verbosity,
            env=env,
        )

        secrets_manager = None
        all_secrets = existing_secrets
        if env_file:
            secrets_manager = SecretsManager(env_file, config_manager)
            required_secrets, all_secrets = secrets_manager.get_secrets(set(enabled_services), set(enabled_sources))
            secrets_manager.validate_secrets(required_secrets)
            secrets_manager.write_secrets_to_files(deployment_dir, all_secrets)
        elif "grafana" in enabled_services:
            raise click.ClickException(
                "Grafana is enabled for this deployment. Please provide --env-file so "
                "Grafana assets can be rendered."
            )
        else:
            secrets_manager = SecretsManager(None, config_manager)

        compose_config = ServiceBuilder.build_compose_config(
            name=name,
            verbosity=verbosity,
            base_dir=deployment_dir,
            enabled_services=enabled_services,
            enabled_sources=enabled_sources,
            secrets=all_secrets,
            podman=podman,
            gpu_ids=gpu_ids,
            host_mode=host_mode,
            tag=tag,
        )

        template_manager = TemplateManager(env, verbosity)
        template_manager.prepare_deployment_files(
            compose_config,
            config_manager,
            secrets_manager,
            host_mode=host_mode,
            allow_port_reuse=True,
        )

    deployment_manager = DeploymentManager(use_podman=podman)
    deployment_manager.restart_service(
        deployment_dir=deployment_dir,
        service_name=service,
        build=not no_build,
        no_deps=not with_deps,
        force_recreate=True
    )
    
@click.command()
def list_services():
    """List all available services"""
    
    click.echo("Available ARCHI services:\n")
    
    # Application services
    app_services = service_registry.get_application_services()
    if app_services:
        click.echo("Application Services:")
        for name, service_def in app_services.items():
            click.echo(f"  {name:20} {service_def.description}")
        click.echo()
    
    # Integration services
    integration_services = service_registry.get_integration_services()
    if integration_services:
        click.echo("Integration Services:")
        for name, service_def in integration_services.items():
            click.echo(f"  {name:20} {service_def.description}")
        click.echo()
    
    # Data sources
    click.echo("Data Sources:")
    for name in source_registry.names():
        if name == 'links':
            continue
        definition = source_registry.get(name)
        click.echo(f"  {name:20}{definition.description}")


@click.command()
def list_deployments():
    """List all existing deployments"""
    
    archi_dir = Path(ARCHI_DIR)

    if not archi_dir.exists():
        click.echo("No deployments found")
        return
    
    deployments = [d for d in archi_dir.iterdir() 
                  if d.is_dir() and d.name.startswith('archi-')]
    
    if not deployments:
        click.echo("No deployments found")
        return
    
    click.echo("Existing deployments:")
    for deployment in deployments:
        name = deployment.name.replace('archi-', '')
        
        # Try to get running status
        try:
            compose_file = deployment / "compose.yaml"
            if compose_file.exists():
                click.echo(f"  {name}")
            else:
                click.echo(f"  {name} (incomplete)")
        except Exception:
            click.echo(f"  {name} (status unknown)")


@click.command()
@click.option('--name', '-n', type=str, required=True, help="Name of the archi deployment")
@click.option('--config', '-c', 'config_file', type=str, help="Path to .yaml archi configuration")
@click.option('--config-dir', '-cd', 'config_dir', type=str, help="Path to configs directory")
@click.option('--env-file', '-e', type=str, required=False, help="Path to .env file with 'secrets")
@click.option('--hostmode', 'host_mode', is_flag=True, help="Use host network mode")
@click.option('--podman', '-p', is_flag=True, help="Use Podman instead of Docker")
@click.option('--gpu-ids', callback=parse_gpu_ids_option, help='GPU configuration: "all" or comma-separated IDs')
@click.option('--force', '-f', is_flag=True, help="Force deployment creation, overwriting existing deployment")
@click.option('--tag', '-t', type=str, default="2000", help="Image tag for built containers")
@click.option('--verbosity', '-v', type=int, default=3, help="Logging verbosity level (0-4)")
def evaluate(name: str, config_file: str, config_dir: str, env_file: str, force: bool, verbosity: int, **other_flags):
    """Create an ARCHI deployment with selected services and data sources."""
    if not (bool(config_file) ^ bool(config_dir)): 
        raise click.ClickException(f"Must specify only one of config files or config dir")
    if config_dir: 
        config_path = Path(config_dir)
        config_files = [str(item) for item in config_path.iterdir() if item.is_file()]
    else: 
        config_files = [item for item in config_file.split(",")]

    print("Starting ARCHI benchmarking process...")
    setup_cli_logging(verbosity=verbosity)
    logger = get_logger(__name__)

    # Check if Docker is available when --podman is not specified
    if not other_flags.get('podman', False) and not check_docker_available():
        raise click.ClickException(
            "Docker is not available on this system. "
            "Please install Docker or use the '--podman' option to use Podman instead.\n"
            "Example: archi evaluate --name mybot --podman ..."
        )

    gpu = other_flags.get("gpu-ids") != None

    try: 
        base_dir = Path(ARCHI_DIR) / f"archi-{name}"
        handle_existing_deployment(base_dir, name, force, False, other_flags.get('podman', False))

        if base_dir.exists():
            raise click.ClickException(
                    f"Benchmarking runtime '{name}' already exists at {base_dir}"
                    )

        config_manager = ConfigurationManager(config_files,env)
        secrets_manager = SecretsManager(env_file, config_manager)

        # Services for benchmarking: PostgreSQL is required
        enabled_services = ["postgres", "benchmarking"]

        # Resolve enabled sources from config (no CLI source overrides).
        # Keep links enabled by default.
        config_defined_sources = config_manager.get_enabled_sources()
        config_disabled_sources = config_manager.get_disabled_sources()
        enabled_sources = list(dict.fromkeys(["links"] + config_defined_sources))
        enabled_sources = [src for src in enabled_sources if src not in config_disabled_sources]
        enabled_sources = source_registry.resolve_dependencies(enabled_sources)

        disabled_conflicts = sorted(set(enabled_sources) & set(config_disabled_sources))
        if disabled_conflicts:
            raise click.ClickException(
                f"Cannot enable sources due to disabled dependencies in config: {', '.join(disabled_conflicts)}"
            )

        config_manager.validate_configs(enabled_services, enabled_sources)

        required_secrets, all_secrets = secrets_manager.get_secrets(set(enabled_services), set(enabled_sources))
        secrets_manager.validate_secrets(required_secrets)
        config_manager.set_sources_enabled(enabled_sources)

        benchmarking_configs = config_manager.get_interface_config("benchmarking")

        other_flags['benchmarking'] = True
        other_flags['query_file'] = benchmarking_configs.get('queries_path', ".")
        other_flags['benchmarking_dest'] = os.path.abspath(benchmarking_configs.get('out_dir', '.'))
        other_flags['host_mode'] = other_flags.get('host_mode', False)

        compose_config = ServiceBuilder.build_compose_config(
                name=name, verbosity=verbosity, base_dir=base_dir, 
                enabled_services=enabled_services, enabled_sources=enabled_sources, secrets=all_secrets,
                **other_flags
                )


        template_manager = TemplateManager(env, verbosity)
        base_dir.mkdir(parents=True, exist_ok=True)
        
        secrets_manager.write_secrets_to_files(base_dir, all_secrets)

        volume_manager = VolumeManager(compose_config.use_podman)
        volume_manager.create_required_volumes(compose_config, config_manager.config)
        
        template_manager.prepare_deployment_files(
            compose_config,
            config_manager,
            secrets_manager,
            **other_flags,
        )

        deployment_manager = DeploymentManager(compose_config.use_podman)
        deployment_manager.start_deployment(base_dir)
    except Exception as e:
        if verbosity >=4: 
            traceback.print_exc()
        else: 
            raise click.ClickException(f"Failed due to the following exception: {e}")


def main():
    """
    Entrypoint for archi cli tool implemented using Click.
    """
    # cli.add_command(help)
    cli.add_command(create)
    cli.add_command(delete)
    cli.add_command(restart)
    cli.add_command(list_services)
    cli.add_command(list_deployments)
    cli.add_command(evaluate)
    cli()
