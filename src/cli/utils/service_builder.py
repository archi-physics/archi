from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

from src.cli.service_registry import service_registry


@dataclass
class ServiceState:
    """Holds compose-facing configuration for a single service."""

    enabled: bool = False
    image_name: str = ""
    tag: str = ""
    container_name: str = ""
    volume_name: str = ""
    required_secrets: List[str] = field(default_factory=list)
    required_config_fields: List[str] = field(default_factory=list)
    port_host: Optional[int] = None
    port_container: Optional[int] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "enabled": self.enabled,
            "image_name": self.image_name,
            "tag": self.tag,
            "container_name": self.container_name,
            "volume_name": self.volume_name,
            "required_secrets": self.required_secrets,
            "required_config_fields": self.required_config_fields,
            "port_host": self.port_host,
            "port_container": self.port_container,
        }


class DeploymentPlan:
    """Immutable-ish view of the services, sources, and secrets for a deployment."""

    def __init__(
        self,
        name: str,
        base_dir: Path,
        tag: str,
        use_podman: bool,
        gpu_ids: Optional[str],
        host_mode: bool,
        verbosity: int,
        benchmarking_dest: str,
    ) -> None:
        self.name = name
        self.base_dir = base_dir
        self.tag = tag
        self.use_podman = use_podman
        self.gpu_ids = gpu_ids
        self.host_mode = host_mode
        self.verbosity = verbosity
        self.benchmarking_dest = benchmarking_dest

        self.enabled_sources: Set[str] = set()
        self._required_secrets: Set[str] = set()

        # Track services in a consistent order for template rendering
        self.services: Dict[str, ServiceState] = {
            "data-manager": ServiceState(),
            "postgres": ServiceState(),
            "chatbot": ServiceState(),
            "grafana": ServiceState(),
            "uploader": ServiceState(),
            "grader": ServiceState(),
            "piazza": ServiceState(),
            "mattermost": ServiceState(),
            "redmine-mailer": ServiceState(),
            "benchmarking": ServiceState(),
            "vllm-server": ServiceState(),
        }

        self.use_redmine: bool = False
        self.use_jira: bool = False

    def enable_service(self, name: str, **config: object) -> None:
        if name not in self.services:
            raise ValueError(f"Unknown service: {name}")
        self.services[name] = ServiceState(enabled=True, **config)

    def register_required_secrets(self, secrets: Set[str]) -> None:
        self._required_secrets.update(secrets)

    def get_service(self, name: str) -> ServiceState:
        if name not in self.services:
            raise ValueError(f"Unknown service: {name}")
        return self.services[name]

    def get_enabled_services(self) -> List[str]:
        return [name for name, state in self.services.items() if state.enabled]

    def get_required_volumes(self) -> List[str]:
        volumes: Set[str] = set()
        for state in self.services.values():
            if state.enabled and state.volume_name:
                volumes.add(state.volume_name)
        if self.gpu_ids:
            volumes.add("archi-models")
        return sorted(volumes)

    def get_required_secrets(self) -> List[str]:
        secrets: Set[str] = set(self._required_secrets)
        for state in self.services.values():
            if state.enabled:
                secrets.update(state.required_secrets)
        return sorted(secrets)

    def to_template_vars(self) -> Dict[str, object]:
        data: Dict[str, object] = {
            "name": self.name,
            "tag": self.tag,
            "use_podman": self.use_podman,
            "gpu_ids": self.gpu_ids,
            "host_mode": self.host_mode,
            "verbosity": self.verbosity,
            "use_redmine": self.use_redmine,
            "use_jira": self.use_jira,
            "required_volumes": self.get_required_volumes(),
            "required_secrets": sorted(self._required_secrets),
            "benchmarking_dest": self.benchmarking_dest,
            "enabled_sources": sorted(self.enabled_sources),
        }

        for name, state in self.services.items():
            state_dict = state.to_dict()
            data[name] = state_dict
            if state.enabled:
                key_prefix = name.replace("-", "_")
                data[f"{key_prefix}_enabled"] = True
                data[f"{key_prefix}_image"] = state.image_name
                data[f"{key_prefix}_tag"] = state.tag
                data[f"{key_prefix}_container_name"] = state.container_name
                data[f"{key_prefix}_volume_name"] = state.volume_name
        return data


class ServiceBuilder:
    """Utility helpers for driving service + source enablement from the registry."""

    @staticmethod
    def get_available_services() -> Dict[str, str]:
        available_services = service_registry.get_application_services()
        integration_services = service_registry.get_integration_services()
        compute_services = service_registry.get_services_by_category('compute')
        return {
            **{name: svc.description for name, svc in available_services.items()},
            **{name: svc.description for name, svc in integration_services.items()},
            **{name: svc.description for name, svc in compute_services.items()},
        }

    @staticmethod
    def build_compose_config(
        name: str,
        verbosity: int,
        base_dir: Path,
        enabled_services: List[str],
        enabled_sources: Optional[List[str]] = None,
        secrets: Optional[Set[str]] = None,
        **other_flags: object,
    ) -> DeploymentPlan:
        enabled_sources = enabled_sources or []

        tag = other_flags.get("tag", "2000")
        use_podman = other_flags.get("podman", False)
        gpu_ids = other_flags.get("gpu_ids")
        host_mode = other_flags.get("hostmode", other_flags.get("host_mode", False))
        benchmarking_dest = other_flags.get("benchmarking_dest", "")

        plan = DeploymentPlan(
            name=name,
            base_dir=base_dir,
            tag=tag,
            use_podman=use_podman,
            gpu_ids=gpu_ids,
            host_mode=host_mode,
            verbosity=verbosity,
            benchmarking_dest=benchmarking_dest,
        )

        plan.enabled_sources = set(enabled_sources)
        plan.use_redmine = "redmine" in plan.enabled_sources
        plan.use_jira = "jira" in plan.enabled_sources
        if secrets:
            plan.register_required_secrets(secrets)

        all_services = service_registry.resolve_dependencies(enabled_services)
        for service_name in all_services:
            if service_name not in service_registry.get_all_services():
                continue
            definition = service_registry.get_service(service_name)
            config: Dict[str, object] = {
                "required_secrets": list(definition.required_secrets),
                "required_config_fields": list(definition.required_config_fields),
                "port_host": definition.default_host_port,
                "port_container": definition.default_container_port,
            }
            if definition.requires_image:
                config.update(
                    {
                        "image_name": definition.get_image_name(name),
                        "tag": tag,
                        "container_name": definition.get_container_name(name),
                    }
                )
            volume_name = definition.get_volume_name(name)
            if volume_name:
                config["volume_name"] = volume_name
            plan.enable_service(service_name, **config)

        return plan
