import os
from functools import reduce
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

from src.cli.managers.templates_manager import BASE_CONFIG_TEMPLATE
from src.cli.source_registry import source_registry
from src.utils.logging import get_logger

logger = get_logger(__name__)

STATIC_FIELDS = ['global', 'services']

class ConfigurationManager:
    """Manages archi configuration loading and validation"""
    
    def __init__(self, config_paths_list: List[str], env):
        self.configs = []
        for config_filepath in config_paths_list:
            config_filepath = Path(config_filepath)
            try:
                config = self._load_config(config_filepath)
                self._append(config)
            except Exception as e:
                logger.error(f'Config {config_filepath} could not be loaded due to {str(e)}')

        assert(len(self.configs)>0)        
        self.config = self.configs[0]

        self.env = env
    
    def _load_config(self, config_filepath) -> Dict[str, Any]:
        """Load and validate basic structure of config file"""
        if not config_filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_filepath}")

        with open(config_filepath, 'r') as f:
            config = yaml.safe_load(f)

        if not config:
            raise ValueError("Configuration file is empty or invalid")

        if "archi" in config:
            raise ValueError("The 'archi' section is no longer supported in config.yaml.")

        # Track origin for relative-path resolution (e.g., prompts).
        config["_config_path"] = str(config_filepath)

        return config
    
    def _append(self,config):
        """Appends configuration to the config list if the static portions are equivalent to the previous one."""

        if len(self.configs)==0:
            self.configs = [config]
        else:
            previous_config = self.configs[-1]

            #This does not assume the static_fields to be required 
            for static_field in STATIC_FIELDS:
                if static_field in previous_config.keys():
                    if not static_field in config.keys():
                        raise ValueError(f"The field {static_field} must be present in all configurations.")

                    if previous_config[static_field] != config[static_field]:
                        raise ValueError(f"The field {static_field} must be consistent across all configurations.")

            self.configs.append(config)
    
    def _get_static_required_fields_for_services(self, services: List[str]) -> List[str]:
        """Get required configuration fields based on enabled services"""
        if not services:
            return []  # No validation needed if no services selected

        # Base fields always required
        requirements = [
            'name', 
        ]

        # Services that have additional required fields
        service_fields = self._get_service_fields()

        # Add service-specific fields
        for service in services:
            if service in service_fields:
                requirements.extend(service_fields[service])
                
        return requirements
    
    def _get_service_fields(self):
        """Generates dictionary of service fields for services that have additional required fields"""
        template = self.env.get_template(BASE_CONFIG_TEMPLATE)
        default_config = template.render()
        default_config = yaml.safe_load(default_config)

        services = default_config['services']
        service_fields = {}
        optional_keys = {"agents_dir"}

        for service_name, service_configs in services.items():
            blank_configs = [
                key for key, value in service_configs.items()
                if ((value is None) or (value == '')) and key not in optional_keys
            ]
            service_fields[service_name] = [f'services.{service_name}.{key}' for key in blank_configs]

        return service_fields
    
    def _get_active_pipeline_requirements(self,config) -> List[str]:
        """Legacy pipeline requirements (archi section removed)."""
        return []
    
    def _validate_config(self, required_fields: List[str], config) -> None:
        """Validate that all required fields are present in config"""
        for field in required_fields:
            keys = field.split('.')
            value = config
            for key in keys:
                if key not in value:
                    raise ValueError(f"Missing required field: '{field}' in the configuration")
                value = value[key]

    def validate_configs(self, services: List[str], sources: Optional[List[str]] = None):
        """Validate that all required fields are present in each config"""

        sources = source_registry.resolve_dependencies(sources or [])
        static_requirements = self._get_static_required_fields_for_services(services)

        for config in self.configs:
            pipeline_requirements = self._get_active_pipeline_requirements(config)
            required_fields = static_requirements + pipeline_requirements
            self._validate_config(required_fields, config)
            self._validate_chat_app_config(config, services)
            self._validate_benchmarking_config(config, services)
            self._validate_source_fields(config, sources)

        self._collect_embedding_metadata()
        self._collect_input_lists()

    def _validate_chat_app_config(self, config: Dict[str, Any], services: List[str]) -> None:
        if not services or "chatbot" not in services:
            return
        services_cfg = config.get("services", {}) or {}
        chat_cfg = services_cfg.get("chat_app", {}) or {}

        if "provider" in chat_cfg or "model" in chat_cfg:
            raise ValueError(
                "Legacy keys detected: 'services.chat_app.provider'/'services.chat_app.model'. "
                "Use 'services.chat_app.default_provider' and 'services.chat_app.default_model' instead."
            )
        if "agent_dir" in chat_cfg and "agents_dir" not in chat_cfg:
            raise ValueError("Missing required field: 'services.chat_app.agents_dir' (did you mean 'agent_dir'?)")

        required = [
            ("agent_class", "services.chat_app.agent_class"),
            ("agents_dir", "services.chat_app.agents_dir"),
            ("default_provider", "services.chat_app.default_provider"),
            ("default_model", "services.chat_app.default_model"),
        ]
        for key, path in required:
            value = chat_cfg.get(key)
            if not value:
                raise ValueError(f"Missing required field: '{path}' in the configuration")

        agents_dir = Path(str(chat_cfg.get("agents_dir"))).expanduser()
        if agents_dir.exists():
            if not agents_dir.is_dir():
                raise ValueError(f"agents_dir must be a directory: '{agents_dir}'")
            if not list(agents_dir.glob("*.md")):
                raise ValueError(f"agents_dir must contain at least one .md file: '{agents_dir}'")

        # Guard against self-contradictory provider config:
        # default_provider cannot be explicitly disabled in providers.<name>.enabled.
        default_provider = str(chat_cfg.get("default_provider", "")).strip().lower()
        providers_cfg = chat_cfg.get("providers", {}) or {}
        default_provider_cfg = providers_cfg.get(default_provider, {}) if isinstance(providers_cfg, dict) else {}
        if isinstance(default_provider_cfg, dict) and default_provider_cfg.get("enabled") is False:
            raise ValueError(
                "Invalid chat config: services.chat_app.default_provider "
                f"'{default_provider}' is explicitly disabled via "
                f"services.chat_app.providers.{default_provider}.enabled=false"
            )

        timeout_path = "services.chat_app.client_timeout_seconds"
        timeout_raw = chat_cfg.get("client_timeout_seconds", 600)
        if isinstance(timeout_raw, bool):
            raise ValueError(f"Invalid field: '{timeout_path}' must be a positive number of seconds")
        try:
            timeout_value = float(timeout_raw)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid field: '{timeout_path}' must be a positive number of seconds")
        if timeout_value <= 0:
            raise ValueError(f"Invalid field: '{timeout_path}' must be > 0")
        if timeout_value > 86400:
            raise ValueError(f"Invalid field: '{timeout_path}' must be <= 86400 seconds")

    def _validate_benchmarking_config(self, config: Dict[str, Any], services: List[str]) -> None:
        if not services or "benchmarking" not in services:
            return

        services_cfg = config.get("services", {}) or {}
        benchmarking_cfg = services_cfg.get("benchmarking", {}) or {}

        required = [
            ("agent_class", "services.benchmarking.agent_class"),
            ("agent_md_file", "services.benchmarking.agent_md_file"),
            ("provider", "services.benchmarking.provider"),
            ("model", "services.benchmarking.model"),
        ]
        for key, path in required:
            value = benchmarking_cfg.get(key)
            if not value:
                raise ValueError(f"Missing required field: '{path}' in the configuration")

        if "agents_dir" in benchmarking_cfg:
            raise ValueError(
                "Unsupported field: 'services.benchmarking.agents_dir'. "
                "Use 'services.benchmarking.agent_md_file' instead."
            )
        if benchmarking_cfg.get("provider") == "local" and not benchmarking_cfg.get("ollama_url"):
            raise ValueError(
                "Missing required field: 'services.benchmarking.ollama_url' when provider is 'local'"
            )

        agent_md_file = Path(str(benchmarking_cfg.get("agent_md_file"))).expanduser()
        config_path = Path(str(config.get("_config_path", ""))).expanduser()
        if not agent_md_file.is_absolute() and config_path:
            candidate = (config_path.parent / agent_md_file).resolve()
            if candidate.exists():
                agent_md_file = candidate
        if not agent_md_file.exists():
            raise ValueError(f"agent_md_file not found: '{agent_md_file}'")
        if not agent_md_file.is_file():
            raise ValueError(f"agent_md_file must be a file: '{agent_md_file}'")
        if agent_md_file.suffix.lower() != ".md":
            raise ValueError(f"agent_md_file must be a markdown file (.md): '{agent_md_file}'")

    def _validate_source_fields(self, config: Dict[str, Any], sources: List[str]) -> None:
        if not sources:
            return

        for field in source_registry.required_config_fields(sources):
            value = self._get_value_from_path(config, field)
            if value in (None, ''):
                raise ValueError(f"Missing required field: '{field}' in the configuration")
            if isinstance(value, list) and not value and not field.endswith('input_lists'):
                raise ValueError(f"Missing required field: '{field}' in the configuration")

    def _get_value_from_path(self, config: Dict[str, Any], path: str) -> Any:
        value: Any = config
        for key in path.split('.'):
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                raise ValueError(f"Missing required field: '{path}' in the configuration")
        return value

    def _collect_embedding_metadata(self) -> None:
        embedding_models_used = list({conf.get('data_manager', {}).get('embedding_name', '') for conf in self.configs})
        self.embedding_models_used = ' '.join([model for model in embedding_models_used if model])

    def _collect_input_lists(self) -> None:
        collected: List[str] = []
        for conf in self.configs:
            data_manager = conf.get('data_manager', {})
            sources_section = data_manager.get('sources', {}) or {}
            links_section = sources_section.get('links', {}) if isinstance(sources_section, dict) else {}
            lists = links_section.get('input_lists') or []
            if isinstance(lists, list):
                collected.extend(lists)
        self.input_list = sorted(set(collected)) if collected else []

    def get_enabled_sources(self) -> List[str]:
        """Return sources marked as enabled across all configs."""
        valid_names = set(source_registry.names())
        enabled: Set[str] = set()

        for conf in self.configs:
            sources_section = conf.get('data_manager', {}).get('sources', {}) or {}
            for name, entry in sources_section.items():
                if name not in valid_names:
                    continue
                if isinstance(entry, dict):
                    if entry.get('enabled'):
                        enabled.add(name)
                elif isinstance(entry, bool) and entry:
                    enabled.add(name)

        return sorted(enabled)

    def get_disabled_sources(self) -> List[str]:
        """Return sources explicitly disabled across configs."""
        valid_names = set(source_registry.names())
        disabled: Set[str] = set()

        for conf in self.configs:
            sources_section = conf.get('data_manager', {}).get('sources', {}) or {}
            for name, entry in sources_section.items():
                if name not in valid_names:
                    continue
                if isinstance(entry, dict):
                    if entry.get('enabled') is False:
                        disabled.add(name)
                elif isinstance(entry, bool) and entry is False:
                    disabled.add(name)

        return sorted(disabled)

    def set_sources_enabled(self, enabled_sources: List[str]) -> None:
        enabled_set = set(enabled_sources or [])
        managed_sources = [name for name in source_registry.names() if name != 'links']

        for conf in self.configs:
            data_manager = conf.setdefault('data_manager', {})
            sources_section = data_manager.setdefault('sources', {})

            for name in managed_sources:
                entry = sources_section.setdefault(name, {})
                if name in enabled_set:
                    entry['enabled'] = True
                elif 'enabled' not in entry:
                    entry['enabled'] = False

            links_entry = sources_section.setdefault('links', {})
            links_entry.setdefault('enabled', True)
            links_entry.setdefault('input_lists', links_entry.get('input_lists', []))

    
    def get_configs(self) -> Dict[str, Any]:
        """Get the loaded configuration"""
        return self.configs
    
    def get_pipeline_configs(self) -> Dict[str, Any]:
        """Legacy pipeline configuration accessor (archi section removed)."""
        return [{}]
    
    def get_models_configs(self) -> Dict[str, Any]:
        """Legacy models configuration accessor (archi section removed)."""
        return []
    
    def get_prompts_config(self) -> Dict[str, Any]:
        """Legacy prompts configuration accessor (archi section removed)."""
        return []
    
    def get_interface_config(self, interface_name: str) -> Dict[str, Any]:
        """Get configuration for a specific interface"""
        return self.config.get("services", {}).get(interface_name, {})
    
    def get_embedding_name(self):
        return self.embedding_models_used
    
    def get_input_lists(self):
        return self.input_list
    
    def _get_all_models(self, config): 
        return set()
