"""
Config access helpers that read from ConfigService / Postgres only.
"""

from typing import Any, Dict

from src.utils.postgres_service_factory import PostgresServiceFactory
from src.utils.config_service import ConfigService


class ConfigNotReadyError(RuntimeError):
    pass


def _config_service() -> ConfigService:
    factory = PostgresServiceFactory.get_instance()
    if not factory:
        raise ConfigNotReadyError("PostgresServiceFactory not initialized. Set it before accessing config.")
    return factory.config_service


def get_static_config():
    cfg = _config_service().get_static_config()
    if cfg is None:
        raise ConfigNotReadyError("Static config not initialized in Postgres.")
    return cfg


def get_dynamic_config():
    return _config_service().get_dynamic_config()


def get_global_config() -> Dict[str, Any]:
    return get_static_config().global_config or {}


def get_services_config() -> Dict[str, Any]:
    return get_static_config().services_config or {}


def get_data_manager_config(*, resolve_embeddings: bool = False) -> Dict[str, Any]:
    """
    Return the data_manager config.

    Set resolve_embeddings=True to map embedding_class_map 'class' entries
    from string names to actual callables using ConfigService.
    """
    static = get_static_config()
    data_manager = dict(static.data_manager_config or {})

    if resolve_embeddings:
        try:
            resolved_map = _config_service().get_embedding_class_map(resolved=True)
            if resolved_map:
                data_manager["embedding_class_map"] = resolved_map
        except Exception:
            # Leave unresolved; caller can handle if needed
            pass

    return data_manager


def get_archi_config() -> Dict[str, Any]:
    return get_static_config().archi_config or {}


def get_mcp_servers_config() -> Dict[str, Any]:
    return get_static_config().mcp_servers_config or {}


def get_full_config(*, resolve_embeddings: bool = False) -> Dict[str, Any]:
    """
    Return the full merged config.

    Set resolve_embeddings=True to map embedding_class_map 'class' entries
    from string names to actual callables using ConfigService.
    """
    static = get_static_config()

    data_manager_config = dict(static.data_manager_config or {})
    if resolve_embeddings:
        try:
            resolved_map = _config_service().get_embedding_class_map(resolved=True)
            if resolved_map:
                data_manager_config["embedding_class_map"] = resolved_map
        except Exception:
            pass

    return {
        "name": static.deployment_name,
        "config_version": static.config_version,
        "global": static.global_config,
        "services": static.services_config,
        "data_manager": data_manager_config,
        "archi": static.archi_config,
        "sources": static.sources_config,
        "mcp_servers": static.mcp_servers_config or {},
        "available_pipelines": static.available_pipelines,
        "available_models": static.available_models,
        "available_providers": static.available_providers,
    }
