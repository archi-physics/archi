import os
import warnings

import yaml

from src.utils.a2rchi_settings_store import load_a2rchi_settings

# DEFINITIONS
CONFIGS_PATH = "/root/A2rchi/configs/"

def _build_pg_config_from_env() -> dict:
    host = os.getenv("A2RCHI_PG_HOST") or os.getenv("PGHOST")
    port = os.getenv("A2RCHI_PG_PORT") or os.getenv("PGPORT")
    user = os.getenv("A2RCHI_PG_USER") or os.getenv("PGUSER")
    database = os.getenv("A2RCHI_PG_DATABASE") or os.getenv("PGDATABASE")
    if not host or not user or not database:
        return {}
    return {
        "host": host,
        "port": int(port) if port else 5432,
        "user": user,
        "database": database,
        "password": _read_pg_password(),
    }


def _read_pg_password() -> str:
    secret_filepath = os.getenv("PG_PASSWORD_FILE")
    if secret_filepath:
        with open(secret_filepath, "r") as f:
            return f.read().strip()
    return os.getenv("PG_PASSWORD", "")


def _load_config_from_postgres(name: str = None) -> dict:
    pg_config = _build_pg_config_from_env()
    if not pg_config:
        raise RuntimeError("Postgres configuration missing for runtime config loading.")

    try:
        import psycopg2
    except Exception as exc:
        raise RuntimeError("psycopg2 unavailable; cannot load runtime config.") from exc

    query = "SELECT config FROM configs ORDER BY config_id DESC LIMIT 1"
    params = ()
    if name:
        query = """
            SELECT config
            FROM configs
            WHERE config_name = %s
            ORDER BY config_id DESC
            LIMIT 1
        """
        params = (name,)

    try:
        with psycopg2.connect(**pg_config) as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                row = cursor.fetchone()
    except Exception as exc:
        raise RuntimeError(f"Failed to load config from Postgres: {exc}") from exc

    if not row or not row[0]:
        target = f"'{name}'" if name else "default"
        raise RuntimeError(f"No config found in Postgres for {target}.")

    payload = yaml.safe_load(row[0]) or {}
    if not isinstance(payload, dict):
        raise ValueError("Invalid config payload retrieved from Postgres.")
    _ensure_required_sections(payload)
    return payload


def _ensure_required_sections(payload: dict) -> None:
    required = ("global", "services", "data_manager", "a2rchi")
    missing = [section for section in required if section not in payload]
    if missing:
        raise RuntimeError(
            "Config in Postgres is missing required sections: "
            + ", ".join(missing)
        )


def load_runtime_config(map: bool = False, name: str = None):
    """
    Load the configuration specified by name, or the first one by default.
    Optionally maps models to the corresponding class.
    """

    if name is None:
        env_name = os.getenv("A2RCHI_CONFIG_NAME")
        config = _load_config_from_postgres(env_name)
    else:
        config = _load_config_from_postgres(name)

    config["_config_path"] = None

    config["a2rchi"] = load_a2rchi_settings()

    # change the model class parameter from a string to an actual class
    if map:

        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_openai import OpenAIEmbeddings

        from src.a2rchi.models import (VLLM, AnthropicLLM, DumbLLM,
                                       HuggingFaceImageLLM, HuggingFaceOpenLLM,
                                       LlamaLLM, OllamaInterface, OpenAILLM)
        from src.data_manager.collectors.scrapers.integrations.sso_scraper import \
            CERNSSOScraper
        
        MODEL_MAPPING = {
            "AnthropicLLM": AnthropicLLM,
            "OpenAIGPT4": OpenAILLM,
            "OpenAIGPT35": OpenAILLM,
            "DumbLLM": DumbLLM,
            "LlamaLLM": LlamaLLM,
            "HuggingFaceOpenLLM": HuggingFaceOpenLLM,
            "HuggingFaceImageLLM": HuggingFaceImageLLM,
            "VLLM": VLLM,
            "OllamaInterface": OllamaInterface, 
        }
        a2rchi_map = config.get("a2rchi", {}) or {}
        model_class_map = a2rchi_map.get("model_class_map", {}) or {}
        for model in model_class_map.keys():
            model_class_map[model]["class"] = MODEL_MAPPING[model]

        EMBEDDING_MAPPING = {
            "OpenAIEmbeddings": OpenAIEmbeddings,
            "HuggingFaceEmbeddings": HuggingFaceEmbeddings
        }
        for model in config["data_manager"]["embedding_class_map"].keys():
            config["data_manager"]["embedding_class_map"][model]["class"] = EMBEDDING_MAPPING[model]

        # change the SSO class parameter from a string to an actual class
        sso_section = config.get('utils', {}).get('sso', {}) or {}
        sources_sso = config.get('data_manager', {}).get('sources', {}).get('sso', {}) or {}
        active_sso_config = None
        if sources_sso.get('enabled'):
            active_sso_config = sources_sso
        elif sso_section.get('enabled'):
            active_sso_config = sso_section

        if active_sso_config:
            SSO_MAPPING = {
                'CERNSSOScraper': CERNSSOScraper,
            }
            sso_class_map = active_sso_config.get('sso_class_map', {})
            for sso_class in sso_class_map.keys():
                if sso_class in SSO_MAPPING:
                    sso_class_map[sso_class]['class'] = SSO_MAPPING[sso_class]

    return config

def load_runtime_global_config(name: str = None):
    """
    Load the global part of the config stored in Postgres.
    This is assumed to be static for a deployment.
    """

    config = _load_config_from_postgres(name or os.getenv("A2RCHI_CONFIG_NAME"))
    return config["global"]

def load_runtime_data_manager_config(name: str = None):
    """
    Load the data_manager part of the config stored in Postgres.
    This is assumed to be static for a deployment.
    """

    config = _load_config_from_postgres(name or os.getenv("A2RCHI_CONFIG_NAME"))
    return config["data_manager"]

def load_runtime_services_config(name: str = None):
    """
    Load the services part of the config stored in Postgres.
    This is assumed to be static for a deployment.
    """

    config = _load_config_from_postgres(name or os.getenv("A2RCHI_CONFIG_NAME"))
    return config["services"]

def get_runtime_config_names():
    """
    Gets the available configurations names.
    """

    pg_config = _build_pg_config_from_env()
    if not pg_config:
        raise RuntimeError("Postgres configuration missing for config listing.")

    try:
        import psycopg2
    except Exception as exc:
        raise RuntimeError("psycopg2 unavailable; cannot list configs.") from exc

    try:
        with psycopg2.connect(**pg_config) as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT DISTINCT config_name FROM configs ORDER BY config_name;")
                rows = cursor.fetchall()
    except Exception as exc:
        raise RuntimeError(f"Failed to list configs from Postgres: {exc}") from exc

    return [row[0] for row in rows]


def load_config(map: bool = False, name: str = None):
    warnings.warn(
        "load_config is deprecated; use load_runtime_config instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return load_runtime_config(map=map, name=name)


def load_global_config(name: str = None):
    warnings.warn(
        "load_global_config is deprecated; use load_runtime_global_config instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return load_runtime_global_config(name=name)


def load_data_manager_config(name: str = None):
    warnings.warn(
        "load_data_manager_config is deprecated; use load_runtime_data_manager_config instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return load_runtime_data_manager_config(name=name)


def load_services_config(name: str = None):
    warnings.warn(
        "load_services_config is deprecated; use load_runtime_services_config instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return load_runtime_services_config(name=name)


def get_config_names():
    warnings.warn(
        "get_config_names is deprecated; use get_runtime_config_names instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_runtime_config_names()
