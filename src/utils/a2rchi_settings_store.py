import copy
import logging
import os
from typing import Dict, List, Tuple

from src.utils.env import read_secret

LOGGER = logging.getLogger(__name__)

TABLE_SETTINGS = "a2rchi_settings"
TABLE_MODEL_OPTIONS = "a2rchi_model_options"
TABLE_PIPELINE_OPTIONS = "a2rchi_pipeline_options"
TABLE_AGENT_OPTIONS = "a2rchi_agent_options"
TABLE_TOOL_OPTIONS = "a2rchi_tool_options"
TABLE_MCP_SERVER_OPTIONS = "a2rchi_mcp_server_options"


def load_a2rchi_settings() -> Dict:
    """
    Load the a2rchi settings from Postgres.
    Fails fast when Postgres is unavailable or empty.
    """
    pg_config = _build_pg_config()
    if not pg_config:
        raise RuntimeError("Postgres configuration missing for a2rchi settings.")

    psycopg2 = _load_psycopg2()
    if psycopg2 is None:
        raise RuntimeError("psycopg2 unavailable; cannot load a2rchi settings.")

    try:
        conn = psycopg2.connect(**pg_config)
    except Exception as exc:
        raise RuntimeError(f"Failed to connect to Postgres for settings: {exc}") from exc

    try:
        with conn:
            with conn.cursor() as cursor:
                _ensure_tables(cursor)
                settings_id, settings_payload = _fetch_latest_settings(cursor)
                if settings_payload is None:
                    raise RuntimeError("No a2rchi settings found in Postgres.")

                (
                    model_opts,
                    pipeline_opts,
                    agent_opts,
                    tool_opts,
                    mcp_opts,
                ) = _fetch_option_catalogs(cursor)
                if not model_opts and not pipeline_opts and not agent_opts:
                    _seed_options_from_a2rchi(cursor, settings_payload)
                    (
                        model_opts,
                        pipeline_opts,
                        agent_opts,
                        tool_opts,
                        mcp_opts,
                    ) = _fetch_option_catalogs(cursor)

                composed = compose_a2rchi_settings(
                    settings_payload,
                    model_opts,
                    pipeline_opts,
                    agent_opts,
                    tool_opts,
                    mcp_opts,
                )
                _upsert_settings(cursor, settings_id, settings_payload, composed)
                return composed
    finally:
        conn.close()


def compose_a2rchi_settings(
    base_a2rchi: Dict,
    model_options: List[Tuple[str, Dict]],
    pipeline_options: List[Tuple[str, Dict, bool]],
    agent_options: List[Tuple[str, Dict, bool]],
    tool_options: List[Tuple[str, str, bool]],
    mcp_server_options: List[Tuple[str, Dict, bool]],
) -> Dict:
    """
    Build a2rchi settings by composing option catalogs onto a base settings dict.
    """
    composed = copy.deepcopy(base_a2rchi)

    if model_options:
        composed["model_class_map"] = {name: config for name, config in model_options}

    if pipeline_options or agent_options:
        pipeline_map = {name: config for name, config, _ in pipeline_options}
        pipeline_map.update({name: config for name, config, _ in agent_options})
        composed["pipeline_map"] = pipeline_map

    if pipeline_options:
        enabled = [name for name, _, is_enabled in pipeline_options if is_enabled]
        composed["pipelines"] = enabled

    if tool_options:
        tool_map: Dict[str, List[str]] = {}
        for agent_name, tool_name, enabled in tool_options:
            if enabled:
                tool_map.setdefault(agent_name, []).append(tool_name)
        pipeline_map = composed.get("pipeline_map", {})
        for agent_name, tools in tool_map.items():
            pipeline_map.setdefault(agent_name, {})["tools"] = {"enabled": tools}
        composed["pipeline_map"] = pipeline_map

    if mcp_server_options:
        mcp_servers: Dict[str, Dict] = {}
        for name, config, enabled in mcp_server_options:
            if enabled:
                mcp_servers[name] = config
        composed["mcp_servers"] = mcp_servers

    return composed


def derive_option_catalogs(
    a2rchi: Dict,
) -> Tuple[
    List[Tuple[str, Dict]],
    List[Tuple[str, Dict, bool]],
    List[Tuple[str, Dict, bool]],
    List[Tuple[str, str, bool]],
    List[Tuple[str, Dict, bool]],
]:
    """
    Derive option catalogs (models, pipelines, agents) from an a2rchi config dict.
    """
    model_class_map = a2rchi.get("model_class_map") or {}
    pipeline_map = a2rchi.get("pipeline_map") or {}
    enabled_pipelines = set(a2rchi.get("pipelines") or [])

    model_options = [(name, config) for name, config in model_class_map.items()]
    pipeline_options = []
    agent_options = []
    for name, config in pipeline_map.items():
        entry = (name, config, name in enabled_pipelines)
        if _is_agent_name(name):
            agent_options.append(entry)
        else:
            pipeline_options.append(entry)

    tool_options: List[Tuple[str, str, bool]] = []
    for name, config in pipeline_map.items():
        if not _is_agent_name(name) or not isinstance(config, dict):
            continue
        tools_cfg = config.get("tools", {}) or {}
        for tool_name in tools_cfg.get("enabled") or []:
            tool_options.append((name, tool_name, True))

    mcp_servers = a2rchi.get("mcp_servers", {}) or {}
    mcp_server_options: List[Tuple[str, Dict, bool]] = []
    for name, cfg in mcp_servers.items():
        enabled = True
        resolved_cfg = cfg
        if isinstance(cfg, dict) and "enabled" in cfg:
            enabled = bool(cfg.get("enabled"))
            resolved_cfg = {k: v for k, v in cfg.items() if k != "enabled"}
        if isinstance(resolved_cfg, dict):
            mcp_server_options.append((name, resolved_cfg, enabled))

    return (
        model_options,
        pipeline_options,
        agent_options,
        tool_options,
        mcp_server_options,
    )


def _is_agent_name(name: str) -> bool:
    return name.lower().endswith("agent")


def _build_pg_config() -> Dict:
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
        "password": read_secret("PG_PASSWORD"),
    }


def _ensure_tables(cursor) -> None:
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_SETTINGS} (
            settings_id SERIAL PRIMARY KEY,
            a2rchi JSONB NOT NULL,
            updated_at TIMESTAMP NOT NULL DEFAULT NOW()
        );
        """
    )
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_MODEL_OPTIONS} (
            name TEXT PRIMARY KEY,
            config JSONB NOT NULL
        );
        """
    )
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_PIPELINE_OPTIONS} (
            name TEXT PRIMARY KEY,
            config JSONB NOT NULL,
            enabled BOOLEAN NOT NULL DEFAULT FALSE
        );
        """
    )
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_AGENT_OPTIONS} (
            name TEXT PRIMARY KEY,
            config JSONB NOT NULL,
            enabled BOOLEAN NOT NULL DEFAULT FALSE
        );
        """
    )
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_TOOL_OPTIONS} (
            agent_name TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            enabled BOOLEAN NOT NULL DEFAULT TRUE,
            PRIMARY KEY (agent_name, tool_name)
        );
        """
    )
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_MCP_SERVER_OPTIONS} (
            name TEXT PRIMARY KEY,
            config JSONB NOT NULL,
            enabled BOOLEAN NOT NULL DEFAULT TRUE
        );
        """
    )


def _fetch_latest_settings(cursor) -> Tuple[int, Dict]:
    cursor.execute(
        f"""
        SELECT settings_id, a2rchi
        FROM {TABLE_SETTINGS}
        ORDER BY settings_id DESC
        LIMIT 1;
        """
    )
    row = cursor.fetchone()
    if not row:
        return None, None
    return row[0], row[1]


def _fetch_option_catalogs(cursor):
    cursor.execute(f"SELECT name, config FROM {TABLE_MODEL_OPTIONS};")
    model_options = [(row[0], row[1]) for row in cursor.fetchall()]

    cursor.execute(f"SELECT name, config, enabled FROM {TABLE_PIPELINE_OPTIONS};")
    pipeline_options = [(row[0], row[1], row[2]) for row in cursor.fetchall()]

    cursor.execute(f"SELECT name, config, enabled FROM {TABLE_AGENT_OPTIONS};")
    agent_options = [(row[0], row[1], row[2]) for row in cursor.fetchall()]

    cursor.execute(f"SELECT agent_name, tool_name, enabled FROM {TABLE_TOOL_OPTIONS};")
    tool_options = [(row[0], row[1], row[2]) for row in cursor.fetchall()]

    cursor.execute(f"SELECT name, config, enabled FROM {TABLE_MCP_SERVER_OPTIONS};")
    mcp_server_options = [(row[0], row[1], row[2]) for row in cursor.fetchall()]

    return (
        model_options,
        pipeline_options,
        agent_options,
        tool_options,
        mcp_server_options,
    )


def _seed_options_from_a2rchi(cursor, a2rchi: Dict) -> None:
    psycopg2 = _load_psycopg2()
    if psycopg2 is None:
        return
    (
        model_options,
        pipeline_options,
        agent_options,
        tool_options,
        mcp_server_options,
    ) = derive_option_catalogs(a2rchi)

    if model_options:
        psycopg2.extras.execute_values(
            cursor,
            f"""
            INSERT INTO {TABLE_MODEL_OPTIONS} (name, config)
            VALUES %s
            ON CONFLICT (name) DO UPDATE
            SET config = EXCLUDED.config;
            """,
            [
                (name, psycopg2.extras.Json(config))
                for name, config in model_options
            ],
        )

    if pipeline_options:
        psycopg2.extras.execute_values(
            cursor,
            f"""
            INSERT INTO {TABLE_PIPELINE_OPTIONS} (name, config, enabled)
            VALUES %s
            ON CONFLICT (name) DO UPDATE
            SET config = EXCLUDED.config,
                enabled = EXCLUDED.enabled;
            """,
            [
                (name, psycopg2.extras.Json(config), enabled)
                for name, config, enabled in pipeline_options
            ],
        )

    if agent_options:
        psycopg2.extras.execute_values(
            cursor,
            f"""
            INSERT INTO {TABLE_AGENT_OPTIONS} (name, config, enabled)
            VALUES %s
            ON CONFLICT (name) DO UPDATE
            SET config = EXCLUDED.config,
                enabled = EXCLUDED.enabled;
            """,
            [
                (name, psycopg2.extras.Json(config), enabled)
                for name, config, enabled in agent_options
            ],
        )

    if tool_options:
        psycopg2.extras.execute_values(
            cursor,
            f"""
            INSERT INTO {TABLE_TOOL_OPTIONS} (agent_name, tool_name, enabled)
            VALUES %s
            ON CONFLICT (agent_name, tool_name) DO UPDATE
            SET enabled = EXCLUDED.enabled;
            """,
            [(agent, tool, enabled) for agent, tool, enabled in tool_options],
        )

    if mcp_server_options:
        psycopg2.extras.execute_values(
            cursor,
            f"""
            INSERT INTO {TABLE_MCP_SERVER_OPTIONS} (name, config, enabled)
            VALUES %s
            ON CONFLICT (name) DO UPDATE
            SET config = EXCLUDED.config,
                enabled = EXCLUDED.enabled;
            """,
            [
                (name, psycopg2.extras.Json(config), enabled)
                for name, config, enabled in mcp_server_options
            ],
        )


def _upsert_settings(cursor, settings_id, current_payload, new_payload) -> None:
    psycopg2 = _load_psycopg2()
    if psycopg2 is None:
        return
    if current_payload == new_payload and settings_id is not None:
        return

    if settings_id is None:
        cursor.execute(
            f"INSERT INTO {TABLE_SETTINGS} (a2rchi) VALUES (%s);",
            (psycopg2.extras.Json(new_payload),),
        )
        return

    cursor.execute(
        f"""
        UPDATE {TABLE_SETTINGS}
        SET a2rchi = %s,
            updated_at = NOW()
        WHERE settings_id = %s;
        """,
        (psycopg2.extras.Json(new_payload), settings_id),
    )


def _load_psycopg2():
    try:
        import psycopg2
        import psycopg2.extras
        return psycopg2
    except Exception:
        return None
