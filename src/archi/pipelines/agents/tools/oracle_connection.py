"""OracleConnectionManager - manages connections to multiple Oracle databases.

Provides lazy connection pooling, config parsing, and password resolution
for the Oracle agent tools. Uses python-oracledb in thick mode when Oracle
Instant Client is available (required for NNE), falling back to thin mode.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional

from src.utils.logging import get_logger
from src.utils.env import read_secret

logger = get_logger(__name__)

# Lazy import: only attempt when Oracle databases are configured
_oracledb = None
_thick_mode_attempted = False


def _get_oracledb():
    """Lazily import oracledb and initialize thick mode if available."""
    global _oracledb, _thick_mode_attempted
    if _oracledb is None:
        try:
            import oracledb
            _oracledb = oracledb
        except ImportError:
            raise ImportError(
                "python-oracledb is required for Oracle database tools. "
                "Install it with: pip install oracledb>=2.0.0"
            )

    if not _thick_mode_attempted:
        _thick_mode_attempted = True
        try:
            lib_dir = os.environ.get("ORACLE_CLIENT_LIB")
            _oracledb.init_oracle_client(lib_dir=lib_dir)
            logger.info("Oracle thick mode enabled (NNE supported)")
        except _oracledb.ProgrammingError:
            # Already initialized — OK
            pass
        except Exception as e:
            logger.warning(
                "Could not enable Oracle thick mode (NNE may not work): %s. "
                "Falling back to thin mode.", e
            )

    return _oracledb


@dataclass
class OracleDBConfig:
    """Configuration for a single Oracle database connection."""

    name: str
    dsn: str
    user: str
    password_secret: str
    description: str = ""
    allowed_schemas: List[str] = field(default_factory=list)
    max_rows: int = 100
    query_timeout_seconds: int = 30
    pool_min: int = 1
    pool_max: int = 5


class OracleConnectionManager:
    """Manages connection pools for multiple Oracle databases.

    Pools are created lazily on first use. Each database is identified
    by its config name (e.g., "orders_prod", "inventory").

    Example:
        >>> mgr = OracleConnectionManager.from_config(oracle_databases)
        >>> with mgr.get_connection("orders_prod") as conn:
        ...     cursor = conn.cursor()
        ...     cursor.execute("SELECT 1 FROM DUAL")
        >>> mgr.close_all()
    """

    def __init__(self, db_configs: Dict[str, OracleDBConfig]) -> None:
        self._configs = db_configs
        self._pools: Dict[str, Any] = {}

    @classmethod
    def from_config(cls, oracle_databases: Dict[str, Any]) -> Optional["OracleConnectionManager"]:
        """Parse oracle database entries into an OracleConnectionManager.

        Args:
            oracle_databases: Mapping of database names to their config dicts,
                e.g. ``{"tier0_prod": {"dsn": "...", "user": "...", ...}}``.
                Typically sourced from ``services.chat_app.tools.oracle_databases``.

        Returns:
            OracleConnectionManager if valid databases are found, None otherwise.
        """
        if not oracle_databases:
            return None

        db_configs: Dict[str, OracleDBConfig] = {}
        for db_name, db_cfg in oracle_databases.items():
            if not isinstance(db_cfg, dict):
                logger.warning("Skipping invalid oracle_databases entry: %s", db_name)
                continue

            dsn = db_cfg.get("dsn")
            user = db_cfg.get("user")
            password_secret = db_cfg.get("password_secret")

            if not dsn or not user or not password_secret:
                logger.warning(
                    "Skipping Oracle database '%s': missing required field (dsn, user, or password_secret)",
                    db_name,
                )
                continue

            db_configs[db_name] = OracleDBConfig(
                name=db_name,
                dsn=dsn,
                user=user,
                password_secret=password_secret,
                description=db_cfg.get("description", ""),
                allowed_schemas=[
                    s.upper() for s in (db_cfg.get("allowed_schemas") or [])
                ],
                max_rows=db_cfg.get("max_rows", 100),
                query_timeout_seconds=db_cfg.get("query_timeout_seconds", 30),
                pool_min=db_cfg.get("pool_min", 1),
                pool_max=db_cfg.get("pool_max", 5),
            )

        if not db_configs:
            return None

        return cls(db_configs)

    def _ensure_pool(self, db_name: str) -> Any:
        """Create a connection pool for the named database if it doesn't exist."""
        if db_name in self._pools:
            return self._pools[db_name]

        if db_name not in self._configs:
            raise ValueError(
                f"Unknown Oracle database '{db_name}'. "
                f"Available: {', '.join(sorted(self._configs.keys()))}"
            )

        oracledb = _get_oracledb()
        cfg = self._configs[db_name]

        password = read_secret(cfg.password_secret)
        if not password:
            raise ValueError(
                f"Could not resolve password for Oracle database '{db_name}' "
                f"(secret: {cfg.password_secret}). "
                f"Set the {cfg.password_secret} environment variable or "
                f"{cfg.password_secret}_FILE Docker secret."
            )

        pool = oracledb.create_pool(
            user=cfg.user,
            password=password,
            dsn=cfg.dsn,
            min=cfg.pool_min,
            max=cfg.pool_max,
        )
        self._pools[db_name] = pool
        logger.info(
            "Created Oracle connection pool for '%s' (min=%d, max=%d)",
            db_name, cfg.pool_min, cfg.pool_max,
        )
        return pool

    @contextmanager
    def get_connection(self, db_name: str) -> Generator:
        """Acquire a connection from the pool for the named database.

        Sets call_timeout from the database config.

        Yields:
            An oracledb connection.
        """
        pool = self._ensure_pool(db_name)
        cfg = self._configs[db_name]
        conn = pool.acquire()
        try:
            conn.call_timeout = cfg.query_timeout_seconds * 1000  # milliseconds
            yield conn
        finally:
            pool.release(conn)

    def get_db_config(self, db_name: str) -> OracleDBConfig:
        """Return the config for a named database.

        Raises:
            ValueError: If db_name is not configured.
        """
        if db_name not in self._configs:
            raise ValueError(
                f"Unknown Oracle database '{db_name}'. "
                f"Available: {', '.join(sorted(self._configs.keys()))}"
            )
        return self._configs[db_name]

    def list_databases(self) -> List[Dict[str, str]]:
        """Return names and descriptions of all configured databases."""
        result = []
        for name, cfg in sorted(self._configs.items()):
            entry = {"name": name}
            if cfg.description:
                entry["description"] = cfg.description
            result.append(entry)
        return result

    def has_databases(self) -> bool:
        """Return True if any Oracle databases are configured."""
        return bool(self._configs)

    def close_all(self) -> None:
        """Close all connection pools."""
        for db_name, pool in self._pools.items():
            try:
                pool.close(force=True)
                logger.info("Closed Oracle connection pool for '%s'", db_name)
            except Exception as e:
                logger.warning("Error closing Oracle pool '%s': %s", db_name, e)
        self._pools.clear()
