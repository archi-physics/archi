"""
Unit tests for Oracle agent tools.

Tests cover:
- SQL safety utilities (validate_select_only, check_schema_allowlist, ensure_row_limit)
- Result formatting (markdown table rendering, truncation)
- OracleConnectionManager (config parsing, list_databases, error handling)
- Query tool (full flow with mocked connection)
- Schema discovery tool (list/tables/columns modes)

NOTE: We use importlib.util to load oracle_* modules directly by file path,
bypassing the src.archi.pipelines.__init__ import chain which pulls in
langchain_classic and other heavy dependencies not needed by these tests.
"""

import importlib.util
import os
import sys
import types
import pytest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Bootstrap: register parent packages as empty modules in sys.modules so
# that absolute imports like "from src.archi.pipelines.agents.tools.X import Y"
# inside the oracle modules resolve without triggering __init__.py chains.
# ---------------------------------------------------------------------------

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

_PACKAGES_TO_STUB = [
    "src",
    "src.archi",
    "src.archi.pipelines",
    "src.archi.pipelines.agents",
    "src.archi.pipelines.agents.tools",
    "src.utils",
]

for _pkg in _PACKAGES_TO_STUB:
    if _pkg not in sys.modules:
        mod = types.ModuleType(_pkg)
        mod.__path__ = [os.path.join(_ROOT, *_pkg.split("."))]
        mod.__package__ = _pkg
        sys.modules[_pkg] = mod


def _load_module(name: str, filepath: str):
    """Load a module from an absolute file path and register it in sys.modules."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load src.utils.logging and src.utils.env (needed by oracle_connection)
_src = os.path.join(_ROOT, "src")

_utils_logging = _load_module(
    "src.utils.logging",
    os.path.join(_src, "utils", "logging.py"),
)

_utils_env = _load_module(
    "src.utils.env",
    os.path.join(_src, "utils", "env.py"),
)

# Load the Oracle modules under test
_safety = _load_module(
    "src.archi.pipelines.agents.tools.oracle_safety",
    os.path.join(_src, "archi", "pipelines", "agents", "tools", "oracle_safety.py"),
)

_connection = _load_module(
    "src.archi.pipelines.agents.tools.oracle_connection",
    os.path.join(_src, "archi", "pipelines", "agents", "tools", "oracle_connection.py"),
)

_query = _load_module(
    "src.archi.pipelines.agents.tools.oracle_query",
    os.path.join(_src, "archi", "pipelines", "agents", "tools", "oracle_query.py"),
)

# Pull out symbols for convenience
SQLValidationError = _safety.SQLValidationError
validate_select_only = _safety.validate_select_only
check_schema_allowlist = _safety.check_schema_allowlist
ensure_row_limit = _safety.ensure_row_limit

OracleConnectionManager = _connection.OracleConnectionManager
OracleDBConfig = _connection.OracleDBConfig

format_results_as_markdown = _query.format_results_as_markdown
create_oracle_query_tool = _query.create_oracle_query_tool
create_oracle_schema_tool = _query.create_oracle_schema_tool


# =============================================================================
# 9.1 - SQL Safety Utilities
# =============================================================================


class TestValidateSelectOnly:
    """Test validate_select_only rejects non-SELECT statements."""

    def test_simple_select(self):
        validate_select_only("SELECT * FROM my_table")

    def test_with_clause(self):
        validate_select_only("WITH cte AS (SELECT 1 FROM DUAL) SELECT * FROM cte")

    def test_select_with_leading_comment(self):
        validate_select_only("-- some comment\nSELECT 1 FROM DUAL")

    def test_empty_query_rejected(self):
        with pytest.raises(SQLValidationError, match="Empty"):
            validate_select_only("")

    def test_whitespace_only_rejected(self):
        with pytest.raises(SQLValidationError, match="Empty"):
            validate_select_only("   ")

    @pytest.mark.parametrize("keyword", [
        "INSERT", "UPDATE", "DELETE", "MERGE",
        "CREATE", "ALTER", "DROP", "TRUNCATE",
        "GRANT", "REVOKE", "EXECUTE", "CALL",
        "BEGIN", "DECLARE", "COMMIT", "ROLLBACK",
    ])
    def test_forbidden_keywords_rejected(self, keyword):
        with pytest.raises(SQLValidationError, match="Forbidden keyword"):
            validate_select_only(f"SELECT 1 FROM DUAL; {keyword} INTO foo VALUES (1)")

    def test_insert_leading_rejected(self):
        with pytest.raises(SQLValidationError, match="must start with SELECT"):
            validate_select_only("INSERT INTO foo VALUES (1)")

    def test_drop_leading_rejected(self):
        with pytest.raises(SQLValidationError, match="must start with SELECT"):
            validate_select_only("DROP TABLE foo")

    def test_trailing_semicolon_stripped(self):
        validate_select_only("SELECT 1 FROM DUAL;")


class TestCheckSchemaAllowlist:
    """Test check_schema_allowlist validates schema-qualified references."""

    def test_allowed_schema_passes(self):
        check_schema_allowlist(
            "SELECT * FROM CMS_WMBS.jobs",
            ["CMS_WMBS"],
        )

    def test_disallowed_schema_rejected(self):
        with pytest.raises(SQLValidationError, match="not allowed"):
            check_schema_allowlist(
                "SELECT * FROM SECRET_SCHEMA.passwords",
                ["CMS_WMBS"],
            )

    def test_none_allowlist_allows_all(self):
        check_schema_allowlist("SELECT * FROM ANY_SCHEMA.any_table", None)

    def test_empty_allowlist_allows_all(self):
        check_schema_allowlist("SELECT * FROM ANY_SCHEMA.any_table", [])

    def test_case_insensitive_matching(self):
        check_schema_allowlist(
            "SELECT * FROM cms_wmbs.jobs",
            ["CMS_WMBS"],
        )

    def test_dual_always_allowed(self):
        check_schema_allowlist(
            "SELECT * FROM DUAL.something",
            ["CMS_WMBS"],
        )

    def test_multiple_schemas_mixed(self):
        with pytest.raises(SQLValidationError, match="BAD_SCHEMA"):
            check_schema_allowlist(
                "SELECT a.x, b.y FROM CMS_WMBS.jobs a JOIN BAD_SCHEMA.stuff b ON a.id=b.id",
                ["CMS_WMBS"],
            )

    def test_multiple_allowed_schemas(self):
        check_schema_allowlist(
            "SELECT * FROM SCHEMA_A.t1 JOIN SCHEMA_B.t2 ON t1.id = t2.id",
            ["SCHEMA_A", "SCHEMA_B"],
        )


class TestEnsureRowLimit:
    """Test ensure_row_limit appends FETCH FIRST when missing."""

    def test_adds_fetch_first(self):
        result = ensure_row_limit("SELECT * FROM foo", 100)
        assert "FETCH FIRST 100 ROWS ONLY" in result

    def test_preserves_existing_fetch_first(self):
        sql = "SELECT * FROM foo FETCH FIRST 50 ROWS ONLY"
        result = ensure_row_limit(sql, 100)
        assert "FETCH FIRST 100" not in result
        assert "FETCH FIRST 50 ROWS ONLY" in result

    def test_preserves_rownum(self):
        sql = "SELECT * FROM foo WHERE ROWNUM <= 50"
        result = ensure_row_limit(sql, 100)
        assert "FETCH FIRST" not in result

    def test_strips_trailing_semicolon(self):
        result = ensure_row_limit("SELECT * FROM foo;", 100)
        assert not result.rstrip().endswith(";")
        assert "FETCH FIRST 100 ROWS ONLY" in result


# =============================================================================
# 9.2 - Result Formatting
# =============================================================================


class TestFormatResultsAsMarkdown:
    """Test format_results_as_markdown renders tables correctly."""

    def test_basic_table(self):
        columns = ["ID", "NAME"]
        rows = [(1, "Alice"), (2, "Bob")]
        result = format_results_as_markdown(columns, rows)
        assert "| ID | NAME |" in result
        assert "| --- | --- |" in result
        assert "| 1 | Alice |" in result
        assert "| 2 | Bob |" in result

    def test_empty_rows(self):
        result = format_results_as_markdown(["ID"], [])
        assert result == "Query returned 0 rows."

    def test_null_values(self):
        result = format_results_as_markdown(["COL"], [(None,)])
        assert "| NULL |" in result

    def test_truncation(self):
        columns = ["ID", "DATA"]
        rows = [(i, "x" * 50) for i in range(100)]
        result = format_results_as_markdown(columns, rows, max_chars=500)
        assert "output truncated" in result
        assert "Showing" in result
        assert "of 100 rows" in result

    def test_single_row_never_truncated(self):
        """Even if a single row exceeds max_chars, it should still be shown."""
        columns = ["DATA"]
        rows = [("x" * 5000,)]
        result = format_results_as_markdown(columns, rows, max_chars=100)
        assert "x" * 100 in result


# =============================================================================
# 9.3 - OracleConnectionManager
# =============================================================================


class TestOracleConnectionManager:
    """Test OracleConnectionManager config parsing and management."""

    def test_from_config_with_valid_databases(self):
        oracle_databases = {
            "test_db": {
                "dsn": "localhost:1521/testservice",
                "user": "testuser",
                "password_secret": "TEST_PASSWORD",
                "description": "Test database",
                "allowed_schemas": ["SCHEMA_A"],
                "max_rows": 200,
                "query_timeout_seconds": 60,
            }
        }
        mgr = OracleConnectionManager.from_config(oracle_databases)
        assert mgr is not None
        assert mgr.has_databases()

    def test_from_config_returns_none_when_empty(self):
        assert OracleConnectionManager.from_config({}) is None

    def test_from_config_skips_invalid_entries(self):
        oracle_databases = {
            "valid_db": {
                "dsn": "host:1521/svc",
                "user": "user",
                "password_secret": "SECRET",
            },
            "bad_db": {
                "dsn": "host:1521/svc",
                # missing user and password_secret
            },
            "not_a_dict": "garbage",
        }
        mgr = OracleConnectionManager.from_config(oracle_databases)
        assert mgr is not None
        dbs = mgr.list_databases()
        assert len(dbs) == 1
        assert dbs[0]["name"] == "valid_db"

    def test_list_databases(self):
        oracle_databases = {
            "db_a": {
                "dsn": "h:1521/s",
                "user": "u",
                "password_secret": "P",
                "description": "Database A",
            },
            "db_b": {
                "dsn": "h:1521/s",
                "user": "u",
                "password_secret": "P",
            },
        }
        mgr = OracleConnectionManager.from_config(oracle_databases)
        dbs = mgr.list_databases()
        assert len(dbs) == 2
        names = {d["name"] for d in dbs}
        assert names == {"db_a", "db_b"}
        db_a = next(d for d in dbs if d["name"] == "db_a")
        db_b = next(d for d in dbs if d["name"] == "db_b")
        assert db_a["description"] == "Database A"
        assert "description" not in db_b

    def test_get_db_config_unknown_raises(self):
        oracle_databases = {
            "real_db": {
                "dsn": "h:1521/s",
                "user": "u",
                "password_secret": "P",
            }
        }
        mgr = OracleConnectionManager.from_config(oracle_databases)
        with pytest.raises(ValueError, match="Unknown Oracle database"):
            mgr.get_db_config("nonexistent")

    def test_config_defaults(self):
        oracle_databases = {
            "db": {
                "dsn": "h:1521/s",
                "user": "u",
                "password_secret": "P",
            }
        }
        mgr = OracleConnectionManager.from_config(oracle_databases)
        cfg = mgr.get_db_config("db")
        assert cfg.max_rows == 100
        assert cfg.query_timeout_seconds == 30
        assert cfg.pool_min == 1
        assert cfg.pool_max == 5
        assert cfg.allowed_schemas == []
        assert cfg.description == ""


# =============================================================================
# 9.4 - Query Tool
# =============================================================================


class TestOracleQueryTool:
    """Test create_oracle_query_tool with mocked Oracle connections."""

    @pytest.fixture
    def manager(self):
        cfg = OracleDBConfig(
            name="test_db",
            dsn="host:1521/svc",
            user="user",
            password_secret="SECRET",
            description="Test DB",
            allowed_schemas=["MY_SCHEMA"],
            max_rows=100,
            query_timeout_seconds=30,
        )
        return OracleConnectionManager({"test_db": cfg})

    @pytest.fixture
    def mock_connection(self):
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value = cursor
        return conn, cursor

    def test_valid_query_returns_markdown(self, manager, mock_connection):
        conn, cursor = mock_connection
        cursor.description = [("ID",), ("NAME",)]
        cursor.fetchall.return_value = [(1, "Alice"), (2, "Bob")]

        with patch.object(manager, "get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__ = MagicMock(return_value=conn)
            mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)

            tool = create_oracle_query_tool(manager)
            result = tool.invoke({"db_name": "test_db", "sql": "SELECT * FROM MY_SCHEMA.users"})

        assert "| ID | NAME |" in result
        assert "| 1 | Alice |" in result

    def test_dml_rejected(self, manager):
        tool = create_oracle_query_tool(manager)
        result = tool.invoke({"db_name": "test_db", "sql": "DELETE FROM MY_SCHEMA.users"})
        assert "rejected" in result.lower()

    def test_disallowed_schema_rejected(self, manager):
        tool = create_oracle_query_tool(manager)
        result = tool.invoke({"db_name": "test_db", "sql": "SELECT * FROM SECRET.data"})
        assert "rejected" in result.lower() or "not allowed" in result.lower()

    def test_unknown_db_returns_available(self, manager):
        tool = create_oracle_query_tool(manager)
        result = tool.invoke({"db_name": "nonexistent", "sql": "SELECT 1 FROM DUAL"})
        assert "test_db" in result

    def test_empty_db_name_lists_available(self, manager):
        tool = create_oracle_query_tool(manager)
        result = tool.invoke({"db_name": "", "sql": "SELECT 1 FROM DUAL"})
        assert "test_db" in result

    def test_empty_sql_rejected(self, manager):
        tool = create_oracle_query_tool(manager)
        result = tool.invoke({"db_name": "test_db", "sql": ""})
        assert "provide" in result.lower()

    def test_empty_result_returns_zero_rows(self, manager, mock_connection):
        conn, cursor = mock_connection
        cursor.description = [("ID",)]
        cursor.fetchall.return_value = []

        with patch.object(manager, "get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__ = MagicMock(return_value=conn)
            mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)

            tool = create_oracle_query_tool(manager)
            result = tool.invoke({"db_name": "test_db", "sql": "SELECT * FROM MY_SCHEMA.empty_table"})

        assert "0 rows" in result

    def test_timeout_error_handled(self, manager):
        with patch.object(manager, "get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__ = MagicMock(
                side_effect=Exception("DPI-1067: call timeout of 30000 ms exceeded")
            )
            mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)

            tool = create_oracle_query_tool(manager)
            result = tool.invoke({"db_name": "test_db", "sql": "SELECT * FROM MY_SCHEMA.big_table"})

        assert "timed out" in result.lower()


# =============================================================================
# 9.5 - Schema Discovery Tool
# =============================================================================


class TestOracleSchemaDiscoveryTool:
    """Test create_oracle_schema_tool with mocked Oracle connections."""

    @pytest.fixture
    def manager(self):
        cfg = OracleDBConfig(
            name="test_db",
            dsn="host:1521/svc",
            user="user",
            password_secret="SECRET",
            description="Test DB",
            allowed_schemas=["MY_SCHEMA"],
            max_rows=100,
            query_timeout_seconds=30,
        )
        return OracleConnectionManager({"test_db": cfg})

    @pytest.fixture
    def mock_connection(self):
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value = cursor
        return conn, cursor

    def test_list_mode(self, manager):
        tool = create_oracle_schema_tool(manager)
        result = tool.invoke({"db_name": "list"})
        assert "test_db" in result
        assert "Test DB" in result

    def test_list_mode_empty_db_name(self, manager):
        tool = create_oracle_schema_tool(manager)
        result = tool.invoke({"db_name": ""})
        assert "test_db" in result

    def test_tables_mode(self, manager, mock_connection):
        conn, cursor = mock_connection
        cursor.fetchall.return_value = [
            ("MY_SCHEMA", "USERS", "TABLE"),
            ("MY_SCHEMA", "ORDERS", "TABLE"),
            ("MY_SCHEMA", "USER_SUMMARY", "VIEW"),
        ]

        with patch.object(manager, "get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__ = MagicMock(return_value=conn)
            mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)

            tool = create_oracle_schema_tool(manager)
            result = tool.invoke({"db_name": "test_db"})

        assert "USERS" in result
        assert "ORDERS" in result
        assert "USER_SUMMARY" in result
        assert "MY_SCHEMA" in result

    def test_columns_mode(self, manager, mock_connection):
        conn, cursor = mock_connection
        cursor.fetchall.return_value = [
            ("MY_SCHEMA", "USERS", "ID", "NUMBER", 22, "N"),
            ("MY_SCHEMA", "USERS", "NAME", "VARCHAR2", 100, "Y"),
            ("MY_SCHEMA", "USERS", "EMAIL", "VARCHAR2", 255, "Y"),
        ]

        with patch.object(manager, "get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__ = MagicMock(return_value=conn)
            mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)

            tool = create_oracle_schema_tool(manager)
            result = tool.invoke({"db_name": "test_db", "table_name": "MY_SCHEMA.USERS"})

        assert "ID" in result
        assert "NAME" in result
        assert "VARCHAR2(100)" in result
        assert "| Column | Type | Nullable |" in result

    def test_disallowed_schema_in_columns_mode(self, manager):
        tool = create_oracle_schema_tool(manager)
        result = tool.invoke({"db_name": "test_db", "table_name": "SECRET.PASSWORDS"})
        assert "not allowed" in result.lower()

    def test_unknown_db_returns_error(self, manager):
        tool = create_oracle_schema_tool(manager)
        result = tool.invoke({"db_name": "nonexistent"})
        assert "Unknown" in result or "test_db" in result

    def test_table_not_found(self, manager, mock_connection):
        conn, cursor = mock_connection
        cursor.fetchall.return_value = []

        with patch.object(manager, "get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__ = MagicMock(return_value=conn)
            mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)

            tool = create_oracle_schema_tool(manager)
            result = tool.invoke({"db_name": "test_db", "table_name": "MY_SCHEMA.NONEXISTENT"})

        assert "not found" in result.lower()
