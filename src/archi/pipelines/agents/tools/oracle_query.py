"""Oracle agent tools for querying and discovering Oracle database schemas.

Provides two LangChain tools:
- query_oracle_db: Execute read-only SQL queries against named Oracle databases.
- describe_oracle_schema: Discover databases, tables, and columns.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Tuple

from langchain.tools import tool

from src.archi.pipelines.agents.tools.oracle_connection import OracleConnectionManager
from src.archi.pipelines.agents.tools.oracle_safety import (
    SQLValidationError,
    check_schema_allowlist,
    ensure_row_limit,
    validate_select_only,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------


_ORACLE_TYPE_NAMES = {
    "DB_TYPE_VARCHAR": "VARCHAR2",
    "DB_TYPE_CHAR": "CHAR",
    "DB_TYPE_NVARCHAR": "NVARCHAR2",
    "DB_TYPE_NUMBER": "NUMBER",
    "DB_TYPE_DATE": "DATE",
    "DB_TYPE_TIMESTAMP": "TIMESTAMP",
    "DB_TYPE_CLOB": "CLOB",
    "DB_TYPE_BLOB": "BLOB",
    "DB_TYPE_RAW": "RAW",
}


def _type_label(type_obj: Any) -> str:
    """Convert an oracledb type object to a readable label."""
    name = getattr(type_obj, "name", None) or str(type_obj)
    return _ORACLE_TYPE_NAMES.get(name, name)


def _column_summary(cursor_description: Sequence[Tuple]) -> str:
    """Build a one-line summary of column names and types from cursor.description."""
    parts = []
    for col in cursor_description:
        col_name = col[0]
        if len(col) > 1 and col[1] is not None:
            col_type = _type_label(col[1])
            parts.append(f"{col_name} {col_type}")
        else:
            parts.append(col_name)
    return ", ".join(parts)


def format_results_as_markdown(
    columns: List[str],
    rows: List[Tuple],
    *,
    max_chars: int = 4000,
    cursor_description: Optional[Sequence[Tuple]] = None,
) -> str:
    """Render query results as a markdown table.

    If the rendered output exceeds *max_chars*, rows are truncated and a
    note is appended indicating how many rows are shown vs total.
    """
    if not rows:
        return "Query returned 0 rows."

    total_rows = len(rows)
    lines: List[str] = []

    # Column type summary
    if cursor_description:
        summary = _column_summary(cursor_description)
        lines.append(f"{total_rows} rows returned ({len(columns)} columns: {summary}).\n")
    else:
        lines.append(f"{total_rows} rows returned.\n")

    # Build header
    header = "| " + " | ".join(str(c) for c in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    lines.extend([header, separator])
    base_len = sum(len(l) + 1 for l in lines)

    shown = 0
    current_len = base_len
    row_lines: List[str] = []

    for row in rows:
        row_line = "| " + " | ".join(_cell(v) for v in row) + " |"
        line_len = len(row_line) + 1  # +1 for newline
        if current_len + line_len > max_chars and shown > 0:
            break
        row_lines.append(row_line)
        current_len += line_len
        shown += 1

    lines.extend(row_lines)

    if shown < total_rows:
        lines.append(f"\nShowing {shown} of {total_rows} rows (output truncated)")

    return "\n".join(lines)


def _cell(value) -> str:
    """Format a single cell value for markdown."""
    if value is None:
        return "NULL"
    return str(value)


# ---------------------------------------------------------------------------
# Query tool
# ---------------------------------------------------------------------------


def create_oracle_query_tool(
    manager: OracleConnectionManager,
    *,
    name: str = "query_oracle_db",
    description: Optional[str] = None,
) -> Callable:
    """Create a LangChain tool for executing read-only Oracle SQL queries.

    Args:
        manager: OracleConnectionManager with configured databases.
        name: Tool name.
        description: Override tool description.

    Returns:
        A LangChain @tool callable.
    """
    db_list = manager.list_databases()
    db_summary = ", ".join(
        f"{d['name']}" + (f" ({d['description']})" if d.get("description") else "")
        for d in db_list
    )

    tool_description = description or (
        "Execute a read-only SQL query against an Oracle database. "
        "Only SELECT queries are allowed.\n"
        f"Available databases: {db_summary}\n"
        "Input: db_name (string) and sql (string).\n"
        "Use describe_oracle_schema first to discover tables and columns.\n"
        "This is an Oracle database - use Oracle SQL syntax "
        "(FETCH FIRST N ROWS ONLY instead of LIMIT, use DUAL for constants, etc.)."
    )

    @tool(name, description=tool_description)
    def _query_oracle_db(db_name: str, sql: str) -> str:
        if not db_name or not db_name.strip():
            names = [d["name"] for d in manager.list_databases()]
            return f"Please provide a db_name. Available databases: {', '.join(names)}"

        if not sql or not sql.strip():
            return "Please provide a SQL query."

        db_name = db_name.strip()

        # Resolve config
        try:
            cfg = manager.get_db_config(db_name)
        except ValueError as e:
            return str(e)

        # Safety checks
        try:
            validate_select_only(sql)
        except SQLValidationError as e:
            return f"Query rejected: {e}"

        try:
            check_schema_allowlist(sql, cfg.allowed_schemas or None)
        except SQLValidationError as e:
            return f"Query rejected: {e}"

        safe_sql = ensure_row_limit(sql, cfg.max_rows)

        # Execute
        try:
            with manager.get_connection(db_name) as conn:
                cursor = conn.cursor()
                cursor.execute(safe_sql)
                desc = cursor.description or []
                columns = [col[0] for col in desc]
                rows = cursor.fetchall()
        except Exception as e:
            err_str = str(e)
            if "DPI-1067" in err_str or "call timeout" in err_str.lower():
                return (
                    f"Query timed out after {cfg.query_timeout_seconds} seconds. "
                    "Try a simpler query or add row limits."
                )
            logger.warning("Oracle query error on '%s': %s", db_name, e)
            hint = ""
            if "ORA-00942" in err_str:
                hint = " Hint: use the oracle_schema tool to discover available table names before querying."
            elif "ORA-00904" in err_str:
                hint = " Hint: use the oracle_schema tool with table_name to check available column names."
            elif "ORA-01722" in err_str:
                hint = " Hint: use the oracle_schema tool to check column data types — you may be comparing a NUMBER column with a string."
            return f"Oracle query error: {err_str}{hint}"

        if not rows:
            return "Query returned 0 rows."

        return format_results_as_markdown(columns, rows, max_chars=4000, cursor_description=desc)

    return _query_oracle_db


# ---------------------------------------------------------------------------
# Schema discovery tool
# ---------------------------------------------------------------------------


def create_oracle_schema_tool(
    manager: OracleConnectionManager,
    *,
    name: str = "describe_oracle_schema",
    description: Optional[str] = None,
) -> Callable:
    """Create a LangChain tool for discovering Oracle database schemas.

    Supports three modes:
    - List databases: db_name omitted or "list"
    - List tables: db_name provided, table_name omitted
    - Describe columns: db_name and table_name provided

    Args:
        manager: OracleConnectionManager with configured databases.
        name: Tool name.
        description: Override tool description.

    Returns:
        A LangChain @tool callable.
    """
    db_list = manager.list_databases()
    db_summary = ", ".join(
        f"{d['name']}" + (f" ({d['description']})" if d.get("description") else "")
        for d in db_list
    )

    tool_description = description or (
        "Discover Oracle database schemas: list databases, tables, or columns.\n"
        f"Available databases: {db_summary}\n"
        "Modes:\n"
        '- db_name="list" or omitted → list all databases\n'
        "- db_name=<name>, table_name omitted → list tables/views in that database\n"
        '- db_name=<name>, table_name="SCHEMA.TABLE" → describe columns\n'
        "Use this before query_oracle_db to discover what data is available."
    )

    @tool(name, description=tool_description)
    def _describe_oracle_schema(
        db_name: str = "list",
        table_name: str = "",
    ) -> str:
        db_name = (db_name or "").strip()
        table_name = (table_name or "").strip()

        # --- Mode 1: List databases ---
        if not db_name or db_name.lower() == "list":
            dbs = manager.list_databases()
            if not dbs:
                return "No Oracle databases are configured."
            lines = ["Available Oracle databases:"]
            for db in dbs:
                line = f"- {db['name']}"
                if db.get("description"):
                    line += f": {db['description']}"
                lines.append(line)
            return "\n".join(lines)

        # Validate db_name
        try:
            cfg = manager.get_db_config(db_name)
        except ValueError as e:
            return str(e)

        schema_filter = cfg.allowed_schemas

        # --- Mode 2: List tables ---
        if not table_name:
            return _list_tables(manager, db_name, schema_filter)

        # --- Mode 3: Describe columns ---
        # Parse SCHEMA.TABLE or just TABLE
        if "." in table_name:
            schema_part, table_part = table_name.split(".", 1)
            schema_part = schema_part.upper()
            table_part = table_part.upper()
            # Check allowlist
            if schema_filter and schema_part not in [s.upper() for s in schema_filter]:
                return (
                    f"Schema '{schema_part}' is not allowed. "
                    f"Allowed schemas: {', '.join(sorted(schema_filter))}"
                )
        else:
            schema_part = None
            table_part = table_name.upper()

        return _describe_columns(manager, db_name, schema_part, table_part, schema_filter)

    return _describe_oracle_schema


def _list_tables(
    manager: OracleConnectionManager,
    db_name: str,
    schema_filter: Optional[List[str]],
) -> str:
    """List tables and views in the database, filtered by allowed schemas."""
    if schema_filter:
        schema_in = ", ".join(f"'{s.upper()}'" for s in schema_filter)
        sql = (
            f"SELECT owner, table_name, 'TABLE' AS object_type "
            f"FROM ALL_TABLES WHERE owner IN ({schema_in}) "
            f"UNION ALL "
            f"SELECT owner, view_name, 'VIEW' "
            f"FROM ALL_VIEWS WHERE owner IN ({schema_in}) "
            f"ORDER BY 1, 3, 2"
        )
    else:
        sql = (
            "SELECT owner, table_name, 'TABLE' AS object_type "
            "FROM ALL_TABLES WHERE owner NOT IN ('SYS','SYSTEM','MDSYS','CTXSYS','XDB','WMSYS','ORDDATA','ORDSYS') "
            "UNION ALL "
            "SELECT owner, view_name, 'VIEW' "
            "FROM ALL_VIEWS WHERE owner NOT IN ('SYS','SYSTEM','MDSYS','CTXSYS','XDB','WMSYS','ORDDATA','ORDSYS') "
            "ORDER BY 1, 3, 2"
        )
    sql += "\nFETCH FIRST 500 ROWS ONLY"

    try:
        with manager.get_connection(db_name) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
    except Exception as e:
        err_str = str(e)
        if "DPI-1067" in err_str or "call timeout" in err_str.lower():
            return "Schema query timed out. The database may have a very large catalog."
        return f"Error listing tables: {err_str}"

    if not rows:
        return f"No tables or views found in database '{db_name}'."

    # Group by schema
    current_schema = None
    lines = [f"Tables and views in '{db_name}':"]
    for owner, name, obj_type in rows:
        if owner != current_schema:
            current_schema = owner
            lines.append(f"\n**{owner}**")
        lines.append(f"  - {name} ({obj_type})")

    return "\n".join(lines)


def _describe_columns(
    manager: OracleConnectionManager,
    db_name: str,
    schema: Optional[str],
    table: str,
    schema_filter: Optional[List[str]],
) -> str:
    """Describe columns of a specific table."""
    conditions = [f"table_name = '{table}'"]
    if schema:
        conditions.append(f"owner = '{schema}'")
    elif schema_filter:
        schema_in = ", ".join(f"'{s.upper()}'" for s in schema_filter)
        conditions.append(f"owner IN ({schema_in})")

    where = " AND ".join(conditions)
    sql = (
        f"SELECT owner, table_name, column_name, data_type, "
        f"data_length, nullable "
        f"FROM ALL_TAB_COLUMNS WHERE {where} "
        f"ORDER BY owner, table_name, column_id"
    )

    try:
        with manager.get_connection(db_name) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
    except Exception as e:
        err_str = str(e)
        if "DPI-1067" in err_str or "call timeout" in err_str.lower():
            return "Schema query timed out."
        return f"Error describing table: {err_str}"

    if not rows:
        target = f"{schema}.{table}" if schema else table
        return f"Table '{target}' not found in database '{db_name}'."

    lines = []
    current_table = None
    for owner, tbl, col, dtype, dlen, nullable in rows:
        full_table = f"{owner}.{tbl}"
        if full_table != current_table:
            current_table = full_table
            lines.append(f"**{full_table}**")
            lines.append("| Column | Type | Nullable |")
            lines.append("| --- | --- | --- |")
        null_str = "Yes" if nullable == "Y" else "No"
        type_str = dtype
        if dtype in ("VARCHAR2", "CHAR", "NVARCHAR2", "RAW"):
            type_str = f"{dtype}({dlen})"
        lines.append(f"| {col} | {type_str} | {null_str} |")

    return "\n".join(lines)
