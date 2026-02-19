"""SQL safety utilities for Oracle agent tools.

Validates agent-generated SQL before execution to enforce:
- Read-only (SELECT only)
- Schema allowlists
- Row limits
"""

from __future__ import annotations

import re
from typing import List, Optional


# Statements that are never allowed
_FORBIDDEN_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|MERGE|CREATE|ALTER|DROP|TRUNCATE|GRANT|REVOKE|"
    r"EXECUTE|CALL|BEGIN|DECLARE|COMMIT|ROLLBACK|SAVEPOINT)\b",
    re.IGNORECASE,
)

# Match the leading statement type (ignoring comments and whitespace)
_LEADING_SELECT = re.compile(
    r"^\s*(?:--[^\n]*\n\s*)*(SELECT|WITH)\b",
    re.IGNORECASE,
)

# Detect existing row limit clauses
_HAS_FETCH = re.compile(r"\bFETCH\s+FIRST\b", re.IGNORECASE)
_HAS_ROWNUM = re.compile(r"\bROWNUM\b", re.IGNORECASE)

# Extract schema-qualified table references in FROM/JOIN context: SCHEMA.TABLE
# Only matches SCHEMA.TABLE that appear after FROM or JOIN keywords to avoid
# false positives from alias.column references like "a.id" or "t1.name".
_TABLE_CONTEXT_REF = re.compile(
    r"(?:FROM|JOIN)\s+"
    r"([A-Za-z_][A-Za-z0-9_$#]*)\s*\.\s*([A-Za-z_][A-Za-z0-9_$#]*)",
    re.IGNORECASE,
)


class SQLValidationError(Exception):
    """Raised when SQL fails safety validation."""
    pass


def validate_select_only(sql: str) -> None:
    """Validate that the SQL is a SELECT or WITH...SELECT statement.

    Raises:
        SQLValidationError: If the statement is not a SELECT query.
    """
    stripped = sql.strip().rstrip(";").strip()
    if not stripped:
        raise SQLValidationError("Empty SQL query.")

    if not _LEADING_SELECT.match(stripped):
        raise SQLValidationError(
            "Only SELECT queries are allowed. "
            "The query must start with SELECT or WITH."
        )

    # Check for forbidden keywords that could indicate DML/DDL
    # within the query (e.g., subquery with INSERT)
    forbidden = _FORBIDDEN_KEYWORDS.search(stripped)
    if forbidden:
        raise SQLValidationError(
            f"Only SELECT queries are allowed. "
            f"Forbidden keyword found: {forbidden.group(0).upper()}"
        )


def check_schema_allowlist(
    sql: str, allowed_schemas: Optional[List[str]]
) -> None:
    """Verify that all schema-qualified table references use allowed schemas.

    Args:
        sql: The SQL query to check.
        allowed_schemas: List of uppercase schema names. If None or empty,
            no restriction is applied.

    Raises:
        SQLValidationError: If a disallowed schema is referenced.
    """
    if not allowed_schemas:
        return

    allowed_upper = {s.upper() for s in allowed_schemas}
    # Known Oracle catalog prefixes that should always be accessible
    safe_prefixes = {"DUAL"}

    for match in _TABLE_CONTEXT_REF.finditer(sql):
        schema = match.group(1).upper()
        if schema in safe_prefixes:
            continue
        if schema not in allowed_upper:
            raise SQLValidationError(
                f"Schema '{schema}' is not allowed. "
                f"Allowed schemas: {', '.join(sorted(allowed_upper))}"
            )


def ensure_row_limit(sql: str, max_rows: int) -> str:
    """Append FETCH FIRST N ROWS ONLY if no row limit is present.

    Args:
        sql: The SQL query.
        max_rows: Maximum rows to return.

    Returns:
        The SQL query, possibly with a FETCH FIRST clause appended.
    """
    stripped = sql.strip().rstrip(";").strip()

    if _HAS_FETCH.search(stripped) or _HAS_ROWNUM.search(stripped):
        return stripped

    return f"{stripped}\nFETCH FIRST {max_rows} ROWS ONLY"
