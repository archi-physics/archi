"""
RBAC Permission Enum - Authoritative list of all permission strings.

Permissions are grouped into nested enums by category. Each inner class is a
StrEnum, so members compare equal to their string values and can be used
anywhere a plain string is expected without calling .value.

Usage:
    from src.utils.rbac.permission_enum import Permission

    @require_permission(Permission.Upload.DOCUMENTS)
    def upload(): ...

    if has_permission(Permission.Config.MODIFY):
        ...
"""

from enum import Enum


class Permission:
    """Namespace for all RBAC permission strings, grouped by category."""

    class Chat(str, Enum):
        QUERY = "chat:query"
        HISTORY = "chat:history"
        FEEDBACK = "chat:feedback"

    class Documents(str, Enum):
        VIEW = "documents:view"
        SELECT = "documents:select"

    class Upload(str, Enum):
        DOCUMENTS = "upload:documents"
        PAGE = "upload:page"
        FILE = "upload:file"
        URL = "upload:url"
        GIT = "upload:git"
        JIRA = "upload:jira"
        EMBED = "upload:embed"

    class Sources(str, Enum):
        VIEW = "sources:view"
        SELECT = "sources:select"

    class Config(str, Enum):
        VIEW = "config:view"
        MODIFY = "config:modify"

    class ApiKeys(str, Enum):
        MANAGE = "api-keys:manage"

    class Tools(str, Enum):
        HTTP_GET = "tools:http_get"

    class Metrics(str, Enum):
        VIEW = "view:metrics"

    class Admin(str, Enum):
        SYSTEM = "admin:system"
        USERS = "admin:users"
        DATABASE = "database:admin"
