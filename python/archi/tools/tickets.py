"""Live tools: change a ticket's status or assignment in Redmine, JIRA or ServiceNow.

These are the mediators of the ticket-triage action classes `set_ticket_status` and `assign_ticket`
(`bundles/packs/ticket-triage/action_classes.yaml`). Each changes one field of one ticket in an external
system, so it runs at L2 only: the ops runtime calls it after a person approved the escalation, and the
tool checks that approval again itself before it sends anything (`archi.tools._live.require_approval`:
the approval must be for this proposal, this action class and this exact payload). Then:

1. it reads the ticket. When the field already holds the requested value it changes nothing and succeeds
   (a repeat with the same idempotency key, or a person who got there first);
2. when the field no longer holds what it held at proposal time (`expected_status`, `expected`), it
   changes nothing and fails: the proposal is stale;
3. it writes the field with a note on the ticket that names the approver, the escalation and the proposal,
   so the ticket's own history shows the decision the escalation ledger holds;
4. it reads the ticket again and fails unless the field now holds the value.

A write that times out or loses its connection may have applied, so it returns `unknown`, never `failed`.
When the field changed but the note could not be posted (JIRA writes the note in a second call), the
result is `succeeded` and says the note is missing: the change happened, and calling it failed would hide
that.

    set_ticket_status(payload={system, ticket, ticket_url, [table], status, expected_status, note, endpoint}, ...)
    assign_ticket(payload={system, ticket, ticket_url, [table], field, value, expected, note, endpoint}, ...)

`endpoint` is `{url, [site], token_env | user_env + password_env}`: the API base URL, the root of the
ticket links that server publishes when it differs from the API URL (a local proxy in front of the real
server), and the names of the environment variables that hold the credential (the values are read here,
never put in a payload, a log, an error or a result). The URL must be https; plain http is accepted only
for a loopback host (a local proxy or a test server), where the credential does not leave the machine. A
URL may not carry a user name or password. Redirects are not followed.

`ticket_url` is the link of the ticket the proposal was made for, as its reader published it. The tool
writes nothing unless it lies under the endpoint (`site`, else `url`): two servers can hold tickets with
the same key (two JIRA servers both with a project OPS), and the key alone cannot say which one a person
approved a change to.

    redmine     X-Redmine-API-Key from token_env. Status by name (GET /issue_statuses.json); assignee by
                name among the project's members (GET /projects/<id>/memberships.json).
    jira        Bearer token_env (a personal access token) or basic user_env/password_env. Status by a
                transition whose target has that name; assignee by user search, exactly one match.
    servicenow  basic user_env/password_env, or Bearer token_env. Fields are written as display values
                (state name, group or person name), in the table the payload names.

The HTTP session is a seam: any object with `request(method, url, headers=, json=, params=, timeout=,
allow_redirects=)` returning a response with `status_code`, `json()` and `text` (a `requests.Session`
does). The unit tests and the scratch run pass their own and never reach a real ticket system.
"""
from __future__ import annotations

import base64
import os
import re
from typing import Any, Callable, Mapping, Optional
from urllib.parse import quote, urlsplit

from archi.enrichment.anonymizer import redact_email_addresses
from archi.tools._live import ActRefused, ToolError, Verifier, outcome, require_approval, validate_payload

__all__ = ["ActRefused", "ToolError", "PAYLOAD_SCHEMAS", "TOOLS", "set_ticket_status", "assign_ticket"]

TIMEOUT = 30
LOOPBACK = ("127.0.0.1", "localhost", "::1")
#: a ticket key as each system writes it: a Redmine id, a JIRA key, a ServiceNow number
KEY_PATTERNS = {
    "redmine": re.compile(r"[0-9]{1,12}"),
    "jira": re.compile(r"[A-Z][A-Z0-9_]{0,31}-[0-9]{1,9}"),
    "servicenow": re.compile(r"[A-Z]{2,8}[0-9]{4,12}"),
}
ONE_LINE = r"^[^\x00-\x1f\x7f]+$"
#: an http(s) base URL with no user name or password in it
BASE_URL = r"^https?://[^\s/?#@]+(/[^\s?#]*)?$"

_ENDPOINT = {
    "type": "object",
    "required": ["url"],
    "additionalProperties": False,
    "properties": {
        "url": {"type": "string", "pattern": BASE_URL},
        "site": {"type": "string", "pattern": BASE_URL},
        "token_env": {"type": "string", "pattern": "^[A-Z_][A-Z0-9_]*$"},
        "user_env": {"type": "string", "pattern": "^[A-Z_][A-Z0-9_]*$"},
        "password_env": {"type": "string", "pattern": "^[A-Z_][A-Z0-9_]*$"},
    },
}
_COMMON = {
    "system": {"enum": ["redmine", "jira", "servicenow"]},
    "ticket": {"type": "string", "pattern": r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$"},
    "ticket_url": {"type": "string", "maxLength": 2000, "pattern": r"^https?://[^\s@]+$"},
    "table": {"enum": ["incident", "sc_request"]},
    "note": {"type": "string", "minLength": 1, "maxLength": 4000},
    "endpoint": _ENDPOINT,
}

PAYLOAD_SCHEMAS: dict[str, dict[str, Any]] = {
    "set_ticket_status": {
        "type": "object",
        "required": ["system", "ticket", "ticket_url", "status", "expected_status", "note", "endpoint"],
        "additionalProperties": False,
        "properties": {
            **_COMMON,
            "status": {"type": "string", "minLength": 1, "maxLength": 100, "pattern": ONE_LINE},
            "expected_status": {"type": "string", "maxLength": 100},
        },
    },
    "assign_ticket": {
        "type": "object",
        "required": ["system", "ticket", "ticket_url", "field", "value", "expected", "note", "endpoint"],
        "additionalProperties": False,
        "properties": {
            **_COMMON,
            "field": {"enum": ["assignee", "assignment_group"]},
            "value": {"type": "string", "minLength": 1, "maxLength": 200, "pattern": ONE_LINE},
            "expected": {"type": "string", "maxLength": 200},
        },
    },
}

#: What the pack ships, in one place: the tests hold this equal to pack.yaml and action_classes.yaml.
TOOLS: dict[str, dict[str, Any]] = {
    "set_ticket_status": {"kind": "act", "idempotent": True},
    "assign_ticket": {"kind": "act", "idempotent": True},
}


class TicketToolError(RuntimeError):
    """The ticket system refused or could not be read. Carries no credential."""


class WriteUnknown(RuntimeError):
    """A write whose answer never came: it may have applied."""


def _same(a: Any, b: Any) -> bool:
    return str(a or "").strip().casefold() == str(b or "").strip().casefold()


class NoteFailed(RuntimeError):
    """The field was written but the note naming the approval was not (JIRA: a second call)."""


def _base_url(url: str) -> str:
    parts = urlsplit(url)
    if parts.username is not None or parts.password is not None:
        raise ToolError("endpoint url may not carry a user name or password; name credential variables instead")
    if parts.scheme == "https" or (parts.scheme == "http" and parts.hostname in LOOPBACK):
        return url.rstrip("/")
    raise ToolError(f"endpoint url must be https (plain http only to a loopback host); got {parts.scheme}://{parts.hostname}")


def _root(url: str) -> tuple[str, str, str]:
    """(scheme, host[:port], path without a trailing slash), scheme and host in lower case."""
    parts = urlsplit(url)
    return parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/")


def under(ticket_url: str, root: str) -> bool:
    """True when `ticket_url` is a link under `root` (same scheme and host, path below root's path)."""
    t_scheme, t_host, t_path = _root(ticket_url)
    r_scheme, r_host, r_path = _root(root)
    return bool(t_host) and (t_scheme, t_host) == (r_scheme, r_host) and t_path.startswith(r_path + "/")


def _check_site(data: Mapping[str, Any]) -> None:
    root = data["endpoint"].get("site") or data["endpoint"]["url"]
    if not under(data["ticket_url"], root):
        raise ToolError(f"the ticket link {data['ticket_url']} is not under {root}: this endpoint is another "
                        "server than the one the ticket was read from; nothing was sent")


def _safe(text: Any, limit: int = 200) -> str:
    """Server text for a result: email addresses removed, one line, cut to `limit`."""
    return " ".join(redact_email_addresses(str(text or "")).split())[:limit]


def _display(value: Any) -> str:
    """A field ServiceNow or JIRA returns as a string or as an object with a display name."""
    if isinstance(value, Mapping):
        for key in ("display_value", "displayName", "name", "value"):
            if value.get(key):
                return redact_email_addresses(str(value[key])).strip()
        return ""
    return "" if value is None else redact_email_addresses(str(value)).strip()


def _holds(current: Mapping[str, Any], field: str, value: Any) -> bool:
    """Whether the read ticket's `field` is `value`: its display value or one of the other names the system
    gives the same thing (JIRA: a user's display name, user name or key)."""
    return any(_same(v, value) for v in (current.get(field), *current.get(f"{field}_aliases", ())))


class _Client:
    def __init__(self, endpoint: Mapping[str, Any], env: Mapping[str, str], session: Any):
        self.base = _base_url(str(endpoint["url"]))
        self.headers = {"Accept": "application/json", "Content-Type": "application/json", **self.auth(endpoint, env)}
        if session is None:
            import requests

            session = requests.Session()
        self.session = session

    @staticmethod
    def _need(env: Mapping[str, str], name: Optional[str], what: str) -> str:
        if not name:
            raise ToolError(f"endpoint names no {what} variable")
        value = env.get(name)
        if not value:
            raise ToolError(f"environment variable {name} ({what}) is not set")
        return value

    def basic_or_bearer(self, endpoint: Mapping[str, Any], env: Mapping[str, str]) -> dict[str, str]:
        if endpoint.get("token_env"):
            return {"Authorization": f"Bearer {self._need(env, endpoint['token_env'], 'token')}"}
        user = self._need(env, endpoint.get("user_env"), "user")
        password = self._need(env, endpoint.get("password_env"), "password")
        return {"Authorization": "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()}

    def auth(self, endpoint: Mapping[str, Any], env: Mapping[str, str]) -> dict[str, str]:
        return self.basic_or_bearer(endpoint, env)

    def call(self, method: str, path: str, *, json: Any = None, params: Optional[Mapping[str, Any]] = None,
             write: bool = False) -> Any:
        try:
            response = self.session.request(method, self.base + path, headers=self.headers, json=json,
                                            params=params, timeout=TIMEOUT, allow_redirects=False)
        except Exception as exc:  # timeout, connection reset: a write may have applied
            if write:
                raise WriteUnknown(f"{method} {path}: {type(exc).__name__}") from None
            raise TicketToolError(f"{method} {path}: {type(exc).__name__}") from None
        if response.status_code >= 300:
            raise TicketToolError(f"{method} {path}: HTTP {response.status_code}: {_safe(response.text)}")
        if response.status_code == 204 or not str(response.text or "").strip():
            return None
        return response.json()


class Redmine(_Client):
    def auth(self, endpoint, env):
        return {"X-Redmine-API-Key": self._need(env, endpoint.get("token_env"), "token")}

    def read(self, key: str, table: Optional[str]) -> dict[str, Any]:
        issue = (self.call("GET", f"/issues/{quote(key)}.json") or {}).get("issue") or {}
        return {"status": _display(issue.get("status")), "assignee": _display(issue.get("assigned_to")),
                "project": str((issue.get("project") or {}).get("id") or "")}

    def set_status(self, key, table, current, status, note):
        statuses = (self.call("GET", "/issue_statuses.json") or {}).get("issue_statuses") or []
        found = [s for s in statuses if _same(s.get("name"), status)]
        if len(found) != 1:
            raise TicketToolError(f"Redmine has {len(found)} statuses named {status!r}, not one")
        self.call("PUT", f"/issues/{quote(key)}.json", json={"issue": {"status_id": found[0]["id"], "notes": note}},
                  write=True)

    def assign(self, key, table, current, field, value, note):
        members = (self.call("GET", f"/projects/{quote(current['project'])}/memberships.json",
                             params={"limit": 100}) or {}).get("memberships") or []
        found = {m[k]["id"] for m in members for k in ("user", "group") if _same((m.get(k) or {}).get("name"), value)}
        if len(found) != 1:
            raise TicketToolError(f"the project has {len(found)} members named {value!r}, not one")
        self.call("PUT", f"/issues/{quote(key)}.json", json={"issue": {"assigned_to_id": found.pop(), "notes": note}},
                  write=True)


class Jira(_Client):
    def read(self, key, table):
        fields = (self.call("GET", f"/rest/api/2/issue/{quote(key)}", params={"fields": "status,assignee"}) or {}).get(
            "fields") or {}
        assignee = fields.get("assignee") or {}
        names = [str(assignee.get(k) or "") for k in ("displayName", "name", "key")]
        return {"status": _display(fields.get("status")), "assignee": next((n for n in names if n), ""),
                "assignee_aliases": tuple(n for n in names if n)}

    def comment(self, key, note):
        """The note naming the approval. JIRA writes it in its own call, after the change: when it fails, the
        change stands, so this raises NoteFailed, never TicketToolError or WriteUnknown."""
        try:
            self.call("POST", f"/rest/api/2/issue/{quote(key)}/comment", json={"body": note}, write=True)
        except (TicketToolError, WriteUnknown) as exc:
            raise NoteFailed(str(exc)) from None

    def set_status(self, key, table, current, status, note):
        moves = (self.call("GET", f"/rest/api/2/issue/{quote(key)}/transitions") or {}).get("transitions") or []
        found = [t for t in moves if _same((t.get("to") or {}).get("name"), status)]
        if len(found) != 1:
            raise TicketToolError(f"{key} has {len(found)} transitions to {status!r} from {current['status']!r}, not one")
        self.call("POST", f"/rest/api/2/issue/{quote(key)}/transitions", json={"transition": {"id": found[0]["id"]}},
                  write=True)
        self.comment(key, note)

    def assign(self, key, table, current, field, value, note):
        users = self.call("GET", "/rest/api/2/user/search", params={"username": value}) or []
        found = [u for u in users if _same(u.get("displayName"), value) or _same(u.get("name"), value)]
        if len(found) != 1:
            raise TicketToolError(f"JIRA has {len(found)} users named {value!r}, not one")
        self.call("PUT", f"/rest/api/2/issue/{quote(key)}/assignee", json={"name": found[0]["name"]}, write=True)
        self.comment(key, note)


class ServiceNow(_Client):
    def record(self, key, table):
        rows = (self.call("GET", f"/api/now/table/{table}", params={
            "sysparm_query": f"number={key}", "sysparm_fields": "sys_id,state,assigned_to,assignment_group",
            "sysparm_display_value": "true", "sysparm_limit": 2}) or {}).get("result") or []
        if len(rows) != 1:
            raise TicketToolError(f"ServiceNow {table} has {len(rows)} records numbered {key}, not one")
        return rows[0]

    def read(self, key, table):
        row = self.record(key, table)
        return {"status": _display(row.get("state")), "assignee": _display(row.get("assigned_to")),
                "assignment_group": _display(row.get("assignment_group")), "sys_id": _display(row.get("sys_id"))}

    def patch(self, table, current, body):
        self.call("PATCH", f"/api/now/table/{table}/{quote(current['sys_id'])}",
                  params={"sysparm_input_display_value": "true"}, json=body, write=True)

    def set_status(self, key, table, current, status, note):
        self.patch(table, current, {"state": status, "work_notes": note})

    def assign(self, key, table, current, field, value, note):
        self.patch(table, current, {"assigned_to" if field == "assignee" else field: value, "work_notes": note})


CLIENTS = {"redmine": Redmine, "jira": Jira, "servicenow": ServiceNow}


def _checked(payload: Any, action_class: str) -> dict[str, Any]:
    data = validate_payload(payload, PAYLOAD_SCHEMAS[action_class])
    system = data["system"]
    if not KEY_PATTERNS[system].fullmatch(data["ticket"]):
        raise ToolError(f"{data['ticket']!r} is not a {system} ticket key")
    if (system == "servicenow") != ("table" in data):
        raise ToolError("a ServiceNow ticket needs its table (incident or sc_request); the other systems take none")
    if data.get("field") == "assignment_group" and system != "servicenow":
        raise ToolError("assignment_group is a ServiceNow field")
    for key in ("status", "value"):
        if key in data and any(ord(ch) < 32 or ch == "\x7f" for ch in data[key]):
            raise ToolError(f"{key} may not hold control characters")
    _base_url(data["endpoint"]["url"])  # https (or loopback http), no credentials: before anything else uses it
    _check_site(data)
    return data


def _act(action_class: str, payload: Any, context: Optional[Mapping[str, Any]], session: Any,
         env: Optional[Mapping[str, str]], verify: Optional[Verifier], field_of: Callable[[dict], str],
         target_of: Callable[[dict], str], expected_of: Callable[[dict], str], write: Callable[..., None]
         ) -> dict[str, Any]:
    data = _checked(payload, action_class)
    approval = require_approval(context, verify, payload=data, action_class=action_class)
    client = CLIENTS[data["system"]](data["endpoint"], os.environ if env is None else env, session)
    key, table, field, target = data["ticket"], data.get("table"), field_of(data), target_of(data)
    label = f"{data['system']} {key}"
    try:
        current = client.read(key, table)
    except TicketToolError as exc:
        return outcome("failed", f"could not read {label} ({exc}); nothing changed")
    if _holds(current, field, target):
        return outcome("succeeded", f"{label}: {field} is already {target}; nothing changed")
    if not _holds(current, field, expected_of(data)):
        return outcome("failed", f"{label}: {field} is {current.get(field)!r}, not {expected_of(data)!r} as when it "
                                 "was proposed; the proposal is stale, nothing changed")
    note = (f"{data['note']}\n\nApproved by {approval.decided_by} in escalation {approval.escalation_id} "
            f"(proposal {approval.proposal_id}); applied by the archi ticket-triage automation.")
    missing_note = ""
    try:
        write(client, data, current, note)
    except NoteFailed as exc:
        missing_note = f"; the note naming the approval could not be posted ({exc}), add it by hand"
    except WriteUnknown as exc:
        return outcome("unknown", f"{label}: the write got no answer ({exc}); it may have applied, check the ticket")
    except TicketToolError as exc:
        return outcome("failed", f"{label}: {exc}")
    try:
        after = client.read(key, table)
    except TicketToolError as exc:
        return outcome("unknown", f"{label}: written, but the check read failed ({exc}); check the ticket{missing_note}")
    if not _holds(after, field, target):
        return outcome("failed", f"{label}: written, but {field} reads {after.get(field)!r}, not {target!r}{missing_note}")
    noted = missing_note or f"; noted on the ticket with escalation {approval.escalation_id}"
    return outcome("succeeded", f"{label}: {field} {current.get(field) or '(empty)'!s} -> {target}{noted}")


def set_ticket_status(*, payload: Mapping[str, Any], context: Mapping[str, Any], session: Any = None,
                      env: Optional[Mapping[str, str]] = None, verify: Optional[Verifier] = None) -> dict[str, Any]:
    """Move one ticket to `status` after an approved escalation (see the module docstring)."""
    return _act("set_ticket_status", payload, context, session, env, verify,
                field_of=lambda d: "status", target_of=lambda d: d["status"],
                expected_of=lambda d: d["expected_status"],
                write=lambda c, d, cur, note: c.set_status(d["ticket"], d.get("table"), cur, d["status"], note))


def assign_ticket(*, payload: Mapping[str, Any], context: Mapping[str, Any], session: Any = None,
                  env: Optional[Mapping[str, str]] = None, verify: Optional[Verifier] = None) -> dict[str, Any]:
    """Set one ticket's assignee (ServiceNow: assignment group) after an approved escalation."""
    return _act("assign_ticket", payload, context, session, env, verify,
                field_of=lambda d: d["field"], target_of=lambda d: d["value"], expected_of=lambda d: d["expected"],
                write=lambda c, d, cur, note: c.assign(d["ticket"], d.get("table"), cur, d["field"], d["value"], note))
