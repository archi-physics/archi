"""Fakes for the ticket-triage tests and the scratch run: a ticket system and a graph, both in memory.

`FakeTicketSystem` answers the Redmine, JIRA and ServiceNow REST calls `archi.tools.tickets` makes, keeps
each ticket's status, assignee and notes, and records every request. `session()` hands it to a tool as
its HTTP session; `serve()` puts it on a loopback port for the scratch run, where the real tool in the
okg worker reaches it over HTTP (`python ticket_fakes.py serve <state.json> <port>`).

`Graph` is the part of `archi.ops.okg_io.OpsIO` that `archi.triage` reads (filter by type, by attribute
and by id; a lexical search; the ticket cursors and signals), over rows shaped like `okg filter --jsonl`.
"""
from __future__ import annotations

import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, unquote, urlsplit

TOKEN = "fixture-token"  # what the fake systems accept; not a real credential
HEADERS = {
    "redmine": ("X-Redmine-API-Key", TOKEN),
    "jira": ("Authorization", f"Bearer {TOKEN}"),
    "servicenow": ("Authorization", f"Bearer {TOKEN}"),
}


def default_state() -> dict:
    """A small queue per system. Status names are the systems' own; members are the assignable people."""
    return {
        "redmine": {
            "statuses": [{"id": 1, "name": "New"}, {"id": 2, "name": "In Progress"}, {"id": 5, "name": "Closed"}],
            "members": [{"user": {"id": 12, "name": "Grace Hopper"}}, {"user": {"id": 14, "name": "Alan Turing"}},
                        {"group": {"id": 30, "name": "Storage Team"}}],
            "issues": {},
        },
        "jira": {
            "users": [{"name": "ghopper", "displayName": "Grace Hopper"}, {"name": "aturing", "displayName": "Alan Turing"}],
            "transitions": {"Open": [{"id": "21", "to": {"name": "In Progress"}}, {"id": "31", "to": {"name": "Closed"}}],
                            "In Progress": [{"id": "31", "to": {"name": "Closed"}}]},
            "issues": {},
        },
        "servicenow": {"records": {}},
    }


class FakeTicketSystem:
    def __init__(self, state: Optional[dict] = None, path: Optional[Path] = None):
        self.path = path
        self.state = state if state is not None else (json.loads(path.read_text()) if path and path.is_file()
                                                      else default_state())
        self.requests: list[tuple[str, str, Any]] = []
        self.fail_writes = False  # answer every write with a dropped connection (a write whose answer never came)
        self.lock = threading.Lock()

    # --- tickets -------------------------------------------------------------------------

    def add(self, system: str, key: str, **fields: Any) -> None:
        if system == "servicenow":
            self.state["servicenow"]["records"][key] = {"table": fields.pop("table", "incident"),
                                                        "sys_id": f"sys{len(self.state['servicenow']['records']) + 1:04d}",
                                                        "notes": [], **fields}
        else:
            self.state[system]["issues"][key] = {"notes": [], "project_id": 7, **fields}
        self.save()

    def ticket(self, system: str, key: str) -> dict:
        return (self.state["servicenow"]["records"] if system == "servicenow" else self.state[system]["issues"])[key]

    def writes(self) -> list[tuple[str, str, Any]]:
        return [r for r in self.requests if r[0] in ("PUT", "POST", "PATCH")]

    def save(self) -> None:
        if self.path is not None:
            self.path.write_text(json.dumps(self.state, indent=1))

    # --- the REST surface ----------------------------------------------------------------

    def handle(self, method: str, url: str, headers: dict, params: Optional[dict], body: Any) -> tuple[int, Any]:
        with self.lock:
            parts = urlsplit(url)
            path = unquote(parts.path)
            params = {**{k: v[0] for k, v in parse_qs(parts.query).items()}, **(params or {})}
            self.requests.append((method, path, body))
            if self.path is not None and self.path.is_file():
                self.state = json.loads(self.path.read_text())
            if method != "GET" and self.fail_writes:
                raise ConnectionError("connection reset")
            for system, (name, value) in HEADERS.items():
                if headers.get(name) == value:
                    break
            else:
                return 401, {"error": "unauthorised"}
            try:
                answer = self.route(method, path, params, body)
            except KeyError as exc:
                return 404, {"error": f"not found: {exc}"}
            self.save()
            return answer

    def route(self, method: str, path: str, params: dict, body: Any) -> tuple[int, Any]:
        rm, jr, sn = self.state["redmine"], self.state["jira"], self.state["servicenow"]
        if path == "/issue_statuses.json":
            return 200, {"issue_statuses": rm["statuses"]}
        m = re.fullmatch(r"/projects/(\d+)/memberships\.json", path)
        if m:
            return 200, {"memberships": rm["members"]}
        m = re.fullmatch(r"/issues/(\d+)\.json", path)
        if m:
            issue = rm["issues"][m.group(1)]
            if method == "GET":
                return 200, {"issue": {"id": int(m.group(1)), "project": {"id": issue["project_id"]},
                                       "status": {"name": issue["status"]},
                                       **({"assigned_to": {"name": issue["assignee"]}} if issue.get("assignee") else {})}}
            change = body["issue"]
            if "status_id" in change:
                issue["status"] = next(s["name"] for s in rm["statuses"] if s["id"] == change["status_id"])
            if "assigned_to_id" in change:
                issue["assignee"] = next(m[k]["name"] for m in rm["members"] for k in ("user", "group")
                                         if (m.get(k) or {}).get("id") == change["assigned_to_id"])
            issue["notes"].append(change.get("notes", ""))
            return 204, None
        m = re.fullmatch(r"/rest/api/2/issue/([A-Z0-9_]+-\d+)(/\w+)?", path)
        if m:
            issue, tail = jr["issues"][m.group(1)], m.group(2)
            if tail is None and method == "GET":
                user = next((u for u in jr["users"] if u["displayName"] == issue.get("assignee")), None)
                return 200, {"fields": {"status": {"name": issue["status"]}, "assignee": user}}
            if tail == "/transitions" and method == "GET":
                return 200, {"transitions": jr["transitions"].get(issue["status"], [])}
            if tail == "/transitions":
                moves = jr["transitions"].get(issue["status"], [])
                issue["status"] = next(t["to"]["name"] for t in moves if t["id"] == body["transition"]["id"])
                return 204, None
            if tail == "/assignee":
                issue["assignee"] = next(u["displayName"] for u in jr["users"] if u["name"] == body["name"])
                return 204, None
            if tail == "/comment":
                issue["notes"].append(body["body"])
                return 201, {"id": str(len(issue["notes"]))}
        if path == "/rest/api/2/user/search":
            q = str(params.get("username", "")).casefold()
            return 200, [u for u in jr["users"] if q in u["name"].casefold() or q in u["displayName"].casefold()]
        m = re.fullmatch(r"/api/now/table/(incident|sc_request)(?:/(\w+))?", path)
        if m:
            table, sys_id = m.groups()
            if sys_id is None:
                number = str(params.get("sysparm_query", "")).removeprefix("number=")
                rec = sn["records"].get(number)
                rows = [rec] if rec and rec["table"] == table else []
                return 200, {"result": [{"sys_id": r["sys_id"], "state": r["state"], "assigned_to": r.get("assignee", ""),
                                         "assignment_group": r.get("assignment_group", "")} for r in rows]}
            rec = next(r for r in sn["records"].values() if r["sys_id"] == sys_id)
            for field, value in body.items():
                if field in ("work_notes", "comments"):
                    rec["notes"].append(value)
                else:
                    rec[{"state": "state", "assigned_to": "assignee"}.get(field, field)] = value
            return 200, {"result": {"sys_id": sys_id}}
        raise KeyError(path)

    # --- as an HTTP session (unit tests) and as a server (scratch run) ---------------------

    def session(self) -> "FakeSession":
        return FakeSession(self)


class FakeResponse:
    def __init__(self, status: int, body: Any):
        self.status_code = status
        self.text = "" if body is None else json.dumps(body)
        self._body = body

    def json(self) -> Any:
        return self._body


class FakeSession:
    def __init__(self, system: FakeTicketSystem):
        self.system = system
        self.calls: list[dict] = []

    def request(self, method, url, headers=None, json=None, params=None, timeout=None, allow_redirects=True):
        self.calls.append({"method": method, "url": url, "allow_redirects": allow_redirects, "timeout": timeout})
        status, body = self.system.handle(method, url, dict(headers or {}), params, json)
        return FakeResponse(status, body)


def serve(state_path: Path, port: int) -> None:
    """Serve a FakeTicketSystem kept in `state_path` on 127.0.0.1:`port` until killed."""
    system = FakeTicketSystem(path=state_path)

    class Handler(BaseHTTPRequestHandler):
        def _answer(self, method: str) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length)) if length else None
            try:
                status, out = system.handle(method, self.path, dict(self.headers), None, body)
            except ConnectionError:
                self.connection.close()
                return
            data = b"" if out is None else json.dumps(out).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            self._answer("GET")

        def do_PUT(self):  # noqa: N802
            self._answer("PUT")

        def do_POST(self):  # noqa: N802
            self._answer("POST")

        def do_PATCH(self):  # noqa: N802
            self._answer("PATCH")

        def log_message(self, *args):
            pass

    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


# --- the fixture queue -----------------------------------------------------------------

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "ticket_triage"
#: cache directory under the data root, per system (what each pack's source reads)
CACHE_DIRS = {"redmine": "tickets", "jira": "jira", "servicenow": "servicenow"}


def load_cache(system: str) -> list[dict]:
    return json.loads((FIXTURES / f"{CACHE_DIRS[system]}.json").read_text())


def arrival(system: str, key: str, subject: str, description: str, updated: str, *, assignee: str = "",
            status: Optional[str] = None, group: str = "", tracker: str = "redmine") -> dict:
    """A new ticket in the cache shape of its pack, unassigned unless `assignee` (or `group`) is given.
    `system` names the cache (CACHE_DIRS: `redmine` is the tickets pack's cache); a ticket in that cache is
    a `tracker` ticket (`redmine`, or `jira` through the generic JIRA reader)."""
    if system == "redmine":
        return {"system": tracker, "key": key, "subject": subject, "description": description,
                "project": "Cluster Ops", "tracker": "Support", "status": status or "New", "priority": "High",
                "author": "Ada Lovelace", "assignee": assignee, "created": updated, "updated": updated, "closed": "",
                "notes": []}
    if system == "jira":
        return {"key": key, "project": "OPS", "summary": subject, "description": description, "status": status or "Open",
                "resolution": "", "priority": "Major", "issue_type": "Incident", "reporter": "Ada Lovelace",
                "assignee": assignee, "components": [], "labels": [], "environment": "", "parent_key": "",
                "comment_count": 0, "fix_versions": [], "issue_links": [], "subtasks": [], "recent_comments": [],
                "created": updated, "updated": updated}
    return {"table": "incident", "number": key, "sys_id": f"sys{key[-4:]}", "short_description": subject,
            "description": description, "state": status or "New", "priority": "2 - High", "category": "",
            "assignment_group": group, "assigned_to": assignee, "opened_by": "Ada Lovelace", "cmdb_ci": "",
            "opened_at": updated.replace("T", " ").rstrip("Z"), "sys_updated_on": updated.replace("T", " ").rstrip("Z"),
            "resolved_at": "", "close_notes": ""}


#: the link roots the fixture servers publish (the readers' `base_url` / `instance_url`)
SITES = {"redmine": "https://redmine.example.org", "jira": "https://jira.example.org",
         "servicenow": "https://example.service-now.com"}


def to_row(system: str, item: dict, *, site: Optional[str] = None) -> dict:
    """A cache record as the node its pack's source publishes (`okg filter --jsonl` shape), the attributes
    archi.sources.{tickets,jira,servicenow} write, with the link each writes. `site` overrides the server
    (a second JIRA server, say)."""
    if system == "redmine":
        tracker = item.get("system") or "redmine"
        root = site or SITES[tracker]
        link = f"{root}/browse/{item['key']}" if tracker == "jira" else f"{root}/issues/{item['key']}"
        attrs = {"ticket_id": item["key"], "system": tracker, "subject": item["subject"],
                 "description": item["description"], "project": item["project"], "tracker": item["tracker"],
                 "status": item["status"], "assignee": item["assignee"], "updated": item["updated"],
                 "closed": item["closed"], "url": link}
        return {"node_id": f"tracker_ticket:{tracker}:{item['key']}", "subtype": "tracker_ticket", "attrs": attrs}
    if system == "jira":
        attrs = {k: item[k] for k in ("summary", "description", "project", "status", "issue_type", "assignee",
                                      "resolution", "components", "updated")}
        attrs["url"] = f"{site or SITES['jira']}/browse/{item['key']}"
        return {"node_id": f"jira:{item['key']}", "subtype": "jira_issue", "attrs": {"issue_key": item["key"], **attrs}}
    subtype = {"incident": "servicenow_incident", "sc_request": "servicenow_request",
               "kb_knowledge": "servicenow_kb_article"}[item["table"]]
    attrs = {"number": item["number"], "title": item["short_description"],
             "description": item.get("description") or item.get("text") or "", "state": item["state"],
             "updated": item["sys_updated_on"],
             "url": f"{site or SITES['servicenow']}/{item['table']}.do?sys_id={item['sys_id']}"}
    if subtype != "servicenow_kb_article":
        attrs.update(category=item["category"], assignment_group=item["assignment_group"],
                     assignee=item["assigned_to"], config_item=item["cmdb_ci"], resolved=item["resolved_at"],
                     resolution=item["close_notes"])
    return {"node_id": f"{subtype}:{item['number']}", "subtype": subtype, "attrs": attrs}


def queue_rows() -> list[dict]:
    return [to_row(system, item) for system in CACHE_DIRS for item in load_cache(system)]


# --- the graph --------------------------------------------------------------------------


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2}


class Graph:
    """The OpsIO read surface `archi.triage` uses, over `okg filter --jsonl` shaped rows."""

    def __init__(self, rows=(), generation: str = "gen-1"):
        self.rows = {r["node_id"]: r for r in rows}
        self.generation = generation
        self.cursors: dict[tuple[str, str], dict] = {}
        self.signals: dict[tuple[str, str], dict] = {}
        self.searches: list[tuple[str, str]] = []
        self.missing_subtypes: set[str] = set()

    def publish(self, rows=(), generation: Optional[str] = None) -> None:
        for r in rows:
            self.rows[r["node_id"]] = r
        self.generation = generation or f"gen-{int(self.generation.split('-')[1]) + 1}"

    def pin_generation(self) -> str:
        return self.generation

    def _check(self, subtype: str) -> None:
        if subtype in self.missing_subtypes:
            from archi.ops.record import RecordError

            raise RecordError(f"unknown subtype {subtype!r}")

    def read_nodes(self, subtype, generation, where=None):
        self._check(subtype)
        return [r for r in self.rows.values() if r["subtype"] == subtype
                and all(str(r["attrs"].get(k)) == str(v) for k, v in (where or {}).items())]

    def read_by_attr(self, subtype, generation, attr, value):
        return self.read_nodes(subtype, generation, {attr: value})

    def read_ids(self, ids, generation):
        return [self.rows[i] for i in ids if i in self.rows]

    def search_ids(self, query, generation, subtype, limit):
        self._check(subtype)
        self.searches.append((query, subtype))
        want = _words(query)
        scored = []
        for r in self.rows.values():
            if r["subtype"] != subtype:
                continue
            text = " ".join(str(v) for v in r["attrs"].values() if isinstance(v, str))
            score = len(want & _words(text))
            if score:
                scored.append((-score, r["node_id"]))
        return [nid for _, nid in sorted(scored)[:limit]]

    def ticket_cursor(self, automation, source):
        return self.cursors.get((automation, source))

    def set_ticket_cursor(self, automation, source, updated, generation, seen=()):
        self.cursors[(automation, source)] = {"updated": updated, "generation": generation, "seen": list(seen)}

    def add_signal(self, automation, signal_id, entity, title, *, summary=None, source="table"):
        if (automation, signal_id) in self.signals:
            return False
        self.signals[(automation, signal_id)] = {"id": signal_id, "entity": entity, "title": title, "summary": summary,
                                                 "source": source, "attempt": 1, "status": "new"}
        return True

    def signal_ids_for(self, automation, entity):
        return [sid for (a, sid), s in self.signals.items() if a == automation and s["entity"] == entity]

    def supersede_signals(self, automation, entity, keep):
        out = []
        for (a, sid), s in self.signals.items():
            if a == automation and s["entity"] == entity and s["status"] == "new" and sid != keep:
                s["status"] = "superseded"
                out.append(sid)
        return sorted(out)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "serve":
        serve(Path(sys.argv[2]), int(sys.argv[3]))
    else:
        sys.exit("usage: ticket_fakes.py serve <state.json> <port>")
