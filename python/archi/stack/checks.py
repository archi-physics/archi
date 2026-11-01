"""`archi doctor`'s checks: okg's own lint and doctor, plus one check per
quickstart trap archi absorbs (docs/docs/operating/footguns.md lists them all).

Each check returns a `Result` with one remediation line when it fails. The
probes that touch the outside world (git, the database, HTTP, compose, okg,
MCP) go through `Probes`, so every check has a failing fixture in
python/tests/cli/test_doctor.py without a running stack.
"""
from __future__ import annotations

import asyncio
import re
import stat
import subprocess
from datetime import datetime, timezone
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import yaml

from archi.stack import compose as stack_compose
from archi.stack.instance import (
    COMPOSE_FILE,
    DEPLOYMENTS_DIR,
    ENV_FILE,
    GENERATED_KEYS,
    SECRET_KEYS,
    SECRETS_DIR,
    Instance,
    graph_dsn,
)

PASS, FAIL, WARN, SKIP = "pass", "fail", "warn", "skip"


@dataclass
class Result:
    id: str
    status: str
    message: str
    remediation: str = ""
    footgun: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


def _git(directory: Path, *args: str, input: Optional[str] = None) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(directory), *args], capture_output=True, text=True, input=input)


def _db_query(dsn: str, sql: str) -> list[tuple]:
    import psycopg

    with psycopg.connect(dsn, connect_timeout=5) as conn:
        return list(conn.execute(sql).fetchall())


def _mcp_probe(url: str, token: str) -> dict[str, Any]:
    """initialize, tools/list and one read through the instance's MCP endpoint."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async def run() -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {token}"}
        async with streamablehttp_client(url, headers=headers) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = [t.name for t in (await session.list_tools()).tools]
                overview = await session.call_tool("inspect", {"target": {"kind": "graph"}})
                text = "".join(getattr(c, "text", "") for c in overview.content)
                return {"tools": tools, "is_error": bool(overview.isError), "text": text}

    return asyncio.run(run())


@dataclass
class Probes:
    git: Callable[..., subprocess.CompletedProcess] = _git
    db_query: Callable[[str, str], list[tuple]] = _db_query
    http_status: Callable[..., int] = stack_compose.http_status
    services: Optional[Callable[[], dict[str, dict[str, Any]]]] = None
    okg_json: Optional[Callable[..., dict[str, Any]]] = None
    mcp: Callable[[str, str], dict[str, Any]] = _mcp_probe
    environ: Optional[Mapping[str, str]] = None
    installed_okg: Optional[Callable[[], Optional[str]]] = None
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)


@dataclass
class Context:
    instance: Instance
    secrets: Mapping[str, str]
    probes: Probes

    @property
    def deployment(self) -> Path:
        return self.instance.deployment_dir

    def manifest(self) -> dict[str, Any]:
        return yaml.safe_load((self.deployment / "deployment.yaml").read_text(encoding="utf-8")) or {}

    def registry(self) -> dict[str, Any]:
        path = self.deployment / "source_registry.yaml"
        return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("sources") or {}

    def compose(self) -> dict[str, Any]:
        return yaml.safe_load((self.instance.dir / COMPOSE_FILE).read_text(encoding="utf-8")) or {}


Check = Callable[[Context], Result]
CHECKS: list[tuple[str, str, Check]] = []


def check(check_id: str, footgun: str):
    """Register a check; `footgun` is its ledger id in docs/docs/operating/footguns.md."""

    def wrap(fn: Check) -> Check:
        CHECKS.append((check_id, footgun, fn))
        return fn

    return wrap


def _is_pathish(key: str) -> bool:
    return key.endswith(("_root", "_path", "_dir"))


def _relative(value: Any) -> bool:
    text = str(value or "")
    return bool(text) and not text.startswith("${") and not Path(text).is_absolute()


# ---------------------------------------------------------------------------
# The instance directory
# ---------------------------------------------------------------------------


@check("instance-git", "FG-03")
def instance_git(ctx: Context) -> Result:
    head = ctx.probes.git(ctx.instance.dir, "rev-parse", "--verify", "HEAD")
    if head.returncode != 0:
        return Result("instance-git", FAIL, f"{ctx.instance.dir} is not a git repository with a commit",
                      f"git -C {ctx.instance.dir} init && git -C {ctx.instance.dir} commit "
                      "--allow-empty -m baseline  (okg publishes only from committed config)")
    return Result("instance-git", PASS, f"git repository at {head.stdout.strip()[:12]}")


@check("deployment-committed", "FG-08")
def deployment_committed(ctx: Context) -> Result:
    status = ctx.probes.git(ctx.instance.dir, "status", "--porcelain", "--", DEPLOYMENTS_DIR)
    dirty = [line[3:] for line in status.stdout.splitlines() if line.strip()]
    if status.returncode != 0 or dirty:
        return Result("deployment-committed", FAIL,
                      f"uncommitted deployment files: {', '.join(dirty[:5]) or status.stderr.strip()}",
                      f"git -C {ctx.instance.dir} add -A {DEPLOYMENTS_DIR} && git -C {ctx.instance.dir} "
                      "commit -m 'deployment change'  (okg refuses to publish uncommitted config)",
                      data={"dirty": dirty})
    return Result("deployment-committed", PASS, "deployment configuration is committed")


@check("env-secrets", "FG-14")
def env_secrets(ctx: Context) -> Result:
    path = ctx.instance.env_path
    if not path.is_file():
        return Result("env-secrets", FAIL, f"{path} is missing",
                      "the instance lost its secrets; start over: `archi down --purge --yes`, then `archi init`")
    missing = [k for k in (*SECRET_KEYS, "ARCHI_OKG_MCP_PASSWORD") if not ctx.secrets.get(k)]
    mode = stat.S_IMODE(path.stat().st_mode)
    if missing:
        return Result("env-secrets", FAIL, f"{ENV_FILE} lacks {', '.join(missing)}",
                      "`archi init` did not finish; start over with `archi down --purge --yes` and `archi init`",
                      data={"missing": missing})
    if mode & 0o077:
        return Result("env-secrets", FAIL, f"{ENV_FILE} is readable by others (mode {oct(mode)})",
                      f"chmod 600 {path}")
    return Result("env-secrets", PASS, f"{len(SECRET_KEYS) + 1} secrets in {ENV_FILE}, mode 0600")


#: A `.env` key whose value is a secret: one archi generated, or a credential by its name.
#: A cookie-file path or a URL in `.env` is not scanned (it may well appear in committed answers).
_SECRET_NAME = re.compile(r"(TOKEN|SECRET|PASSWORD|PASSWD|API_KEY|_KEY)$")


def _is_secret_key(key: str) -> bool:
    return key in GENERATED_KEYS or bool(_SECRET_NAME.search(key))


@check("secrets-not-committed", "FG-14")
def secrets_not_committed(ctx: Context) -> Result:
    git = ctx.probes.git
    tracked = git(ctx.instance.dir, "ls-files", "--", ENV_FILE, SECRETS_DIR).stdout.split()
    ignored = git(ctx.instance.dir, "check-ignore", "-q", "--no-index", ENV_FILE)
    leaked = []
    for key, value in ctx.secrets.items():
        if not value or not _is_secret_key(key):
            continue
        # The value goes to git on stdin (-f -), never on a command line `ps` shows.
        if git(ctx.instance.dir, "grep", "-q", "-F", "-f", "-", "HEAD", input=value + "\n").returncode == 0:
            leaked.append(key)
    if tracked or ignored.returncode != 0 or leaked:
        what = (f"tracked: {', '.join(tracked)}; " if tracked else "") + \
               ("" if ignored.returncode == 0 else f"{ENV_FILE} is not git-ignored; ") + \
               (f"values committed: {', '.join(leaked)}" if leaked else "")
        return Result("secrets-not-committed", FAIL, what.strip("; "),
                      f"git rm --cached {ENV_FILE}; add {ENV_FILE} and {SECRETS_DIR}/ to .gitignore; "
                      "rotate every committed secret",
                      data={"tracked": tracked, "leaked": leaked})
    return Result("secrets-not-committed", PASS, "no secret is tracked or committed")


@check("data-root", "FG-06")
def data_root(ctx: Context) -> Result:
    root = ctx.instance.data_root
    answers = (ctx.manifest().get("profile") or {}).get("init_answers") or {}
    recorded = answers.get("archi_data_root")
    problems = []
    if not root.is_absolute():
        problems.append(f"data root {root} is relative")
    if root == ctx.instance.dir or root.is_relative_to(ctx.instance.dir):
        problems.append(f"data root {root} is inside the instance")
    if recorded is not None and Path(str(recorded)) != root:
        problems.append(f"the deployment's archi_data_root is {recorded}, not {root}")
    if problems:
        return Result("data-root", FAIL, "; ".join(problems),
                      "keep the data root absolute and outside the instance: `okg install --force` "
                      "deletes the deployment, and a relative root resolves against okg's working directory")
    return Result("data-root", PASS, f"data root {root}, outside the instance")


@check("absolute-paths", "FG-07")
def absolute_paths(ctx: Context) -> Result:
    answers = (ctx.manifest().get("profile") or {}).get("init_answers") or {}
    relative = [f"answer {k}={v}" for k, v in answers.items() if _is_pathish(k) and _relative(v)]
    for sid, entry in ctx.registry().items():
        for key, value in ((entry or {}).get("params") or {}).items():
            if _is_pathish(key) and _relative(value):
                relative.append(f"{sid}.{key}={value}")
    if relative:
        return Result("absolute-paths", FAIL, "relative paths: " + ", ".join(relative[:6]),
                      "make them absolute in deployments/<name>/ (a relative path resolves against "
                      "the directory okg runs from), commit, then `okg catalog ownership claim`",
                      data={"relative": relative})
    return Result("absolute-paths", PASS, "every path answer and source path is absolute")


@check("profiles-env", "FG-04")
def profiles_env(ctx: Context) -> Result:
    env = ctx.probes.environ if ctx.probes.environ is not None else __import__("os").environ
    value = env.get("OKG_PROFILES_DIR")
    if value:
        return Result("profiles-env", WARN, f"OKG_PROFILES_DIR={value} is set",
                      "unset OKG_PROFILES_DIR: archi runs okg from the instance, which finds "
                      "./profiles/; by hand, the variable can make okg pick a different profile")
    return Result("profiles-env", PASS, "OKG_PROFILES_DIR is not set")


@check("okg-version", "FG-01")
def okg_version_check(ctx: Context) -> Result:
    from archi.stack.okg import okg_version

    installed = (ctx.probes.installed_okg or okg_version)()
    want = ctx.instance.okg_version
    if installed is None:
        return Result("okg-version", FAIL, "okg is not installed in this environment",
                      f"pip install the AKMON wheel: {stack_compose.okg_wheel(want)}")
    if installed != want:
        return Result("okg-version", FAIL,
                      f"okg {installed} is installed; the instance's MCP image and graph database run {want}",
                      f"pip install {stack_compose.okg_wheel(want)}  (or re-init on {installed})")
    return Result("okg-version", PASS, f"okg {installed}, as the stack runs")


# ---------------------------------------------------------------------------
# The compose file
# ---------------------------------------------------------------------------


def _service(ctx: Context, name: str) -> dict[str, Any]:
    return (ctx.compose().get("services") or {}).get(name) or {}


def _size_bytes(text: str) -> int:
    text = str(text).strip().lower()
    units = {"k": 1 << 10, "m": 1 << 20, "g": 1 << 30}
    if text and text[-1] in units:
        return int(float(text[:-1]) * units[text[-1]])
    return int(text or 0)


@check("graph-db-shm", "FG-02")
def graph_db_shm(ctx: Context) -> Result:
    shm = _service(ctx, "graph-db").get("shm_size")
    if not shm or _size_bytes(shm) < (1 << 30):
        return Result("graph-db-shm", FAIL, f"graph-db shm_size is {shm or 'the 64 MB default'}",
                      f"set services.graph-db.shm_size: {stack_compose.GRAPH_DB_SHM} (archi up re-renders "
                      "compose.yaml); index builds fail partway with less")
    return Result("graph-db-shm", PASS, f"graph-db shm_size {shm}")


@check("ports-loopback", "FG-17")
def ports_loopback(ctx: Context) -> Result:
    open_ports = []
    for name, service in (ctx.compose().get("services") or {}).items():
        for spec in service.get("ports") or []:
            if not str(spec).startswith("127.0.0.1:"):
                open_ports.append(f"{name}: {spec}")
    if open_ports:
        return Result("ports-loopback", FAIL, "published beyond loopback: " + ", ".join(open_ports),
                      "publish on 127.0.0.1 only; the chat database holds every conversation, and "
                      "other machines have no business reaching the graph or MCP")
    return Result("ports-loopback", PASS, "every published port is on 127.0.0.1")


@check("chat-db-distinct", "FG-15")
def chat_db_distinct(ctx: Context) -> Result:
    chat = _service(ctx, "chat-db")
    url = (_service(ctx, "webui").get("environment") or {}).get("DATABASE_URL", "")
    if not chat or "@chat-db:" not in url or "graph-db" in url:
        return Result("chat-db-distinct", FAIL, "Open WebUI does not use its own chat-db service",
                      "the chat app needs its own Postgres (okg refuses a shared one); restore the "
                      "chat-db service and DATABASE_URL with `archi up`")
    return Result("chat-db-distinct", PASS, "Open WebUI uses its own chat-db")


@check("mcp-port", "FG-20")
def mcp_port(ctx: Context) -> Result:
    declared = stack_compose.mcp_port(ctx.deployment)
    ports = _service(ctx, "mcp").get("ports") or []
    served = {int(str(spec).rsplit(":", 1)[1]) for spec in ports} if ports else set()
    if declared not in served:
        return Result("mcp-port", FAIL,
                      f"chat.mcp.port is {declared}; the mcp service serves {sorted(served) or 'nothing'}",
                      "run `archi up`: it renders the mcp service from chat.mcp.port, which "
                      "`okg chat sync` dials and nowhere else")
    return Result("mcp-port", PASS, f"MCP served on chat.mcp.port {declared}")


@check("selected-sources-fed", "FG-05")
def selected_sources_fed(ctx: Context) -> Result:
    empty = []
    for sid, entry in ctx.registry().items():
        params = (entry or {}).get("params") or {}
        for key in ("url", "repo_url"):
            if key in params and not str(params[key] or "").strip():
                empty.append(f"{sid}.{key}")
    if empty:
        return Result("selected-sources-fed", FAIL, "selected sources with no input: " + ", ".join(empty),
                      "answer them (archi add <pack> --answer ...) or remove the source: every selected "
                      "source must finish or nothing publishes",
                      data={"empty": empty})
    return Result("selected-sources-fed", PASS, "every selected source has its input")


# ---------------------------------------------------------------------------
# The running stack
# ---------------------------------------------------------------------------


@check("graph-db-preload", "FG-02")
def graph_db_preload(ctx: Context) -> Result:
    dsn = graph_dsn(ctx.instance, ctx.secrets)
    try:
        preload = ctx.probes.db_query(dsn, "SHOW shared_preload_libraries")[0][0]
        cron_db = ctx.probes.db_query(dsn, "SHOW cron.database_name")[0][0]
    except Exception as exc:  # noqa: BLE001 - any failure is the finding
        return Result("graph-db-preload", FAIL, f"cannot query the graph database: {exc}".strip(),
                      "`archi up` (is the graph-db service running?)")
    have = {p.strip() for p in str(preload).split(",")}
    missing = [lib for lib in stack_compose.PRELOAD_LIBRARIES if lib not in have]
    if missing or cron_db != ctx.instance.database:
        return Result("graph-db-preload", FAIL,
                      f"preload {preload!r}, cron.database_name {cron_db!r}",
                      "the graph database must start with shared_preload_libraries="
                      f"{','.join(stack_compose.PRELOAD_LIBRARIES)} and cron.database_name="
                      f"{ctx.instance.database}; `archi up` renders them")
    return Result("graph-db-preload", PASS, f"preloaded {preload}")


@check("role-passwords", "FG-10")
def role_passwords(ctx: Context) -> Result:
    path = ctx.instance.role_secrets_path
    if not path.is_file():
        return Result("role-passwords", FAIL, f"{path} is missing: okg's login roles have no password",
                      f"okg db role-passwords --deployment {ctx.instance.name} --apply  "
                      f"(with OKG_ROLE_SECRETS_FILE={path})")
    from archi.stack.okg import role_password

    dsn = graph_dsn(ctx.instance, ctx.secrets).replace(
        f"postgres:{ctx.secrets.get('ARCHI_GRAPH_DB_PASSWORD', '')}@",
        f"okg_mcp:{role_password(path, 'okg_mcp')}@",
    )
    try:
        user = ctx.probes.db_query(dsn, "SELECT current_user")[0][0]
    except Exception as exc:  # noqa: BLE001
        return Result("role-passwords", FAIL, f"okg_mcp cannot log in: {exc}".strip(),
                      f"okg db role-passwords --deployment {ctx.instance.name} --apply, then copy the "
                      "okg_mcp password into .env as ARCHI_OKG_MCP_PASSWORD and `archi up`")
    return Result("role-passwords", PASS, f"okg_mcp logs in as {user}")


@check("published", "FG-09")
def published(ctx: Context) -> Result:
    if ctx.probes.okg_json is None:
        return Result("published", SKIP, "no okg runner")
    out = ctx.probes.okg_json("status", "--deployment", ctx.instance.name)
    data = out.get("data") or {}
    pub = data.get("publishing") or data
    gen = pub.get("latest_published_generation")
    if not gen:
        return Result("published", FAIL, "no published generation",
                      f"okg status --deployment {ctx.instance.name}; if a run failed partway "
                      "(`staged source work was not published`), start over: `archi down --purge --yes`")
    return Result("published", PASS, f"published {gen}",
                  data={"nodes": pub.get("latest_published_node_count"),
                        "edges": pub.get("latest_published_edge_count")})


@check("services", "FG-16")
def services(ctx: Context) -> Result:
    if ctx.probes.services is None:
        return Result("services", SKIP, "no container runtime")
    running = ctx.probes.services()
    want = ("graph-db", "chat-db", "mcp", "webui")
    bad = [f"{s}: {running.get(s, {}).get('state') or 'absent'}" for s in want
           if running.get(s, {}).get("state") != "running"
           or running.get(s, {}).get("health") not in ("", "healthy")]
    if bad:
        return Result("services", FAIL, "not running: " + ", ".join(bad),
                      f"archi up --instance {ctx.instance.dir}", data={"services": running})
    return Result("services", PASS, "graph-db, chat-db, mcp and webui are running")


@check("mcp-answers", "FG-19")
def mcp_answers(ctx: Context) -> Result:
    url = f"http://127.0.0.1:{ctx.instance.ports.mcp}/mcp"
    try:
        out = ctx.probes.mcp(url, ctx.secrets.get("OKG_CHAT_MCP_TOKEN", ""))
    except Exception as exc:  # noqa: BLE001
        return Result("mcp-answers", FAIL, f"MCP at {url} did not answer: {exc!r}"[:400],
                      "archi up; then `docker compose logs mcp` (or podman) in the instance")
    if out.get("is_error") or "search" not in out.get("tools", []):
        return Result("mcp-answers", FAIL, f"MCP answered without a usable graph: {out.get('text', '')[:300]}",
                      "archi doctor's published check first; then restart mcp: archi up")
    return Result("mcp-answers", PASS, f"MCP serves {', '.join(out['tools'])}")


@check("webui-up", "FG-18")
def webui_up(ctx: Context) -> Result:
    url = f"http://127.0.0.1:{ctx.instance.ports.webui}/health"
    status = ctx.probes.http_status(url)
    if status != 200:
        return Result("webui-up", FAIL, f"Open WebUI health is {status or 'unreachable'} at {url}",
                      "archi up (the first start loads an embedding model and can take minutes)")
    return Result("webui-up", PASS, f"Open WebUI answers at {url[:-7]}")


# ---------------------------------------------------------------------------
# okg's own checks
# ---------------------------------------------------------------------------


def okg_lint(ctx: Context) -> Result:
    if ctx.probes.okg_json is None:
        return Result("okg-lint", SKIP, "no okg runner")
    out = ctx.probes.okg_json("deployment", "lint", ctx.instance.name)
    findings = (out.get("data") or {}).get("findings") or []
    errors = [f for f in findings if f.get("severity") in ("error", "fatal")]
    if out.get("returncode") or errors:
        first = errors[0] if errors else {}
        return Result("okg-lint", FAIL, f"okg deployment lint: {first.get('message', out.get('returncode'))}",
                      first.get("next_action") or f"okg deployment lint {ctx.instance.name}",
                      data={"errors": errors})
    return Result("okg-lint", PASS, f"okg deployment lint: {len(findings)} non-error finding(s)")


def okg_doctor(ctx: Context) -> Result:
    if ctx.probes.okg_json is None:
        return Result("okg-doctor", SKIP, "no okg runner")
    out = ctx.probes.okg_json("doctor", "--deployment", ctx.instance.name)
    data = out.get("data") or {}
    errors = data.get("errors") or []
    if out.get("returncode") or errors:
        first = errors[0] if errors else {}
        message = first.get("message") if isinstance(first, dict) else str(first)
        return Result("okg-doctor", FAIL, f"okg doctor: {message or out.get('returncode')}",
                      (first.get("fix_hint") if isinstance(first, dict) else None)
                      or f"okg doctor --deployment {ctx.instance.name}",
                      data={"errors": errors})
    return Result("okg-doctor", PASS, "okg doctor: ok")


def source_freshness(ctx: Context) -> Result:
    """Each source's age, from okg's watermarks, against its pack's `max_staleness`.

    `data.sources` lists `{source, age, max_staleness, ok}` (plus seconds and
    a reason) for automations and context gates.
    """
    from archi.bundles.lock import LockError, read_lock
    from archi.stack.freshness import evaluate

    if ctx.probes.okg_json is None:
        return Result("source-freshness", SKIP, "no okg runner")
    try:
        slos = read_lock(ctx.instance.dir).get("composed", {}).get("freshness") or {}
    except LockError as exc:
        return Result("source-freshness", SKIP, f"no pack freshness SLOs to check: {exc}")
    if not slos:
        return Result("source-freshness", SKIP, "no selected pack declares freshness")
    out = ctx.probes.okg_json("doctor", "--check", "sources", "--deployment", ctx.instance.name)
    report = out.get("data") or {}
    if out.get("returncode") not in (0, None) and not report.get("sources"):
        return Result("source-freshness", FAIL, "okg could not report source watermarks",
                      f"okg doctor --check sources --deployment {ctx.instance.name} --json")
    if not report.get("runtime_available"):
        return Result("source-freshness", FAIL,
                      f"okg cannot read the watermarks: {report.get('runtime_error') or 'runtime unavailable'}",
                      "archi up; then archi doctor again (the graph database must answer)")
    rows = evaluate(slos, report.get("sources") or [], ctx.probes.now())
    late = [r for r in rows if not r["ok"]]
    if late:
        first = late[0]
        more = f" (+{len(late) - 1} more)" if len(late) > 1 else ""
        return Result("source-freshness", FAIL,
                      f"{len(late)} of {len(rows)} sources over their max_staleness: "
                      f"{first['source']}: {first['reason']}{more}",
                      "archi ingest (run in the instance); to relax the "
                      "limit, edit the pack's freshness: and re-render",
                      data={"sources": rows})
    return Result("source-freshness", PASS, f"all {len(rows)} sources are within their max_staleness",
                  data={"sources": rows})


def automation_names(ctx: Context) -> Result:
    """Every `automations:` name in archi.yaml is an automation pack the instance selected."""
    from archi.bundles.lock import LockError, read_lock
    from archi.ops.levels import unknown_automation_message, unknown_automations

    settings = ctx.instance.automations
    if not settings:
        return Result("automation-names", PASS, "archi.yaml sets no automation levels")
    try:
        locked = read_lock(ctx.instance.dir).get("composed", {}).get("automations") or {}
    except LockError as exc:
        return Result("automation-names", SKIP, f"cannot compare with the selected packs: {exc}")
    unknown = unknown_automations(settings, locked)
    if unknown:
        return Result("automation-names", FAIL, unknown_automation_message(unknown, locked),
                      "fix the name under automations: in archi.yaml (it must be a selected automation pack), "
                      "or add the pack with archi add",
                      data={"unknown": unknown, "selected": sorted(locked)})
    return Result("automation-names", PASS, f"{len(settings)} automation setting(s) match selected packs")


def run_checks(ctx: Context) -> list[Result]:
    results = [okg_lint(ctx), okg_doctor(ctx), source_freshness(ctx), automation_names(ctx)]
    for check_id, footgun, fn in CHECKS:
        try:
            result = fn(ctx)
        except Exception as exc:  # noqa: BLE001 - a crashing check is a failed check
            result = Result(check_id, FAIL, f"check crashed: {exc!r}"[:400],
                            "report this as an archi bug, with `archi doctor --json`")
        result.footgun = footgun
        results.append(result)
    return results
