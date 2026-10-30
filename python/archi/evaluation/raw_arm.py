"""The raw-document baseline of `archi eval --context`: two agent arms (PACT r3-context-quality).

Both arms put the same model, under the same caps and prompt, in front of a question;
they differ only in where the knowledge is:

    agent-bash    one tool, bash, run in a bubblewrap sandbox: no network, the raw corpus
                  (`archi.evaluation.raw_corpus`) mounted read-only at /corpus, a tmpfs
                  HOME, an empty environment, rg, jq and python3 on PATH when the host has
                  them, and resource limits (processes, memory, file size, CPU time). Each
                  command may run 120 s and print 64 KiB.
    agent-graph   the instance's MCP tools.

The prompt is one text with one paragraph, the one saying where the knowledge lives, that
differs between the arms (`PROMPT`, `KNOWLEDGE`).

Scoring is the same for both (`score_evidence`), over the tool responses the model saw
(the visible part of a cut response counts; tool inputs and the final answer never do):

- a gold record is **read** when part of its content appears in a response: a run of six
  words from one of its field values or lines, or a whole value of four words or more (or
  of sixteen characters or more). Its path, natural key or node id alone only **lists**
  it: listed records are counted and reported, never credited;
- evidence recall per try: the share of the question's available gold records read. A
  gold record with no files in the corpus (ops-record, a live source), or with no content
  distinctive enough to recognise, is unavailable: it leaves the denominator, in both
  arms, and is listed;
- MRR: 1 / the rank of the first gold record read among the distinct records the agent
  met (other records count when their path or node id appears);
- the answer score: a judge model compares the answer with the question's answer key. It
  runs only with `--judge` and only for questions that have a key; with no keys at all the
  report says "no answer keys", never 0.

A try that ends in a model error is listed and counted per arm, and left out of the
averages. A 4xx from the model endpoint other than 429 or a context-length error aborts
the run; 429 and 5xx are retried with backoff, at most three times.

A gold id is resolved to its record through the MCP `inspect` tool (its source and node
id, matched against the corpus's keys.json), or through the question's `gold_sources`. A
gold id that does not resolve refuses the run before any model call, with the id and the
reason: it is never scored as a miss.

Every model call, agent and judge, answered or not, is written to a JSONL spend ledger
with its tokens and cost; a call that failed after it was sent, or a reply without token
usage, is charged at the call's bound. The run refuses to start without a spend cap, and
stops before a call whose cost bound (input estimated from the request's size, output at
most `max_output_tokens`) would cross it; the report then says what completed.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from archi.evaluation.agent import (ChatModel, ChatReply, ModelError, ToolSpec, _describe, assistant_message,
                                    openai_tools)

AGENT_BASH, AGENT_GRAPH = "agent-bash", "agent-graph"
AGENT_ARMS = (AGENT_BASH, AGENT_GRAPH)
#: The ledger file name in a run's directory.
LEDGER = "ledger.jsonl"
DEFAULT_CAP_USD = 10.0
NO_ANSWER_KEYS = "no answer keys"
JUDGE_OFF = "judge off"
INSPECT = "inspect"
#: okg's JSON result format, as the no-model arms ask for it.
JSON_META = {"okg/result_format": "json"}
#: Retries of a 429 or 5xx reply, after the first attempt.
MAX_RETRIES = 3
#: Prices below these (dollars per million input / output tokens) are refused without
#: --allow-low-price: a typo in a price would make the cap meaningless.
LOW_PRICE = (0.01, 0.04)


@dataclass(frozen=True)
class Caps:
    max_turns: int = 30
    max_tool_calls: int = 200
    try_seconds: float = 900.0
    #: a model call is not started with less of the try left than this
    min_call_seconds: float = 60.0
    command_seconds: float = 120.0
    #: the most of one tool response the model sees (and the scorer reads), both arms
    output_bytes: int = 64 * 1024
    #: sandbox limits per command: processes beyond what the user already runs, address
    #: space, size of any file written, CPU seconds (the wall limit is command_seconds)
    max_procs: int = 256
    memory_bytes: int = 4 * 1024 ** 3
    file_bytes: int = 256 * 1024 ** 2


# --- the prompt ------------------------------------------------------------------------------

PROMPT = """\
You answer questions from the operators of a scientific computing facility.

{knowledge}

Find the records that answer the question, read them, and answer only from what you read. \
If what you find does not answer the question, say so rather than guess. You have at most \
{turns} turns and {calls} tool calls for this question.

End your answer with a line "Sources:" followed by the records you used."""

KNOWLEDGE = {
    AGENT_BASH: (
        "Where the knowledge is: the directory /corpus, read-only, holds the raw records this "
        "facility's knowledge graph was built from, one file per record, one folder per source "
        "({folders}). Your one tool, bash, runs a command in a sandbox with no network; {tools} "
        "A command may run for {seconds} seconds and print {kib} KiB; one that goes over either "
        "is stopped. Name each record you used by its file path."
    ),
    AGENT_GRAPH: (
        "Where the knowledge is: this facility's knowledge graph, which you reach through the "
        "tools you are given. Name each record you used by its id, as the tools show it."
    ),
}


def build_prompt(arm: str, caps: Caps, *, folders: Sequence[str] = (), tools: Sequence[str] = ()) -> str:
    if arm == AGENT_BASH:
        named = ", ".join(tools)
        tool_text = (f"{named} {'is' if len(tools) == 1 else 'are'} installed beside the usual shell commands."
                     if tools else "only the usual shell commands are installed.")
        knowledge = KNOWLEDGE[arm].format(folders=", ".join(folders) or "none", tools=tool_text,
                                          seconds=int(caps.command_seconds), kib=caps.output_bytes // 1024)
    else:
        knowledge = KNOWLEDGE[arm]
    return PROMPT.format(knowledge=knowledge, turns=caps.max_turns, calls=caps.max_tool_calls)


# --- money -----------------------------------------------------------------------------------


class SpendError(RuntimeError):
    """The run may not start: no usable spend cap or price."""


class CapReached(Exception):
    """The next model call could cross the spend cap."""


class FatalModelError(ModelError):
    """The model endpoint refused the request in a way a retry cannot fix (a 4xx other than 429
    or a context-length error): the whole run stops."""


@dataclass(frozen=True)
class Price:
    """US dollars per million tokens."""

    input_per_m: float
    output_per_m: float
    cached_input_per_m: Optional[float] = None

    @classmethod
    def parse(cls, text: str) -> "Price":
        """`IN/OUT` or `IN/OUT/CACHED`, dollars per million tokens."""
        parts = str(text).split("/")
        try:
            values = [float(p) for p in parts]
        except ValueError:
            values = []
        if len(values) not in (2, 3) or not all(math.isfinite(v) and v >= 0 for v in values):
            raise SpendError(f"price {text!r} is not IN/OUT or IN/OUT/CACHED in dollars per million tokens")
        return cls(*values)

    def check_plausible(self, *, allow_low: bool, what: str = "--model-price") -> None:
        """Refuse a price so small the cap would mean nothing, unless the operator says so."""
        if allow_low:
            return
        if self.input_per_m < LOW_PRICE[0] or self.output_per_m < LOW_PRICE[1]:
            raise SpendError(
                f"{what} {self.input_per_m:g}/{self.output_per_m:g} is below ${LOW_PRICE[0]:g} input or "
                f"${LOW_PRICE[1]:g} output per million tokens; that is likely a typo (prices are per MILLION "
                "tokens). Pass --allow-low-price if it is right, for example for a local model")

    def cost(self, prompt: int, completion: int, cached: int = 0) -> float:
        cached = max(0, min(cached, prompt))
        cached_rate = self.input_per_m if self.cached_input_per_m is None else self.cached_input_per_m
        return ((prompt - cached) * self.input_per_m + cached * cached_rate + completion * self.output_per_m) / 1e6

    def bound(self, prompt: int, completion: int) -> float:
        return (prompt * self.input_per_m + completion * self.output_per_m) / 1e6


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Ledger:
    """One JSONL row per model call, answered or not. One per invocation: its cap covers every
    question set; `redirect` moves where the next rows go (one file per run directory) while the
    spend carries over. Refuses to exist without a cap."""

    def __init__(self, path: Path, cap_usd: Optional[float]) -> None:
        if cap_usd is None or isinstance(cap_usd, bool) or not isinstance(cap_usd, (int, float)) \
                or not math.isfinite(cap_usd) or cap_usd <= 0:
            raise SpendError(f"the agent arms refuse to start without a spend cap (got {cap_usd!r}); "
                             "pass --cap DOLLARS")
        self.cap = float(cap_usd)
        self.spent = 0.0
        self.calls = 0
        self.stopped: Optional[str] = None
        self.warnings: list[str] = []
        self.redirect(path)

    def redirect(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch()

    def check(self, bound: float) -> None:
        if self.stopped:
            raise CapReached(self.stopped)
        if self.spent + bound > self.cap:
            self.stopped = (f"the next call could cost up to ${bound:.4f}, and ${self.spent:.4f} of the "
                            f"${self.cap:.2f} cap is spent")
            raise CapReached(self.stopped)

    def _write(self, row: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if self.spent > self.cap and not self.stopped:
            self.stopped = f"the spend reached ${self.spent:.4f}, over the ${self.cap:.2f} cap"

    def record(self, *, arm: str, role: str, question: str, attempt: int, model: str, reply: ChatReply,
               price: Price, bound: float) -> float:
        if reply.usage_reported:
            cost, note = price.cost(reply.prompt_tokens, reply.completion_tokens, reply.cached_tokens), None
        else:
            cost, note = bound, "the reply carried no token usage; charged at the call's bound"
            self.warnings.append(f"{arm} {question} try {attempt} ({role}): {note}")
        self.spent += cost
        self.calls += 1
        row = {"at": _now(), "arm": arm, "role": role, "question": question, "try": attempt, "model": model,
               "outcome": "answered", "prompt_tokens": reply.prompt_tokens, "cached_tokens": reply.cached_tokens,
               "completion_tokens": reply.completion_tokens, "reasoning_tokens": reply.reasoning_tokens,
               "finish_reason": reply.finish_reason or None,
               "cost_usd": round(cost, 6), "bound_usd": round(bound, 6), "spent_usd": round(self.spent, 6)}
        if note:
            row["note"] = note
        self._write(row)
        return cost

    def record_failure(self, *, arm: str, role: str, question: str, attempt: int, model: str, bound: float,
                       error: ModelError) -> float:
        """A call that failed after it was sent may still be billed: charge it at its bound."""
        self.spent += bound
        self.calls += 1
        self._write({"at": _now(), "arm": arm, "role": role, "question": question, "try": attempt, "model": model,
                     "outcome": "failed", "status": error.status, "error": str(error)[:300],
                     "cost_usd": round(bound, 6), "bound_usd": round(bound, 6), "spent_usd": round(self.spent, 6),
                     "note": "failed after it was sent; charged at the call's bound"})
        return bound


def estimate_tokens(messages: list, tools: list) -> int:
    """A bound on the request's input tokens: one per three bytes of its JSON (BPE tokens
    average about four)."""
    return math.ceil(len(json.dumps([messages, tools]).encode("utf-8")) / 3)


def _context_length(error: ModelError) -> bool:
    text = str(error).lower()
    return error.status == 400 and any(s in text for s in (
        "context_length", "context length", "maximum context", "too many tokens", "prompt is too long"))


def _retryable(error: ModelError) -> bool:
    return error.status == 429 or (error.status is not None and 500 <= error.status < 600)


def _fatal(error: ModelError) -> bool:
    return (error.status is not None and 400 <= error.status < 500 and error.status != 429
            and not _context_length(error))


@dataclass
class Metered:
    """A model behind the ledger: each call is checked against the cap first and recorded after,
    answered or not; 429 and 5xx are retried with backoff."""

    model: ChatModel
    price: Price
    ledger: Ledger
    max_output_tokens: int = 16000
    sleep: Callable[[float], None] = time.sleep

    def bound_for(self, messages: list, tools: list) -> float:
        return self.price.bound(estimate_tokens(messages, tools), self.max_output_tokens)

    def complete(self, messages: list, tools: list, *, arm: str, role: str, question: str, attempt: int,
                 seconds: Optional[float] = None) -> ChatReply:
        """`seconds` bounds the call's HTTP timeout (a try's time left), on a model that has one.
        Raises CapReached before a call that could cross the cap, FatalModelError on a 4xx that a
        retry cannot fix, ModelError otherwise."""
        started = time.monotonic()
        retries = 0
        while True:
            bound = self.bound_for(messages, tools)
            self.ledger.check(bound)
            left = None if seconds is None else seconds - (time.monotonic() - started)
            saved = getattr(self.model, "timeout", None)
            if left is not None and isinstance(saved, (int, float)):
                self.model.timeout = max(1.0, min(saved, left))
            try:
                reply = self.model.complete(messages, tools)
            except ModelError as exc:
                if exc.sent:
                    self.ledger.record_failure(arm=arm, role=role, question=question, attempt=attempt,
                                               model=self.model.name, bound=bound, error=exc)
                if _fatal(exc):
                    raise FatalModelError(str(exc), status=exc.status, sent=exc.sent) from exc
                if _retryable(exc) and retries < MAX_RETRIES:
                    retries += 1
                    delay = min(60.0, exc.retry_after if exc.retry_after else 2.0 * 2 ** retries)
                    if left is not None and delay >= left:
                        raise
                    self.sleep(delay)
                    continue
                raise
            finally:
                if left is not None and isinstance(saved, (int, float)):
                    self.model.timeout = saved
            self.ledger.record(arm=arm, role=role, question=question, attempt=attempt, model=self.model.name,
                               reply=reply, price=self.price, bound=bound)
            return reply


def read_key_file(variable: str, environ: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """The API key in the file whose path the environment variable `variable` holds, or None
    when the variable is unset. The key is never logged; a problem names only the variable."""
    env = os.environ if environ is None else environ
    where = env.get(variable)
    if not where:
        return None
    try:
        key = Path(where).read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise SpendError(f"cannot read the key file named by ${variable} ({type(exc).__name__})") from None
    if not key:
        raise SpendError(f"the key file named by ${variable} is empty")
    return key


# --- the sandbox -----------------------------------------------------------------------------


class SandboxError(RuntimeError):
    """The bash arm cannot run here."""


class SandboxBroken(SandboxError):
    """The sandbox stopped working during the run (its systemd scope can no longer start): the run
    aborts rather than hand the model the wrapper's error as a command's output."""


TOOLS = ("rg", "jq", "python3")
#: Where tools are found inside the sandbox without binding anything: /usr is mounted whole.
_SYSTEM_DIRS = ("/usr/local/bin", "/usr/bin", "/bin")
_EXTRA_TOOLS = "/opt/tools"
#: The agent's HOME inside the sandbox: its own tmpfs, empty for every command.
SANDBOX_HOME = "/agent-home"
#: Sets each limit inside the sandbox before the command runs: `lim <flag> <value>` lowers a
#: limit only (a host's lower hard limit stands), and a limit that cannot be set refuses the
#: command (exit 126). bubblewrap 0.6.3 cannot size its tmpfs mounts (`--size` came later), so
#: the file-size limit is what bounds a write to /tmp or HOME. RLIMIT_NPROC counts processes per
#: user namespace on current kernels (5.14 and later), so inside bwrap's new namespace the limit
#: is `max_procs` itself; on a kernel that counts the user's processes on the whole host, a
#: sandbox whose probe cannot fork falls back to the user's count plus `max_procs`.
_LIMITS = (
    'lim() { h=$(ulimit -H "$1"); if [ "$h" = unlimited ] || [ "$h" -gt "$2" ]; then '
    'ulimit "$1" "$2" || exit 126; fi; }; '
    'lim -u {procs} && lim -v {memory_kib} && lim -f {file_kib} && lim -t {cpu} || exit 126; '
    'exec "$0" --noprofile --norc -c "$1"'
)


#: The rlimits above bound each process, so a command's processes together, and what it writes to
#: its tmpfs mounts, could still take max_procs x memory_bytes. Where the user's systemd manager
#: runs, each command also runs in its own transient scope, whose MemoryMax (tmpfs pages
#: included) and TasksMax bound the command as a whole. `BashSandbox.limit_scope` says which held.
SCOPE_COMMAND, SCOPE_PROCESS = "command", "process"
#: What systemd-run --user needs to find the user's manager; nothing else reaches it, and
#: bubblewrap clears the environment before the command starts.
_SCOPE_ENV = ("XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS")
#: How systemd-run reports, before the command starts, that it could not make the scope
_SCOPE_FAILED = re.compile(r"\A(Failed to (connect to bus|create bus connection|start transient scope unit)"
                           r"|Job for run-\S+\.scope failed)")


def find_bwrap() -> Optional[str]:
    return shutil.which("bwrap")


def _user_tasks() -> int:
    """Threads the current user runs on this host (what RLIMIT_NPROC counts)."""
    uid, count = os.getuid(), 0
    try:
        entries = os.listdir("/proc")
    except OSError:
        return 0
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            if os.stat(f"/proc/{entry}").st_uid == uid:
                count += len(os.listdir(f"/proc/{entry}/task"))
        except OSError:
            continue
    return count


@dataclass
class BashResult:
    text: str
    exit_code: Optional[int]
    timed_out: bool = False
    cut: bool = False
    #: whether the command printed anything (before the harness's own notes)
    produced: bool = False

    @property
    def ok(self) -> bool:
        """A clean run: exit 0 within every limit."""
        return self.exit_code == 0 and not self.timed_out and not self.cut

    @property
    def responded(self) -> bool:
        """Whether this is a response the scorer reads: anything the command printed, whatever its
        exit status, and the visible part of output that was cut."""
        return self.produced


class BashSandbox:
    """`bash -c` in bubblewrap: no network, an empty environment, the corpus read-only at /corpus,
    resource limits. Starting one runs `true` through the real argv once and refuses on failure."""

    def __init__(self, corpus: Path, *, bwrap: Optional[str] = None, command_seconds: float = 120.0,
                 output_bytes: int = 64 * 1024, max_procs: int = 256, memory_bytes: int = 4 * 1024 ** 3,
                 file_bytes: int = 256 * 1024 ** 2) -> None:
        self.bwrap = bwrap or find_bwrap()
        if not self.bwrap:
            raise SandboxError("bubblewrap (bwrap) is not installed on this host; the agent-bash arm refuses "
                               "to run without its sandbox")
        try:
            self.corpus = Path(corpus).resolve(strict=True)
        except OSError as exc:
            raise SandboxError(f"the corpus directory {corpus} does not exist") from exc
        self.command_seconds = command_seconds
        self.output_bytes = output_bytes
        self.max_procs = max_procs
        self.memory_bytes = memory_bytes
        self.file_bytes = file_bytes
        self.binds: list[tuple[str, str]] = []
        self.tools: list[str] = []
        for name in TOOLS:
            if any(Path(d, name).is_file() for d in _SYSTEM_DIRS):
                self.tools.append(name)
                continue
            found = shutil.which(name)
            if found and name != "python3":  # a single binary can be bound; a python needs its tree
                self.binds.append((str(Path(found).resolve()), f"{_EXTRA_TOOLS}/{name}"))
                self.tools.append(name)
        shell = next((p for p in ("/usr/bin/bash", "/bin/bash") if Path(p).is_file()), None)
        if shell is None:
            raise SandboxError("no bash at /usr/bin/bash or /bin/bash")
        self.shell = shell
        #: "namespace" (the process limit is max_procs) or "host" (the user's count plus max_procs)
        self.nproc_mode = "namespace"
        #: SCOPE_COMMAND (a systemd scope bounds each command's memory and tasks as a whole) or
        #: SCOPE_PROCESS (only the per-process rlimits), and why not the first
        self.limit_scope = SCOPE_PROCESS
        self.scope_note = ""
        self._scope: list[str] = []
        self._scope_env: dict[str, str] = {}
        seconds = min(30.0, command_seconds)
        try:
            probe = self.run("true", seconds=seconds)
        except OSError as exc:
            raise SandboxError(f"the sandbox cannot start ({self.bwrap}): {exc}") from exc
        if not probe.ok:
            raise SandboxError(f"the sandbox cannot start: `true` in it gave exit {probe.exit_code}: "
                               f"{probe.text.strip()[:300]}")
        if not self.run("/usr/bin/env true | cat", seconds=seconds).ok:
            self.nproc_mode = "host"
            again = self.run("/usr/bin/env true | cat", seconds=seconds)
            if not again.ok:
                raise SandboxError(f"the sandbox cannot start a pipeline under its process limit: exit "
                                   f"{again.exit_code}: {again.text.strip()[:300]}")
        self._try_scope(seconds)

    def _try_scope(self, seconds: float) -> None:
        systemd_run = shutil.which("systemd-run")
        if not systemd_run:
            self.scope_note = "systemd-run is not installed"
            return
        self._scope = [systemd_run, "--user", "--scope", "--quiet", "--collect",
                       "-p", f"MemoryMax={self.memory_bytes}", "-p", "MemorySwapMax=0",
                       "-p", f"TasksMax={self.max_procs}", "--"]
        self._scope_env = {k: os.environ[k] for k in _SCOPE_ENV if os.environ.get(k)}
        try:
            probe = self.run("/usr/bin/env true | cat", seconds=seconds)
            why = "" if probe.ok else f"exit {probe.exit_code}: {probe.text.strip()[:200]}"
        except SandboxBroken as exc:
            why = str(exc)
        if not why:
            self.limit_scope = SCOPE_COMMAND
            return
        self._scope, self._scope_env = [], {}
        self.scope_note = f"systemd-run --user --scope did not run a command ({why})"

    def limits(self) -> str:
        procs = self.max_procs + (_user_tasks() if self.nproc_mode == "host" else 0)
        return (_LIMITS.replace("{procs}", str(procs))
                .replace("{memory_kib}", str(self.memory_bytes // 1024))
                .replace("{file_kib}", str(max(1, self.file_bytes // 1024)))
                .replace("{cpu}", str(max(1, math.ceil(self.command_seconds)))))

    def argv(self, command: str) -> list[str]:
        args = [
            *self._scope,
            self.bwrap,
            "--unshare-all", "--unshare-net", "--die-with-parent", "--new-session",
            "--clearenv",
            "--setenv", "PATH", f"/usr/local/bin:/usr/bin:/bin:{_EXTRA_TOOLS}",
            "--setenv", "HOME", SANDBOX_HOME,
            "--setenv", "TMPDIR", "/tmp",
            "--setenv", "LANG", "C.UTF-8",
            "--ro-bind", "/usr", "/usr",
        ]
        for top in ("/bin", "/sbin", "/lib", "/lib64", "/lib32"):
            path = Path(top)
            if path.is_symlink():
                args += ["--symlink", os.readlink(top), top]
            elif path.is_dir():
                args += ["--ro-bind", top, top]
        args += ["--ro-bind-try", "/etc/alternatives", "/etc/alternatives",
                 "--ro-bind-try", "/etc/ld.so.cache", "/etc/ld.so.cache",
                 "--proc", "/proc", "--dev", "/dev",
                 "--tmpfs", "/tmp", "--tmpfs", SANDBOX_HOME]
        for source, target in self.binds:
            args += ["--ro-bind", source, target]
        args += ["--ro-bind", str(self.corpus), "/corpus", "--chdir", "/corpus",
                 "--", self.shell, "--noprofile", "--norc", "-c", self.limits(), self.shell, command]
        return args

    def _confirm_broken(self, text: str) -> None:
        """A command whose output starts like systemd-run's own failure: raise SandboxBroken only
        if `true` cannot start in a scope either (else the command printed that text itself, as a
        `cat` of an ops log can)."""
        try:
            check = subprocess.run(self.argv("true"), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, env=dict(self._scope_env), timeout=30)
            if check.returncode == 0:
                return
            why = check.stdout.decode("utf-8", "replace").strip() or f"exit {check.returncode}"
        except subprocess.TimeoutExpired:
            why = "`true` did not finish in 30 s"
        raise SandboxBroken(f"systemd-run could not start the command's scope: {text.strip()[:200]}; "
                            f"a check with `true` also failed: {why[:200]}")

    def run(self, command: str, *, seconds: Optional[float] = None) -> BashResult:
        limit = self.command_seconds if seconds is None else max(0.1, min(seconds, self.command_seconds))
        proc = subprocess.Popen(self.argv(command), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, env=dict(self._scope_env), start_new_session=True)
        chunks: list[bytes] = []
        state = {"size": 0, "cut": False}

        def drain() -> None:
            assert proc.stdout is not None
            while True:
                block = proc.stdout.read1(65536) if hasattr(proc.stdout, "read1") else proc.stdout.read(65536)
                if not block:
                    return
                room = self.output_bytes - state["size"]
                if room > 0:
                    chunks.append(block[:room])
                state["size"] += len(block)
                if state["size"] > self.output_bytes and not state["cut"]:
                    state["cut"] = True
                    _kill(proc)

        reader = threading.Thread(target=drain, daemon=True)
        reader.start()
        timed_out = False
        try:
            proc.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill(proc)
            proc.wait()
        reader.join(timeout=5)
        text = b"".join(chunks).decode("utf-8", "replace")
        if self._scope and proc.returncode and not timed_out and _SCOPE_FAILED.match(text):
            self._confirm_broken(text)
        produced = bool(text.strip())
        notes = []
        if state["cut"]:
            notes.append(f"[output cut at {self.output_bytes} bytes; the command was stopped]")
        if timed_out:
            notes.append(f"[the command was stopped after {limit:g} seconds]")
        elif proc.returncode and not state["cut"]:
            notes.append(f"[exit code {proc.returncode}]")
        if notes:
            text = text + ("\n" if text and not text.endswith("\n") else "") + "\n".join(notes)
        return BashResult(text, None if timed_out else proc.returncode, timed_out, state["cut"], produced)


def _kill(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


BASH_SPEC = ToolSpec(
    "bash",
    "Run one bash command in the sandbox and return what it printed (stdout and stderr together). "
    "The working directory is /corpus.",
    {"type": "object", "properties": {"command": {"type": "string", "description": "the command to run"}},
     "required": ["command"]},
)


class BashTools:
    """The bash arm's tool box (the `ToolBox` shape the MCP one has). A command that printed
    anything is a response, whatever its exit status; one that printed nothing is not."""

    instructions: Optional[str] = None

    def __init__(self, sandbox: BashSandbox) -> None:
        self.sandbox = sandbox

    async def list_tools(self) -> list[ToolSpec]:
        return [BASH_SPEC]

    async def call(self, name: str, arguments: Mapping[str, Any], *, seconds: Optional[float] = None
                   ) -> tuple[str, bool]:
        command = arguments.get("command") if isinstance(arguments, Mapping) else None
        if name != "bash" or not isinstance(command, str) or not command.strip():
            return "error: call bash with {\"command\": \"...\"}", True
        result = await asyncio.to_thread(self.sandbox.run, command, seconds=seconds)
        return result.text, not result.responded


# --- one try ---------------------------------------------------------------------------------

ANSWERED, CAPPED, ERROR, STOPPED = "answered", "capped", "error", "stopped_at_cap"
#: the reason of a try whose model ran out of output tokens before answering
OUTPUT_LIMIT = "the model reached max_output_tokens before it answered (finish_reason length)"


@dataclass
class ToolEvent:
    name: str
    arguments: Mapping[str, Any]
    #: the response as the model saw it (at most `Caps.output_bytes`)
    text: str
    #: whether the scorer reads it: an MCP call the server did not report as an error, a bash
    #: command that printed anything
    ok: bool


@dataclass
class TryResult:
    arm: str
    question: str
    attempt: int
    status: str = ANSWERED
    reason: Optional[str] = None
    answer: str = ""
    events: list[ToolEvent] = field(default_factory=list)
    turns: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    seconds: float = 0.0
    evidence: Optional["Evidence"] = None
    judge: Optional[dict[str, Any]] = None

    @property
    def tool_calls(self) -> int:
        return len(self.events)

    @property
    def scored(self) -> bool:
        """A try stopped by the spend cap or ended by a model error is listed, not scored."""
        return self.status not in (STOPPED, ERROR) and self.evidence is not None and self.evidence.available > 0

    def to_dict(self) -> dict[str, Any]:
        out = {"try": self.attempt, "status": self.status, "reason": self.reason, "turns": self.turns,
               "tool_calls": self.tool_calls, "unread_tool_calls": sum(not e.ok for e in self.events),
               "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
               "cost_usd": round(self.cost, 6), "seconds": round(self.seconds, 1), "answer": self.answer,
               "calls": [{"tool": e.name, "arguments": dict(e.arguments), "ok": e.ok, "bytes": len(e.text)}
                         for e in self.events]}
        if self.evidence is not None:
            out["evidence"] = self.evidence.to_dict()
        if self.judge is not None:
            out["judge"] = self.judge
        return out


def _cut(text: str, limit: int) -> str:
    data = text.encode("utf-8")
    if len(data) <= limit:
        return text
    return data[:limit].decode("utf-8", "ignore") + f"\n[cut at {limit} bytes]"


async def run_try(model: Metered, tools: Any, specs: list[ToolSpec], *, arm: str, system: str, question_id: str,
                  question: str, attempt: int, caps: Caps, clock: Callable[[], float] = time.monotonic) -> TryResult:
    """One question, once, through the loop: model, tool calls, results back, until it answers or
    a cap ends it. A model failure ends it as an error; the spend cap ends it as stopped; a
    FatalModelError propagates (it stops the run)."""
    result = TryResult(arm, question_id, attempt)
    started = clock()
    deadline = started + caps.try_seconds
    schema = openai_tools(specs)
    known = {s.name for s in specs}
    messages: list[dict] = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    spent_before = model.ledger.spent

    def cap(reason: str) -> None:
        result.status, result.reason = CAPPED, reason

    try:
        while True:
            if result.turns >= caps.max_turns:
                cap(f"{caps.max_turns} model turns")
                break
            left = deadline - clock()
            if left < max(caps.min_call_seconds, 0.0) or left <= 0:
                cap(f"{caps.try_seconds:g} seconds (a call is not started with under "
                    f"{caps.min_call_seconds:g} s left)")
                break
            reply = await asyncio.to_thread(model.complete, messages, schema, arm=arm, role="agent",
                                            question=question_id, attempt=attempt, seconds=left)
            result.turns += 1
            result.prompt_tokens += reply.prompt_tokens
            result.completion_tokens += reply.completion_tokens
            if not reply.tool_calls:
                if reply.finish_reason == "length" and not (reply.content or "").strip():
                    result.status, result.reason = ERROR, OUTPUT_LIMIT
                else:
                    result.answer = reply.content
                break
            messages.append(assistant_message(reply))
            for call in reply.tool_calls:
                if result.tool_calls >= caps.max_tool_calls:
                    cap(f"{caps.max_tool_calls} tool calls")
                    break
                remaining = deadline - clock()
                if remaining <= 0:
                    cap(f"{caps.try_seconds:g} seconds")
                    break
                if call.name not in known:
                    text, is_error = f"error: no tool named {call.name!r}; tools are {sorted(known)}", True
                else:
                    try:
                        if isinstance(tools, BashTools):
                            text, is_error = await tools.call(call.name, call.arguments, seconds=remaining)
                        else:
                            text, is_error = await asyncio.wait_for(tools.call(call.name, call.arguments),
                                                                    timeout=remaining)
                    except asyncio.TimeoutError:
                        text, is_error = f"error: the call did not finish in the {remaining:.0f} s the try had left", True
                    except SandboxError:
                        raise
                    except Exception as exc:  # noqa: BLE001 - a failing tool is data for the model
                        text, is_error = f"error: {_describe(exc)}", True
                shown = _cut(text, caps.output_bytes)
                result.events.append(ToolEvent(call.name, dict(call.arguments), shown, not is_error))
                messages.append({"role": "tool", "tool_call_id": call.id, "content": shown})
            if result.status == CAPPED:
                break
    except CapReached as exc:
        result.status, result.reason = STOPPED, str(exc)
    except FatalModelError:
        raise
    except ModelError as exc:
        result.status, result.reason = ERROR, str(exc)
    result.cost = model.ledger.spent - spent_before
    result.seconds = clock() - started
    return result


# --- gold records ----------------------------------------------------------------------------

RESOLVED, OVERRIDE, UNAVAILABLE = "resolved", "override", "unavailable"
NO_CONTENT = "its record has no field value or line distinctive enough to recognise in a tool response"
ALL_SHARED = ("every value or line of its record that is long enough to recognise also appears in another "
              "corpus record, so no tool response can show that this one was read")


@dataclass(frozen=True)
class GoldRecord:
    node_id: str
    status: str
    source: Optional[str] = None
    key: Optional[str] = None
    path: Optional[str] = None
    reason: str = ""
    #: one entry per value or line of the record: the word runs that show it (see `content_fingerprints`)
    units: tuple = ()
    #: fingerprints dropped because another corpus record holds the same text (see `drop_shared`)
    shared: int = 0

    @property
    def available(self) -> bool:
        return self.status != UNAVAILABLE

    @property
    def needed(self) -> int:
        """How many of its values or lines a response must show for the record to count as read."""
        return min(READ_UNITS, len(self.units))

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "source": self.source, "key": self.key, "path": self.path,
                "reason": self.reason or None, "units": len(self.units),
                "fingerprints": sum(len(u) for u in self.units), "shared_fingerprints_dropped": self.shared}


class GoldError(RuntimeError):
    """Gold ids that did not resolve: the run refuses to score."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("the agent arms refuse to run: " + str(len(problems)) + " gold id(s) do not resolve to a "
                         "corpus record (a gold id that does not resolve is never scored as a miss):\n"
                         + "\n".join(f"  - {p}" for p in problems))
        self.problems = problems


def _find_node(payload: Any, node_id: str) -> Optional[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        if payload.get("node_id") == node_id:
            return payload
        for value in payload.values():
            found = _find_node(value, node_id)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _find_node(value, node_id)
            if found is not None:
                return found
    return None


@dataclass(frozen=True)
class NodeInfo:
    node_id: str
    subtype: str
    source: str
    attrs: Mapping[str, Any]
    generation: Optional[str] = None


async def inspect_node(tools: Any, node_id: str, *, generation: Optional[str] = None) -> tuple[Optional[NodeInfo], str]:
    """(the node, "") or (None, why not), through the MCP inspect tool."""
    arguments: dict[str, Any] = {"target": {"kind": "node", "node_id": node_id}}
    if generation is not None:
        arguments["generation_id"] = generation
    try:
        text, is_error = await tools.call(INSPECT, arguments, meta=JSON_META)
    except Exception as exc:  # noqa: BLE001
        return None, f"inspect failed: {_describe(exc)}"
    if is_error:
        return None, f"inspect reported an error: {text.strip()[:200]}"
    try:
        payload = json.loads(text)
    except ValueError:
        return None, f"inspect did not answer in JSON: {text.strip()[:200]}"
    node = _find_node(payload, node_id)
    if node is None:
        return None, f"inspect did not return the node: {text.strip()[:200]}"
    provenance = node.get("provenance")
    source = (provenance.get("source") if isinstance(provenance, Mapping) else None) or node.get("source")
    if not isinstance(source, str) or not source:
        return None, "inspect returned the node without its source"
    attrs = node.get("attrs") if isinstance(node.get("attrs"), Mapping) else {}
    seen = (payload.get("pinned_generation_id") or payload.get("generation_id")) if isinstance(payload, Mapping) else None
    return NodeInfo(node_id, str(node.get("subtype") or ""), source, attrs, None if seen is None else str(seen)), ""


#: Attributes that carry a record's natural key, by subtype, for records whose node id does not
#: (a documentation page's id is a hash of its URL). Every other subtype resolves by its id.
KEY_ATTRS: dict[str, tuple[str, ...]] = {
    "documentation_page": ("url",),
}


def key_candidates(info: NodeInfo) -> list[str]:
    """Names a node may be filed under in keys.json, most specific first: its id, the id's
    tails, then the key attributes its subtype has (`KEY_ATTRS`)."""
    nid = info.node_id
    out = [nid]
    if ":" in nid:
        first = nid.split(":", 1)[1]
        out += [first, first.replace(":", "/"), nid.rsplit(":", 1)[1]]
    for name in KEY_ATTRS.get(info.subtype, ()):
        value = info.attrs.get(name)
        if isinstance(value, str) and value.strip():
            out.append(value.strip())
    return list(dict.fromkeys(out))


def normal_source(name: str) -> str:
    """okg writes a producer as `<deployment>.<source>`, and spells it with `-` or `_`; keys.json
    uses the registry name. Compare them lower-cased, with `-` read as `_`."""
    return name.strip().lower().replace("-", "_")


def source_candidates(source: str) -> list[str]:
    out = [source]
    if "." in source:
        out += [source.split(".", 1)[1], source.rsplit(".", 1)[1]]
    return list(dict.fromkeys(out))


class GoldMap:
    """Every scored question's gold ids, resolved to corpus records or marked unavailable."""

    def __init__(self, records: dict[str, list[GoldRecord]], generations: Sequence[str] = (),
                 live_atoms: Optional[dict[str, list[dict[str, str]]]] = None) -> None:
        self.records = records
        #: the generations inspect read while resolving
        self.generations = list(dict.fromkeys(g for g in generations if g))
        #: per question, the atoms only a live system answers (unavailable in both arms)
        self.live_atoms = live_atoms or {}

    def to_dict(self) -> dict[str, Any]:
        return {qid: {g.node_id: g.to_dict() for g in golds} for qid, golds in self.records.items()}


def _lookup(keys: Mapping[str, Any], source: str, names: Sequence[str]) -> tuple[Optional[str], Optional[str]]:
    table = keys["sources"][source]
    for name in names:
        path = table["lookup"].get(name)
        if path:
            return path, table["natural_key"].get(path, name)
    return None, None


def _left_out(keys: Mapping[str, Any], names: Sequence[str]) -> Optional[str]:
    left = keys.get("left_out") or {}
    by_normal = {normal_source(k): k for k in left}
    for name in names:
        hit = by_normal.get(normal_source(name))
        if hit:
            return f"{hit}: {left[hit]}"
    return None


# --- what a record looks like in a response ---------------------------------------------------

SHINGLE_WORDS = 6
MIN_VALUE_WORDS = 4
MAX_SHINGLES = 20000
#: A record counts as read once the responses show two of its values or lines (or its only one):
#: one line a grep matched, or one excerpt, alone only lists it.
READ_UNITS = 2
#: Fields that name a record rather than hold its content: a search row shows them beside every
#: hit, as `ls` shows a file name, so, like the key, they never count toward reading a record.
TITLE_FIELDS = frozenset({"summary", "title", "subject", "short_description", "topic", "headline", "label", "name",
                          "displayName"})
_WORDS = re.compile(r"[a-z0-9]+")
#: JSON string escapes, as both a `cat` of a corpus file and an MCP JSON reply show them
_UNICODE_ESCAPE = re.compile(r"\\u[0-9a-fA-F]{4}")
_ESCAPE = re.compile(r"\\[nrtbf\"/\\]")


def words(text: str) -> list[str]:
    return _WORDS.findall(_ESCAPE.sub(" ", _UNICODE_ESCAPE.sub(" ", text)).lower())


def _leaves(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        return [s for k, v in value.items() if k not in TITLE_FIELDS for s in _leaves(v)]
    if isinstance(value, list):
        return [s for v in value for s in _leaves(v)]
    if isinstance(value, str):
        return [value]
    return []


def content_fingerprints(data: bytes, *, exclude: Sequence[str] = ()) -> tuple[tuple[str, ...], ...]:
    """The units that identify a record's content, one per string value of a JSON record (field
    names and `TITLE_FIELDS` values never) or per line of a text record, each as the word runs that show it: a value of six
    words or more as its six-word runs, one of four or five words whole. Shorter values carry no
    unit. The record's own key, id and path are excluded: naming a record is listing it, not
    reading it."""
    text = data.decode("utf-8", "replace")
    try:
        parts = _leaves(json.loads(text))  # its title fields left out (`TITLE_FIELDS`)
    except ValueError:
        parts = text.splitlines()
    skip = {" ".join(words(x)) for x in exclude if x}
    units: list[tuple[str, ...]] = []
    total = 0
    for part in parts:
        w = words(part)
        joined = " ".join(w)
        if not w or joined in skip or total >= MAX_SHINGLES:
            continue
        if len(w) >= SHINGLE_WORDS:
            unit = tuple(dict.fromkeys(" ".join(w[i:i + SHINGLE_WORDS]) for i in range(len(w) - SHINGLE_WORDS + 1)))
        elif len(w) >= MIN_VALUE_WORDS:
            unit = (joined,)
        else:
            continue
        units.append(unit[:MAX_SHINGLES - total])
        total += len(units[-1])
    return tuple(dict.fromkeys(units))


def _with_content(record: GoldRecord, records_dir: Optional[Path]) -> GoldRecord:
    if records_dir is None or not record.path or not record.available:
        return record
    try:
        data = (Path(records_dir) / record.path).read_bytes()
    except OSError as exc:
        return GoldRecord(record.node_id, UNAVAILABLE, record.source, record.key, record.path,
                          reason=f"its file cannot be read ({type(exc).__name__})")
    units = content_fingerprints(data, exclude=(record.key or "", record.node_id, record.path))
    if not units:
        return GoldRecord(record.node_id, UNAVAILABLE, record.source, record.key, record.path, reason=NO_CONTENT)
    return GoldRecord(record.node_id, record.status, record.source, record.key, record.path, record.reason, units)


def _runs(w: list[str], sizes: Mapping[str, frozenset]) -> set[str]:
    """The word runs of `w` that start with a word in `sizes` and have one of its lengths."""
    out = set()
    for i, first in enumerate(w):
        for n in sizes.get(first, ()):
            if i + n <= len(w):
                out.add(" ".join(w[i:i + n]))
    return out


def drop_shared(records: Mapping[str, list[GoldRecord]], records_dir: Path,
                manifest: frozenset[str]) -> dict[str, list[GoldRecord]]:
    """Each gold record without the fingerprints another corpus record also holds (a
    `from __future__` line, a TWiki META line, an assignee's name): text many records share would
    credit the gold record whichever of them was read. One pass over the corpus, reading each
    file as `cat` shows it. A record left with no unit is unavailable."""
    wanted = {fp for golds in records.values() for g in golds if g.available for u in g.units for fp in u}
    if not wanted:
        return dict(records)
    lengths: dict[str, set[int]] = {}
    for fp in wanted:
        parts = fp.split(" ")
        lengths.setdefault(parts[0], set()).add(len(parts))
    sizes = {first: frozenset(ns) for first, ns in lengths.items()}
    holders: dict[str, set[str]] = {}
    for rel in sorted(manifest):
        try:
            data = (Path(records_dir) / rel).read_bytes()
        except OSError:
            continue
        for fp in _runs(words(data.decode("utf-8", "replace")), sizes) & wanted:
            seen = holders.setdefault(fp, set())
            if len(seen) < 2:
                seen.add(rel)
    out: dict[str, list[GoldRecord]] = {}
    for qid, golds in records.items():
        kept_golds = []
        for g in golds:
            if not g.available:
                kept_golds.append(g)
                continue
            units, dropped = [], 0
            for unit in g.units:
                kept = tuple(fp for fp in unit if not (holders.get(fp, set()) - {g.path}))
                dropped += len(unit) - len(kept)
                if kept:
                    units.append(kept)
            if units:
                kept_golds.append(GoldRecord(g.node_id, g.status, g.source, g.key, g.path, g.reason,
                                             tuple(units), dropped))
            else:
                kept_golds.append(GoldRecord(g.node_id, UNAVAILABLE, g.source, g.key, g.path, ALL_SHARED,
                                             shared=dropped))
        out[qid] = kept_golds
    return out


def record_for(info: NodeInfo, keys: Mapping[str, Any]) -> "GoldRecord | str":
    """The corpus record of an inspected node, an unavailable record (its source is left out of
    the corpus), or the reason it has none."""
    from archi.evaluation.raw_corpus import OPS_RECORD, OPS_RECORD_PRODUCER

    sources = keys.get("sources") or {}
    by_normal = {normal_source(name): name for name in sources}
    gid = info.node_id
    if info.source == OPS_RECORD_PRODUCER or info.subtype.startswith("ops_"):
        return GoldRecord(gid, UNAVAILABLE, OPS_RECORD, reason=_left_out(keys, [OPS_RECORD]) or "ops-record is left out")
    names = source_candidates(info.source)
    source = next((by_normal[normal_source(n)] for n in names if normal_source(n) in by_normal), None)
    if source is None:
        left = _left_out(keys, names)
        if left:
            return GoldRecord(gid, UNAVAILABLE, info.source, reason=left)
        return (f"its source {info.source} is neither in the corpus nor listed as left out; rebuild the "
                "corpus with `archi eval --raw-corpus`")
    candidates = key_candidates(info)
    path, natural = _lookup(keys, source, candidates)
    if path is None:
        return (f"no record of {source} in the corpus has its key (tried {', '.join(candidates[:4])}); "
                "the corpus and the graph differ, or add gold_sources for it")
    return GoldRecord(gid, RESOLVED, source, natural, path)


async def resolve_gold(tools: Any, questions: Sequence[Any], keys: Mapping[str, Any],
                       manifest: frozenset[str], records_dir: Optional[Path] = None, *,
                       drop_shared_text: bool = True) -> GoldMap:
    """Resolve every gold id, or raise GoldError naming each one that does not resolve. With
    `records_dir`, each record's content fingerprints are read from its file, less those another
    corpus record also holds (`drop_shared`). A caller that holds an MCP session open passes
    `drop_shared_text=False` and runs `drop_shared` itself once the session is closed: the pass
    reads the whole corpus."""
    sources = keys.get("sources") or {}
    by_normal = {normal_source(name): name for name in sources}
    out: dict[str, list[GoldRecord]] = {}
    problems: list[str] = []
    generations: list[str] = []
    live_atoms: dict[str, list[dict[str, str]]] = {}
    for question in questions:
        golds = []
        atoms = tuple(getattr(question, "atoms", ()) or ())
        live = [a for a in atoms if a.live_only]
        if live:
            live_atoms[question.id] = [{"atom": a.id, "answer_source": a.answer_source} for a in live]
        # A gold id that only live-only atoms carry is unavailable; one a file-backed atom also
        # carries ("ticket+live_rucio", "doc or ticket") counts through that file.
        live_gold = {g for a in live for g in a.gold} - {g for a in atoms if not a.live_only for g in a.gold}
        for gid in question.gold:
            if gid in live_gold:
                sources_ = sorted({a.answer_source for a in live if gid in a.gold})
                golds.append(GoldRecord(gid, UNAVAILABLE, reason=f"answered only from a live system "
                                                                 f"({'; '.join(sources_)}); no file holds it"))
                continue
            where = (question.gold_sources or {}).get(gid)
            if where is not None:
                if "path" in where:
                    if where["path"] not in manifest:
                        problems.append(f"{question.id}: {gid}: gold_sources path {where['path']} is not in the corpus")
                        continue
                    natural = next((t["natural_key"].get(where["path"]) for name, t in sources.items()
                                    if where["path"] in t["natural_key"]), None)
                    owner = next((name for name, t in sources.items() if where["path"] in t["natural_key"]), None)
                    golds.append(_with_content(GoldRecord(gid, OVERRIDE, owner, natural, where["path"]), records_dir))
                    continue
                source = by_normal.get(normal_source(where["source"]))
                if source is None:
                    left = _left_out(keys, [where["source"]])
                    if left:
                        golds.append(GoldRecord(gid, UNAVAILABLE, where["source"], where["key"], reason=left))
                    else:
                        problems.append(f"{question.id}: {gid}: gold_sources source {where['source']} is not in the corpus")
                    continue
                path, natural = _lookup(keys, source, [where["key"]])
                if path is None:
                    problems.append(f"{question.id}: {gid}: gold_sources key {where['key']!r} is not a record of "
                                    f"{source} in the corpus")
                    continue
                golds.append(_with_content(GoldRecord(gid, OVERRIDE, source, natural, path), records_dir))
                continue
            info, why = await inspect_node(tools, gid)
            if info is None:
                problems.append(f"{question.id}: {gid}: {why}")
                continue
            if info.generation:
                generations.append(info.generation)
            record = record_for(info, keys)
            if isinstance(record, str):
                problems.append(f"{question.id}: {gid}: {record}")
                continue
            golds.append(_with_content(record, records_dir))
        out[question.id] = golds
    if problems:
        raise GoldError(problems)
    if records_dir is not None and drop_shared_text:
        out = drop_shared(out, records_dir, manifest)
    return GoldMap(out, generations, live_atoms)


# --- scoring ---------------------------------------------------------------------------------

#: A natural key shorter than this does not even list a record: "gpu" or "T2_US_X" would match
#: too much text. No key ever credits a record; only its content does.
MIN_KEY_CHARS = 6
_WORD = r"A-Za-z0-9_"
_PATH_TOKEN = re.compile(r"[\w.%@+=,~-]+(?:/[\w.%@+=,~-]+)+")
#: okg row heads (`[n] <id> (subtype)`) and JSON `"node_id": "<id>"` fields
_ROW_HEAD = re.compile(r"^\[\d+\] (\S+) \(", re.M)
_JSON_ID = re.compile(r'"node_id"\s*:\s*"([^"]+)"')


def _bounded(term: str, *, left: str, right: str) -> re.Pattern:
    return re.compile(rf"(?<![{left}]){re.escape(term)}(?![{right}])")


@dataclass
class Evidence:
    #: gold records read (their content appeared)
    found: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    unavailable: list[str] = field(default_factory=list)
    #: gold records named (path, key or id) without their content: not credited
    listed: list[str] = field(default_factory=list)
    first_rank: Optional[int] = None
    records_read: int = 0

    @property
    def available(self) -> int:
        return len(self.found) + len(self.missed)

    @property
    def recall(self) -> Optional[float]:
        return len(self.found) / self.available if self.available else None

    @property
    def reciprocal_rank(self) -> float:
        return 1.0 / self.first_rank if self.first_rank else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"recall": None if self.recall is None else round(self.recall, 4),
                "reciprocal_rank": round(self.reciprocal_rank, 4), "first_rank": self.first_rank,
                "found": self.found, "missed": self.missed, "listed": self.listed,
                "unavailable": self.unavailable, "records_met": self.records_read}


def _name_patterns(gold: GoldRecord) -> list[re.Pattern]:
    patterns = [_bounded(gold.node_id, left=_WORD + ":", right=_WORD + "-")]
    if gold.path:
        patterns.append(_bounded(gold.path, left=_WORD + ".%-", right=_WORD + "%-"))
    if gold.key and len(gold.key) >= MIN_KEY_CHARS:
        patterns.append(_bounded(gold.key, left=_WORD, right=_WORD))
    return patterns


def _units_in(gold: GoldRecord, norm: str, grams: set) -> set[int]:
    """The indexes of the gold record's units this response shows."""
    return {i for i, unit in enumerate(gold.units)
            if any(fp in grams if fp.count(" ") == SHINGLE_WORDS - 1 else f" {fp} " in norm for fp in unit)}


def score_evidence(events: Sequence[ToolEvent], gold: Sequence[GoldRecord], manifest: frozenset[str]) -> Evidence:
    """Evidence recall, listed records and the first gold record's rank, from the tool responses
    the scorer reads (`ToolEvent.ok`). A record is read once the responses show `READ_UNITS` of its
    values or lines (its only one, if it has one). Tool inputs (`ToolEvent.arguments`) and the final
    answer are never read."""
    evidence = Evidence(unavailable=[g.node_id for g in gold if not g.available])
    targets = [g for g in gold if g.available]
    names = {g.node_id: _name_patterns(g) for g in targets}
    alias = {}
    for g in targets:
        alias[g.node_id] = g.node_id
        if g.path:
            alias[g.path] = g.node_id
    read: set[str] = set()
    shown: dict[str, set[int]] = {g.node_id: set() for g in targets}
    named: set[str] = set()
    order: list[str] = []
    first_read: Optional[int] = None
    for event in events:
        if not event.ok:
            continue
        w = words(event.text)
        norm = " " + " ".join(w) + " "
        grams = {" ".join(w[i:i + SHINGLE_WORDS]) for i in range(len(w) - SHINGLE_WORDS + 1)}
        mentions: list[tuple[int, str]] = []  # (offset, identity) of records met by name
        gold_at: dict[str, int] = {}
        for g in targets:
            for pattern in names[g.node_id]:
                match = pattern.search(event.text)
                if match:
                    named.add(g.node_id)
                    gold_at[g.node_id] = min(gold_at.get(g.node_id, match.start()), match.start())
        for match in _PATH_TOKEN.finditer(event.text):
            token = match.group(0)
            for prefix in ("./", "corpus/"):
                if token.startswith(prefix):
                    token = token[len(prefix):]
            if token in manifest and token not in alias:
                mentions.append((match.start(), "path:" + token))
        for regex in (_ROW_HEAD, _JSON_ID):
            for match in regex.finditer(event.text):
                if match.group(1) not in alias:
                    mentions.append((match.start(1), "id:" + match.group(1)))
        # A gold record is read once the responses so far show `needed` of its units; it is placed
        # where this response first names it, else after everything this response names.
        for g in targets:
            if g.node_id in read:
                continue
            shown[g.node_id] |= _units_in(g, norm, grams)
            if g.units and len(shown[g.node_id]) >= g.needed:
                read.add(g.node_id)
                mentions.append((gold_at.get(g.node_id, len(event.text)), "gold:" + g.node_id))
        for _, identity in sorted(mentions):
            if identity not in order:
                order.append(identity)
            if identity.startswith("gold:") and first_read is None:
                first_read = len([i for i in order if not i.startswith("gold:")]) + 1
    evidence.found = [g.node_id for g in targets if g.node_id in read]
    evidence.missed = [g.node_id for g in targets if g.node_id not in read]
    evidence.listed = [g.node_id for g in targets if g.node_id in named and g.node_id not in read]
    evidence.records_read = len(order)
    evidence.first_rank = first_read
    return evidence


# --- the judge -------------------------------------------------------------------------------

JUDGE_PROMPT = """\
You grade one answer to an operations question against a reference answer an operator wrote.

Score 1 when the answer states what the reference states (wording may differ). Score 0.5 when \
it states part of it, or hedges between the reference and something else. Score 0 when it is \
wrong, says it could not find the answer, or misses the point. Ignore the "Sources:" line and \
the style.

Reply with JSON only: {"score": 0 or 0.5 or 1, "reason": "one sentence"}"""


@dataclass
class Judge:
    model: Metered
    passes: int = 1

    def grade(self, *, arm: str, question_id: str, attempt: int, question: str, key: str, answer: str
              ) -> dict[str, Any]:
        """{"score": mean of the passes that parsed, or None; "passes": [...]}. An empty answer is
        a 0 without a call."""
        if not answer.strip():
            return {"score": 0.0, "passes": [], "note": "no answer"}
        messages = [{"role": "system", "content": JUDGE_PROMPT},
                    {"role": "user", "content": f"Question: {question}\n\nReference answer: {key}\n\n"
                                                f"Answer to grade:\n{answer}"}]
        passes = []
        for _ in range(self.passes):
            try:
                reply = self.model.complete(messages, [], arm=arm, role="judge", question=question_id,
                                            attempt=attempt)
            except FatalModelError:
                raise
            except ModelError as exc:
                passes.append({"score": None, "error": str(exc)[:300]})
                continue
            if reply.finish_reason == "length" and not (reply.content or "").strip():
                passes.append({"score": None, "error": OUTPUT_LIMIT})
                continue
            passes.append(_verdict(reply.content))
        scores = [p["score"] for p in passes if p.get("score") is not None]
        return {"score": sum(scores) / len(scores) if scores else None, "passes": passes}


def _verdict(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text or "", re.S)
    try:
        data = json.loads(match.group(0)) if match else None
    except ValueError:
        data = None
    score = data.get("score") if isinstance(data, Mapping) else None
    if isinstance(score, bool) or score not in (0, 0.5, 1):
        return {"score": None, "error": f"unreadable verdict: {(text or '')[:200]}"}
    return {"score": float(score), "reason": str(data.get("reason") or "")[:300]}


# --- the run ---------------------------------------------------------------------------------


@dataclass
class AgentPlan:
    arms: tuple[str, ...]
    model: Metered
    caps: Caps
    tries: int
    corpus: Any  # raw_corpus.Corpus, verified
    sandbox: Optional[BashSandbox] = None
    judge: Optional[Judge] = None
    judge_requested: bool = False
    reasoning_effort: Optional[str] = None
    model_url: str = ""
    #: "responses" or "chat": the API the agent calls used
    model_api: str = ""


@dataclass
class AgentReport:
    plan: AgentPlan
    gold: GoldMap
    tries: dict[str, dict[str, list[TryResult]]] = field(default_factory=dict)  # question -> arm -> tries
    planned: int = 0
    stopped: Optional[str] = None
    #: why the run was aborted (a model endpoint refusal no retry can fix), or None
    aborted: Optional[str] = None
    started_at: str = ""
    finished_at: str = ""
    questions: list = field(default_factory=list)
    #: the generations the graph arm read during the context run, to compare with the corpus's
    evaluated_generations: list = field(default_factory=list)
    spent_before: float = 0.0
    calls_before: int = 0

    @property
    def answer_keys(self) -> int:
        return sum(1 for q in self.questions if q.answer)

    def paired(self) -> set[tuple[str, int]]:
        """The (question, try) pairs every arm ran to the end without an error: a spend stop, an
        abort or an error try can leave one arm with a try the other lacks, and recall, MRR and the
        answer score compare the arms on these pairs only."""
        out = set()
        for qid, per_arm in self.tries.items():
            for attempt in range(1, self.plan.tries + 1):
                if all(any(t.attempt == attempt and t.status not in (STOPPED, ERROR) for t in per_arm.get(arm, []))
                       for arm in self.plan.arms):
                    out.add((qid, attempt))
        return out

    def summary(self, arm: str) -> dict[str, Any]:
        rows = [t for per_arm in self.tries.values() for t in per_arm.get(arm, [])]
        paired = self.paired()
        unpaired = [t for t in rows if t.scored and (t.question, t.attempt) not in paired]
        scored = [t for t in rows if t.scored and (t.question, t.attempt) in paired]
        recalls = [t.evidence.recall for t in scored]
        rrs = [t.evidence.reciprocal_rank for t in scored]
        completed = [t for t in rows if t.status != STOPPED]
        if not self.answer_keys:
            answer: Any = NO_ANSWER_KEYS
        elif self.plan.judge is None:
            answer = JUDGE_OFF
        else:
            graded = [t.judge["score"] for t in completed if (t.question, t.attempt) in paired
                      and t.judge and t.judge.get("score") is not None]
            answer = round(sum(graded) / len(graded), 4) if graded else None
        listed = [len(t.evidence.listed) for t in scored]
        return {
            "tries": len(rows),
            "tries_scored": len(scored),
            "tries_unpaired": len(unpaired),
            "evidence_recall": round(sum(recalls) / len(recalls), 4) if recalls else None,
            "mrr": round(sum(rrs) / len(rrs), 4) if rrs else None,
            "listed_not_read_per_try": round(sum(listed) / len(listed), 2) if listed else None,
            "answer_score": answer,
            "judge_errors": sum(1 for t in completed if t.judge and t.judge.get("score") is None),
            "cost_usd": round(sum(t.cost for t in rows), 6),
            "cost_per_try_usd": round(sum(t.cost for t in completed) / len(completed), 6) if completed else None,
            "tool_calls_per_try": round(sum(t.tool_calls for t in completed) / len(completed), 2) if completed else None,
            "turns_per_try": round(sum(t.turns for t in completed) / len(completed), 2) if completed else None,
            "failures": sum(t.status == ERROR for t in rows),
            "error_tries": [{"question": t.question, "try": t.attempt, "reason": t.reason}
                            for t in rows if t.status == ERROR],
            "capped": sum(t.status == CAPPED for t in rows),
            "stopped_at_cap": sum(t.status == STOPPED for t in rows),
        }

    def to_dict(self) -> dict[str, Any]:
        plan = self.plan
        ledger = plan.model.ledger
        spent = ledger.spent - self.spent_before
        judge_cost = 0.0
        if plan.judge is not None:
            judge_cost = spent - sum(t.cost for per in self.tries.values() for ts in per.values() for t in ts)
        corpus_generation = getattr(plan.corpus, "generation", None)
        evaluated = list(dict.fromkeys(g for g in [*self.evaluated_generations, *self.gold.generations] if g))
        warning = None
        if corpus_generation is None:
            warning = "the corpus records no published generation; it cannot be checked against the graph"
        elif evaluated and evaluated != [corpus_generation]:
            warning = (f"the corpus was built when generation {corpus_generation} was published, but this run "
                       f"read {', '.join(evaluated)}: the two arms may not see the same records")
        return {
            "arms": {arm: self.summary(arm) for arm in plan.arms},
            "model": plan.model.model.name,
            "model_url": plan.model_url,
            "model_api": plan.model_api,
            "reasoning_effort": plan.reasoning_effort,
            "price_usd_per_million": vars(plan.model.price),
            "caps": {"max_turns": plan.caps.max_turns, "max_tool_calls": plan.caps.max_tool_calls,
                     "try_seconds": plan.caps.try_seconds, "min_call_seconds": plan.caps.min_call_seconds,
                     "command_seconds": plan.caps.command_seconds, "output_bytes": plan.caps.output_bytes,
                     "max_output_tokens": plan.model.max_output_tokens, "max_procs": plan.caps.max_procs,
                     "memory_bytes": plan.caps.memory_bytes, "file_bytes": plan.caps.file_bytes},
            "tries_per_question": plan.tries,
            "prompt": {arm: build_prompt(arm, plan.caps, folders=_folders(plan.corpus),
                                         tools=plan.sandbox.tools if plan.sandbox else ()) for arm in plan.arms},
            "sandbox": None if plan.sandbox is None else {"bwrap": plan.sandbox.bwrap, "tools": plan.sandbox.tools,
                                                         "missing_tools": [t for t in TOOLS if t not in plan.sandbox.tools],
                                                         "limit_scope": plan.sandbox.limit_scope,
                                                         "limit_scope_note": plan.sandbox.scope_note or None},
            "corpus": {"path": str(plan.corpus.path), "digest": plan.corpus.digest,
                       "generation": corpus_generation, "evaluated_generations": evaluated,
                       "generation_warning": warning,
                       "lock_sha256": plan.corpus.lock_sha256, "built_at": plan.corpus.built_at,
                       "coverage": getattr(plan.corpus, "coverage", ""),
                       "sources": sorted(plan.corpus.sources), "left_out": dict(plan.corpus.left_out)},
            "ops_record": "left out of both agent arms: " + str(plan.corpus.left_out.get("ops-record", "")),
            "judge": {"requested": plan.judge_requested, "enabled": plan.judge is not None,
                      "model": plan.judge.model.model.name if plan.judge else None,
                      "passes": plan.judge.passes if plan.judge else None,
                      "max_output_tokens": plan.judge.model.max_output_tokens if plan.judge else None,
                      "answer_keys": self.answer_keys, "cost_usd": round(judge_cost, 6)},
            "spend": {"cap_usd": ledger.cap, "spent_usd": round(spent, 6),
                      "spent_this_invocation_usd": round(ledger.spent, 6),
                      "calls": ledger.calls - self.calls_before,
                      "stopped_at_cap": self.stopped is not None, "stop_reason": self.stopped,
                      "warnings": list(ledger.warnings), "ledger": LEDGER},
            "aborted": self.aborted,
            "live_only_atoms": self.gold.live_atoms,
            "tries_planned": self.planned,
            "tries_run": sum(len(ts) for per in self.tries.values() for ts in per.values()),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "gold": self.gold.to_dict(),
            "gold_unavailable_by_reason": _count_reasons(self.gold),
            "questions": [{"id": qid, "arms": {arm: [t.to_dict() for t in ts] for arm, ts in per.items()}}
                          for qid, per in self.tries.items()],
        }


def _count_reasons(gold: GoldMap) -> dict[str, int]:
    """How many gold records each reason made unavailable (a record with nothing but a title, one
    whose text other records share, one from a left-out source), so a shrinking denominator shows."""
    counts: dict[str, int] = {}
    for golds in gold.records.values():
        for g in golds:
            if not g.available:
                counts[g.reason] = counts.get(g.reason, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def _folders(corpus: Any) -> list[str]:
    try:
        return sorted(p.name for p in corpus.records.iterdir() if p.is_dir())
    except OSError:
        return []


async def run_agent_arms(questions: Sequence[Any], plan: AgentPlan, mcp: Any, gold: GoldMap, *,
                         manifest: frozenset[str], evaluated_generations: Sequence[str] = (),
                         progress: Optional[Callable[[TryResult, int, int], None]] = None) -> AgentReport:
    """Every scored question, `plan.tries` times, in each arm. The arms take turns per (question,
    try), and which arm goes first alternates, so neither arm is always the one a spend stop cuts
    short; the summary scores only the pairs every arm finished (`AgentReport.paired`). A
    FatalModelError aborts the run; the report says so and keeps what completed."""
    ledger = plan.model.ledger
    report = AgentReport(plan, gold, questions=list(questions), started_at=_now(),
                         evaluated_generations=list(evaluated_generations), spent_before=ledger.spent,
                         calls_before=ledger.calls)
    folders = _folders(plan.corpus)
    systems = {arm: build_prompt(arm, plan.caps, folders=folders,
                                 tools=plan.sandbox.tools if plan.sandbox else ()) for arm in plan.arms}
    boxes: dict[str, Any] = {}
    specs: dict[str, list[ToolSpec]] = {}
    for arm in plan.arms:
        if arm == AGENT_BASH:
            if plan.sandbox is None:
                raise SandboxError("the agent-bash arm needs its sandbox")
            boxes[arm] = BashTools(plan.sandbox)
        else:
            boxes[arm] = mcp
        specs[arm] = await boxes[arm].list_tools()
    todo = [q for q in questions if any(g.available for g in gold.records.get(q.id, []))]
    report.planned = len(todo) * plan.tries * len(plan.arms)
    done = 0
    turn = 0
    try:
        for question in todo:
            per_arm = report.tries.setdefault(question.id, {})
            for attempt in range(1, plan.tries + 1):
                first = turn % len(plan.arms)
                turn += 1
                for arm in plan.arms[first:] + plan.arms[:first]:
                    if ledger.stopped:
                        report.stopped = ledger.stopped
                        break
                    result = await run_try(plan.model, boxes[arm], specs[arm], arm=arm, system=systems[arm],
                                           question_id=question.id, question=question.question, attempt=attempt,
                                           caps=plan.caps)
                    per_arm.setdefault(arm, []).append(result)
                    result.evidence = score_evidence(result.events, gold.records[question.id], manifest)
                    if plan.judge is not None and question.answer and result.status not in (STOPPED, ERROR):
                        try:
                            result.judge = await asyncio.to_thread(
                                plan.judge.grade, arm=arm, question_id=question.id, attempt=attempt,
                                question=question.question, key=question.answer, answer=result.answer)
                        except CapReached as exc:
                            result.judge = {"score": None, "error": f"stopped at the spend cap: {exc}"}
                    done += 1
                    if progress:
                        progress(result, done, report.planned)
                if report.stopped:
                    break
            if report.stopped:
                break
    except FatalModelError as exc:
        report.aborted = (f"the model endpoint refused a request in a way a retry cannot fix (HTTP {exc.status}): "
                          f"{str(exc)[:300]}")
    except SandboxBroken as exc:
        report.aborted = f"the bash sandbox stopped working during the run: {exc}"
    if ledger.stopped and report.stopped is None:
        report.stopped = ledger.stopped
    report.finished_at = _now()
    return report


def one_call_bounds(plan: AgentPlan, questions: Sequence[Any], *, context_tokens: int = 128_000) -> dict[str, float]:
    """What one call can cost: the first call of a try (the system prompt and the longest question)
    and a call with `context_tokens` of context; the judge's when there is one. Printed before
    the first call so a wrong price shows before money is spent."""
    longest = max((q.question for q in questions), key=len, default="")
    out = {}
    for arm in plan.arms:
        system = build_prompt(arm, plan.caps, folders=_folders(plan.corpus),
                              tools=plan.sandbox.tools if plan.sandbox else ())
        schema = openai_tools([BASH_SPEC]) if arm == AGENT_BASH else []
        out[f"{arm} first call"] = plan.model.bound_for(
            [{"role": "system", "content": system}, {"role": "user", "content": longest}], schema)
    out[f"agent call with {context_tokens // 1000}k tokens of context"] = plan.model.price.bound(
        context_tokens, plan.model.max_output_tokens)
    if plan.judge is not None:
        out["judge call"] = plan.judge.model.price.bound(8000, plan.judge.model.max_output_tokens)
    return out


# --- the report section ----------------------------------------------------------------------


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "none"
    if isinstance(value, str):
        return value
    return f"{value:.{digits}f}"


def render_agent_markdown(agents: Mapping[str, Any], data: Mapping[str, Any]) -> list[str]:
    """The agent arms beside the no-model arms, for report.md."""
    k = data.get("k")
    lines = ["## All arms side by side", "",
             "| arm | what it is | evidence recall | MRR | listed, not read | answer score | cost per try "
             "| tool calls per try | failures |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, s in (data.get("arms") or {}).items():
        lines.append(f"| {name} | search call, no model | {_fmt(s['recall_at_k'])} (recall@{k}) | {_fmt(s['mrr'])} "
                     f"| n/a | n/a | $0 | 1 | {s['errors']} failed calls |")
    what = {AGENT_BASH: "agent, bash over the raw files", AGENT_GRAPH: "agent, the instance's MCP tools"}
    for name, s in agents["arms"].items():
        cost = "none" if s["cost_per_try_usd"] is None else f"${s['cost_per_try_usd']:.4f}"
        failures = f"{s['failures']} errors, {s['capped']} capped" + (
            f", {s['stopped_at_cap']} stopped at the spend cap" if s["stopped_at_cap"] else "") + (
            f", {s['tries_unpaired']} unpaired (not scored)" if s.get("tries_unpaired") else "")
        lines.append(f"| {name} | {what.get(name, 'agent')} | {_fmt(s['evidence_recall'])} | {_fmt(s['mrr'])} "
                     f"| {_fmt(s['listed_not_read_per_try'], 2)} per try "
                     f"| {_fmt(s['answer_score'])} | {cost} | {_fmt(s['tool_calls_per_try'], 1)} | {failures} |")
    spend = agents["spend"]
    corpus = agents["corpus"]
    caps = agents["caps"]
    lines += [
        "",
        "Evidence recall and MRR count a gold record only when its content appears in a tool response; "
        "naming it (path, key or id) lists it, uncredited. Error tries are left out of both and listed below.",
        "",
        f"- agent arms: model {agents['model']} (reasoning {agents['reasoning_effort'] or 'default'}"
        + (f", {agents['model_api']} API" if agents.get("model_api") else "") + "), "
        f"{agents['tries_per_question']} tries per question; caps {caps['max_turns']} turns, "
        f"{caps['max_tool_calls']} tool calls, {caps['try_seconds']:g} s per try; bash commands "
        f"{caps['command_seconds']:g} s and {caps['output_bytes'] // 1024} KiB",
        f"- corpus: `{corpus['digest']}` at {corpus['path']} (snapshot lock `{corpus['lock_sha256'][:16]}`), "
        f"built at generation {corpus['generation'] or 'unknown'}",
        f"- coverage: {corpus['coverage'] or ', '.join(corpus['sources'])}",
        f"- left out of the corpus: " + "; ".join(f"{k}: {v}" for k, v in corpus["left_out"].items()),
        f"- {agents['ops_record']}",
        f"- spend: ${spend['spent_usd']:.4f} over {spend['calls']} model calls (ledger `{spend['ledger']}`); "
        f"${spend['spent_this_invocation_usd']:.4f} of the ${spend['cap_usd']:.2f} cap this invocation"
        + (f"; **stopped at the cap**: {spend['stop_reason']}; {agents['tries_run']} of "
           f"{agents['tries_planned']} tries ran" if spend["stopped_at_cap"] else ""),
        f"- judge: " + (NO_ANSWER_KEYS if not agents["judge"]["answer_keys"] else
                        (f"{agents['judge']['model']}, {agents['judge']['passes']} pass(es), "
                         f"{agents['judge']['answer_keys']} answer keys" if agents["judge"]["enabled"] else JUDGE_OFF)),
    ]
    if corpus.get("generation_warning"):
        lines.append(f"- **warning**: {corpus['generation_warning']}")
    if agents.get("aborted"):
        lines.append(f"- **aborted**: {agents['aborted']}")
    for warning in spend.get("warnings") or []:
        lines.append(f"- spend warning: {warning}")
    if agents.get("sandbox") and agents["sandbox"]["missing_tools"]:
        lines.append(f"- not installed on this host, so not in the sandbox: {', '.join(agents['sandbox']['missing_tools'])}")
    if agents.get("sandbox") and agents["sandbox"].get("limit_scope") == SCOPE_PROCESS:
        lines.append("- sandbox memory and process limits held per process, not per command: "
                     f"{agents['sandbox']['limit_scope_note']}")
    live = agents.get("live_only_atoms") or {}
    if live:
        lines.append("- atoms only a live system answers (left out of evidence recall in both arms): " + "; ".join(
            f"{qid} {a['atom']} ({a['answer_source']})" for qid, atoms in live.items() for a in atoms))
    unavailable = sorted({gid for per in agents["gold"].values() for gid, g in per.items() if g["status"] == UNAVAILABLE})
    if unavailable:
        lines.append(f"- gold records unavailable to the agent arms (not counted as misses): {', '.join(unavailable)}")
        for reason, count in (agents.get("gold_unavailable_by_reason") or {}).items():
            lines.append(f"  - {count}: {reason}")
    for name, s in agents["arms"].items():
        for row in s["error_tries"]:
            lines.append(f"- error try, {name}: {row['question']} try {row['try']}: {row['reason']}")
    return lines
