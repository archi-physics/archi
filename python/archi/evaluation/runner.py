"""Run a smoke set and write report.md and report.json."""
from __future__ import annotations

import errno
import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from archi.evaluation.agent import (
    DEFAULT_MAX_STEPS,
    DEFAULT_MAX_TOOL_CHARS,
    ChatModel,
    ToolBox,
    ToolError,
    ToolSpec,
    _describe,
    ask,
    build_system_prompt,
    prompt_hash,
    sha,
)
from archi.evaluation.smoke import NodeInfo, SmokeSet, check_answer, merge_nodes, nodes_in, unresolved

REPORT_SCHEMA = "archi-eval-report/1"
PASS, FAIL, EXECUTION_FAILED = "pass", "fail", "execution_failed"
#: How many dated runs per bundle `archi eval` keeps in its default directory (`--keep`).
DEFAULT_KEEP = 20
#: The pointer file `publish_report` writes beside the dated runs.
LATEST_FILE = "latest.json"
LATEST_SCHEMA = "archi-eval-latest/1"
#: A run directory `publish_report` made: the run's UTC start, compact, with `-2`, `-3`, ... on a collision.
RUN_DIR_RE = re.compile(r"^(\d{8}T\d{6}Z)(?:-(\d+))?$")
#: the read tool the runner uses to learn a cited node's subtype and source, and how
#: many cited nodes per question it looks up
INSPECT = "inspect"
MAX_LOOKUPS = 5

STOP_GAP_NOTE = (
    "This is the $0 smoke harness (PACT r3-bundle-evals). The eval framework tracked in "
    "#619 replaces it; the question files and this report format are kept so they can move across."
)


@dataclass
class QuestionResult:
    id: str
    question: str
    status: str
    reasons: list[str] = field(default_factory=list)
    cited: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    answer: str = ""
    error: Optional[str] = None
    #: cited nodes the runner inspected itself (no model) to learn their subtype or
    #: source: {"id", "ok"} and, when the lookup failed, "error"
    lookups: list[dict] = field(default_factory=list)


@dataclass
class Report:
    bundle: str
    model: str
    model_url: str
    mcp_url: str
    pass_threshold: int
    smoke_file: str
    tools: list[str] = field(default_factory=list)
    prompt_hash: str = ""
    #: what the model was given besides the question, each as a hash (None: not given)
    mcp_instructions_hash: Optional[str] = None
    chat_prompt_hash: Optional[str] = None
    chat_prompt_file: Optional[str] = None
    system_prompt_hash: Optional[str] = None
    questions: list[QuestionResult] = field(default_factory=list)
    started_at: str = ""
    seconds: float = 0.0
    error: Optional[str] = None

    @property
    def passed(self) -> int:
        return sum(q.status == PASS for q in self.questions)

    @property
    def status(self) -> str:
        """`execution_failed` when nothing could run, else `pass` or `fail` against the threshold."""
        if self.error or not self.questions or all(q.status == EXECUTION_FAILED for q in self.questions):
            return EXECUTION_FAILED
        return PASS if self.passed >= self.pass_threshold else FAIL

    @property
    def exit_code(self) -> int:
        return {PASS: 0, FAIL: 1, EXECUTION_FAILED: 2}[self.status]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": REPORT_SCHEMA,
            "note": STOP_GAP_NOTE,
            "bundle": self.bundle,
            "status": self.status,
            "passed": self.passed,
            "total": len(self.questions),
            "pass_threshold": self.pass_threshold,
            "model": self.model,
            "model_url": self.model_url,
            "mcp_url": self.mcp_url,
            "tools": self.tools,
            "prompt_hash": self.prompt_hash,
            "mcp_instructions_hash": self.mcp_instructions_hash,
            "chat_prompt_hash": self.chat_prompt_hash,
            "chat_prompt_file": self.chat_prompt_file,
            "system_prompt_hash": self.system_prompt_hash,
            "smoke_file": self.smoke_file,
            "started_at": self.started_at,
            "seconds": round(self.seconds, 1),
            "prompt_tokens": sum(q.prompt_tokens for q in self.questions),
            "completion_tokens": sum(q.completion_tokens for q in self.questions),
            "error": self.error,
            "questions": [
                {
                    "id": q.id,
                    "question": q.question,
                    "status": q.status,
                    "reasons": q.reasons,
                    "cited": q.cited,
                    "lookups": q.lookups,
                    "tool_calls": q.tool_calls,
                    "prompt_tokens": q.prompt_tokens,
                    "completion_tokens": q.completion_tokens,
                    "seconds": round(q.seconds, 1),
                    "answer": q.answer,
                    "error": q.error,
                }
                for q in self.questions
            ],
        }


ToolBoxFactory = Callable[[], Any]  # an async context manager yielding a ToolBox


async def run_smoke(
    smoke: SmokeSet,
    model: ChatModel,
    open_tools: ToolBoxFactory,
    *,
    model_url: str = "",
    mcp_url: str = "",
    chat_prompt: Optional[str] = None,
    chat_prompt_file: str = "",
    max_steps: int = DEFAULT_MAX_STEPS,
    max_tool_chars: int = DEFAULT_MAX_TOOL_CHARS,
    progress: Optional[Callable[[QuestionResult, int, int], None]] = None,
) -> Report:
    report = Report(
        bundle=smoke.bundle,
        model=model.name,
        model_url=model_url,
        mcp_url=mcp_url,
        pass_threshold=smoke.pass_threshold,
        smoke_file=str(smoke.path or ""),
        started_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    started = time.monotonic()
    try:
        async with open_tools() as tools:
            specs: list[ToolSpec] = await tools.list_tools()
            report.tools = sorted(s.name for s in specs)
            instructions = getattr(tools, "instructions", None)
            system = build_system_prompt(instructions, chat_prompt)
            report.prompt_hash = prompt_hash(specs, system)
            report.mcp_instructions_hash = sha(instructions)
            report.chat_prompt_hash = sha(chat_prompt)
            report.chat_prompt_file = chat_prompt_file or None
            report.system_prompt_hash = sha(system)
            if not specs:
                report.error = "the MCP endpoint offers no tools, so the agent has nothing to read with"
                report.questions = [
                    QuestionResult(q.id, q.question, EXECUTION_FAILED, error=report.error)
                    for q in smoke.questions
                ]
            else:
                for i, question in enumerate(smoke.questions, 1):
                    result = await _one(question, model, tools, specs, system, max_steps, max_tool_chars)
                    report.questions.append(result)
                    if progress:
                        progress(result, i, len(smoke.questions))
    except ToolError as exc:
        _abandon(report, smoke, f"MCP endpoint not usable: {exc}")
    except Exception as exc:  # noqa: BLE001 - any other failure is a run that could not execute
        _abandon(report, smoke, f"the run stopped: {_describe(exc)}")
    report.seconds = time.monotonic() - started
    return report


def _abandon(report: Report, smoke: SmokeSet, error: str) -> None:
    """Record why the run stopped and mark every question not yet answered as not executed,
    so the report is still written and the exit code is 2, not 1."""
    report.error = error
    done = {q.id for q in report.questions}
    report.questions += [
        QuestionResult(q.id, q.question, EXECUTION_FAILED, error=error) for q in smoke.questions if q.id not in done
    ]


async def _one(question, model, tools: ToolBox, specs, system, max_steps, max_tool_chars) -> QuestionResult:
    trace = await ask(model, tools, specs, question.question, system=system, max_steps=max_steps,
                      max_tool_chars=max_tool_chars)
    result = QuestionResult(
        id=question.id,
        question=question.question,
        status=FAIL,
        tool_calls=[
            {"name": t.name, "arguments": dict(t.arguments), "chars": t.chars, "is_error": t.is_error}
            for t in trace.tool_calls
        ],
        prompt_tokens=trace.prompt_tokens,
        completion_tokens=trace.completion_tokens,
        seconds=trace.seconds,
        answer=trace.answer,
        error=trace.error,
    )
    if trace.error:
        result.status = EXECUTION_FAILED
        return result
    if not trace.finished:
        result.reasons = ["no answer: the step limit ran out while the model was still calling tools"]
        return result
    seen: dict[str, NodeInfo] = {}
    for text in trace.results:
        merge_nodes(seen, nodes_in(text))
    if any(spec.name == INSPECT for spec in specs):
        for nid in unresolved(question, trace.answer, seen)[:MAX_LOOKUPS]:
            # Only what the lookup says about this node counts: the ids it mentions
            # in passing were never returned to the model.
            try:
                text, is_error = await tools.call(INSPECT, {"target": {"kind": "node", "node_id": nid}})
            except Exception as exc:  # the node stays as the model's tools left it; the report says why
                result.lookups.append({"id": nid, "ok": False, "error": _describe(exc)})
                continue
            info = None if is_error else nodes_in(text).get(nid)
            if info is None:
                result.lookups.append({"id": nid, "ok": False,
                                       "error": text[:200] if is_error else "the reply did not describe the node"})
                continue
            merge_nodes(seen, {nid: info})
            result.lookups.append({"id": nid, "ok": True})
    verdict = check_answer(question, trace.answer, seen)
    result.cited = [{"id": n.id, "subtype": n.subtype, "source": n.source} for n in verdict.cited]
    result.reasons = verdict.reasons
    result.status = PASS if verdict.passed else FAIL
    return result


# --- writing ---------------------------------------------------------------------------


def render_markdown(report: Report) -> str:
    data = report.to_dict()
    lines = [
        f"# archi eval: {report.bundle}",
        "",
        f"**{report.status}**: {report.passed} of {len(report.questions)} questions passed "
        f"(threshold {report.pass_threshold}).",
        "",
        f"> {STOP_GAP_NOTE}",
        "",
        f"- model: `{report.model}` at {report.model_url}",
        f"- MCP: {report.mcp_url}; tools: {', '.join(report.tools) or 'none'}",
        f"- prompt hash: `{report.prompt_hash or 'none'}`; system message `{report.system_prompt_hash}`",
        f"- MCP server instructions: `{report.mcp_instructions_hash}`; chat prompt: "
        f"`{report.chat_prompt_hash}` ({report.chat_prompt_file or 'none'})",
        f"- started {report.started_at}; wall time {data['seconds']} s; "
        f"tokens {data['prompt_tokens']} in / {data['completion_tokens']} out",
        f"- smoke file: {report.smoke_file}",
    ]
    if report.error:
        lines += ["", f"**Error:** {report.error}"]
    lines += ["", "| question | result | tool calls | cited | tokens in/out | seconds |", "|---|---|---|---|---|---|"]
    for q in report.questions:
        cited = ", ".join(f"`{c['id']}`" for c in q.cited[:3]) + (" ..." if len(q.cited) > 3 else "")
        lines.append(
            f"| {q.id} | {q.status} | {len(q.tool_calls)} | {cited or '-'} | "
            f"{q.prompt_tokens}/{q.completion_tokens} | {q.seconds:.1f} |"
        )
    for q in report.questions:
        lines += ["", f"## {q.id}: {q.status}", "", f"Q: {q.question}", ""]
        if q.reasons:
            lines += [f"- {r}" for r in q.reasons] + [""]
        if q.error:
            lines += [f"Error: {q.error}", ""]
        if q.tool_calls:
            lines.append("Tool calls: " + "; ".join(f"`{t['name']}({json.dumps(t['arguments'])})`" for t in q.tool_calls))
            lines.append("")
        if q.answer:
            lines += ["Answer:", "", *[f"> {ln}" for ln in q.answer.strip().splitlines()]]
    return "\n".join(lines).rstrip() + "\n"


def write_report(report: Report, out: Path) -> tuple[Path, Path]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    md, js = out / "report.md", out / "report.json"
    _write_atomic(md, render_markdown(report))
    _write_atomic(js, json.dumps(report.to_dict(), indent=2) + "\n")
    return md, js


def publish_report(report: Report, root: Path, *, keep: int = DEFAULT_KEEP) -> tuple[Path, Path, list[Path]]:
    """Write one run to `<root>/<YYYYMMDDTHHMMSSZ>/` and keep the newest `keep` runs (0 keeps all).

    The default output of `archi eval`, so every run stays and instance-health can compare it with the
    earlier ones (#854). The stamp is the run's `started_at`; a second run in the same second gets `-2`,
    `-3`, ... Both reports go into a hidden `<root>/.<stamp>.tmp-<pid>` first and the directory is
    renamed into place only when they are complete; readers skip names that start with a dot.
    `<root>/latest.json` then names the run written last (with overlapping runs, not necessarily the one
    with the latest stamp). instance-health counts two runs with the same bundle, model and start second
    once, so a `-2` run stays on disk but does not add to the trend. Pruning deletes only directories whose names match
    `RUN_DIR_RE`, oldest first, never the run just written; anything else under `root` (a `report.json`
    an older archi wrote straight into it, a folder a person made) is left alone.

    Returns (report.md, report.json, the run directories it deleted).
    """
    if keep < 0:
        raise ValueError("keep must be 0 (keep all) or more")
    stamp = (report.started_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    stamp = stamp.replace("-", "").replace(":", "")
    if not RUN_DIR_RE.match(stamp):
        raise ValueError(f"started_at {report.started_at!r} is not a UTC time like 2026-10-30T12:00:00Z")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / f".{stamp}.tmp-{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    try:
        write_report(report, tmp)
        n = 1
        while True:
            target = root / (stamp if n == 1 else f"{stamp}-{n}")
            if not target.exists():
                try:
                    os.rename(tmp, target)
                    break
                except OSError as exc:
                    if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                        raise
            n += 1
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    _write_atomic(root / LATEST_FILE, json.dumps({
        "schema": LATEST_SCHEMA, "run": target.name, "report": f"{target.name}/report.json",
        "started_at": report.started_at, "status": report.status,
    }, indent=2) + "\n")
    deleted: list[Path] = []
    if keep:
        runs = sorted((p for p in root.iterdir()
                       if RUN_DIR_RE.match(p.name) and p.is_dir() and not p.is_symlink()), key=_run_order)
        for old in [p for p in runs if p != target][:max(0, len(runs) - keep)]:
            # another run of the same bundle may be pruning the same directory at the same time
            shutil.rmtree(old, ignore_errors=True)
            if not old.exists():
                deleted.append(old)
    return target / "report.md", target / "report.json", deleted


def _run_order(path: Path) -> tuple[str, int]:
    m = RUN_DIR_RE.match(path.name)
    return m.group(1), int(m.group(2) or 1)


def _write_atomic(path: Path, text: str) -> None:
    """Write next to `path`, then rename: a reader (instance-health) never sees half a report."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
