"""Context evals: retrieval scored on its own (PACT r3-context-quality, ADR 0002 section 6.1).

`archi eval --context` asks each question of a set straight to the instance's MCP
search tool, with no model in between, and checks whether the gold evidence ids
come back in the top k. Each question runs in two arms:

    graph     the search an operator's agent makes: the tool with the server's defaults
    baseline  the raw-document baseline: the same tool asked to rank each text unit on
              its own words (`rank_by: self`), with no roll-up to records and no concept
              expansion, so it reads like keyword search over the raw document text

A hit counts for a gold id when its row's id is that id, or when the row is a passage
whose record (`in <id>` on the row line) is that id, so a baseline that returns chunks
is scored against record-level gold the same way. Ids in a row's quoted title or
excerpt never count.

The question file keeps the smoke format's shape and adds `gold`:

    bundle: cern-team
    k: 10                  # optional; the cut-off for recall@k (default 10)
    min_recall: 0.6        # optional; exit 1 when the graph arm's recall@k is below it
    corpus: what the questions are answerable from (a note for readers)
    arms:                  # optional; each overrides the default arm of that name
      baseline: {tool: search, arguments: {method: lexical, where: {type: document_chunk}}}
    questions:
      - id: fts3-what-is
        question: What is FTS3?
        gold: [documentation_page:22, ...]   # the node ids that answer it

A smoke file with `gold` added to its questions is accepted: `pass_threshold`,
`must_cite`, `must_contain` and `must_contain_any` are read and ignored. A question
without `gold` is listed and not scored.

The score the context gate reads (`min_context_eval_score`) is the graph arm's
recall@k. `record_score` writes it under the instance as
`eval/reports/context-<bundle>-<time>.json`, the file `archi.ops.context_gate` and the
instance-health monitor read.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import yaml

from archi.evaluation.agent import ToolError, ToolSpec, _describe
from archi.ops.context_gate import CONTEXT_EVAL_DIR, CONTEXT_EVAL_SCHEMA

CONTEXT_FILE = "context.yaml"
REPORT_SCHEMA = "archi-context-eval-report/1"
DEFAULT_K = 10
GRAPH, BASELINE = "graph", "baseline"
PASS, FAIL, EXECUTION_FAILED = "pass", "fail", "execution_failed"

#: The arms every run has, and what each sends besides `query` and `limit`.
DEFAULT_ARMS: dict[str, dict[str, Any]] = {
    GRAPH: {"tool": "search", "arguments": {}},
    BASELINE: {"tool": "search", "arguments": {"method": "lexical", "rank_by": "self", "expand_concepts": False}},
}
#: Arguments the harness sets itself; an arm may not.
RESERVED_ARGS = ("query", "limit")
SMOKE_KEYS = {"pass_threshold"}
SMOKE_QUESTION_KEYS = {"must_cite", "must_contain", "must_contain_any"}

NOTE = (
    "Context eval (PACT r3-context-quality): the instance's MCP search tool is called directly, "
    "with no model, and each arm is scored on whether the gold evidence ids come back in the top k."
)


class ContextSetError(ValueError):
    """The context question file is not usable."""


@dataclass(frozen=True)
class Arm:
    name: str
    tool: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True)
class ContextQuestion:
    id: str
    question: str
    gold: tuple[str, ...] = ()


@dataclass(frozen=True)
class ContextSet:
    bundle: str
    k: int
    questions: tuple[ContextQuestion, ...]
    arms: tuple[Arm, ...]
    min_recall: Optional[float] = None
    corpus: str = ""
    path: Optional[Path] = None

    @property
    def scored(self) -> tuple[ContextQuestion, ...]:
        return tuple(q for q in self.questions if q.gold)


def _strings(where: str, raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or not all(isinstance(x, str) and x.strip() for x in raw):
        raise ContextSetError(f"{where} must be a list of non-empty strings")
    return tuple(dict.fromkeys(x.strip() for x in raw))


def _arms(where: str, raw: Any) -> tuple[Arm, ...]:
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ContextSetError(f"{where}: `arms` must be a mapping of arm name to {{tool, arguments}}")
    merged: dict[str, Mapping[str, Any]] = {**DEFAULT_ARMS, **raw}
    arms = []
    for name, spec in merged.items():
        label = f"{where}: arms.{name}"
        if not isinstance(name, str) or not name:
            raise ContextSetError(f"{where}: arm names must be non-empty strings")
        if not isinstance(spec, Mapping) or set(spec) - {"tool", "arguments"}:
            raise ContextSetError(f"{label} takes only `tool` and `arguments`")
        tool = spec.get("tool", "search")
        args = spec.get("arguments") or {}
        if not isinstance(tool, str) or not tool:
            raise ContextSetError(f"{label}.tool must be a tool name")
        if not isinstance(args, Mapping):
            raise ContextSetError(f"{label}.arguments must be a mapping")
        reserved = sorted(set(args) & set(RESERVED_ARGS))
        if reserved:
            raise ContextSetError(f"{label}.arguments may not set {reserved}; the harness sends the question and k")
        arms.append(Arm(name, tool, dict(args)))
    return tuple(arms)


def parse_context(raw: Any, path: Optional[Path] = None) -> ContextSet:
    where = str(path) if path else "context set"
    if not isinstance(raw, Mapping):
        raise ContextSetError(f"{where}: expected a mapping")
    unknown = set(raw) - {"bundle", "k", "min_recall", "corpus", "arms", "questions"} - SMOKE_KEYS
    if unknown:
        raise ContextSetError(f"{where}: unknown keys {sorted(unknown)}")
    bundle = raw.get("bundle")
    if not isinstance(bundle, str) or not bundle:
        raise ContextSetError(f"{where}: `bundle` is required")
    k = raw.get("k", DEFAULT_K)
    if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= 100:
        raise ContextSetError(f"{where}: `k` must be an integer from 1 to 100")
    min_recall = raw.get("min_recall")
    if min_recall is not None:
        if isinstance(min_recall, bool) or not isinstance(min_recall, (int, float)) or not 0 <= min_recall <= 1:
            raise ContextSetError(f"{where}: `min_recall` must be a number from 0 to 1")
        min_recall = float(min_recall)
    rows = raw.get("questions")
    if not isinstance(rows, list) or not rows:
        raise ContextSetError(f"{where}: `questions` must be a non-empty list")
    questions: list[ContextQuestion] = []
    seen: set[str] = set()
    for i, row in enumerate(rows):
        label = f"{where}: questions[{i}]"
        if not isinstance(row, Mapping):
            raise ContextSetError(f"{label} must be a mapping")
        extra = set(row) - {"id", "question", "gold"} - SMOKE_QUESTION_KEYS
        if extra:
            raise ContextSetError(f"{label}: unknown keys {sorted(extra)}")
        qid, text = row.get("id"), row.get("question")
        if not isinstance(qid, str) or not qid or not isinstance(text, str) or not text.strip():
            raise ContextSetError(f"{label}: `id` and `question` are required strings")
        if qid in seen:
            raise ContextSetError(f"{label}: duplicate id {qid!r}")
        seen.add(qid)
        questions.append(ContextQuestion(qid, text.strip(), _strings(f"{label}: gold", row.get("gold"))))
    if not any(q.gold for q in questions):
        raise ContextSetError(f"{where}: no question has `gold` ids, so nothing can be scored")
    return ContextSet(bundle, k, tuple(questions), _arms(where, raw.get("arms")), min_recall,
                      str(raw.get("corpus") or ""), path)


def load_context(path: Path) -> ContextSet:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ContextSetError(f"{path}: {exc}") from exc
    return parse_context(raw, path)


def context_path(bundle_dir: Path) -> Path:
    return Path(bundle_dir) / "eval" / CONTEXT_FILE


# --- reading the ranking out of a search result ---------------------------------------

#: `[3] chunk:ab12 (document_chunk) · ... · in documentation_page:22 · ...`, okg's row line
ROW_HEAD = re.compile(r"^[ \t]*\[(\d+)\][ \t]+(\S+)[ \t]+\(", re.M)
#: the record a listed passage belongs to: the ` · in <id>` fact of its row line
PARENT = re.compile(r"\s·\s+in\s+(\S+?)(?=\s+·|\s*$)")
#: a quoted title or value on the row line, blanked before the parent is read
QUOTED = re.compile(r'"(?:[^"\\]|\\.)*"')


def ranked_ids(text: str) -> list[tuple[str, ...]]:
    """The ids each result row stands for, best first: its own id, then its record's if it names one.

    Reads okg's text rows, or a JSON page with a `results` list of `{node_id, parent}`.
    Only the row line is read, never a title or an excerpt, so an id merely mentioned
    in a document's text is not a hit.
    """
    stripped = text.lstrip()
    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
        except ValueError:
            payload = None
        if isinstance(payload, Mapping) and isinstance(payload.get("results"), list):
            out = []
            for row in payload["results"]:
                if not isinstance(row, Mapping) or not row.get("node_id"):
                    continue
                ids = [str(row["node_id"])]
                parent = row.get("parent")
                if isinstance(parent, Mapping) and parent.get("node_id"):
                    ids.append(str(parent["node_id"]))
                elif isinstance(parent, str) and parent:
                    ids.append(parent)
                out.append(tuple(ids))
            return out
    rows: list[tuple[int, tuple[str, ...]]] = []
    for match in ROW_HEAD.finditer(text):
        end = text.find("\n", match.start())
        line = QUOTED.sub('""', text[match.start(): end if end >= 0 else len(text)])
        parent = PARENT.search(line)
        ids = (match.group(2), parent.group(1)) if parent else (match.group(2),)
        rows.append((int(match.group(1)), ids))
    rows.sort(key=lambda r: r[0])
    return [ids for _, ids in rows]


@dataclass(frozen=True)
class Hits:
    found: tuple[str, ...]
    missed: tuple[str, ...]
    #: 1-based rank of the first row that holds a gold id, or None
    first_rank: Optional[int]

    @property
    def recall(self) -> float:
        total = len(self.found) + len(self.missed)
        return len(self.found) / total if total else 0.0

    @property
    def reciprocal_rank(self) -> float:
        return 1.0 / self.first_rank if self.first_rank else 0.0


def score_hits(gold: tuple[str, ...], ranking: list[tuple[str, ...]], k: int) -> Hits:
    """recall@k and the first-hit rank of `gold` in the top `k` rows."""
    top = ranking[:k]
    found = [g for g in gold if any(g in ids for ids in top)]
    first = next((i for i, ids in enumerate(top, 1) if set(ids) & set(gold)), None)
    return Hits(tuple(found), tuple(g for g in gold if g not in found), first)


# --- the run ---------------------------------------------------------------------------


@dataclass
class ArmResult:
    arm: str
    status: str  # pass (a gold id came back), fail (none did), execution_failed (the call failed)
    recall: float = 0.0
    reciprocal_rank: float = 0.0
    found: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    first_rank: Optional[int] = None
    returned: list[str] = field(default_factory=list)
    seconds: float = 0.0
    error: Optional[str] = None


@dataclass
class QuestionResult:
    id: str
    question: str
    gold: list[str]
    arms: dict[str, ArmResult] = field(default_factory=dict)


@dataclass
class ArmSummary:
    arm: str
    tool: str
    arguments: Mapping[str, Any]
    scored: int
    errors: int
    recall_at_k: float
    mrr: float

    def to_dict(self) -> dict[str, Any]:
        return {"arm": self.arm, "tool": self.tool, "arguments": dict(self.arguments), "scored": self.scored,
                "errors": self.errors, "recall_at_k": round(self.recall_at_k, 4), "mrr": round(self.mrr, 4)}


@dataclass
class ContextReport:
    bundle: str
    k: int
    mcp_url: str
    questions_file: str
    arms: tuple[Arm, ...]
    min_recall: Optional[float] = None
    tools: list[str] = field(default_factory=list)
    questions: list[QuestionResult] = field(default_factory=list)
    unscored: list[str] = field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""
    seconds: float = 0.0
    error: Optional[str] = None

    def summary(self, arm: Arm) -> ArmSummary:
        rows = [q.arms[arm.name] for q in self.questions if arm.name in q.arms]
        n = len(rows)
        return ArmSummary(
            arm.name, arm.tool, arm.arguments, n, sum(r.status == EXECUTION_FAILED for r in rows),
            sum(r.recall for r in rows) / n if n else 0.0,
            sum(r.reciprocal_rank for r in rows) / n if n else 0.0,
        )

    @property
    def score(self) -> Optional[float]:
        """The graph arm's recall@k; None when the run could not execute."""
        if self.status == EXECUTION_FAILED:
            return None
        return self.summary(self._arm(GRAPH)).recall_at_k

    def _arm(self, name: str) -> Arm:
        return next(a for a in self.arms if a.name == name)

    @property
    def status(self) -> str:
        """`execution_failed` when the graph arm produced no result at all; else `pass`, or `fail`
        below `min_recall`. Calls that failed count as misses and are listed under `errors`."""
        graph = [q.arms.get(GRAPH) for q in self.questions]
        if self.error or not graph or all(r is None or r.status == EXECUTION_FAILED for r in graph):
            return EXECUTION_FAILED
        recall = self.summary(self._arm(GRAPH)).recall_at_k
        if self.min_recall is not None and recall < self.min_recall:
            return FAIL
        return PASS

    @property
    def exit_code(self) -> int:
        return {PASS: 0, FAIL: 1, EXECUTION_FAILED: 2}[self.status]

    def to_dict(self) -> dict[str, Any]:
        summaries = {a.name: self.summary(a).to_dict() for a in self.arms}
        graph, base = summaries[GRAPH], summaries[BASELINE]
        return {
            "schema": REPORT_SCHEMA,
            "note": NOTE,
            "bundle": self.bundle,
            "status": self.status,
            "score": None if self.score is None else round(self.score, 4),
            "metric": f"recall@{self.k} (graph arm)",
            "k": self.k,
            "min_recall": self.min_recall,
            "baseline_delta": round(graph["recall_at_k"] - base["recall_at_k"], 4),
            "arms": summaries,
            "mcp_url": self.mcp_url,
            "tools": self.tools,
            "questions_file": self.questions_file,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "seconds": round(self.seconds, 1),
            "error": self.error,
            "unscored": self.unscored,
            "questions": [
                {
                    "id": q.id,
                    "question": q.question,
                    "gold": q.gold,
                    "arms": {
                        name: {
                            "status": r.status, "recall": round(r.recall, 4),
                            "reciprocal_rank": round(r.reciprocal_rank, 4), "first_rank": r.first_rank,
                            "found": r.found, "missed": r.missed, "returned": r.returned,
                            "seconds": round(r.seconds, 2), "error": r.error,
                        }
                        for name, r in q.arms.items()
                    },
                }
                for q in self.questions
            ],
        }


ToolBoxFactory = Callable[[], Any]  # an async context manager yielding a ToolBox


async def run_context(
    cset: ContextSet,
    open_tools: ToolBoxFactory,
    *,
    mcp_url: str = "",
    progress: Optional[Callable[[QuestionResult, int, int], None]] = None,
) -> ContextReport:
    report = ContextReport(
        bundle=cset.bundle, k=cset.k, mcp_url=mcp_url, questions_file=str(cset.path or ""), arms=cset.arms,
        min_recall=cset.min_recall, unscored=[q.id for q in cset.questions if not q.gold],
        started_at=_now(),
    )
    started = time.monotonic()
    scored = cset.scored
    try:
        async with open_tools() as tools:
            specs: list[ToolSpec] = await tools.list_tools()
            report.tools = sorted(s.name for s in specs)
            missing = sorted({a.tool for a in cset.arms} - set(report.tools))
            if missing:
                report.error = (f"the MCP endpoint does not offer {', '.join(missing)} "
                                f"(it offers: {', '.join(report.tools) or 'nothing'})")
            else:
                for i, question in enumerate(scored, 1):
                    result = QuestionResult(question.id, question.question, list(question.gold))
                    for arm in cset.arms:
                        result.arms[arm.name] = await _one(tools, arm, question, cset.k)
                    report.questions.append(result)
                    if progress:
                        progress(result, i, len(scored))
    except ToolError as exc:
        report.error = f"MCP endpoint not usable: {exc}"
    except Exception as exc:  # noqa: BLE001 - any other failure is a run that could not execute
        report.error = f"the run stopped: {_describe(exc)}"
    report.seconds = time.monotonic() - started
    report.finished_at = _now()
    return report


async def _one(tools, arm: Arm, question: ContextQuestion, k: int) -> ArmResult:
    arguments = {**arm.arguments, "query": question.question, "limit": k}
    started = time.monotonic()
    try:
        text, is_error = await tools.call(arm.tool, arguments)
    except Exception as exc:  # noqa: BLE001 - a failed call is recorded, not raised
        text, is_error = f"{_describe(exc)}", True
    seconds = time.monotonic() - started
    if is_error:
        return ArmResult(arm.name, EXECUTION_FAILED, missed=list(question.gold), seconds=seconds,
                         error=text.strip()[:500] or "the tool reported an error")
    ranking = ranked_ids(text)
    hits = score_hits(question.gold, ranking, k)
    return ArmResult(
        arm.name, PASS if hits.found else FAIL, hits.recall, hits.reciprocal_rank, list(hits.found),
        list(hits.missed), hits.first_rank, [" in ".join(ids) for ids in ranking[:k]], seconds,
    )


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- writing ---------------------------------------------------------------------------


def render_markdown(report: ContextReport) -> str:
    data = report.to_dict()
    score = "none" if data["score"] is None else f"{data['score']:.3f}"
    lines = [
        f"# archi eval --context: {report.bundle}",
        "",
        f"**{report.status}**: graph arm recall@{report.k} {score} over {len(report.questions)} questions"
        + (f" (minimum {report.min_recall:g})" if report.min_recall is not None else "")
        + f"; baseline delta {data['baseline_delta']:+.3f}.",
        "",
        f"> {NOTE}",
        "",
        f"- MCP: {report.mcp_url}; tools: {', '.join(report.tools) or 'none'}",
        f"- started {report.started_at}; finished {report.finished_at}; wall time {data['seconds']} s",
        f"- questions file: {report.questions_file}",
    ]
    if report.unscored:
        lines.append(f"- not scored (no gold ids): {', '.join(report.unscored)}")
    if report.error:
        lines += ["", f"**Error:** {report.error}"]
    lines += ["", f"| arm | tool and arguments | recall@{report.k} | MRR | errors |", "|---|---|---|---|---|"]
    for arm in report.arms:
        s = data["arms"][arm.name]
        lines.append(f"| {arm.name} | `{arm.tool}({json.dumps(s['arguments'], sort_keys=True)})` | "
                     f"{s['recall_at_k']:.3f} | {s['mrr']:.3f} | {s['errors']} |")
    names = [a.name for a in report.arms]
    lines += ["", "| question | gold | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    for q in report.questions:
        cells = []
        for name in names:
            r = q.arms.get(name)
            cells.append("-" if r is None else ("error" if r.status == EXECUTION_FAILED
                                               else f"{len(r.found)}/{len(q.gold)}"
                                               + (f" @{r.first_rank}" if r.first_rank else "")))
        lines.append(f"| {q.id} | {len(q.gold)} | " + " | ".join(cells) + " |")
    for q in report.questions:
        lines += ["", f"## {q.id}", "", f"Q: {q.question}", "", f"Gold: {', '.join(f'`{g}`' for g in q.gold)}", ""]
        for name in names:
            r = q.arms.get(name)
            if r is None:
                continue
            if r.error:
                lines.append(f"- {name}: error: {r.error}")
            else:
                lines.append(f"- {name}: found {', '.join(f'`{g}`' for g in r.found) or 'none'}; missed "
                             f"{', '.join(f'`{g}`' for g in r.missed) or 'none'}; {len(r.returned)} rows")
    return "\n".join(lines).rstrip() + "\n"


def write_report(report: ContextReport, out: Path) -> tuple[Path, Path]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    md, js = out / "report.md", out / "report.json"
    md.write_text(render_markdown(report), encoding="utf-8")
    js.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    return md, js


def record_score(report: ContextReport, instance_dir: Path, *, report_path: Optional[Path] = None) -> Optional[Path]:
    """Write the score where the context gate and the instance-health monitor read it:
    `<instance>/eval/reports/context-<bundle>-<time>.json`. Nothing is written for a run
    that could not execute: it has no score."""
    score = report.score
    if score is None:
        return None
    directory = Path(instance_dir) / CONTEXT_EVAL_DIR
    directory.mkdir(parents=True, exist_ok=True)
    stamp = report.finished_at.replace("-", "").replace(":", "")
    path = directory / f"context-{report.bundle}-{stamp}.json"
    data = report.to_dict()
    record = {
        "schema": CONTEXT_EVAL_SCHEMA,
        "kind": "context",
        "bundle": report.bundle,
        "finished_at": report.finished_at,
        "score": round(score, 4),
        "metric": data["metric"],
        "k": report.k,
        "baseline_recall_at_k": data["arms"][BASELINE]["recall_at_k"],
        "questions": len(report.questions),
        "questions_file": report.questions_file,
        "report": str(report_path) if report_path else None,
    }
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return path
