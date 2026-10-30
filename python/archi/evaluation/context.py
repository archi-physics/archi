"""Context evals: retrieval scored on its own (PACT r3-context-quality, ADR 0002 section 6.1).

`archi eval --context` asks each question of a set straight to the instance's MCP
search tool, with no model in between, and checks whether the gold evidence ids
come back in the top k. Each question runs in two arms:

    graph     the search an operator's agent makes: the tool with the server's defaults,
              which rolls passages up to their records
    baseline  a lexical baseline: the same tool asked for its lexical index over chunks and
              attributes, each text unit ranked on its own words (`rank_by: self`), with no
              roll-up to records and no concept expansion. It is the closest the MCP surface
              comes to keyword search over the raw documents, not a separate raw-document index.

The graph arm is lexical search too (okg's default method), with concept expansion and
record roll-up on, so the difference between the arms measures those two, not graph
links. Chunk rows can fill the baseline's k slots where the graph arm returns one row per
record.

A question file may override the graph arm; such a run is informational only. On an
instance, where the run would be offered to the context gate, the CLI refuses it, and the
gate refuses any report whose graph arm is not the default.

A hit counts for a gold id when its row's id is that id, or when the row is a passage
whose record is that id (`parent` in JSON, `in <id>` on a text row line). Ids in a row's
title or excerpt never count.

Calls ask okg for JSON (request `_meta` `okg/result_format: json`, served by okg 0.2.0);
a server that answers in text is read from its row lines (`[n] <id> (<subtype>)` at the
start of a line) instead. The generation each call read is taken from the reply.

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
        answer: FTS3 is the file transfer service ...   # optional: an answer key for the judge
        gold_sources:          # optional, agent arms: where a gold id's record is in the raw
          documentation_page:22: {source: docsite, key: "https://..."}   # corpus, instead of
                                                                         # asking MCP inspect
        atoms:                 # optional, agent arms: the facts an answer needs
          - id: a1
            answer_source: doc or ticket   # live_rucio, live_dbs or monitoring alone: unavailable
            gold: [documentation_page:22]  # the gold ids that carry this fact

A smoke file with `gold` added to its questions is accepted: `pass_threshold`,
`must_cite`, `must_contain` and `must_contain_any` are read and ignored. A question
without `gold` is listed and not scored. Operator question sets are private to their
instance and are named at run time; none is kept in this repository.

Each run is written to `<data root>/eval/context/<bundle>/<UTC stamp>/report.{json,md}`.
The context gate (`archi.ops.context_gate.read_context_eval`) reads the newest one of a
bundle; its score is the graph arm's recall@k.

With `--agents`, two agent arms run beside these two (`archi.evaluation.raw_arm`): a bash
agent over the raw files the graph was built from, and an agent with the instance's MCP
tools, on the same model, caps and scorer. Their results go under `agent_arms` in the
report, never under `arms`, so the gate and the run's status read the graph arm alone.
"""
from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import yaml

from archi.evaluation.agent import ToolError, ToolSpec, _describe
from archi.ops.context_gate import CONTEXT_EVAL_REPORT, CONTEXT_REPORT_SCHEMA, bundle_path_name

CONTEXT_FILE = "context.yaml"
REPORT_SCHEMA = CONTEXT_REPORT_SCHEMA
DEFAULT_K = 10
GRAPH, BASELINE = "graph", "baseline"
PASS, FAIL, EXECUTION_FAILED = "pass", "fail", "execution_failed"
#: Asks okg to serve the result as compact JSON instead of its text rendering.
RESULT_META = {"okg/result_format": "json"}

#: The arms every run has, and what each sends besides `query` and `limit`.
DEFAULT_ARMS: dict[str, dict[str, Any]] = {
    GRAPH: {"tool": "search", "arguments": {}},
    BASELINE: {"tool": "search", "arguments": {"method": "lexical", "rank_by": "self", "expand_concepts": False}},
}
ARM_DESCRIPTIONS = {
    GRAPH: ("the server's default search, as an operator's agent calls it: lexical too, with concept expansion "
            "and passages rolled up to their records, so the delta measures those, not graph links"),
    BASELINE: ("lexical baseline: the server's lexical index over chunks and attributes, each text unit ranked "
               "on its own words, no roll-up to records, no concept expansion. Chunk rows can fill its k slots "
               "where the graph arm rolls up to records, so part of the delta is that roll-up"),
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
    description: str = "set by the question file"


@dataclass(frozen=True)
class ContextQuestion:
    id: str
    question: str
    gold: tuple[str, ...] = ()
    #: an answer key for the agent arms' judge, or None
    answer: Optional[str] = None
    #: gold id -> where its record is in the raw corpus ({source, key} or {path}), overriding
    #: resolution through the MCP inspect tool
    gold_sources: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    #: the facts an answer needs, each with where its answer comes from (`Atom`)
    atoms: tuple["Atom", ...] = ()


#: `answer_source` kinds only a live system answers: no file holds them, in either arm.
LIVE_SOURCES = frozenset({"live_rucio", "live_dbs", "monitoring"})
_SOURCE_SPLIT = re.compile(r"\s*(?:\+|,|/|;|\bor\b|\band\b)\s*")


@dataclass(frozen=True)
class Atom:
    id: str
    answer_source: str = ""
    gold: tuple[str, ...] = ()
    text: str = ""
    required: bool = True

    @property
    def kinds(self) -> tuple[str, ...]:
        return tuple(k for k in _SOURCE_SPLIT.split(self.answer_source.strip().lower()) if k)

    @property
    def live_only(self) -> bool:
        """Answered only from live systems ("live_rucio", "monitoring", "ticket+live_rucio" is not)."""
        return bool(self.kinds) and all(k in LIVE_SOURCES for k in self.kinds)


@dataclass(frozen=True)
class ContextSet:
    bundle: str
    k: int
    questions: tuple[ContextQuestion, ...]
    arms: tuple[Arm, ...]
    min_recall: Optional[float] = None
    corpus: str = ""
    path: Optional[Path] = None
    #: `sha256:<hex>` of the question file's bytes (of the parsed mapping when there is no file)
    sha256: str = ""

    @property
    def scored(self) -> tuple[ContextQuestion, ...]:
        return tuple(q for q in self.questions if q.gold)

    @property
    def graph_overridden(self) -> bool:
        """Whether the file replaced the default graph arm (such a run is informational only)."""
        graph = next(a for a in self.arms if a.name == GRAPH)
        default = DEFAULT_ARMS[GRAPH]
        return graph.tool != default["tool"] or dict(graph.arguments) != default["arguments"]


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
        description = ARM_DESCRIPTIONS[name] if name in DEFAULT_ARMS and name not in raw else "set by the question file"
        arms.append(Arm(name, tool, dict(args), description))
    return tuple(arms)


def _atoms(label: str, raw: Any, gold: tuple[str, ...]) -> tuple[Atom, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ContextSetError(f"{label}: `atoms` must be a list of {{id, answer_source, gold}}")
    out, seen = [], set()
    for i, atom in enumerate(raw):
        where = f"{label}: atoms[{i}]"
        if not isinstance(atom, Mapping) or set(atom) - {"id", "answer_source", "gold", "text", "required"}:
            raise ContextSetError(f"{where} takes id, answer_source, gold, text and required")
        aid = atom.get("id")
        if not isinstance(aid, str) or not aid or aid in seen:
            raise ContextSetError(f"{where}: `id` must be a unique non-empty string")
        seen.add(aid)
        source = atom.get("answer_source", "")
        if not isinstance(source, str):
            raise ContextSetError(f"{where}: `answer_source` must be a string")
        atom_gold = _strings(f"{where}: gold", atom.get("gold"))
        stray = [g for g in atom_gold if g not in gold]
        if stray:
            raise ContextSetError(f"{where}: gold {stray} is not among the question's gold ids")
        required = atom.get("required", True)
        if not isinstance(required, bool):
            raise ContextSetError(f"{where}: `required` must be true or false")
        out.append(Atom(aid, source.strip(), atom_gold, str(atom.get("text") or ""), required))
    return tuple(out)


def _gold_sources(label: str, raw: Any, gold: tuple[str, ...]) -> dict[str, dict[str, str]]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ContextSetError(f"{label}: `gold_sources` must map gold ids to {{source, key}} or {{path}}")
    out = {}
    for gid, where in raw.items():
        if gid not in gold:
            raise ContextSetError(f"{label}: gold_sources names {gid!r}, which is not one of its gold ids")
        ok = isinstance(where, Mapping) and set(where) in ({"source", "key"}, {"path"}) and all(
            isinstance(v, str) and v.strip() for v in where.values())
        if not ok:
            raise ContextSetError(f"{label}: gold_sources.{gid} must be {{source, key}} or {{path}}, as strings")
        out[str(gid)] = {k: str(v).strip() for k, v in where.items()}
    return out


def parse_context(raw: Any, path: Optional[Path] = None, *, sha256: str = "") -> ContextSet:
    where = str(path) if path else "context set"
    if not isinstance(raw, Mapping):
        raise ContextSetError(f"{where}: expected a mapping")
    unknown = set(raw) - {"bundle", "k", "min_recall", "corpus", "arms", "questions"} - SMOKE_KEYS
    if unknown:
        raise ContextSetError(f"{where}: unknown keys {sorted(unknown)}")
    bundle = raw.get("bundle")
    if not isinstance(bundle, str) or not bundle:
        raise ContextSetError(f"{where}: `bundle` is required")
    try:
        bundle_path_name(bundle)
    except ValueError as exc:
        raise ContextSetError(f"{where}: {exc}") from exc
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
        extra = set(row) - {"id", "question", "gold", "answer", "gold_sources", "atoms"} - SMOKE_QUESTION_KEYS
        if extra:
            raise ContextSetError(f"{label}: unknown keys {sorted(extra)}")
        qid, text = row.get("id"), row.get("question")
        if not isinstance(qid, str) or not qid or not isinstance(text, str) or not text.strip():
            raise ContextSetError(f"{label}: `id` and `question` are required strings")
        if qid in seen:
            raise ContextSetError(f"{label}: duplicate id {qid!r}")
        seen.add(qid)
        gold = _strings(f"{label}: gold", row.get("gold"))
        answer = row.get("answer")
        if answer is not None and (not isinstance(answer, str) or not answer.strip()):
            raise ContextSetError(f"{label}: `answer` must be a non-empty string")
        questions.append(ContextQuestion(qid, text.strip(), gold, answer.strip() if answer else None,
                                         _gold_sources(label, row.get("gold_sources"), gold),
                                         _atoms(label, row.get("atoms"), gold)))
    if not any(q.gold for q in questions):
        raise ContextSetError(f"{where}: no question has `gold` ids, so nothing can be scored")
    if not sha256:
        body = json.dumps(raw, sort_keys=True, default=str).encode("utf-8")
        sha256 = "sha256:" + hashlib.sha256(body).hexdigest()
    return ContextSet(bundle, k, tuple(questions), _arms(where, raw.get("arms")), min_recall,
                      str(raw.get("corpus") or ""), path, sha256)


def load_context(path: Path) -> ContextSet:
    path = Path(path)
    try:
        data = path.read_bytes()
        raw = yaml.safe_load(data.decode("utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ContextSetError(f"{path}: {exc}") from exc
    return parse_context(raw, path, sha256="sha256:" + hashlib.sha256(data).hexdigest())


def context_path(bundle_dir: Path) -> Path:
    return Path(bundle_dir) / "eval" / CONTEXT_FILE


# --- reading the ranking out of a search result ---------------------------------------

#: `[3] chunk:ab12 (document_chunk) · ... · in documentation_page:22 · ...`, okg's row line. Only a line
#: that starts with the marker is a row: an indented `[2] ...` is a line of an excerpt.
ROW_HEAD = re.compile(r"^\[(\d+)\] (\S+) \(", re.M)
#: the record a listed passage belongs to: the ` · in <id>` fact of its row line
PARENT = re.compile(r"\s·\s+in\s+(\S+?)(?=\s+·|\s*$)")
#: a quoted title or value on the row line, blanked before the parent is read
QUOTED = re.compile(r'"(?:[^"\\]|\\.)*"')
#: the `generation <id>` fact of a text reply's status line
GENERATION = re.compile(r"(?:^|\s)·\s+generation\s+(\S+?)(?=\s+·|\s*$)")


@dataclass(frozen=True)
class Page:
    #: per row, best first: its own id, then its record's if it names one
    ranking: list[tuple[str, ...]]
    generation: Optional[str] = None
    format: str = "text"


def read_page(text: str) -> Page:
    """The ranking and the generation of one search reply, JSON or okg's text rendering."""
    stripped = text.lstrip()
    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
        except ValueError:
            payload = None
        if isinstance(payload, Mapping):
            rows = payload.get("results")
            out = []
            for row in rows if isinstance(rows, list) else []:
                if not isinstance(row, Mapping) or not row.get("node_id"):
                    continue
                ids = [str(row["node_id"])]
                parent = row.get("parent")
                if isinstance(parent, Mapping) and parent.get("node_id"):
                    ids.append(str(parent["node_id"]))
                elif isinstance(parent, str) and parent:
                    ids.append(parent)
                out.append(tuple(ids))
            generation = (payload.get("pinned_generation_id") or payload.get("generation_id")
                          or payload.get("session_default_generation_id"))
            return Page(out, None if generation is None else str(generation), "json")
    rows: list[tuple[int, tuple[str, ...]]] = []
    for match in ROW_HEAD.finditer(text):
        end = text.find("\n", match.start())
        line = QUOTED.sub('""', text[match.start(): end if end >= 0 else len(text)])
        parent = PARENT.search(line)
        ids = (match.group(2), parent.group(1)) if parent else (match.group(2),)
        rows.append((int(match.group(1)), ids))
    rows.sort(key=lambda r: r[0])
    status = QUOTED.sub('""', text.split("\n", 1)[0])
    generation = GENERATION.search(status)
    return Page([ids for _, ids in rows], generation.group(1) if generation else None, "text")


def ranked_ids(text: str) -> list[tuple[str, ...]]:
    return read_page(text).ranking


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
    generation: Optional[str] = None
    format: Optional[str] = None


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
    description: str
    scored: int
    errors: int
    recall_at_k: float
    mrr: float

    def to_dict(self) -> dict[str, Any]:
        return {"arm": self.arm, "tool": self.tool, "arguments": dict(self.arguments),
                "description": self.description, "scored": self.scored, "errors": self.errors,
                "recall_at_k": round(self.recall_at_k, 4), "mrr": round(self.mrr, 4)}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


@dataclass
class ContextReport:
    bundle: str
    k: int
    mcp_url: str
    questions_file: str
    questions_sha256: str
    arms: tuple[Arm, ...]
    min_recall: Optional[float] = None
    #: scored questions in the file; every one must have a result row for the gate to use the run
    question_count: int = 0
    tools: list[str] = field(default_factory=list)
    questions: list[QuestionResult] = field(default_factory=list)
    unscored: list[str] = field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""
    seconds: float = 0.0
    error: Optional[str] = None
    #: the agent arms' section (`archi.evaluation.raw_arm.AgentReport.to_dict()`), when they ran
    agents: Optional[dict[str, Any]] = None

    def summary(self, arm: Arm) -> ArmSummary:
        rows = [q.arms[arm.name] for q in self.questions if arm.name in q.arms]
        return ArmSummary(arm.name, arm.tool, arm.arguments, arm.description, len(rows),
                          sum(r.status == EXECUTION_FAILED for r in rows),
                          _mean([r.recall for r in rows]), _mean([r.reciprocal_rank for r in rows]))

    def _arm(self, name: str) -> Arm:
        return next(a for a in self.arms if a.name == name)

    @property
    def errors(self) -> int:
        """Failed MCP calls, every arm."""
        return sum(r.status == EXECUTION_FAILED for q in self.questions for r in q.arms.values())

    @property
    def graph_errors(self) -> int:
        return sum(q.arms[GRAPH].status == EXECUTION_FAILED for q in self.questions if GRAPH in q.arms)

    @property
    def status(self) -> str:
        """`execution_failed` when the run stopped or any graph-arm call failed (a score with misses that
        are really failed calls is not a measurement); else `pass`, or `fail` below `min_recall`."""
        if self.error or not self.questions or self.graph_errors or len(self.questions) != self.question_count:
            return EXECUTION_FAILED
        if self.min_recall is not None and self.summary(self._arm(GRAPH)).recall_at_k < self.min_recall:
            return FAIL
        return PASS

    @property
    def score(self) -> Optional[float]:
        """The graph arm's recall@k; None when the run did not execute cleanly."""
        if self.status == EXECUTION_FAILED:
            return None
        return self.summary(self._arm(GRAPH)).recall_at_k

    @property
    def exit_code(self) -> int:
        return {PASS: 0, FAIL: 1, EXECUTION_FAILED: 2}[self.status]

    def paired(self) -> list[QuestionResult]:
        """The questions on which both the graph and the baseline arm executed."""
        return [q for q in self.questions
                if all(name in q.arms and q.arms[name].status != EXECUTION_FAILED for name in (GRAPH, BASELINE))]

    def baseline_delta(self) -> Optional[float]:
        """Graph minus baseline recall@k, over the paired questions only."""
        pairs = self.paired()
        if not pairs:
            return None
        return _mean([q.arms[GRAPH].recall for q in pairs]) - _mean([q.arms[BASELINE].recall for q in pairs])

    def generations(self) -> list[str]:
        """The generations the graph arm's calls read, in order of first sight."""
        seen = [q.arms[GRAPH].generation for q in self.questions if GRAPH in q.arms]
        return list(dict.fromkeys(g for g in seen if g))

    def to_dict(self) -> dict[str, Any]:
        delta = self.baseline_delta()
        generations = self.generations()
        return {
            "schema": REPORT_SCHEMA,
            "note": NOTE,
            "bundle": self.bundle,
            "status": self.status,
            "score": None if self.score is None else round(self.score, 4),
            "metric": f"recall@{self.k} (graph arm)",
            "k": self.k,
            "min_recall": self.min_recall,
            "errors": self.errors,
            "baseline_delta": None if delta is None else round(delta, 4),
            "baseline_delta_questions": len(self.paired()),
            "generation": generations[0] if len(generations) == 1 else None,
            "generations": generations,
            "arms": {a.name: self.summary(a).to_dict() for a in self.arms},
            "mcp_url": self.mcp_url,
            "tools": self.tools,
            "questions_file": self.questions_file,
            "questions_sha256": self.questions_sha256,
            "question_count": self.question_count,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "seconds": round(self.seconds, 1),
            "error": self.error,
            "unscored": self.unscored,
            **({"agent_arms": self.agents} if self.agents is not None else {}),
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
                            "generation": r.generation, "format": r.format,
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
        bundle=cset.bundle, k=cset.k, mcp_url=mcp_url, questions_file=str(cset.path or ""),
        questions_sha256=cset.sha256, arms=cset.arms, min_recall=cset.min_recall,
        question_count=len(cset.scored), unscored=[q.id for q in cset.questions if not q.gold], started_at=_now(),
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
        text, is_error = await tools.call(arm.tool, arguments, meta=RESULT_META)
    except Exception as exc:  # noqa: BLE001 - a failed call is recorded, not raised
        text, is_error = f"{_describe(exc)}", True
    seconds = time.monotonic() - started
    if is_error:
        return ArmResult(arm.name, EXECUTION_FAILED, missed=list(question.gold), seconds=seconds,
                         error=text.strip()[:500] or "the tool reported an error")
    page = read_page(text)
    hits = score_hits(question.gold, page.ranking, k)
    return ArmResult(
        arm.name, PASS if hits.found else FAIL, hits.recall, hits.reciprocal_rank, list(hits.found),
        list(hits.missed), hits.first_rank, [" in ".join(ids) for ids in page.ranking[:k]], seconds,
        generation=page.generation, format=page.format,
    )


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- writing ---------------------------------------------------------------------------


def render_markdown(report: ContextReport) -> str:
    data = report.to_dict()
    score = "none" if data["score"] is None else f"{data['score']:.3f}"
    delta = ("none (no question on which both arms executed)" if data["baseline_delta"] is None
             else f"{data['baseline_delta']:+.3f} over {data['baseline_delta_questions']} questions")
    lines = [
        f"# archi eval --context: {report.bundle}",
        "",
        f"**{report.status}**: graph arm recall@{report.k} {score} over {len(report.questions)} questions"
        + (f" (minimum {report.min_recall:g})" if report.min_recall is not None else "")
        + f"; baseline delta {delta}; {data['errors']} failed calls.",
        "",
        f"> {NOTE}",
        "",
        f"- MCP: {report.mcp_url}; tools: {', '.join(report.tools) or 'none'}",
        f"- generation read: {data['generation'] or ', '.join(data['generations']) or 'not reported'}",
        f"- started {report.started_at}; finished {report.finished_at}; wall time {data['seconds']} s",
        f"- questions file: {report.questions_file or 'none'} (`{report.questions_sha256}`)",
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
    lines.append("")
    lines += [f"- **{arm.name}**: {arm.description}." for arm in report.arms]
    if report.agents is not None:
        from archi.evaluation.raw_arm import render_agent_markdown

        lines += ["", *render_agent_markdown(report.agents, data)]
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


def _write(directory: Path, report: ContextReport) -> tuple[Path, Path]:
    md, js = directory / "report.md", directory / CONTEXT_EVAL_REPORT
    md.write_text(render_markdown(report), encoding="utf-8")
    js.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    return md, js


def publish_run(report: ContextReport, root: Path, *, carry: Optional[Mapping[str, Path]] = None
                ) -> tuple[Path, Path]:
    """Write one run to `<root>/<YYYYMMDDTHHMMSSZ>` (`-2`, `-3`, ... on a collision), atomically.

    Both reports are written into a hidden `<root>/.<stamp>.tmp-<pid>` first and the directory is
    renamed into place only when they are complete, so a crash never leaves a half-written report
    where the context gate or anything else reads. Readers skip names that start with a dot.
    `carry` maps a file name to a file written during the run (the agent arms' spend ledger),
    moved into the run's directory with the reports."""
    stamp = report.finished_at.replace("-", "").replace(":", "")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / f".{stamp}.tmp-{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    try:
        _write(tmp, report)
        for name, source in (carry or {}).items():
            if Path(source).exists():
                shutil.move(str(source), str(tmp / name))
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
    return target / "report.md", target / CONTEXT_EVAL_REPORT


def write_report(report: ContextReport, out: Path) -> tuple[Path, Path]:
    """Write both reports into `out` (`--out`), each file replaced atomically."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, text in (("report.md", render_markdown(report)),
                       (CONTEXT_EVAL_REPORT, json.dumps(report.to_dict(), indent=2) + "\n")):
        final = out / name
        tmp = out / f".{name}.tmp-{os.getpid()}"
        try:
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, final)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        paths.append(final)
    return paths[0], paths[1]
