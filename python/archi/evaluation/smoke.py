"""The smoke question format and the deterministic answer checks.

`bundles/<bundle>/eval/smoke.yaml`:

    bundle: cern-team
    pass_threshold: 7          # questions that must pass (of the set)
    corpus: what the questions are answerable from (a note for readers)
    questions:
      - id: fts3-what-is
        question: What is FTS3?
        must_cite:             # at least one key; every key given must hold
          ids: [id, ...]       # every one of these node ids is cited
          any_id: [id, ...]    # at least one of these is cited
          subtypes: [t, ...]   # a cited node has one of these subtypes
          sources: [s, ...]    # a cited node was written by one of these sources
                               # (a source is `<deployment>.<name>`; `docsite` matches `gp.docsite`)
        must_contain: [text]   # every string appears in the answer (ignoring case, hyphens, extra spaces)
        must_contain_any: [text]   # at least one appears

A node counts as cited when the answer names its id AND a tool result of the
same run returned that id: an id the model invented cannot pass.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

SMOKE_FILE = "smoke.yaml"
EVAL_DIR = "eval"
CITE_KEYS = ("ids", "any_id", "subtypes", "sources")


class SmokeError(ValueError):
    """The smoke file is not usable."""


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    ids: tuple[str, ...] = ()
    any_id: tuple[str, ...] = ()
    subtypes: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    contains: tuple[str, ...] = ()
    contains_any: tuple[str, ...] = ()


@dataclass(frozen=True)
class SmokeSet:
    bundle: str
    pass_threshold: int
    questions: tuple[Question, ...]
    corpus: str = ""
    path: Optional[Path] = None


@dataclass(frozen=True)
class NodeInfo:
    """What a tool result said about one node. Unknown fields are None."""

    id: str
    subtype: Optional[str] = None
    source: Optional[str] = None
    #: the tool returned this node as a result (a row or the record inspect read), not
    #: only as an id mentioned inside another record's text
    listed: bool = False


def _strings(path: str, raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or not all(isinstance(x, str) and x.strip() for x in raw):
        raise SmokeError(f"{path} must be a list of non-empty strings")
    return tuple(raw)


def parse_smoke(raw: Any, path: Optional[Path] = None) -> SmokeSet:
    where = str(path) if path else "smoke set"
    if not isinstance(raw, Mapping):
        raise SmokeError(f"{where}: expected a mapping")
    unknown = set(raw) - {"bundle", "pass_threshold", "corpus", "questions"}
    if unknown:
        raise SmokeError(f"{where}: unknown keys {sorted(unknown)}")
    bundle = raw.get("bundle")
    if not isinstance(bundle, str) or not bundle:
        raise SmokeError(f"{where}: `bundle` is required")
    rows = raw.get("questions")
    if not isinstance(rows, list) or not rows:
        raise SmokeError(f"{where}: `questions` must be a non-empty list")
    questions: list[Question] = []
    seen: set[str] = set()
    for i, row in enumerate(rows):
        label = f"{where}: questions[{i}]"
        if not isinstance(row, Mapping):
            raise SmokeError(f"{label} must be a mapping")
        extra = set(row) - {"id", "question", "must_cite", "must_contain", "must_contain_any"}
        if extra:
            raise SmokeError(f"{label}: unknown keys {sorted(extra)}")
        qid, text = row.get("id"), row.get("question")
        if not isinstance(qid, str) or not qid or not isinstance(text, str) or not text.strip():
            raise SmokeError(f"{label}: `id` and `question` are required strings")
        if qid in seen:
            raise SmokeError(f"{label}: duplicate id {qid!r}")
        seen.add(qid)
        cite = row.get("must_cite") or {}
        if not isinstance(cite, Mapping) or set(cite) - set(CITE_KEYS):
            raise SmokeError(f"{label}: must_cite takes only {list(CITE_KEYS)}")
        parsed = {k: _strings(f"{label}: must_cite.{k}", cite.get(k)) for k in CITE_KEYS}
        if not any(parsed.values()):
            raise SmokeError(
                f"{label}: must_cite needs at least one check "
                "(a question without one cannot fail on its sources)"
            )
        questions.append(
            Question(
                id=qid,
                question=text.strip(),
                ids=parsed["ids"],
                any_id=parsed["any_id"],
                subtypes=parsed["subtypes"],
                sources=parsed["sources"],
                contains=_strings(f"{label}: must_contain", row.get("must_contain")),
                contains_any=_strings(f"{label}: must_contain_any", row.get("must_contain_any")),
            )
        )
    threshold = raw.get("pass_threshold")
    if not isinstance(threshold, int) or isinstance(threshold, bool) or not 1 <= threshold <= len(questions):
        raise SmokeError(f"{where}: `pass_threshold` must be an integer from 1 to {len(questions)}")
    return SmokeSet(
        bundle=bundle,
        pass_threshold=threshold,
        questions=tuple(questions),
        corpus=str(raw.get("corpus") or ""),
        path=path,
    )


def load_smoke(path: Path) -> SmokeSet:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SmokeError(f"{path}: {exc}") from exc
    return parse_smoke(raw, path)


def smoke_path(bundle_dir: Path) -> Path:
    return Path(bundle_dir) / EVAL_DIR / SMOKE_FILE


# --- reading node ids out of tool results ---------------------------------------------

#: `[3] chunk:ab12 (document_chunk) · source docsite ...`, the row shape of the read tools
ROW = re.compile(r"^\s*\[(\d+)\]\s+(\S+:\S+)\s+\(([^)]+)\)(?:.*? · source (\S+))?.*$", re.M)
#: the same row when the model selected fields that leave out subtype and source
#: (`filter` with `select: [node_id, attrs]` prints `[1] ops_incident:inc-1`)
BARE_ROW = re.compile(r"^\s*\[\d+\]\s+(\S+:\S+)\s*$", re.M)
HEADER_SOURCE = re.compile(r"^\w+ ok · .*? · source (\S+)", re.M)
#: `inspect ok · an:X (analysis_note) · complete`, the header of one record read with inspect;
#: its source is on a line of its own below (`source px.cmssw-releases`, or
#: `trust=untrusted · source px.analysis-notes`), before the first `--- field` block
INSPECT_HEAD = re.compile(r"^inspect ok · (\S+:\S+) \(([^)]+)\)", re.M)
SOURCE_LINE = re.compile(r"^(?:\S+ · )*source (\S+)\s*$", re.M)
#: any `<kind>:<rest>` that looks like a node id (no URL schemes, no times); the rest may
#: hold more colons (`tracker_ticket:redmine:4812`)
BARE_ID = re.compile(r"(?<![\w/:.-])([A-Za-z][A-Za-z0-9_]*:[A-Za-z0-9_.+/=:-]*[A-Za-z0-9_=-])")
NOT_KINDS = {"http", "https", "ftp", "mailto", "file", "ssh", "git", "ok", "error", "trust", "source", "type"}


def nodes_in(text: str) -> dict[str, NodeInfo]:
    """Every node id a tool result names, with subtype and source where the row says them."""
    found: dict[str, NodeInfo] = {}
    header = HEADER_SOURCE.search(text)
    default_source = header.group(1) if header else None
    for _, nid, subtype, source in ROW.findall(text):
        found[nid] = NodeInfo(nid, subtype, source or default_source, listed=True)
    for nid in BARE_ROW.findall(text):
        if nid not in found:
            found[nid] = NodeInfo(nid, listed=True)
    head = INSPECT_HEAD.search(text)
    if head and head.group(1) not in found:
        end = text.find("\n---", head.end())
        line = SOURCE_LINE.search(text, head.end(), end if end >= 0 else len(text))
        found[head.group(1)] = NodeInfo(head.group(1), head.group(2), line.group(1) if line else None, listed=True)
    for nid in BARE_ID.findall(text):
        if nid.split(":", 1)[0].lower() in NOT_KINDS or nid in found:
            continue
        found[nid] = NodeInfo(nid)
    return found


def merge_nodes(into: dict[str, NodeInfo], more: Mapping[str, NodeInfo]) -> None:
    """Add `more` to `into`, keeping what either knew about a node."""
    for nid, info in more.items():
        old = into.get(nid)
        if old is None:
            into[nid] = info
        else:
            into[nid] = NodeInfo(nid, old.subtype or info.subtype, old.source or info.source,
                                 old.listed or info.listed)


# --- the checks -----------------------------------------------------------------------


@dataclass
class Verdict:
    passed: bool
    cited: list[NodeInfo] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


MIN_BARE_ID = 6


def cited_nodes(answer: str, seen: Mapping[str, NodeInfo]) -> list[NodeInfo]:
    """The nodes the answer names that a tool returned.

    A node is named by its full id (`documentation_page:2275b50c`), or by the part
    after the colon when that part is at least six characters long and no other
    returned node ends the same way: small models often drop the prefix.
    """
    tails: dict[str, list[str]] = {}
    for nid in seen:
        tails.setdefault(nid.split(":", 1)[-1], []).append(nid)
    found: dict[str, NodeInfo] = {}
    for nid, info in seen.items():
        if re.search(rf"(?<![\w:]){re.escape(nid)}(?![\w=])", answer):
            found[nid] = info
    for tail, ids in tails.items():
        if len(ids) == 1 and len(tail) >= MIN_BARE_ID and ids[0] not in found:
            if re.search(rf"(?<![\w:]){re.escape(tail)}(?![\w=])", answer):
                found[ids[0]] = seen[ids[0]]
    return sorted(found.values(), key=lambda n: n.id)


def _from_source(source: Optional[str], wanted: tuple[str, ...]) -> bool:
    """A source is named `<deployment>.<pack source>` (`gp.docsite`); `docsite` matches it."""
    return bool(source) and any(source == w or source.endswith("." + w) for w in wanted)


#: hyphens and dashes a model may write between words ("low-level", "low‑level")
_DASHES = re.compile(r"[-‐‑‒–—]")


def _plain(text: str) -> str:
    """For `must_contain`: case, hyphens between words and runs of spaces do not count."""
    return " ".join(_DASHES.sub(" ", text.lower()).split())


def unresolved(question: Question, answer: str, seen: Mapping[str, NodeInfo]) -> list[str]:
    """Cited ids whose subtype or source the question checks but no tool result said.

    A read tool leaves them out when the model selects only some fields
    (`filter` with `select: [node_id, attrs]`), so the runner looks them up. Only
    nodes a tool returned as a result qualify, never an id the model saw only inside
    another record's text, and none at all once the check is already met.
    """
    cited = cited_nodes(answer, seen)
    need_subtype = bool(question.subtypes) and not any(n.subtype in question.subtypes for n in cited)
    need_source = bool(question.sources) and not any(_from_source(n.source, question.sources) for n in cited)
    return [
        n.id for n in cited
        if n.listed and ((need_subtype and n.subtype is None) or (need_source and n.source is None))
    ]


def check_answer(question: Question, answer: str, seen: Mapping[str, NodeInfo]) -> Verdict:
    cited = cited_nodes(answer, seen)
    ids = {n.id for n in cited}
    reasons: list[str] = []
    missing = [i for i in question.ids if i not in ids]
    if missing:
        reasons.append(f"did not cite required ids: {', '.join(missing)}")
    if question.any_id and not ids & set(question.any_id):
        reasons.append(f"cited none of: {', '.join(question.any_id)}")
    if question.subtypes and not any(n.subtype in question.subtypes for n in cited):
        reasons.append(f"cited no node of subtype {' or '.join(question.subtypes)}")
    if question.sources and not any(_from_source(n.source, question.sources) for n in cited):
        reasons.append(f"cited no node from source {' or '.join(question.sources)}")
    low = _plain(answer)
    absent = [s for s in question.contains if _plain(s) not in low]
    if absent:
        reasons.append(f"answer lacks: {', '.join(absent)}")
    if question.contains_any and not any(_plain(s) in low for s in question.contains_any):
        reasons.append(f"answer has none of: {', '.join(question.contains_any)}")
    return Verdict(passed=not reasons, cited=cited, reasons=reasons)
