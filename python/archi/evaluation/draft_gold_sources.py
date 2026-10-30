"""Draft `gold_sources:` for a question set that names its evidence in prose, not node ids.

A benchmark such as CompOps-42 gives each question `atoms` (each with an `answer_source`: ticket,
source_code, doc, repo_config, git_history, live_rucio, live_dbs, monitoring, or a mix such as
"doc or ticket"), a free-text `sources:` list and `verification.evidence` prose. This reads
those and drafts, per question, the structured references the agent arms need:

    a Jira key (ABC-123, jira:ABC-123, .../browse/ABC-123)   -> jira, natural key ABC-123
    repos/<repo>[ @<sha>] <path>, or repos/<repo>/<path>      -> git_repo, key <repo>/<path>
    a repository document named loosely: <repo>/<x>.md,       -> git_repo, resolved against the
      <org>/<repo> <path>.md, <repo> Documentation: <a_b>.md,    checkouts given with --repos
      documentation_page:<repo>:<a_b>.md
    a documentation URL                                      -> docsite, key the URL
    a CMS site name (T0_ to T3_)                             -> cric, key the site name

    python -m archi.evaluation.draft_gold_sources --questions <dir of golden YAML> --out <file> \
        [--repos <dir of git checkouts>]

A loosely named document gives its repository and file only roughly: the repository name
differs from the checkout's (cmsdmops for cmsdmops-Documentation), folders are left out, or
the path is flattened ('_' or '-' for '/' and ' '). Documents are .md, .rst, .txt or
.markdown files. With --repos, each one is resolved to the one checkout file it names
(`high`); one that matches no file or several, and every one when --repos is not given, is
listed under `unresolved` and gives no reference.

The output is a DRAFT for a person to review: every reference carries a confidence note, and
whatever could not be matched is listed under `unresolved`. Atoms get the references of their
file-backed kinds; an atom answered only by a live system gets none and is marked so (the
agent arms count it as unavailable).

The question set and this draft are private to their instance. The draft is written beside
the set and is never committed to this repository: the command refuses an `--out` inside it.
It reads files only: no model, no network.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import yaml

from archi.evaluation.context import LIVE_SOURCES, Atom

#: answer_source kind -> the reference kinds that can carry it
KIND_REFS = {
    "ticket": ("jira",),
    "source_code": ("git",),
    "repo_config": ("git",),
    "git_history": ("git",),
    "doc": ("doc",),
}
SOURCE_FOR = {"jira": "jira", "git": "git_repo", "doc": "docsite", "site": "cric"}

_JIRA = re.compile(r"(?<![A-Za-z0-9_/-])(?:jira:)?([A-Z][A-Z0-9_]{1,15}-\d{1,7})(?![-\w])")
_JIRA_URL = re.compile(r"/browse/([A-Z][A-Z0-9_]{1,15}-\d{1,7})\b")
_REPO_SHA_PATH = re.compile(r"\brepos/([\w.-]+)(?:\s*\([^)]*\))*(?:\s*@\s*([0-9a-f]{7,40}))?(?:\s*\([^)]*\))*"
                            r"\s+([\w.][\w./-]*\.\w+|[\w.-]+/[\w./-]+)(?:\s*\(([^)]*)\))?")
_REPO_PATH = re.compile(r"\brepos/([\w.-]+)/([\w./-]+\.\w+)")
_URL = re.compile(r"https?://[^\s)'\"<>\]]+")
_SITE = re.compile(r"(?<![A-Za-z0-9])(T[0-3]_[A-Z]{2}_[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*)(?![A-Za-z0-9])")
#: a document file name, shortest first so that "a.md vs b.md" stops at a.md
_DOC_FILE = r"[\w-]+(?:\.[\w-]+)*?\.(?:md|rst|txt|markdown)(?![\w-]|\.\w)"
_NOT_AFTER = r"(?<![\w/.:@-])"
#: loosely named repository documents, most specific first: (form, pattern of repo and path).
#: Two words name a repository only as <org>/<repo> or as "<repo> Documentation", the forms the
#: CompOps-42 key uses; any other pair of words is prose.
_LOOSE_DOCS = (
    ("documentation_page:<repo>:<file>", re.compile(r"documentation_page:([\w.-]+):(" + _DOC_FILE + ")")),
    ("<org>/<repo> <path>", re.compile(_NOT_AFTER + r"([\w.-]+(?:/[\w.-]+| [Dd]ocumentation)):?\s+"
                                       r"((?:[\w.-]+/)+[\w-]+(?:[ .][\w-]+)*?\.(?:md|rst|txt|markdown)"
                                       r"(?![\w-]|\.\w))")),
    ("<repo> Documentation: <file>", re.compile(_NOT_AFTER + r"([\w.-]+[ _][Dd]ocumentation):\s+(" + _DOC_FILE + ")")),
    ("<repo>/<path>", re.compile(_NOT_AFTER + r"([\w.-]+)/((?:[\w.-]+/)*" + _DOC_FILE + ")")),
)
#: hosts whose URLs are source repositories, not documentation pages
_CODE_HOSTS = ("github.com", "gitlab.")


class Ref(dict):
    """{gold id, kind, source, key, confidence, where}."""


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _flat(path: str) -> str:
    return re.sub(r"[/ _-]+", "_", path.lower())


def index_checkouts(root: Path) -> dict[str, list[str]]:
    """{checkout name: its file paths, relative and '/'-separated} for each git checkout under root."""
    out: dict[str, list[str]] = {}
    for checkout in sorted(p for p in root.iterdir() if p.is_dir() and (p / ".git").exists()):
        files = []
        for folder, dirs, names in os.walk(checkout):
            dirs[:] = [d for d in dirs if d != ".git"]
            files += [Path(folder, n).relative_to(checkout).as_posix() for n in names]
        out[checkout.name] = sorted(files)
    return out


def _matching_files(names: Iterable[str], path: str, checkouts: Mapping[str, list[str]]) -> list[str]:
    """Files of these checkouts that are `path`: first as written (the whole path or a `/path`
    suffix), and only if none is, once '/', ' ', '_' and '-' all read as one separator, from
    a folder boundary of the file's path (so policies.md never matches dm_policies.md)."""
    exact = [f"{n}/{rel}" for n in names for rel in checkouts[n] if rel == path or rel.endswith("/" + path)]
    if exact:
        return exact
    flat = _flat(path)
    return [f"{n}/{rel}" for n in names for rel in checkouts[n]
            if any(_flat(rel[i:]) == flat for i in [0] + [j + 1 for j, c in enumerate(rel) if c == "/"])]


def resolve_document(repo: str, path: str, checkouts: Mapping[str, list[str]]) -> list[str]:
    """The `<checkout>/<file>` keys a loosely named document can be.

    Names are compared without case or punctuation, and a name under four characters matches
    nothing. Checkouts are tried in rank: the same name, then one name a prefix of the other,
    then one inside the other; the first rank with a matching file wins (_matching_files).
    """
    want = _norm(repo)
    if len(want) < 4:
        return []
    ranks: tuple[list[str], list[str], list[str]] = ([], [], [])
    for name in checkouts:
        have = _norm(name)
        if len(have) < 4:
            continue
        if have == want:
            ranks[0].append(name)
        elif have.startswith(want) or want.startswith(have):
            ranks[1].append(name)
        elif want in have or have in want:
            ranks[2].append(name)
    for names in ranks:
        hits = _matching_files(names, path, checkouts) if names else []
        if hits:
            return hits
    return []


def _texts(item: Mapping[str, Any]) -> list[tuple[str, str]]:
    """(where, text) pairs to scan: each `sources:` entry, then the verification evidence."""
    out = []
    for i, entry in enumerate(item.get("sources") or []):
        out.append((f"sources[{i}]", entry if isinstance(entry, str) else yaml.safe_dump(entry)))
    verification = item.get("verification") or {}
    evidence = verification.get("evidence") if isinstance(verification, Mapping) else None
    if isinstance(evidence, str):
        out.append(("verification.evidence", evidence))
    elif isinstance(evidence, list):
        out += [(f"verification.evidence[{i}]", str(e)) for i, e in enumerate(evidence)]
    return out


def extract(where: str, text: str, checkouts: Optional[Mapping[str, list[str]]] = None,
            problems: Optional[list[str]] = None) -> tuple[list[Ref], bool]:
    """The references in one text, and whether anything in it was recognised.

    A loosely named document gives a reference only when `checkouts` (from index_checkouts)
    resolve it to exactly one file. Otherwise it is recognised but gives none, and why goes
    to `problems`. It is matched last, so a Jira key or site name next to it is kept.
    """
    recognised = False
    refs: list[Ref] = []
    taken: list[tuple[int, int]] = []

    def free(span: tuple[int, int]) -> bool:
        return not any(a < span[1] and span[0] < b for a, b in taken)

    for m in _REPO_SHA_PATH.finditer(text):
        repo, sha, path, detail = m.group(1), m.group(2), m.group(3).rstrip("."), m.group(4)
        note = "high: repos/<repo> <path>" + (f" at {sha}" if sha else "") + (f" ({detail})" if detail else "")
        if sha:
            note += "; the corpus holds the checkout's HEAD, which may differ from this commit"
        refs.append(Ref(id=f"git_repo:{repo}/{path}", kind="git", source="git_repo", key=f"{repo}/{path}",
                        confidence=note, where=where))
        taken.append(m.span())
    for m in _REPO_PATH.finditer(text):
        if free(m.span()):
            refs.append(Ref(id=f"git_repo:{m.group(1)}/{m.group(2)}", kind="git", source="git_repo",
                            key=f"{m.group(1)}/{m.group(2)}", confidence="high: repos/<repo>/<path>", where=where))
            taken.append(m.span())
    for m in _URL.finditer(text):
        url = m.group(0).rstrip(".,;:")
        jira = _JIRA_URL.search(url)
        if jira:
            refs.append(Ref(id=f"jira:{jira.group(1)}", kind="jira", source="jira", key=jira.group(1),
                            confidence="high: a Jira browse URL", where=where))
        elif any(h in url for h in _CODE_HOSTS):
            taken.append(m.span())
            continue  # a repository URL: not a docsite page; the repos/ form names the file
        else:
            refs.append(Ref(id=f"docsite:{url}", kind="doc", source="docsite", key=url,
                            confidence="medium: a URL; check it is a page the docsite cache holds", where=where))
        taken.append(m.span())
    for m in _JIRA.finditer(text):
        if free(m.span()):
            prefixed = text[m.start():m.start() + 5] == "jira:"
            refs.append(Ref(id=f"jira:{m.group(1)}", kind="jira", source="jira", key=m.group(1),
                            confidence="high: jira:<key>" if prefixed else "medium: looks like a Jira key",
                            where=where))
            taken.append(m.span())
    for m in _SITE.finditer(text):
        if free(m.span()):
            refs.append(Ref(id=f"site:{m.group(1)}", kind="site", source="cric", key=m.group(1),
                            confidence="medium: a CMS site name; attach it to an atom by hand if it is evidence",
                            where=where))
            taken.append(m.span())
    for form, pattern in _LOOSE_DOCS:
        for m in pattern.finditer(text):
            if not free(m.span()):
                continue
            taken.append(m.span())
            recognised = True
            repo, path, said = m.group(1), m.group(2), m.group(0)
            hits = resolve_document(repo, path, checkouts) if checkouts is not None else []
            if len(hits) == 1:
                refs.append(Ref(id=f"git_repo:{hits[0]}", kind="git", source="git_repo", key=hits[0],
                                confidence=f"high: {said!r} ({form}) is the one matching checkout file",
                                where=where))
            elif problems is not None:
                if checkouts is None:
                    why = f"names a repository document ({form}); draft again with --repos to resolve it"
                elif hits:
                    why = f"matches {len(hits)} checkout files: {', '.join(hits[:5])}{', ...' if len(hits) > 5 else ''}"
                else:
                    why = f"matches no file in a checkout named like {repo!r}"
                problems.append(f"{where}: {said!r} {why}")
    return refs, recognised or bool(refs)


def draft_item(qid: str, item: Mapping[str, Any],
               checkouts: Optional[Mapping[str, list[str]]] = None) -> dict[str, Any]:
    refs: dict[str, Ref] = {}
    unresolved: list[str] = []
    for where, text in _texts(item):
        found, any_found = extract(where, text, checkouts, unresolved)
        for ref in found:
            refs.setdefault(ref["id"], ref)
        if not any_found and where.startswith("sources["):
            unresolved.append(f"{where}: no reference recognised")
    atoms_out = []
    for i, raw in enumerate(item.get("atoms") or []):
        if not isinstance(raw, Mapping):
            unresolved.append(f"atoms[{i}]: not a mapping")
            continue
        atom = Atom(str(raw.get("id") or f"a{i + 1}"), str(raw.get("answer_source") or ""))
        kinds = atom.kinds
        wanted = {k for kind in kinds for k in KIND_REFS.get(kind, ())}
        unknown = [k for k in kinds if k not in KIND_REFS and k not in LIVE_SOURCES]
        gold = [rid for rid, ref in refs.items() if ref["kind"] in wanted]
        entry: dict[str, Any] = {"id": atom.id, "answer_source": atom.answer_source, "gold": gold}
        if "required" in raw:
            entry["required"] = bool(raw["required"])
        atoms_out.append(entry)
        if atom.live_only:
            entry["note"] = "answered only by a live system: unavailable in both agent arms"
        elif not gold:
            unresolved.append(f"atom {atom.id} ({atom.answer_source or 'no answer_source'}): no "
                              f"{' or '.join(sorted(wanted)) or 'file'} reference found in sources or evidence")
        if unknown:
            unresolved.append(f"atom {atom.id}: answer_source kind(s) {unknown} are not known; check them")
    return {
        "id": qid,
        "gold": list(refs),
        "gold_sources": {rid: {"source": ref["source"], "key": ref["key"]} for rid, ref in refs.items()},
        "atoms": atoms_out,
        "notes": {rid: f"{ref['confidence']} (from {ref['where']})" for rid, ref in refs.items()},
        "unresolved": unresolved,
    }


def _items(directory: Path) -> Iterable[tuple[str, Mapping[str, Any]]]:
    for path in sorted(list(directory.rglob("*.yaml")) + list(directory.rglob("*.yml"))):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        items = data if isinstance(data, list) else [data]
        for i, item in enumerate(items):
            if isinstance(item, Mapping):
                qid = str(item.get("id") or (path.stem if len(items) == 1 else f"{path.stem}-{i + 1}"))
                yield qid, item


HEADER = """\
# DRAFT gold_sources, written by `python -m archi.evaluation.draft_gold_sources`.
# PRIVATE: keep this beside the question set it was drafted from. Never commit it to the
# archi repository. A person reviews every entry (the notes say how sure each one is, and
# `unresolved` lists what needs a hand) before a run uses it.
#
# Gold ids are labels: jira:<key> and site:<name> are also the graph's node ids; git_repo:,
# docsite: ids are not, so the no-model arms cannot find those until a reviewer replaces them
# with node ids. The agent arms find every one through gold_sources.
"""


def _inside_repository(path: Path) -> bool:
    repo = Path(__file__).resolve().parents[3]
    try:
        path.resolve().relative_to(repo)
    except ValueError:
        return False
    return (repo / ".git").exists() or (repo / "python" / "archi").is_dir()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m archi.evaluation.draft_gold_sources",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("--questions", type=Path, required=True, help="a directory of golden question YAML files")
    parser.add_argument("--out", type=Path, required=True, help="the draft to write (outside this repository)")
    parser.add_argument("--repos", type=Path, help="a directory of git checkouts, one per repository, to resolve "
                                                   "loosely named documents against")
    args = parser.parse_args(argv)
    for folder in (args.questions, args.repos):
        if folder is not None and not folder.is_dir():
            print(f"error: {folder} is not a directory", file=sys.stderr)
            return 2
    if _inside_repository(args.out):
        print(f"error: {args.out} is inside the archi repository; the draft is private and is never committed. "
              "Write it beside the question set", file=sys.stderr)
        return 2
    try:
        checkouts = index_checkouts(args.repos) if args.repos else None
        drafted = [draft_item(qid, item, checkouts) for qid, item in _items(args.questions)]
    except (OSError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not drafted:
        print(f"error: no question YAML under {args.questions}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(HEADER + yaml.safe_dump({"questions": drafted}, sort_keys=False, allow_unicode=True),
                        encoding="utf-8")
    refs = sum(len(d["gold"]) for d in drafted)
    open_items = sum(len(d["unresolved"]) for d in drafted)
    print(f"drafted {refs} references for {len(drafted)} questions; {open_items} unresolved items for review; "
          f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
