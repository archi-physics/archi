"""Link documents to the physics entities they name.

The connective layer. It emits 180 entity nodes (`physics_technique`,
`physics_concept`, `physics_topic`) and a `references` edge from every
TWiki page and analysis note whose text names one.

**No model is involved.** An edge exists because a declared alias
matched verbatim in the document's own text, so the evidence for any
edge is a span you can point at. Two notes become reachable from each
other because both name `b tagging`, not because anything inferred a
relationship between them.

WHY THIS RE-READS THE CORPUS. It drives the real `TwikiEOSSource` and
`AnalysisNoteSource` rather than reading the graph or a cache. That
costs a second pass over both mirrors, and buys the one property that
matters: the page set, the note set and the node ids are the SAME ones
those sources publish, by construction rather than by agreement. A
cached page list drifts, and a list that names a document this run did
not produce fails admission for the WHOLE emission, not just the bad
edge.

WHY NOT A CROSS-LINK RULE. okg's `DeclarativeLinker` has a
`word_boundary` matcher that would express this declaratively, but it
runs inside the publish transaction as O(chunks x aliases). On a corpus
of hundreds of thousands of chunks that is far too long to hold the
publisher's advisory lock. Matching at ingest, where the text is already
in hand, keeps the publish transaction short.

EDGES ARE AT DOCUMENT LEVEL, not chunk level. "Which analyses use
b-tagging" is a question about notes; one edge per (document, entity)
keeps the answer legible and the edge count bounded at
documents x entities rather than chunks x entities.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterator

from okg.deployment import (
    ConnectorHealth,
    ConnectorRun,
    ContentHashProbe,
    EdgeFact,
    NodeFact,
    PreflightResult,
)

from archi.sources._physics_entities_vocabulary import (
    ALL,
    SUBTYPE_BY_KIND,
    entity_node_id,
)
from archi.sources._run_modes import COMPLETED_SCOPE_RUN_MODES
from archi.sources._sdk_adapter import ReaderAdapter


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


class PhysicsEntitySource:
    """Emit the entity vocabulary and link documents that name it."""

    name = "physics_entities"
    profile = "reference_catalog"
    change_probe_kind = "content_hash"

    def __init__(
        self,
        *,
        eos_root: str | None = None,
        web_root: str = "CMS",
        an_root: str | None = None,
        physics_filter: bool = True,
        max_notes: int | None = None,
        max_files: int | None = None,
        required: bool = True,
        match_cache_path: str | None = None,
        base: str | None = None,
    ) -> None:
        self.eos_root = eos_root
        self.web_root = web_root
        self.an_root = an_root
        if not isinstance(physics_filter, bool):
            raise ValueError(
                f"physics_filter must be true or false, got {physics_filter!r}"
            )
        self.physics_filter = physics_filter
        self.max_notes = max_notes
        self.max_files = max_files
        self.required = required
        # Where to remember which entities each document matched. Opt-in:
        # unset, the source reads and writes nothing. Give an absolute
        # path outside the deployment directory, which `okg install
        # --force` deletes.
        self.match_cache_path = match_cache_path
        self.base = base
        self.last_link_report: dict[str, Any] | None = None
        self.change_probe = ContentHashProbe(
            content_items=self._probe_content_items,
            config={
                "entities": len(ALL),
                "eos_root": self.eos_root,
                "an_root": self.an_root,
                "physics_filter": self.physics_filter,
            },
            emit_targets=PhysicsEntitySource,
        )

    # -- identity -----------------------------------------------------

    def _vocabulary_payload(self) -> bytes:
        parts = [
            f"{kind}:{code}:{label}:{'|'.join(aliases)}"
            for kind, code, label, aliases in ALL
        ]
        return ";".join(parts).encode("utf-8")

    def _probe_content_items(self) -> list[tuple[str, Any]]:
        # The vocabulary IS part of this source's identity: change an
        # alias and the edges it produces change, so a cached run must
        # not be reused.
        return [("vocabulary", self._vocabulary_payload())]

    # -- matching -----------------------------------------------------

    @classmethod
    def _compiled(cls) -> list[tuple[str, str, re.Pattern[str]]]:
        """One alternation per entity, not one regex per alias.

        Looping each alias separately is far slower over a large corpus.
        Alternation plus a single pass is what makes a second corpus read
        affordable.
        """
        out = []
        for kind, code, _label, aliases in ALL:
            out.append(
                (kind, code, re.compile("|".join(f"(?:{a})" for a in aliases), re.I))
            )
        return out

    # A LITERAL PRE-FILTER WAS TRIED AND REMOVED. Deriving a required
    # lowercase substring per entity and skipping the regex when absent
    # was faster and lost edges: aliases like
    # `\bH ?\\?(to|rightarrow|->) ?\\?tau` have no literal run that is
    # genuinely required, and an over-eager hint drops edges SILENTLY. A
    # single combined alternation over all 180 entities was also tried:
    # `finditer` reports one match per position, so `electron ID` yields
    # the technique and loses the `electron` concept. Both are the
    # failure this layer exists to avoid -- a missing edge is invisible,
    # unlike a slow ingest.
    #
    # Worth revisiting with a real multi-pattern engine (Aho-Corasick
    # over literal anchors, hyperscan); not worth guessing at.

    # -- the match cache ----------------------------------------------
    #
    # Matching is a pure function of (document text, vocabulary): 180
    # compiled alternations, no model, no randomness. So remembering the
    # answer keyed on both is EXACT, not an approximation -- a cached
    # run emits the same edges a cold one does. What it buys is the
    # matcher's time on a corpus where, nightly, almost nothing changed.

    #: Bumped when the on-disk shape changes, so an old file is ignored
    #: rather than misread.
    CACHE_FORMAT = 1

    def _load_match_cache(self, vocab_hash: str) -> dict[str, dict[str, Any]]:
        """Remembered matches, or empty if they cannot be trusted.

        Empty is always a safe answer: it costs a full re-match and
        produces identical output. So every failure mode here -- absent,
        truncated, malformed, written under a different vocabulary or a
        different format -- degrades to re-matching rather than to an
        ingest that fails or, worse, to stale edges.
        """
        if not self.match_cache_path:
            return {}
        path = Path(self.match_cache_path).expanduser()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if not isinstance(payload, dict):
            return {}
        if payload.get("format") != self.CACHE_FORMAT:
            return {}
        # The vocabulary IS half the key. One changed alias can change
        # every document's answer, and a stale match would drop or
        # invent edges silently -- the one failure this layer exists to
        # avoid.
        if payload.get("vocabulary") != vocab_hash:
            return {}
        documents = payload.get("documents")
        return documents if isinstance(documents, dict) else {}

    def _corpus_present(self) -> bool:
        """Are the mirrors this source scans actually mounted?

        `completed_scope` is decided EAGERLY, as an argument to
        ConnectorRun, while the facts generator is lazy -- so the claim
        cannot depend on how many documents were matched. The honest
        thing it can check up front is whether there is a corpus to
        match at all. With `deletion_semantics:
        missing_from_completed_scope`, a complete scope over an unmounted
        mirror is a retraction order for every document-to-entity edge.
        """
        roots = [r for r in (self.eos_root, self.an_root) if r]
        return bool(roots) and all(
            Path(r).expanduser().is_dir() for r in roots
        )

    def _write_match_cache(self, vocab_hash: str,
                           documents: dict[str, dict[str, Any]]) -> None:
        """Replace the cache with what THIS run saw.

        Rewritten rather than merged, so a document the corpus no longer
        produces leaves no entry behind: the file cannot grow without
        bound, and a deleted page cannot be resurrected by a later run
        that trusts it.
        """
        if not self.match_cache_path:
            return
        path = Path(self.match_cache_path).expanduser()
        payload = {
            "format": self.CACHE_FORMAT,
            "vocabulary": vocab_hash,
            "documents": documents,
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            os.replace(tmp, path)
        except OSError:
            # A cache that cannot be written is a slow next run, not a
            # failed ingest. The facts are already correct.
            pass

    #: Documents-reached bands. An entity above `BROAD_AT` is named by so
    #: much of the corpus that it cannot narrow a search on its own; one
    #: below `RARE_BELOW` identifies a handful of analyses. Thresholds are
    #: round numbers sized for a corpus of thousands of notes and pages,
    #: not tuned.
    BROAD_AT = 1000
    RARE_BELOW = 150

    @classmethod
    def _band(cls, reached: int) -> str:
        if reached >= cls.BROAD_AT:
            return "broad"
        if reached <= cls.RARE_BELOW:
            return "rare"
        return "useful"

    def _entity_nodes(
        self,
        revision: dict[str, Any],
        reach: dict[tuple[str, str], dict[str, int]] | None = None,
    ) -> Iterator[Any]:
        reach = reach or {}
        for kind, code, label, aliases in ALL:
            r = reach.get((kind, code), {"notes": 0, "pages": 0})
            documents = r["notes"] + r["pages"]
            yield NodeFact(
                node_id=entity_node_id(kind, code),
                subtype=SUBTYPE_BY_KIND[kind],
                attrs={
                    "code": code,
                    "label": label,
                    "name": label,
                    "entity_kind": kind,
                    "alias_count": len(aliases),
                    # Selectivity, measured in THIS run rather than
                    # carried from a survey that can go stale.
                    "notes_reached": r["notes"],
                    "pages_reached": r["pages"],
                    "documents_reached": documents,
                    "selectivity": self._band(documents),
                    "text": f"{label} ({kind})",
                },
                source_record_id={"entity": f"{kind}:{code}"},
                source_revision=revision,
            )

    # -- the documents ------------------------------------------------

    def _twiki_documents(self) -> Iterator[tuple[str, str]]:
        """(documentation_page node id, text) for every kept topic."""
        if not self.eos_root:
            return
        from archi.sources.twiki import TwikiEOSSource

        source = TwikiEOSSource(
            eos_root=self.eos_root,
            web_root=self.web_root,
            physics_filter=self.physics_filter,
            max_files=self.max_files,
            base=self.base,
        )
        current: str | None = None
        buf: list[str] = []
        for fact in source.run("physics-entities", mode="scope_complete").facts:
            subtype = getattr(fact, "subtype", None)
            if subtype == "documentation_page":
                if current is not None:
                    yield current, "\n".join(buf)
                current, buf = fact.node_id, [str(fact.attrs.get("title") or "")]
            elif subtype == "document_chunk" and current is not None:
                buf.append(str(fact.attrs.get("text") or ""))
        if current is not None:
            yield current, "\n".join(buf)

    def _note_documents(self) -> Iterator[tuple[str, str]]:
        """(analysis_note node id, text) for every note."""
        if not self.an_root:
            return
        from archi.sources.analysis_notes import AnalysisNoteSource

        source = AnalysisNoteSource(an_root=self.an_root, max_notes=self.max_notes)
        current: str | None = None
        buf: list[str] = []
        for fact in source.run("physics-entities", mode="scope_complete").facts:
            subtype = getattr(fact, "subtype", None)
            if subtype == "analysis_note":
                if current is not None:
                    yield current, "\n".join(buf)
                current, buf = fact.node_id, [str(fact.attrs.get("title") or "")]
            elif subtype == "section" and current is not None:
                buf.append(str(fact.attrs.get("body") or ""))
        if current is not None:
            yield current, "\n".join(buf)

    # -- the run ------------------------------------------------------

    def preflight(self, mode: str = "live") -> PreflightResult:
        return PreflightResult(
            source_name=self.name,
            status="ok",
            mode="filesystem",
            required=self.required,
            record_count=len(ALL),
            reason="physics entity vocabulary is declared in code",
        )

    def run(self, run_id: str, *, mode: str = "cursor") -> ConnectorRun:
        revision = {
            "run_id": run_id,
            "content_hash": _sha256(self._vocabulary_payload().decode("utf-8")),
            "n_records": len(ALL),
        }
        patterns = self._compiled()
        report = {"pages": 0, "notes": 0, "edges": 0,
                  "matched": 0, "reused": 0}
        vocab_hash = revision["content_hash"]
        cached = self._load_match_cache(vocab_hash)
        fresh: dict[str, dict[str, Any]] = {}

        def _facts() -> Iterator[Any]:
            # SCAN FIRST, then emit. The entity nodes carry how many
            # documents reach them, and that is not knowable until the
            # scan is done -- so the matches are collected as
            # (document, [codes]) before anything is yielded.
            #
            # Why the counts are on the node at all: a generic entity
            # such as `jet` is named by most notes, while `stop_squark`
            # is named by a handful. A query that ORs entities together
            # therefore selects almost the whole corpus. A ranker needs
            # to know which entities are worth trusting, and the source
            # is the only thing that can count cheaply.
            #
            # Memory: (document id, codes) per document, not the edges
            # themselves.
            matched: list[tuple[str, str, list[tuple[str, str]]]] = []
            reach: dict[tuple[str, str], dict[str, int]] = {}
            for src_subtype, documents in (
                ("documentation_page", self._twiki_documents()),
                ("analysis_note", self._note_documents()),
            ):
                bucket = "pages" if src_subtype == "documentation_page" else "notes"
                for node_id, text in documents:
                    report[bucket] += 1
                    if not text:
                        continue
                    text_hash = _sha256(text)
                    remembered = cached.get(node_id)
                    if remembered is not None and remembered.get("h") == text_hash:
                        # Same text, same vocabulary: the matcher is a
                        # pure function of the two, so the answer cannot
                        # have changed. This is the 70 minutes.
                        hits = [tuple(pair) for pair in remembered.get("e", ())]
                        report["reused"] += 1
                    else:
                        hits = [
                            (kind, code)
                            for kind, code, pattern in patterns
                            if pattern.search(text)
                        ]
                        report["matched"] += 1
                    # Written for every document seen, including ones
                    # that match nothing -- otherwise a document with no
                    # entities is re-matched on every run forever.
                    fresh[node_id] = {"h": text_hash, "e": [list(h) for h in hits]}
                    if not hits:
                        continue
                    matched.append((src_subtype, node_id, hits))
                    for key in hits:
                        r = reach.setdefault(key, {"notes": 0, "pages": 0})
                        r[bucket] += 1
                    report["edges"] += len(hits)

            yield from self._entity_nodes(revision, reach)
            for _src_subtype, node_id, hits in matched:
                for kind, code in hits:
                    yield EdgeFact(
                        src=node_id,
                        dst=entity_node_id(kind, code),
                        edge_type="references",
                        provenance="derived_deterministic",
                        attrs={"kind": "names_entity", "entity_kind": kind},
                        source_record_id={"document": node_id, "entity": code},
                        source_revision=revision,
                    )
            # Only after the whole scan: a generator abandoned part way
            # through has seen part of the corpus, and writing that would
            # prune every document it never reached.
            self._write_match_cache(vocab_hash, fresh)
            self.last_link_report = dict(report)

        return ConnectorRun(
            facts=_facts(),
            # okg decides a source's run mode from its source_class, and
            # a `reference_catalog` gets `release_new` (or
            # `release_unchanged`) -- never `scope_complete`. Deciding
            # the claim from a hand-written set meant this source ran its
            # whole corpus and then declared the scope INCOMPLETE, and
            # admission refused the emission as `partial` on every run
            # after the first. okg's own table is the only
            # thing that knows which modes complete a scope;
            # archi.sources._run_modes mirrors it because a connector may
            # not import okg.substrate (okg lint DEPLOYMENT-001), and
            # archi's test_run_modes asserts the two agree at each pin.
            completed_scope=(
                mode in COMPLETED_SCOPE_RUN_MODES and self._corpus_present()
            ),
            run_mode=mode,
            health=ConnectorHealth(
                status="ok",
                mode="filesystem",
                record_count=len(ALL),
                content_hash=revision["content_hash"],
                reason="CMS physics entity vocabulary and document links",
            ),
        )


class PhysicsEntityAdapter(ReaderAdapter):
    """What ``bundles/packs/cms-analyses/sources/physics_entities.yaml`` names.

    ``profile`` and ``change_probe_kind`` are string LITERALS: the
    substrate reads the first with ``inspect.getattr_static`` and the
    second by parsing this module's AST, both before importing anything.
    """

    reader_class = PhysicsEntitySource
    profile = "reference_catalog"
    change_probe_kind = "content_hash"
