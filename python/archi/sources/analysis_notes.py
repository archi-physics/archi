"""AnalysisNoteSource -- CMS analysis notes as graph facts.

Reads a local mirror of ``gitlab.cern.ch/tdr/notes``: a directory of
``<PAG>-YY-NNN/`` and legacy ``AN-YY-NNN/`` LaTeX trees. The pack README
(``bundles/packs/analysis-notes/README.md``) says how to clone it; the
clone needs CERN credentials, and this reader never touches the network.

The parsing half lives in ``_an_parse``; this module is the connector
shape around it: ``preflight`` / ``run`` returning a ``ConnectorRun`` of
``NodeFact`` / ``EdgeFact``.

Emits, per accepted directory:

* one ``analysis_note`` node,
* one ``section`` node per section .tex, with ``contains`` from the note,
* one ``reference`` node per bib entry, deduplicated across notes by key,
  with ``references`` from the note,
* ``references`` to the note's PAG hub (``physics_analysis_group``),
  which this connector also mints, and
* the bib->TWiki bridge: ``references`` from a note to a
  ``documentation_page`` when a bib entry's URL names a TWiki topic that
  the TWiki connector actually ingested.

The bridge is the one cross-source edge, and it is only as good as its
knowledge of which pages exist. A connector under this SDK cannot read
the graph mid-build, and guessing would emit edges to nodes that are not
there. So the target set is either a file of node ids (``twiki_pages_path``)
or, in the pack, computed by reading the same TWiki dump with the same
reader and filter the twiki source uses (``twiki_eos_root``,
``twiki_web_root``, ``twiki_physics_filter``), so the list describes the
pages that source emits. With neither, the bridge emits nothing and says so
in the run's counters rather than emitting dangling edges. A page named in
the list that the twiki source did not produce fails admission for the
WHOLE emission, not just the bad edge, which is why a stale export is
worse than none.

``min_notes`` is the note-count floor: a mirror with fewer note directories
fails preflight and emits nothing, so a walker whose rules stop matching
cannot publish a fraction of the corpus.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

from okg.deployment import (
    ConnectorHealth,
    ConnectorRun,
    ContentHashProbe,
    EdgeFact,
    ImageRef,
    NodeFact,
    PreflightResult,
)

from archi.sources._an_parse import (
    AnalysisNoteRecord,
    FigureRecord,
    ReferenceRecord,
    SectionRecord,
    bib_twiki_target,
    classify_an_directory,
    classify_section_filename,
    extract_title,
    figure_refs,
    find_bib_file,
    find_figures,
    graphics_paths,
    included_tex_files,
    pag_from_dir,
    parse_bib_file,
    read_text,
    twiki_node_id,
)
from archi.sources._sdk_adapter import ReaderAdapter

DEFAULT_CHUNKER_NAME = "archi.an_section.v1"
#: Deliberately the same window archi's TWiki reader uses
#: (docs.py _CHUNK_SIZE / _CHUNK_OVERLAP). Search ranks chunks against
#: each other, so two layers windowed differently do not compete on
#: equal terms: a note emitted whole never surfaces against windowed
#: wiki text.
CHUNK_SIZE = 4000
CHUNK_OVERLAP = 200


def _chunks(text: str) -> Iterator[tuple[int, int, str]]:
    """``(index, char_offset, text)`` over a fixed overlapping window.

    ``char_offset`` is where ``text`` starts in the section text. Windows
    are stored unstripped, so ``text == section[char_offset:char_offset +
    len(text)]`` and the emitter can state the exact end.
    """
    if not text:
        return
    step = CHUNK_SIZE - CHUNK_OVERLAP
    index = 0
    for offset in range(0, len(text), step):
        window = text[offset:offset + CHUNK_SIZE]
        if not window.strip():
            continue
        yield index, offset, window
        index += 1
        if offset + CHUNK_SIZE >= len(text):
            break


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def an_node_id(an_id: str) -> str:
    return f"an:{an_id}"


def section_node_id(an_id: str, stem: str) -> str:
    return f"an_section:{an_id}:{stem}"


def figure_node_id(an_id: str, key: str) -> str:
    """``key`` is the figure's \\label, or ``<host>#<n>`` when it has none
    (or shares its label with an earlier figure of the same note)."""
    return f"an_figure:{an_id}:{key}"


def reference_node_id(bib_key: str) -> str:
    return f"an_ref:{_sha256(bib_key)[:16]}"


def pag_hub_id(code: str) -> str:
    return f"pag:{code}"


class AnalysisNoteSource:
    """Read CMS analysis-note LaTeX trees from a local mirror."""

    name = "analysis_notes"
    profile = "discovery_crawl"
    change_probe_kind = "content_hash"

    def __init__(
        self,
        *,
        an_root: str | None = None,
        required: bool = True,
        max_notes: int | None = None,
        twiki_pages_path: str | None = None,
        twiki_eos_root: str | None = None,
        twiki_web_root: str = "CMS",
        twiki_physics_filter: bool = False,
        min_notes: int = 1,
        chunker_name: str = DEFAULT_CHUNKER_NAME,
        records: list[AnalysisNoteRecord] | None = None,
        base: str | None = None,
    ) -> None:
        self.an_root = str(Path(an_root).expanduser()) if an_root else ""
        self.required = required
        if max_notes is not None and max_notes < 1:
            raise ValueError(
                f"max_notes must be >= 1 (or None for unlimited), "
                f"got {max_notes}"
            )
        self.max_notes = max_notes
        self.twiki_pages_path = twiki_pages_path
        if twiki_pages_path and twiki_eos_root:
            raise ValueError("give twiki_pages_path or twiki_eos_root, not both")
        if not isinstance(twiki_physics_filter, bool):
            raise ValueError(
                f"twiki_physics_filter must be true or false, got "
                f"{twiki_physics_filter!r}"
            )
        if isinstance(min_notes, bool) or not isinstance(min_notes, int) or min_notes < 0:
            raise ValueError(f"min_notes must be an integer >= 0, got {min_notes!r}")
        self.twiki_eos_root = (
            str(Path(twiki_eos_root).expanduser()) if twiki_eos_root else None
        )
        self.twiki_web_root = twiki_web_root
        self.twiki_physics_filter = twiki_physics_filter
        self.min_notes = min_notes
        self.chunker_name = chunker_name
        self._records = records
        self.base = base
        #: Counters from the last run, for the operator. This module
        #: logs nothing, matching archi.sources.
        self.last_walk_report: dict[str, Any] | None = None
        self.change_probe = ContentHashProbe(
            content_items=self._probe_content_items,
            config={
                "an_root": self.an_root,
                "max_notes": self.max_notes,
                "twiki_pages_path": self.twiki_pages_path,
                "twiki_eos_root": self.twiki_eos_root,
                "twiki_physics_filter": self.twiki_physics_filter,
                "min_notes": self.min_notes,
            },
            emit_targets=AnalysisNoteSource,
        )

    # ── probe ──────────────────────────────────────────────────────────

    def _probe_content_items(self) -> list[tuple[str, Any]]:
        """Every file the walk would read, so a content change is seen.

        Directory listings alone would miss an edit inside a .tex that
        leaves the tree shape identical -- the common case when an
        author revises a section.
        """
        if self._records is not None:
            return [
                (record.an_id, json.dumps(
                    {
                        "an": record.an_id,
                        "title": record.title,
                        "sections": [s.text for s in record.sections],
                        "refs": [r.bib_key for r in record.references],
                    },
                    sort_keys=True,
                ).encode("utf-8"))
                for record in self._records
            ]
        root = Path(self.an_root).expanduser() if self.an_root else None
        if root is None or not root.is_dir():
            return []
        items: list[tuple[str, Any]] = []
        for an_dir in self._note_dirs(root):
            main_tex = an_dir / f"{an_dir.name}.tex"
            section_paths = [
                path for path, _rel, _order in _section_files(an_dir, an_dir.name)
            ]
            for path in sorted(set(
                section_paths
                + ([main_tex] if main_tex.exists() else [])
                + sorted(an_dir.glob("*.bib"))
            )):
                items.append((path.relative_to(root).as_posix(), path))
        if self.twiki_pages_path:
            cache = Path(self.twiki_pages_path).expanduser()
            if cache.is_file():
                items.append(("twiki_pages", cache))
        if self.twiki_eos_root and Path(self.twiki_eos_root).is_dir():
            # The bridge depends on which pages the dump yields, so a topic
            # added, removed or edited must move the probe.
            listing = sorted(
                f"{p.relative_to(self.twiki_eos_root).as_posix()}\0{p.stat().st_size}"
                for p in Path(self.twiki_eos_root).rglob("*.txt")
            )
            items.append(("twiki_dump", "\n".join(listing).encode("utf-8")))
        return items

    # ── lifecycle ──────────────────────────────────────────────────────

    def preflight(self, mode: str = "live") -> PreflightResult:
        root = Path(self.an_root).expanduser() if self.an_root else None
        if root is None or not root.is_dir():
            return PreflightResult(
                source_name=self.name,
                status="cache_missing",
                mode="filesystem",
                required=self.required,
                cache_path=str(root) if root else None,
                reason=(
                    "analysis-note mirror is absent; set an_root to the "
                    "directory of <PAG>-YY-NNN trees (the pack README says "
                    "how to clone gitlab.cern.ch/tdr/notes)"
                ),
            )
        notes = self._note_dirs(root)
        if len(self._all_note_dirs(root)) < self.min_notes:
            return PreflightResult(
                source_name=self.name,
                status="cache_missing",
                mode="filesystem",
                required=self.required,
                record_count=len(notes),
                cache_path=str(root),
                reason=self._floor_reason(root),
            )
        return PreflightResult(
            source_name=self.name,
            status="ok",
            mode="filesystem",
            required=self.required,
            record_count=len(notes),
            cache_path=str(root),
            reason="analysis-note mirror read from local filesystem",
        )

    def run(self, run_id: str, *, mode: str = "cursor") -> ConnectorRun:
        if self._records is not None:
            records = list(self._records)
        else:
            root = Path(self.an_root).expanduser() if self.an_root else None
            if root is None or not root.is_dir():
                return ConnectorRun(
                    facts=[],
                    completed_scope=False,
                    run_mode=mode,
                    health=ConnectorHealth(
                        status="cache_missing",
                        mode="filesystem",
                        reason=(
                            f"analysis-note mirror absent at "
                            f"{root or '<unset>'}; no facts emitted"
                        ),
                    ),
                )
            if len(self._all_note_dirs(root)) < self.min_notes:
                return ConnectorRun(
                    facts=[],
                    completed_scope=False,
                    run_mode=mode,
                    health=ConnectorHealth(
                        status="cache_missing",
                        mode="filesystem",
                        reason=self._floor_reason(root) + "; no facts emitted",
                    ),
                )
            records = self._walk(root)

        twiki_pages = self._twiki_pages()
        content_hash = _records_hash(records)
        revision = {
            "run_id": run_id,
            "content_hash": content_hash,
            "n_records": len(records),
            "an_root": self.an_root,
        }
        report: dict[str, Any] = {
            "notes": len(records),
            "sections": sum(len(r.sections) for r in records),
            "sections_below_top_level": sum(
                "/" in s.filename for r in records for s in r.sections
            ),
            "references": sum(len(r.references) for r in records),
            "figures": 0,
            "figure_panels": 0,
            "figure_panels_missing": 0,
            "figure_refs": 0,
            "bridge_candidates": 0,
            "bridge_emitted": 0,
            "bridge_dropped_unknown_page": 0,
            "twiki_pages_known": len(twiki_pages),
        }

        def _facts() -> Iterator[Any]:
            yield from _facts_for_notes(
                records,
                revision,
                twiki_pages=twiki_pages,
                chunker_name=self.chunker_name,
                report=report,
            )

        self.last_walk_report = report
        # okg's ingest renders each figure's ``images`` against this root
        # and stores the PNGs; the values carry only ``<note dir>/<path>``.
        image_root = (
            Path(self.an_root).expanduser()
            if self.an_root else None
        )
        return ConnectorRun(
            image_root=(
                str(image_root) if image_root and image_root.is_dir() else None
            ),
            facts=_facts(),
            completed_scope=(mode in {"scope_complete", "reconcile"}),
            run_mode=mode,
            health=ConnectorHealth(
                status="ok",
                mode="filesystem" if self._records is None else "fixture",
                record_count=len(records),
                content_hash=content_hash,
                reason="analysis notes read from the local mirror",
            ),
        )

    # ── the walk ───────────────────────────────────────────────────────

    def _all_note_dirs(self, root: Path) -> list[Path]:
        return [
            path for path in sorted(root.iterdir())
            if path.is_dir() and classify_an_directory(path.name) is not None
        ]

    def _note_dirs(self, root: Path) -> list[Path]:
        dirs = self._all_note_dirs(root)
        return dirs[:self.max_notes] if self.max_notes is not None else dirs

    def _floor_reason(self, root: Path) -> str:
        return (
            f"analysis-note mirror at {root} holds "
            f"{len(self._all_note_dirs(root))} note directories, below the "
            f"floor of {self.min_notes} (min_notes); the mirror is partial "
            f"or the directory rules stopped matching"
        )

    def _walk(self, root: Path) -> list[AnalysisNoteRecord]:
        records: list[AnalysisNoteRecord] = []
        for an_dir in self._note_dirs(root):
            an_id = an_dir.name
            kind = classify_an_directory(an_id)
            assert kind is not None  # _note_dirs filtered on it
            main_tex = an_dir / f"{an_id}.tex"
            title = extract_title(main_tex) if main_tex.exists() else an_id

            sections = [
                SectionRecord(
                    an_id=an_id,
                    an_path=str(an_dir),
                    filename=rel,
                    stem=rel.removesuffix(".tex"),
                    order=order,
                    text=read_text(path),
                )
                for path, rel, order in _section_files(an_dir, an_id)
            ]
            if not sections and main_tex.exists():
                # A monolithic note: all the prose is in the main tex and
                # there are no section files at all. Dropping these would
                # quietly omit a large share of the corpus, so emit the
                # main tex as the note's single section.
                sections = [SectionRecord(
                    an_id=an_id,
                    an_path=str(an_dir),
                    filename=main_tex.name,
                    stem=an_id,
                    order=None,
                    text=read_text(main_tex),
                )]
            if not sections:
                continue

            figures = _note_figures(an_dir, an_id, main_tex, sections)

            bib_path = find_bib_file(an_dir, an_id)
            references = (
                parse_bib_file(bib_path, an_id) if bib_path is not None else []
            )

            records.append(AnalysisNoteRecord(
                an_id=an_id,
                an_path=str(an_dir),
                directory_format=kind,
                pag_code=pag_from_dir(an_id) if kind == "pag_coded" else None,
                title=title,
                sections=tuple(sections),
                references=tuple(references),
                figures=tuple(figures),
            ))
        return records

    def _twiki_pages(self) -> frozenset[str]:
        """Node ids of TWiki pages already in the graph, for the bridge."""
        if self.twiki_eos_root:
            return self._twiki_pages_from_dump()
        if not self.twiki_pages_path:
            return frozenset()
        path = Path(self.twiki_pages_path).expanduser()
        if not path.is_file():
            # Configured but absent is an operator error, not a silent
            # downgrade to "no bridge" -- the same contract archi's
            # readers hold for a configured-but-missing cache.
            raise FileNotFoundError(
                f"twiki_pages_path is set to {path} but no such file "
                f"exists; write the list of TWiki node ids there or unset "
                f"twiki_pages_path"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(
                f"{path}: expected a JSON list of TWiki node ids, "
                f"got {type(payload).__name__}"
            )
        return frozenset(str(item) for item in payload)

    def _twiki_pages_from_dump(self) -> frozenset[str]:
        """The pages the twiki source emits from this dump and filter."""
        from archi.sources.twiki import TwikiEOSSource

        root = Path(self.twiki_eos_root)
        if not root.is_dir():
            raise FileNotFoundError(
                f"twiki_eos_root is set to {root} but no such directory exists"
            )
        source = TwikiEOSSource(
            eos_root=str(root),
            web_root=self.twiki_web_root,
            physics_filter=self.twiki_physics_filter,
            base=self.base,
        )
        run = source.run("analysis-notes-bridge", mode="scope_complete")
        return frozenset(
            fact.node_id
            for fact in run.facts
            if getattr(fact, "subtype", None) == "documentation_page"
        )


def _section_files(an_dir: Path, an_id: str) -> list[tuple[Path, str, int | None]]:
    """The files a note's sections come from: ``(path, rel, order)``.

    ``rel`` is the path relative to the note directory, which becomes the
    section's ``tex_path`` and, minus ``.tex``, its stem -- so top-level
    files keep the ids they always had and a ``chapters/Results.tex``
    becomes ``an_section:<AN>:chapters/Results``.

    Top level: every non-main, non-defs .tex, structural files (the
    NN-/App-/letter-prefixed conventions) first. Keeping structural
    files *instead of* the rest would let a single App_*.tex silence
    every unprefixed file in the note.

    Below the top level: only the files the main tex reaches through
    \\input/\\include/\\subfile/\\import. A note that keeps its chapters in
    chapters/ is otherwise read as its main tex alone.

    Empty means the caller falls back to the main tex (a monolithic note).
    """
    structural: list[tuple[Path, str, int | None]] = []
    fallback: list[tuple[Path, str, int | None]] = []
    for tex in sorted(an_dir.glob("*.tex")):
        if tex.name == f"{an_id}.tex":
            continue
        classified = classify_section_filename(tex.name)
        if classified is None:
            continue  # macros/defs file
        order, kind_tag = classified
        (structural if kind_tag == "structural" else fallback).append(
            (tex, tex.name, order)
        )
    nested: list[tuple[Path, str, int | None]] = []
    main_tex = an_dir / f"{an_id}.tex"
    if main_tex.exists():
        root = an_dir.resolve()
        for path in included_tex_files(an_dir, main_tex):
            if path.parent == root:
                continue  # top level: already listed above
            classified = classify_section_filename(path.name)
            if classified is None:
                continue
            rel = path.relative_to(root).as_posix()
            # Reported against the note directory as given, not its
            # resolved form, so callers can relative_to() the mirror root.
            nested.append((an_dir / rel, rel, classified[0]))
    return structural + fallback + nested


def _note_figures(
    an_dir: Path, an_id: str, main_tex: Path, sections: list[SectionRecord],
) -> list[FigureRecord]:
    """Every figure environment of a note, found in its sections -- and in
    its main tex when that is not itself emitted as a section, since a
    note that inputs its chapters often places figures in the main file
    too. Panels resolve against \\graphicspath from any of those files."""
    main_is_section = any(s.filename == main_tex.name for s in sections)
    main_text = read_text(main_tex) if main_tex.exists() and not main_is_section else ""
    search_dirs: list[str] = []
    for text in [main_text, *(s.text for s in sections)]:
        for d in graphics_paths(text):
            if d not in search_dirs:
                search_dirs.append(d)
    figures: list[FigureRecord] = []
    for section in sections:
        figures += find_figures(
            section.text, an_id=an_id, host_stem=section.stem,
            tex_path=section.filename, an_dir=an_dir, search_dirs=search_dirs,
        )
    if main_text:
        figures += find_figures(
            main_text, an_id=an_id, host_stem=None, tex_path=main_tex.name,
            an_dir=an_dir, search_dirs=search_dirs,
        )
    return figures


def _figure_ids(record: AnalysisNoteRecord) -> tuple[list[str], dict[str, str]]:
    """Node id per figure (in record order), and label -> node id.

    A label names a figure only once per note: a second figure reusing it
    (copy-paste is common) falls back to its position, so ids never
    collide and every \\ref resolves to the figure LaTeX would pick."""
    ids: list[str] = []
    by_label: dict[str, str] = {}
    for fig in record.figures:
        host = fig.host_stem if fig.host_stem is not None else record.an_id
        if fig.label and fig.label not in by_label:
            node_id = figure_node_id(record.an_id, fig.label)
        else:
            node_id = figure_node_id(record.an_id, f"{host}#{fig.index}")
        ids.append(node_id)
        for label in fig.labels:
            by_label.setdefault(label, node_id)
    return ids, by_label


def _records_hash(records: list[AnalysisNoteRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(record.an_id.encode("utf-8"))
        digest.update(record.title.encode("utf-8"))
        for section in record.sections:
            digest.update(section.stem.encode("utf-8"))
            digest.update(_sha256(section.text).encode("utf-8"))
        for reference in record.references:
            digest.update(reference.bib_key.encode("utf-8"))
        for fig in record.figures:
            digest.update(json.dumps(
                [fig.tex_path, fig.index, fig.labels, fig.caption,
                 [(p.raw, p.path, p.bytes) for p in fig.panels]],
            ).encode("utf-8"))
    return digest.hexdigest()


# ── emission ───────────────────────────────────────────────────────────

def _facts_for_notes(
    records: list[AnalysisNoteRecord],
    revision: dict[str, Any],
    *,
    twiki_pages: frozenset[str],
    chunker_name: str,
    report: dict[str, Any],
) -> Iterator[NodeFact | EdgeFact]:
    seen_references: dict[str, str] = {}
    seen_pag_hubs: set[str] = set()

    for record in records:
        note_id = an_node_id(record.an_id)
        record_id = {"an_id": record.an_id}
        yield _note_node(record, note_id, record_id, revision)

        if record.pag_code:
            hub_id = pag_hub_id(record.pag_code)
            if hub_id not in seen_pag_hubs:
                seen_pag_hubs.add(hub_id)
                yield NodeFact(
                    node_id=hub_id,
                    subtype="physics_analysis_group",
                    attrs={
                        "label": record.pag_code,
                        "code": record.pag_code,
                        "text": f"{record.pag_code} physics analysis group",
                    },
                    source_record_id={"pag": record.pag_code},
                    source_revision=revision,
                )
            yield EdgeFact(
                src=note_id,
                dst=hub_id,
                edge_type="references",
                provenance="derived_deterministic",
                attrs={"kind": "belongs_to_pag"},
                source_record_id=record_id,
                source_revision=revision,
            )

        # Figures before sections, so a chunk's \\ref edge never points at
        # a node this run has not emitted yet.
        figure_ids, figures_by_label = _figure_ids(record)
        for fig, fig_id in zip(record.figures, figure_ids):
            yield from _figure_facts(record, fig, fig_id, note_id, revision)
            report["figures"] += 1
            report["figure_panels"] += len(fig.panels)
            report["figure_panels_missing"] += sum(not p.exists for p in fig.panels)

        for section in record.sections:
            section_id = section_node_id(record.an_id, section.stem)
            section_record_id = {
                "an_id": record.an_id,
                "path": section.filename,
            }
            attrs: dict[str, Any] = {
                "label": f"{record.an_id}: {section.stem}",
                "an_id": record.an_id,
                "stem": section.stem,
                "tex_path": section.filename,
                # `body`, NOT `text`: okg's stamper rebuilds `attrs.text`
                # from the other declared searchable fields, so prose
                # stored there is overwritten.
                "body": section.text,
                "content_sha256": _sha256(section.text),
                "char_length": len(section.text),
                "chunker_name": chunker_name,
            }
            if section.order is not None:
                attrs["order"] = section.order
            yield NodeFact(
                node_id=section_id,
                subtype="section",
                attrs=attrs,
                source_record_id=section_record_id,
                source_revision=revision,
            )
            yield EdgeFact(
                src=note_id,
                dst=section_id,
                edge_type="contains",
                attrs={"order": section.order} if section.order else {},
                source_record_id=section_record_id,
                source_revision=revision,
            )

            # Chunks, on the same window the TWiki reader uses. Without
            # these a note competes as one long document against
            # windowed wiki text and loses every time.
            for chunk_index, offset, chunk_text in _chunks(section.text):
                chunk_seed = f"{section_id}\\0{chunk_index}\\0{chunk_text}"
                chunk_id = f"chunk:{_sha256(chunk_seed)[:16]}"
                chunk_record_id = {
                    "an_id": record.an_id,
                    "path": section.filename,
                    "chunk_index": chunk_index,
                }
                yield NodeFact(
                    node_id=chunk_id,
                    subtype="document_chunk",
                    attrs={
                        "chunk_id": chunk_id,
                        "content_sha256": _sha256(chunk_text),
                        "text": chunk_text,
                        "char_offset": offset,
                        # The exact end lets okg's whole-record text read
                        # cut the 200-character overlap instead of returning
                        # it twice: it trims only when char_end - char_offset
                        # == len(text) on both neighbours.
                        "char_end": offset + len(chunk_text),
                        "char_length": len(chunk_text),
                        "chunker_name": chunker_name,
                        "heading_path": f"{record.an_id}: {section.stem}",
                    },
                    source_record_id=chunk_record_id,
                    source_revision=revision,
                )
                yield EdgeFact(
                    src=section_id,
                    dst=chunk_id,
                    edge_type="contains",
                    attrs={"chunk_index": chunk_index},
                    source_record_id=chunk_record_id,
                    source_revision=revision,
                )
                # The passage that discusses a figure, linked to it --
                # wherever in the note the figure sits.
                linked: set[str] = set()
                for label in figure_refs(chunk_text):
                    target = figures_by_label.get(label)
                    if target is None or target in linked:
                        continue  # a table/equation/section label, or seen
                    linked.add(target)
                    report["figure_refs"] += 1
                    yield EdgeFact(
                        src=chunk_id,
                        dst=target,
                        edge_type="references",
                        provenance="derived_deterministic",
                        attrs={"kind": "latex_ref", "label": label},
                        source_record_id=chunk_record_id,
                        source_revision=revision,
                    )

        for reference in record.references:
            ref_id = reference_node_id(reference.bib_key)
            if reference.bib_key not in seen_references:
                seen_references[reference.bib_key] = record.an_id
                yield _reference_node(
                    reference, ref_id, record.an_id, revision
                )
            yield EdgeFact(
                src=note_id,
                dst=ref_id,
                edge_type="references",
                provenance="derived_deterministic",
                attrs={"kind": "cites"},
                source_record_id={
                    "an_id": record.an_id,
                    "bib_key": reference.bib_key,
                },
                source_revision=revision,
            )

            target = bib_twiki_target(reference)
            if target is None:
                continue
            report["bridge_candidates"] += 1
            page_id = twiki_node_id(*target)
            if page_id not in twiki_pages:
                # The page was never ingested (filtered out, or the
                # bridge has no page list at all). An edge to a node
                # that is not in the graph is worse than no edge.
                report["bridge_dropped_unknown_page"] += 1
                continue
            report["bridge_emitted"] += 1
            yield EdgeFact(
                src=note_id,
                dst=page_id,
                edge_type="references",
                provenance="derived_deterministic",
                attrs={"kind": "bib_twiki_link"},
                source_record_id={
                    "an_id": record.an_id,
                    "bib_key": reference.bib_key,
                },
                source_revision=revision,
            )


def _figure_facts(
    record: AnalysisNoteRecord,
    fig: FigureRecord,
    fig_id: str,
    note_id: str,
    revision: dict[str, Any],
) -> Iterator[Any]:
    """A figure node and the edge from the section (or note) it sits in.

    The panels are listed on the node rather than emitted as nodes of
    their own: each is a path and a stat. ``images`` holds an okg image
    value for each panel found on disk, in panel order, as a path under
    the mirror root (``<note dir>/<path>``); okg's ingest renders and
    stores them and ``inspect(kind="image")`` serves them.
    This connector never opens an image. ``caption`` and ``label`` are
    the authors' own, as written, and are what search matches."""
    host_id = (
        section_node_id(record.an_id, fig.host_stem)
        if fig.host_stem is not None else note_id
    )
    figure_record_id = {
        "an_id": record.an_id, "path": fig.tex_path, "figure_index": fig.index,
    }
    key = fig_id.split(":", 2)[2]
    # Note directories are direct children of the mirror root (`_note_dirs`).
    note_dir = Path(record.an_path).name
    yield NodeFact(
        node_id=fig_id,
        subtype="note_figure",
        attrs={
            "label": f"{record.an_id}: {key}",
            "an_id": record.an_id,
            "latex_label": fig.label or "",
            "latex_labels": list(fig.labels),
            "caption": fig.caption,
            "tex_path": fig.tex_path,
            "order": fig.index,
            "panel_count": len(fig.panels),
            "panels_missing": sum(not p.exists for p in fig.panels),
            "panels": [
                {"include": p.raw, "path": p.path, "format": p.format,
                 "bytes": p.bytes, "exists": p.exists}
                for p in fig.panels
            ],
            "images": [
                ImageRef.file(f"{note_dir}/{p.path}")
                for p in fig.panels if p.exists and p.path
            ],
        },
        source_record_id=figure_record_id,
        source_revision=revision,
    )
    yield EdgeFact(
        src=host_id,
        dst=fig_id,
        edge_type="contains",
        attrs={"order": fig.index},
        source_record_id=figure_record_id,
        source_revision=revision,
    )


def _note_node(
    record: AnalysisNoteRecord,
    note_id: str,
    record_id: dict[str, Any],
    revision: dict[str, Any],
) -> NodeFact:
    attrs: dict[str, Any] = {
        "label": record.an_id,
        "an_id": record.an_id,
        "title": record.title,
        "path": record.an_path,
        "directory_format": record.directory_format,
        "text": f"{record.an_id} — {record.title}",
        "n_sections": len(record.sections),
        "n_references": len(record.references),
    }
    if record.pag_code:
        attrs["pag_code"] = record.pag_code
    return NodeFact(
        node_id=note_id,
        subtype="analysis_note",
        attrs=attrs,
        source_record_id=record_id,
        source_revision=revision,
    )


def _reference_node(
    reference: ReferenceRecord,
    ref_id: str,
    first_seen_an: str,
    revision: dict[str, Any],
) -> NodeFact:
    text = " — ".join(part for part in (
        reference.title, reference.authors, reference.journal, reference.year,
    ) if part) or reference.bib_key
    return NodeFact(
        node_id=ref_id,
        subtype="reference",
        attrs={
            "label": reference.bib_key,
            "bib_key": reference.bib_key,
            "entry_type": reference.entry_type,
            "title": reference.title,
            "authors": reference.authors,
            "year": reference.year,
            "journal": reference.journal,
            "url": reference.url,
            "doi": reference.doi,
            "report_number": reference.report_number,
            # The split form and the extracted analysis code. A report
            # number field holds several identifiers comma-joined, and
            # the code is what a published CDS summary is named by --
            # this is the join key for the bibliography bridge.
            "report_numbers": list(reference.report_numbers),
            "analysis_code": reference.analysis_code,
            "howpublished": reference.howpublished,
            "text": text,
            "first_seen_an": first_seen_an,
        },
        source_record_id={"bib_key": reference.bib_key},
        source_revision=revision,
    )


# ── registry adapter ───────────────────────────────────────────────────

class AnalysisNoteAdapter(ReaderAdapter):
    """What ``bundles/packs/analysis-notes/sources/analysis_notes.yaml`` names.

    ``profile`` and ``change_probe_kind`` are class-level string LITERALS:
    the substrate reads ``change_probe_kind`` by parsing this module's AST
    and ``profile`` with ``inspect.getattr_static``, both before importing
    anything.
    """

    reader_class = AnalysisNoteSource
    profile = "discovery_crawl"
    change_probe_kind = "content_hash"
