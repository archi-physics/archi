"""Analysis-note parsing: directory conventions, sections, bibliography.

Nothing here imports okg: these are pure functions over a directory of
LaTeX trees, so they can be tested without the connector around them.

The corpus is a directory of `<PAG>-YY-NNN/` and legacy `AN-YY-NNN/`
trees cloned from `gitlab.cern.ch/tdr/notes`. Each holds a main
`<AN-ID>.tex`, per-section `.tex` files under one of several naming
conventions, and usually a `.bib`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# ── Directory & filename conventions ───────────────────────────────────

PAG_CODES: tuple[str, ...] = (
    "HIG", "TOP", "SUS", "SMP", "B2G", "BPH", "BTV", "EXO", "FTR",
    "HIN", "JME", "MUO", "TAU", "TRK", "EGM", "PPS", "FSQ",
)

_PAG_DIR_RE = re.compile(
    r"^(?P<pag>HIG|TOP|SUS|SMP|B2G|BPH|BTV|EXO|FTR|HIN|JME|MUO|TAU|TRK"
    r"|EGM|PPS|FSQ)-\d{2}-\d{3}$"
)
_LEGACY_AN_DIR_RE = re.compile(r"^AN-\d{2,4}-\d{3}$")

# Section-filename conventions. Some notes use NN-Title.tex and
# App-Title.tex; others in tdr/notes use underscore variants and
# lowercase letter prefixes (1_introduction.tex, a_appendix.tex,
# a6_appendix.tex). The walker keeps structural files first and then
# every other non-main, non-defs .tex, so an AN with an unusual layout
# still produces sections rather than vanishing -- and so one
# App_*.tex no longer silences the unprefixed files beside it.
_NUMERIC_PREFIX_RE = re.compile(r"^(?P<order>\d{1,3})[-_].+\.tex$")
_APP_PREFIX_RE = re.compile(r"^App[-_].+\.tex$", re.IGNORECASE)
_LETTER_PREFIX_RE = re.compile(r"^[A-Za-z]\d{0,2}[-_].+\.tex$")

# Per-AN macro files (mydefs.tex, my_defs.tex, topic-defs.tex). Excluded
# from section emission entirely: they carry no prose.
_DEFS_FILE_RE = re.compile(r"^.*[-_]?defs\.tex$", re.IGNORECASE)

_TITLE_RE = re.compile(r"\\title\s*\{([^}]+)\}", re.DOTALL)

#: Cap on bytes read from one section file. A pathological .tex (a
#: pasted table dump, an inlined log) would otherwise put megabytes of
#: noise on one node.
BODY_CAP = 200_000


def classify_an_directory(name: str) -> str | None:
    """Return ``"pag_coded"`` / ``"legacy_an"`` / ``None``."""
    if _PAG_DIR_RE.match(name):
        return "pag_coded"
    if _LEGACY_AN_DIR_RE.match(name):
        return "legacy_an"
    return None


def pag_from_dir(name: str) -> str | None:
    """Return the three-letter PAG token for a modern directory name."""
    match = _PAG_DIR_RE.match(name)
    return match.group("pag") if match else None


def classify_section_filename(filename: str) -> tuple[int | None, str] | None:
    """Classify a section filename.

    ``None`` means a macros/defs file, excluded outright. Otherwise
    ``(order, kind)``: ``order`` is the integer prefix for a numeric
    section and ``None`` elsewhere; ``kind`` is ``"structural"`` for a
    recognised convention and ``"fallback"`` for anything else. The
    caller orders structural files first and keeps both.

    Excluding the main ``<AN-ID>.tex`` is the caller's job: it depends
    on the AN id.
    """
    if _DEFS_FILE_RE.match(filename):
        return None
    match = _NUMERIC_PREFIX_RE.match(filename)
    if match:
        return int(match.group("order")), "structural"
    if _APP_PREFIX_RE.match(filename):
        return None, "structural"
    if _LETTER_PREFIX_RE.match(filename):
        return None, "structural"
    return None, "fallback"


def find_bib_file(an_dir: Path, an_id: str) -> Path | None:
    """The AN's bibliography, if it can be identified unambiguously."""
    canonical = an_dir / f"{an_id}.bib"
    if canonical.exists():
        return canonical
    bibs = sorted(an_dir.glob("*.bib"))
    return bibs[0] if len(bibs) == 1 else None


def extract_title(tex_path: Path) -> str:
    """``\\title{...}`` from the main tex, or the filename stem."""
    try:
        text = tex_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return tex_path.stem
    match = _TITLE_RE.search(text)
    return match.group(1).strip() if match else tex_path.stem


def read_text(path: Path, max_bytes: int = BODY_CAP) -> str:
    try:
        return path.read_bytes()[:max_bytes].decode("utf-8", errors="replace")
    except OSError:
        return ""


# ── Includes ───────────────────────────────────────────────────────────
#
# Many notes keep their prose below the top level (a chapters/ directory
# and a main tex that is a skeleton of \input lines). Globbing
# subdirectories wholesale would also pull in the figure and table
# snippets, vendored templates and abandoned drafts those trees hold, so
# the walk reads a subdirectory file only when the document actually
# includes it.

_INPUT_RE = re.compile(r"\\(?:input|include|subfile)\s*\{([^{}]+)\}")
# \import{dir}{file} resolves against the main directory,
# \subimport{dir}{file} against the including file's directory, and a
# file pulled in either way resolves its own includes against `dir`.
_IMPORT_RE = re.compile(
    r"\\(sub)?(?:import|inputfrom|includefrom|subinputfrom|subincludefrom)"
    r"\*?\s*\{([^{}]*)\}\s*\{([^{}]+)\}"
)
_COMMENT_RE = re.compile(r"(?<!\\)%.*")

#: \input chains deeper than this are treated as a loop or a generated
#: tree rather than followed.
MAX_INCLUDE_DEPTH = 8


def _resolve_include(base: Path, name: str, an_dir: Path) -> Path | None:
    """The .tex file an include argument names, if it exists in the note.

    LaTeX tries ``<name>.tex`` before ``<name>``. Anything resolving
    outside the note directory (``../``, an absolute path, a symlink out)
    is ignored rather than read.
    """
    name = name.strip()
    if not name or "\\" in name or "#" in name:
        return None  # a macro argument; its value is not knowable here
    candidates = [name] if name.endswith(".tex") else [f"{name}.tex", name]
    root = an_dir.resolve()
    for candidate in candidates:
        path = (base / candidate).resolve()
        if path.suffix != ".tex" or not path.is_file():
            continue
        try:
            path.relative_to(root)
        except ValueError:
            return None
        return path
    return None


def included_tex_files(an_dir: Path, main_tex: Path) -> list[Path]:
    """Every .tex the main tex reaches through its includes, in document
    order, followed recursively and deduplicated. The main tex itself is
    not in the list."""
    found: list[Path] = []
    seen = {main_tex.resolve()}

    def visit(path: Path, base: Path, depth: int) -> None:
        if depth > MAX_INCLUDE_DEPTH:
            return
        text = _COMMENT_RE.sub("", read_text(path))
        hits: list[tuple[int, Path, Path]] = []  # (offset, file, its base)
        for m in _INPUT_RE.finditer(text):
            target = _resolve_include(base, m.group(1), an_dir)
            if target is not None:
                hits.append((m.start(), target, base))
        for m in _IMPORT_RE.finditer(text):
            anchor = path.parent if m.group(1) else an_dir
            import_base = (anchor / m.group(2).strip()).resolve()
            target = _resolve_include(import_base, m.group(3), an_dir)
            if target is not None:
                hits.append((m.start(), target, import_base))
        for _offset, target, child_base in sorted(hits, key=lambda h: h[0]):
            if target in seen:
                continue
            seen.add(target)
            found.append(target)
            visit(target, child_base, depth + 1)

    visit(main_tex.resolve(), an_dir.resolve(), 0)
    return found


# ── Figures ────────────────────────────────────────────────────────────
#
# A figure is found, not processed: the environment's caption and label
# as the authors wrote them, and the image files it places, resolved on
# disk with a stat. The images themselves are never opened, so this costs
# what the section walk already costs.

_FIGURE_ENV_RE = re.compile(
    r"\\begin\s*\{(figure\*?|sidewaysfigure\*?|wrapfigure|SCfigure\*?)\}"
)
_INCLUDEGRAPHICS_RE = re.compile(r"\\includegraphics\*?\s*(?:\[[^\]]*\]\s*)?\{")
_CAPTION_RE = re.compile(r"\\caption\*?\s*(?:\[[^\]]*\]\s*)?\{")
_LABEL_RE = re.compile(r"\\label\s*\{([^{}]+)\}")
_GRAPHICSPATH_RE = re.compile(r"\\graphicspath\s*\{")
# \ref and the cleveref / hyperref / varioref spellings. \cref takes a
# comma-separated list.
_FIGURE_REF_RE = re.compile(
    r"\\(?:ref|cref|Cref|crefrange|autoref|Autoref|subref|vref|Vref|"
    r"nameref|pageref)\*?\s*\{([^{}]+)\}"
)
#: Tried in this order when \includegraphics omits the extension, as
#: graphicx does for pdflatex (plus the DVI formats older notes use).
GRAPHICS_EXTENSIONS: tuple[str, ...] = (
    ".pdf", ".png", ".jpg", ".jpeg", ".eps", ".ps", ".svg",
)
#: Captions longer than this are cut: a caption is a pointer to the
#: figure, and the section body keeps the full text.
CAPTION_CAP = 4000


def strip_comments(text: str) -> str:
    """LaTeX with ``%`` comments removed (``\\%`` is kept)."""
    return _COMMENT_RE.sub("", text)


def _braced(text: str, open_at: int) -> tuple[str, int] | None:
    """The contents of the brace group whose ``{`` is at ``open_at``, and
    the index just past its ``}``; ``None`` if it never closes."""
    depth = 0
    i = open_at
    while i < len(text):
        ch = text[i]
        if ch == "\\":
            i += 2  # an escaped brace does not count
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[open_at + 1:i], i + 1
        i += 1
    return None


def graphics_paths(text: str) -> list[str]:
    """Directories named by ``\\graphicspath{{a/}{b/}}``, in order."""
    dirs: list[str] = []
    text = strip_comments(text)
    for m in _GRAPHICSPATH_RE.finditer(text):
        group = _braced(text, m.end() - 1)
        if group is None:
            continue
        inner = group[0]
        pos = 0
        while (start := inner.find("{", pos)) != -1:
            one = _braced(inner, start)
            if one is None:
                break
            if one[0].strip():
                dirs.append(one[0].strip())
            pos = one[1]
    return dirs


def _resolve_graphic(
    raw: str, an_dir: Path, tex_dir: Path, search_dirs: list[str],
) -> PanelRecord:
    """Find the file an ``\\includegraphics`` argument names, by stat only.

    Tried against the note directory (LaTeX's working directory), then
    the including file's directory, each alone and under every
    ``\\graphicspath`` entry; with the extension as written, then each of
    GRAPHICS_EXTENSIONS. Brace-grouped names (``{plots/a.v2}.pdf``) are
    unwrapped. A name built from a macro is recorded, unresolved.
    """
    name = raw.strip().replace("{", "").replace("}", "")
    if not name or "\\" in name or "#" in name:
        return PanelRecord(raw=raw, path=None, format=None, bytes=None, exists=False)
    root = an_dir.resolve()
    has_ext = Path(name).suffix.lower() in GRAPHICS_EXTENSIONS
    suffixes = [""] if has_ext else ["", *GRAPHICS_EXTENSIONS]
    bases: list[Path] = []
    for base in (an_dir, tex_dir):
        for sub in ["", *search_dirs]:
            candidate = (base / sub) if sub else base
            if candidate not in bases:
                bases.append(candidate)
    for base in bases:
        for suffix in suffixes:
            path = (base / f"{name}{suffix}").resolve()
            if path.suffix.lower() not in GRAPHICS_EXTENSIONS:
                continue
            try:
                rel = path.relative_to(root)
            except ValueError:
                continue  # never outside the note
            try:
                st = path.stat()
            except OSError:
                continue
            if not path.is_file():
                continue
            return PanelRecord(
                raw=raw, path=rel.as_posix(), format=path.suffix.lower().lstrip("."),
                bytes=st.st_size, exists=True,
            )
    return PanelRecord(raw=raw, path=None, format=None, bytes=None, exists=False)


def _clean_caption(text: str) -> str:
    return " ".join(text.split())[:CAPTION_CAP]


def find_figures(
    text: str,
    *,
    an_id: str,
    host_stem: str | None,
    tex_path: str,
    an_dir: Path,
    search_dirs: list[str],
) -> list[FigureRecord]:
    """Every figure environment in one file's text, in order."""
    text = strip_comments(text)
    tex_dir = (an_dir / tex_path).parent
    figures: list[FigureRecord] = []
    pos = 0
    while (m := _FIGURE_ENV_RE.search(text, pos)) is not None:
        env = m.group(1)
        end_tag = f"\\end{{{env}}}"
        end = text.find(end_tag, m.end())
        if end == -1:
            break  # unterminated: stop rather than swallow the file
        body = text[m.end():end]
        pos = end + len(end_tag)

        caption = ""
        caption_at = -1
        for cm in _CAPTION_RE.finditer(body):
            group = _braced(body, cm.end() - 1)
            if group is not None:
                caption, caption_at = group[0], cm.start()
                break
        labels = tuple(dict.fromkeys(lm.group(1).strip() for lm in _LABEL_RE.finditer(body)))
        # The figure's own label is the one after its \caption (a label
        # before it usually belongs to a subfigure); else the first.
        label = next(
            (lm.group(1).strip() for lm in _LABEL_RE.finditer(body)
             if caption_at != -1 and lm.start() > caption_at),
            labels[0] if labels else None,
        )
        panels = []
        for gm in _INCLUDEGRAPHICS_RE.finditer(body):
            group = _braced(body, gm.end() - 1)
            if group is not None:
                panels.append(_resolve_graphic(group[0], an_dir, tex_dir, search_dirs))
        figures.append(FigureRecord(
            an_id=an_id, host_stem=host_stem, tex_path=tex_path,
            index=len(figures) + 1, label=label, labels=labels,
            caption=_clean_caption(caption), panels=tuple(panels),
        ))
    return figures


def figure_refs(text: str) -> list[str]:
    """Every label a chunk of LaTeX refers to, in order, deduplicated."""
    out: list[str] = []
    for m in _FIGURE_REF_RE.finditer(strip_comments(text)):
        for label in m.group(1).split(","):
            label = label.strip()
            if label and label not in out:
                out.append(label)
    return out


# ── Bibliography ───────────────────────────────────────────────────────

_BIB_ENTRY_RE = re.compile(
    r"@(?P<etype>[A-Za-z]+)\s*\{\s*(?P<key>[^,\s]+)\s*,(?P<body>.*?)\n\}",
    re.DOTALL,
)
_BIB_FIELD_RE = re.compile(
    r"(?P<name>[A-Za-z_]+)\s*=\s*[{\"](?P<value>.*?)[}\"]\s*,?\s*$",
    re.MULTILINE | re.DOTALL,
)
_BIB_URL_RE = re.compile(r"\\url\{([^}]+)\}")


#: A CMS analysis code inside a bibliography report number, e.g. the
#: ``TOP-99-012`` in ``CMS-TOP-99-012, CERN-EP-2099-078``. This is the
#: key a published CDS summary carries, which is what lets a legacy
#: AN-YY-NNN note reach the published corpus through its own citations.
_CMS_ANALYSIS_CODE_RE = re.compile(
    r"\bCMS-(?P<code>(?:HIG|TOP|SUS|SMP|B2G|BPH|BTV|EXO|FTR|HIN|JME|MUO"
    r"|TAU|TRK|EGM|PPS|FSQ)-\d{2}-\d{3})\b"
)


def split_report_numbers(raw: str) -> tuple[str, ...]:
    """Bibliography report numbers arrive comma-joined.

    ``CMS-TOP-99-012, CERN-EP-2099-078`` is two identifiers in one
    field. Matching the raw string against a CDS record's report number
    finds nothing; splitting it finds the analysis.
    """
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def analysis_code_from_report_numbers(raw: str) -> str:
    """The CMS analysis code a reference names, or ``""``.

    Empty when no part of the field names a recognised PAG-coded
    analysis -- a conference record (``CMS-CR-2099-213``) or a CERN
    preprint number names no analysis, and must not match one.
    """
    for part in split_report_numbers(raw):
        match = _CMS_ANALYSIS_CODE_RE.search(part)
        if match:
            return match.group("code")
    return ""


@dataclass(frozen=True)
class ReferenceRecord:
    an_id: str
    bib_key: str
    entry_type: str
    title: str
    authors: str
    year: str
    journal: str
    url: str
    doi: str
    report_number: str = ""
    howpublished: str = ""

    @property
    def report_numbers(self) -> tuple[str, ...]:
        return split_report_numbers(self.report_number)

    @property
    def analysis_code(self) -> str:
        return analysis_code_from_report_numbers(self.report_number)


def _strip_bib_value(raw: str) -> str:
    value = raw.strip()
    while value.startswith("{") and value.endswith("}"):
        value = value[1:-1].strip()
    return re.sub(r"\s+", " ", value)


def parse_bib_file(bib_path: Path, an_id: str) -> list[ReferenceRecord]:
    """Entries from one .bib, first occurrence of a key winning."""
    try:
        text = bib_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    records: list[ReferenceRecord] = []
    seen: set[str] = set()
    for match in _BIB_ENTRY_RE.finditer(text):
        key = match.group("key")
        if key in seen:
            continue
        seen.add(key)
        fields: dict[str, str] = {}
        for field_match in _BIB_FIELD_RE.finditer(match.group("body")):
            fields[field_match.group("name").lower()] = _strip_bib_value(
                field_match.group("value")
            )
        url = fields.get("url", "")
        if not url:
            url_match = _BIB_URL_RE.search(fields.get("howpublished", ""))
            if url_match:
                url = url_match.group(1)
        records.append(ReferenceRecord(
            an_id=an_id,
            bib_key=key,
            entry_type=match.group("etype").lower(),
            title=fields.get("title", ""),
            authors=fields.get("author", ""),
            year=fields.get("year", ""),
            report_number=fields.get("reportnumber", ""),
            howpublished=fields.get("howpublished", ""),
            journal=fields.get("journal", ""),
            url=url,
            doi=fields.get("doi", ""),
        ))
    return records


# ── Bib URL -> TWiki topic ─────────────────────────────────────────────

_TWIKI_URL_RE = re.compile(
    r"https?://twiki\.cern\.ch/twiki/bin/view/"
    r"(?P<web>CMS|CMSPublic)/(?P<topic>[A-Za-z0-9_]+)"
)


def bib_twiki_target(ref: ReferenceRecord) -> tuple[str, str] | None:
    """``(web, topic)`` when a reference's URL points at a TWiki topic."""
    for source in (ref.url, ref.howpublished):
        if not source:
            continue
        match = _TWIKI_URL_RE.search(source)
        if match:
            return match.group("web"), match.group("topic")
    return None


def twiki_node_id(web: str, topic: str) -> str:
    """The node id archi's TWiki connector mints for a (web, topic).

    Must match ``archi.sources._twiki_parse.twiki_node_id``; the bridge
    edge is useless if it does not. ``tests/packs/test_analysis_notes.py``
    asserts the two agree rather than trusting this comment.
    """
    return f"twiki:{web}:{topic}"


# ── Records ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SectionRecord:
    an_id: str
    an_path: str
    filename: str
    stem: str
    order: int | None
    text: str


@dataclass(frozen=True)
class PanelRecord:
    """One ``\\includegraphics`` of a figure: where the file is, not what
    it shows. Nothing here opens the image."""

    raw: str             # the argument as written
    path: str | None     # resolved, relative to the note dir; None if not found
    format: str | None   # "pdf", "png", "eps", ... from the resolved file
    bytes: int | None
    exists: bool


@dataclass(frozen=True)
class FigureRecord:
    """One ``figure`` environment: the authors' caption and label, and the
    image files it places. ``host_stem`` is the section it sits in, or
    ``None`` when it sits in a main tex that is not emitted as a section."""

    an_id: str
    host_stem: str | None
    tex_path: str
    index: int                # 1-based position among its host's figures
    label: str | None         # the label the figure is referred to by
    labels: tuple[str, ...]   # every \label in the environment (subfigures)
    caption: str
    panels: tuple[PanelRecord, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class AnalysisNoteRecord:
    """One per accepted directory under the AN root."""

    an_id: str
    an_path: str
    directory_format: str  # "pag_coded" | "legacy_an"
    pag_code: str | None   # set for pag_coded, None for legacy_an
    title: str
    sections: tuple[SectionRecord, ...] = field(default_factory=tuple)
    references: tuple[ReferenceRecord, ...] = field(default_factory=tuple)
    figures: tuple[FigureRecord, ...] = field(default_factory=tuple)
