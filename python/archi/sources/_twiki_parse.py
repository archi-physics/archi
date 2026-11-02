"""TWiki markup parser core shared by the two TWiki ingestors.

Consolidated for the archi v3 package (req.w2.sources,
task.w2.sources-twiki) from three references at okg-deployments
``main@f33a9c4``:

- ``cms/cms_sources/twiki_eos.py`` — the canonical base: the
  markup-stripping pipeline, ``%META`` parsing (author / version /
  date / TOPICPARENT), wiki-link and bare-WikiWord extraction, the
  ``twiki:<Web>:<Topic>`` node-id minting, and
  :data:`DEFAULT_SKIP_PATTERNS`.
- ``wisdqm/wisdqm_sources/docs.py`` + ``wisdqm/scripts/
  download_sources.py`` — the fidelity behaviors folded in as flags on
  :func:`strip_twiki`: markdown heading conversion (``---++`` ->
  ``##``), per-line whitespace preservation, embedded HTML table ->
  pipe-table and ``<img>`` -> markdown-image conversion, and the
  viewauth->view URL canonicalization with fragment/query dropping
  (:func:`canonical_twiki_url`, :func:`twiki_page_id_from_url`).
- ``cern-twiki`` — nothing (maintainer decision: its hardening layer
  is excluded from this consolidation).

The archi v2 dev branch (``dev@28b977d1``) TWiki spider parses
rendered HTML DOM; none of that enters the parser core — this module
only ever sees raw TWiki topic markup (EOS snapshot files or the
``?raw=`` endpoint).

Fidelity flags on :func:`strip_twiki` default to the cms flavor
(``heading_style="drop"``, ``whitespace="collapse"``,
``preserve_tables=False``, ``preserve_images=False``) so cms instances
keep byte parity with the ``twiki_eos`` corpus; the wisdqm flavor is
``heading_style="markdown"``, ``whitespace="preserve"``,
``preserve_tables=True``, ``preserve_images=True``.

Decisions: two-ingestor design + fidelity flags approved by the
maintainer 2026-08-12; cern-twiki hardening excluded.

Deliberate parity deviations from the cms original (TWiki is not in
archi's v2 byte-parity corpus; the resulting chunk re-keying was
accepted):

- the cms ``=code=`` unwrap regex (``=([^=\\s][^=]*[^=\\s])=``) pairs
  ``=`` signs across lines and assignments, so ``MAX=5 and MIN=2``
  became ``MAX5 and MIN2``. Ours requires the classic single-line
  inline-code form with non-word boundary context on both sides:
  assignments survive untouched and ``=ls -la=`` still unwraps.
- the cms heading regexes let ``\\s*`` span the newline, so a bare
  ``---++`` marker line absorbed the following line (markdown mode
  even promoted the next line to a heading title). Ours match within
  a single line; a bare marker line is dropped entirely in both
  heading styles.
"""
from __future__ import annotations

import functools
import re
from datetime import datetime, timezone
from html import unescape
from urllib.parse import unquote, urldefrag, urlparse

from archi.sources.docs import _pg_text

HEADING_STYLES = ("drop", "markdown")
WHITESPACE_MODES = ("collapse", "preserve")

# Snapshot filenames that are not real topic pages (cms twiki_eos.py).
# Immutable on purpose: pass a custom tuple to is_real_page (or the
# sources' skip_patterns param) instead of mutating module state.
DEFAULT_SKIP_PATTERNS: tuple[str, ...] = (
    r"^[0-9]+\.txt$",
    r"^[0-9]{6,8}[A-Za-z]",
    r"^[0-9]{1,2}-[0-9]{1,2}-[0-9]{4}",
    r"-replies\.txt$",
    r"^[a-z]",
    r"^Web(Atom|Changes|CreateNewTopic|Index|LeftBar|Notify|Preferences|"
    r"Rss|Search|SearchAdvanced|Statistics|TopicCreator|TopicEditTemplate|"
    r"TopicList|TopMenu|BottomBar)\.txt$",
    r"^(LastViewedTopics|TWeederSummaryViews|SearchResults)\.txt$",
)


@functools.lru_cache(maxsize=None)
def _skip_re(patterns: tuple[str, ...]) -> re.Pattern[str]:
    return re.compile("|".join(f"(?:{p})" for p in patterns))
_META_TOPICINFO_RE = re.compile(r"%META:TOPICINFO\{([^}]*)\}%", re.IGNORECASE)
_META_TOPICPARENT_RE = re.compile(
    r'%META:TOPICPARENT\{[^}]*name="([^"]+)"[^}]*\}%',
    re.IGNORECASE,
)
_META_KV_RE = re.compile(r'(\w+)="([^"]*)"')
_WIKI_LINK_RE = re.compile(
    r"\[\[(?!https?://|mailto:|#|[A-Za-z]+:/)([A-Za-z][\w./]*?)"
    r"(?:\]\[[^\]]*)?\]\]"
)
_BARE_WW_STRIP_PATTERNS = [
    re.compile(r"%META:[^%]+%"),
    re.compile(r"%[A-Z][A-Z0-9_]*\{[^}]*\}%"),
    re.compile(r"<verbatim>.*?</verbatim>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<pre>.*?</pre>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<!--.*?-->", re.DOTALL),
    re.compile(r"\[\[[^\]]+\]\]"),
    re.compile(r"https?://\S+"),
]
_BARE_WIKIWORD_RE = re.compile(
    r"(?<![A-Za-z./_])([A-Z][a-z]+(?:[A-Z][a-z]*)+)(?![a-z])"
)
# Heading title match must stay on the marker's own line ([ \t]* +
# [^\n]*, never \s*/.*$ whose \s* can cross the newline and absorb the
# next line); a bare marker line is dropped entirely in both styles.
_HEADING_RE = re.compile(r"^\-{3}(\+{1,6})!?[ \t]*([^\n]*)(\n?)", re.MULTILINE)
# Inline =code= unwrap: single-line only, non-word boundary context on
# both sides so assignments ('MAX=5 and MIN=2') are never paired up.
_INLINE_CODE_RE = re.compile(
    r"(?<![\w=])=([^=\s\n][^=\n]*[^=\s\n])=(?![\w=])"
)
_HTML_IMG_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
_HTML_TABLE_RE = re.compile(r"<table\b.*?</table>", re.IGNORECASE | re.DOTALL)
_HTML_ROW_RE = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
_HTML_CELL_RE = re.compile(
    r"<(th|td)\b[^>]*>(.*?)</\1>",
    re.IGNORECASE | re.DOTALL,
)
_HTML_ATTR_RE = re.compile(
    r"([A-Za-z_:][-A-Za-z0-9_:.]*)\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s\"'>]+))"
)
LINK_URL_MODES = ("drop", "keep")
_LABELLED_WEB_LINK_RE = re.compile(
    r"\[\[\s*(https?://[^\]\s]+)\s*\]\[([^\]]+)\]\]",
    re.IGNORECASE,
)
_TWIKI_VIEW_MARKERS = ("/twiki/bin/viewauth/", "/twiki/bin/view/")


def is_real_page(
    filename: str,
    patterns: tuple[str, ...] = DEFAULT_SKIP_PATTERNS,
) -> bool:
    """True when *filename* is a real topic page, not a structural one.

    ``patterns`` is a tuple of regex sources (compiled once per unique
    tuple); it defaults to :data:`DEFAULT_SKIP_PATTERNS`.
    """
    return not _skip_re(tuple(patterns)).search(filename)


def strip_twiki(
    text: str,
    *,
    heading_style: str = "drop",
    whitespace: str = "collapse",
    preserve_tables: bool = False,
    preserve_images: bool = False,
    link_urls: str = "drop",
) -> str:
    """Strip raw TWiki markup down to text, with fidelity flags.

    Defaults are the cms flavor (byte parity with the ``twiki_eos``
    corpus). ``heading_style="markdown"`` converts ``---++ Title`` to
    ``## Title`` instead of dropping the marker;
    ``whitespace="preserve"`` keeps line structure (per-line space
    collapse, at most one blank line) instead of collapsing everything
    to one line; ``preserve_tables`` / ``preserve_images`` convert
    embedded HTML ``<table>`` / ``<img>`` to pipe tables / markdown
    images before generic tag stripping (the wisdqm behaviors).
    ``link_urls="keep"`` writes a labelled web link ``[[URL][label]]``
    as ``label (URL)`` instead of the label alone, so an identifier
    that sits only in the URL (a JIRA key in a ``/browse/`` link)
    reaches the text and the extractor. Wiki-topic and ``mailto:``
    links keep the label alone in both modes.
    """
    if heading_style not in HEADING_STYLES:
        raise ValueError(f"heading_style must be one of {HEADING_STYLES}")
    if whitespace not in WHITESPACE_MODES:
        raise ValueError(f"whitespace must be one of {WHITESPACE_MODES}")
    if link_urls not in LINK_URL_MODES:
        raise ValueError(f"link_urls must be one of {LINK_URL_MODES}")
    text = re.sub(r"%META:[^%]+%", " ", text)
    text = re.sub(
        r"%[A-Z][A-Z0-9_]*\{([^}]*)\}%",
        lambda m: " " + m.group(1).replace('"', " ") + " ",
        text,
    )
    text = re.sub(r"%[A-Z][A-Z0-9_]*%", " ", text)
    if link_urls == "keep":
        text = _LABELLED_WEB_LINK_RE.sub(_labelled_web_link_text, text)
    text = re.sub(r"\[\[[^\]]+\]\[([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = _HEADING_RE.sub(
        lambda m: _heading_replacement(m, heading_style),
        text,
    )
    text = re.sub(
        r"</?(verbatim|pre|code|noautolink)>",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    if preserve_tables:
        text = _HTML_TABLE_RE.sub(
            lambda m: "\n" + _html_table_to_pipe(m.group(0)) + "\n",
            text,
        )
    if preserve_images:
        text = _HTML_IMG_RE.sub(_html_img_to_markdown, text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = _INLINE_CODE_RE.sub(r"\1", text)
    text = re.sub(r"(?<=\s)[*_]([^*_\s][^*_]*[^*_\s])[*_](?=\s)", r"\1", text)
    if whitespace == "preserve":
        return _clean_body_text(text)
    text = re.sub(r"\s+", " ", text).strip()
    return _pg_text(text)


def _labelled_web_link_text(match: re.Match[str]) -> str:
    """``[[URL][label]]`` -> ``label (URL)``; a label that is the URL once."""
    url, label = match.group(1).strip(), match.group(2)
    if label.strip() == url:
        return url
    return f"{label} ({url})"


def _heading_replacement(match: re.Match[str], heading_style: str) -> str:
    """One ``---++``-style heading line -> its replacement.

    A bare marker line (no title) is dropped entirely — marker and
    newline — in both styles; it never absorbs the following line.
    """
    title = match.group(2).strip()
    if not title:
        return ""
    if heading_style == "markdown":
        return f"{'#' * len(match.group(1))} {title}{match.group(3)}"
    return f"{match.group(2)}{match.group(3)}"


def parse_meta(body: str) -> dict[str, str]:
    """Author / last_modified / version / parent_topic from %META lines."""
    out = {"author": "", "last_modified": "", "version": "", "parent_topic": ""}
    match = _META_TOPICINFO_RE.search(body)
    if match:
        kvs = dict(_META_KV_RE.findall(match.group(1)))
        out["author"] = kvs.get("author", "")
        out["version"] = kvs.get("version", "")
        raw_date = kvs.get("date", "")
        if raw_date.isdigit():
            try:
                out["last_modified"] = datetime.fromtimestamp(
                    int(raw_date),
                    tz=timezone.utc,
                ).strftime("%Y-%m-%dT%H:%M:%SZ")
            except (OSError, ValueError):
                out["last_modified"] = raw_date
    match = _META_TOPICPARENT_RE.search(body)
    if match:
        out["parent_topic"] = match.group(1)
    return out


def extract_wiki_links(body: str) -> set[str]:
    """Targets of explicit ``[[...]]`` links (dots normalized to slashes)."""
    links: set[str] = set()
    for match in _WIKI_LINK_RE.finditer(body):
        target = match.group(1).strip().replace(".", "/").strip("/")
        if target:
            links.add(target)
    return links


def extract_bare_wikiwords(body: str) -> set[str]:
    """Bare CamelCase WikiWords outside meta/macros/verbatim/links/URLs."""
    if not body:
        return set()
    stripped = body
    for pattern in _BARE_WW_STRIP_PATTERNS:
        stripped = pattern.sub(" ", stripped)
    return {match.group(1) for match in _BARE_WIKIWORD_RE.finditer(stripped)}


def twiki_node_id(page_id: str) -> str:
    """Mint the ``twiki:<Web>:<Topic>`` node id for a ``Web/Topic`` page id."""
    return f"twiki:{page_id.strip('/').replace('/', ':')}"


def topic_page_id(target: str, *, web_name: str, web_root: str = "") -> str:
    """Resolve a link/WikiWord target to a full page id, or ``""``.

    ``web_name`` is the web of the page the target appears on;
    ``web_root`` is the snapshot's root web ("CMS" for the cms EOS
    mirror, empty for a live crawl where webs are fully qualified).
    """
    normalized = target.strip().replace(".", "/").strip("/")
    if not normalized:
        return ""
    if web_root and normalized.startswith(f"{web_root}/"):
        return normalized
    if "/" in normalized:
        return f"{web_root}/{normalized}" if web_root else normalized
    return f"{web_name}/{normalized}" if web_name else ""


def canonical_twiki_url(url: str) -> str:
    """Canonical TWiki view URL: viewauth->view, no fragment, no query."""
    url, _fragment = urldefrag(_pg_text(url))
    if not url:
        return ""
    parsed = urlparse(url)
    path = parsed.path.replace("/twiki/bin/viewauth/", "/twiki/bin/view/")
    return parsed._replace(path=path, query="").geturl()


def twiki_page_id_from_url(url: str) -> str:
    """``Web/Topic`` page id from a TWiki view/viewauth URL, or ``""``."""
    parsed = urlparse(url)
    path = unquote(parsed.path)
    for marker in _TWIKI_VIEW_MARKERS:
        if marker in path:
            return path.split(marker, 1)[1].strip("/")
    return ""


def _html_img_to_markdown(match: re.Match[str]) -> str:
    tag = match.group(0)
    attrs = _html_attrs(tag)
    src = attrs.get("src", "")
    if not src:
        return " "
    alt = attrs.get("alt", "")
    return f"![{_clean_inline_text(alt)}]({src})"


def _html_table_to_pipe(table_html: str) -> str:
    rows: list[list[str]] = []
    for row_match in _HTML_ROW_RE.finditer(table_html):
        cells = [
            _clean_html_cell_text(cell_match.group(2))
            for cell_match in _HTML_CELL_RE.finditer(row_match.group(1))
        ]
        if cells:
            rows.append(cells)
    if not rows:
        return " "
    lines = ["| " + " | ".join(rows[0]) + " |"]
    lines.append("| " + " | ".join("---" for _ in rows[0]) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in rows[1:])
    return "\n".join(lines)


def _html_attrs(tag: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for match in _HTML_ATTR_RE.finditer(tag):
        attrs[match.group(1).lower()] = unescape(
            next(group for group in match.groups()[1:] if group is not None)
        )
    return attrs


def _clean_html_cell_text(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value)
    return _clean_inline_text(unescape(value))


def _clean_inline_text(value: str) -> str:
    return re.sub(r"\s+", " ", _pg_text(value)).strip()


def _clean_body_text(value: str) -> str:
    value = _pg_text(value)
    cleaned_lines = [
        re.sub(r"[ \t]+", " ", line).strip()
        for line in value.splitlines()
    ]
    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
