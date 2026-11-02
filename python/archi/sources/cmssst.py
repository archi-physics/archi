"""Cache-backed CMS Site Support Team (CMS SST) site status source.

One readable, citable page per CMS site, holding the site's *current*
status as the CMS SST site status detail page shows it
(``https://cmssst.web.cern.ch/siteStatus/detail.html?site=<SITE>``). No
history: of the page's five time ranges only "today" is read, and of that
only the status at the snapshot time.

The cache is ``data/cmssst-site-status/records.json``, written by
:mod:`archi.downloaders.cmssst`: a JSON list, one object per site, with the
site JSON's ``site``, ``time`` (snapshot, epoch seconds), ``alert``,
``msg``, ``ggus`` (``[ticket id, opened epoch]`` pairs) and ``metrics``
(metric name -> time range -> list of strings of one-letter status codes).

Decoding. The data carries no words, only letters. The page's renderer
(``detail_lib.js``, function ``fillCanvases``) maps each letter to a colour
and the legend in ``detail.html`` names each colour; :data:`STATUS_WORDS`
is read off those two files (the ``case`` lines of the first canvas
switch, ``detail_lib.js`` lines 968-1126 of the 2026-10-07 copy). Letters
R-V (and W) draw the base colour over the Waiting Room purple, H-L (and M)
over the Morgue brown. The renderer's ``I``-``L`` cases have no ``break``
and fall through to ``M``, so the page draws them all as Morgue; the
decoding here keeps the colour each case sets first.

"Today" is 96 quarter-hour bins from 00:00 UTC of the snapshot day (the
page's ``myTIS`` offset; its GGUS code takes the same UTC midnight). The
bin holding the snapshot time is "now". For the evaluated metrics the
value reported is the latest bin at or before now that is not ``u``
(unknown); a metric with none is "unknown". Downtime and the manual
overrides (``man*Status``) are read at the "now" bin only: for them ``u``
means "none in effect", so a downtime or override that ended earlier
today, or one scheduled for later today, is not reported as current.

Stable text. The page text carries no clock: the snapshot time lives only
in the ``observed_at`` attribute, and statuses do not name the quarter
hour they came from. Chunk ids hash chunk text, so a 15-minute refresh
whose statuses did not change yields identical text and chunk ids, and
nothing is retracted or re-embedded.

Known limits. The words are the page legend's generic ones ("ok (site
state or evaluation good)" for Life Status as for SAM); the page does not
say, for example, that a Life Status of ok means the site is enabled, so
the text does not either.

Facts. Existing subtypes only, no schema change: one ``documentation_page``
per site (``url`` = the detail page URL, ``title`` = ``"<SITE> site
status"``) and its ``document_chunk`` passages joined by ``contains``,
chunked as :mod:`archi.sources.docs` chunks. With ``sites_path`` set to a
CRIC sites cache, the page also ``references`` the ``site:<SITE>`` node --
the ``documentation_page_references_site`` narrowing in
``archi/schemas/bridges/operations.yaml`` -- for sites that cache lists,
and only those, so no edge dangles.

Empty-cache-fails-loudly, as :mod:`archi.sources.gocdb`: a missing,
unreadable, zero-byte, truncated or empty records list is
``cache_missing`` and an error-shaped payload is ``endpoint_failed``; each
gives no facts and ``completed_scope=False`` from both ``preflight()`` and
``run()``. A record that is not an object, has no CMS site name, no
positive ``time`` or a malformed ``metrics``/``ggus`` is skipped and stops
the scope claim. A configured but missing or empty sites cache is refused
the same way.

Registry-entry template -- same prerequisites as
``archi/sources/docs.py``'s template (``documentation_page`` and
``document_chunk`` from ``archi/schemas/sources.yaml`` and the
``extraction`` module; ``site`` and the references narrowing from
``archi/schemas/operations.yaml`` + ``bridges/operations.yaml``). ::

    cmssst_site_status:
      module: archi.sources.cmssst
      class: CMSSSTSiteStatusAdapter
      ownership_id: <instance>.cmssst-site-status
      admission_policy:
        producer_id: <instance>.cmssst-site-status
        producer_kind: source
        trust_label: implicit_legacy_trusted
        admission_mode: fast_track
        authority_scope:
          source_family: <family>
          source_name: cmssst_site_status
        output_signature:
          nodes:
            - {subtype: documentation_page}
            - {subtype: document_chunk}
          edges:
            - {src_subtype: documentation_page, edge_type: contains, dst_subtype: document_chunk}
            # Uncomment together with sites_path below:
            # - {src_subtype: documentation_page, edge_type: references, dst_subtype: site}
        output_scope_summary:
          summary: CMS SST current site status, one page per site, with text chunks
          nodes: [documentation_page, document_chunk]
          edges:
            - documentation_page contains document_chunk
            # Uncomment together with sites_path below:
            # - documentation_page references site
      source_class: discovery_crawl
      record_identity_kind: scoped_locator
      record_identity_fields: [site]
      source_revision_kind: content_hash
      deletion_semantics: missing_from_completed_scope
      publication_mode: published_generation
      required_for_baseline: false
      params:
        records_path: data/cmssst-site-status/records.json
        # Optional: link each page to its CRIC site node.
        # sites_path: data/cric/sites.json
      sync:
        triggers: [manual, reconcile]
        default_event_mode: scope_complete
        reconcile_mode: scope_complete
"""
from __future__ import annotations

import hashlib
import html
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator

from okg.deployment import (
    EdgeFact,
    NodeFact,
    ConnectorHealth,
    PreflightResult,
    ConnectorRun,
)

from archi.auth.cache import (
    content_hash,
    content_hash_change_probe,
)
from archi.sources._cache_report import (
    CacheUnusable,
    read_cache_json,
    skipped_items_status,
    unusable_cache_preflight,
    unusable_cache_run,
)
from archi.sources._sdk_adapter import ReaderAdapter
from archi.sources.cric import _CMS_SITE_NAME, _Skipped, _read_objects
from archi.sources.docs import _chunks

DETAIL_URL = "https://cmssst.web.cern.ch/siteStatus/detail.html?site="
GGUS_TICKET_URL = "https://helpdesk.ggus.eu/#ticket/zoom/"
CHUNKER_NAME = "cmssst_site_status_v1"

#: The site-name check the detail page itself applies (detail.html).
SITE_NAME = re.compile(r"^T[0-9]_[A-Z]{2}_\w+$")

_OK = "ok (site state or evaluation good)"
_WARNING = "warning (site or evaluation has issues)"
_ERROR = "error (site or evaluation failed)"
_DOWNTIME = "downtime"
_UNKNOWN = "unknown (site or evaluation status unknown)"
_WAITING_ROOM = "site not in service (Waiting Room)"
_MORGUE = "site not in service (Morgue)"

#: Status letter -> plain words, from detail_lib.js ``fillCanvases`` (the
#: colour each ``case`` draws) and the detail.html legend (what each
#: colour means). Line numbers are the 2026-10-07 detail_lib.js.
STATUS_WORDS: dict[str, str] = {
    "o": _OK,  # 968, #80FF80
    "w": _WARNING,  # 974, #FFFF00
    "e": _ERROR,  # 980, #FF0000
    "p": "site in partial downtime",  # 986, blue split bar
    "d": "site or service in full downtime",  # 994, #6080FF full bar
    "a": "site or service in unscheduled downtime",  # 1000, blue striped bar
    "r": "site at risk of downtime",  # 1012, thin blue bar
    "u": _UNKNOWN,  # default branch, #F4F4F4
    "R": f"{_OK}; {_WAITING_ROOM}",  # 1018, green over purple
    "S": f"{_WARNING}; {_WAITING_ROOM}",  # 1028, yellow over purple
    "T": f"{_ERROR}; {_WAITING_ROOM}",  # 1038, red over purple
    "U": f"{_DOWNTIME}; {_WAITING_ROOM}",  # 1048, blue over purple
    "V": f"{_UNKNOWN}; {_WAITING_ROOM}",  # 1058, grey over purple
    "W": _WAITING_ROOM,  # 1068, #A000A0 full bar
    "H": f"{_OK}; {_MORGUE}",  # 1074, green over brown
    "I": f"{_WARNING}; {_MORGUE}",  # 1084, yellow over brown
    "J": f"{_ERROR}; {_MORGUE}",  # 1093, red over brown
    "K": f"{_DOWNTIME}; {_MORGUE}",  # 1102, blue over brown
    "L": f"{_UNKNOWN}; {_MORGUE}",  # 1111, grey over brown
    "M": _MORGUE,  # 1120, #663300 full bar
}

#: (metric key, label) in the order the page lists them; labels are the
#: page's own (siteMetricLabel in detail_lib.js) with the short names spelt
#: out.
STATUS_METRICS: tuple[tuple[str, str], ...] = (
    ("LifeStatus", "Life Status"),
    ("manLifeStatus", "Life Status manual override"),
    ("ProdStatus", "Production (Prod) Status"),
    ("manProdStatus", "Production (Prod) Status manual override"),
    ("CrabStatus", "CRAB analysis (Crab) Status"),
    ("manCrabStatus", "CRAB analysis (Crab) Status manual override"),
    ("RucioStatus", "Rucio Status"),
    ("manRucioStatus", "Rucio Status manual override"),
    ("SiteReadiness", "Site Readiness"),
    ("SAMsite", "SAM Status"),
    ("FTSsite", "FTS Status"),
    ("HammerCloud", "HammerCloud"),
    ("Downtime", "Downtime"),
    ("Summary", "Summary"),
)

_BIN_SECONDS = 15 * 60
_BINS_PER_DAY = 96
_TAG = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class SiteStatusRecord:
    site: str
    time: int
    alert: str
    msg: str
    ggus: tuple[tuple[int, int], ...]
    #: metric name -> the joined "today" letter string
    today: tuple[tuple[str, str], ...]

    @property
    def url(self) -> str:
        return DETAIL_URL + self.site

    @property
    def title(self) -> str:
        return f"{self.site} site status"

    @property
    def node_id(self) -> str:
        return f"documentation_page:cmssst:{self.site}"

    @property
    def source_record_id(self) -> dict[str, Any]:
        return {"site": self.site}


class CMSSSTSiteStatusSource:
    """Cache-backed CMS SST site status: one page per site."""

    name = "cmssst_site_status"
    profile = "discovery_crawl"
    change_probe_kind = "content_hash"

    def __init__(
        self,
        *,
        records_path: str = "data/cmssst-site-status/records.json",
        sites_path: str | None = None,
        base: str | None = None,
    ) -> None:
        self.records_path = records_path
        self.sites_path = sites_path
        self.base = base
        self.change_probe = content_hash_change_probe(
            cache_paths=self.cache_paths,
            config={"cache_paths": self.cache_paths},
            emit_targets=CMSSSTSiteStatusSource,
            base=base,
        )

    @property
    def cache_paths(self) -> tuple[str, ...]:
        if self.sites_path:
            return (self.records_path, self.sites_path)
        return (self.records_path,)

    def preflight(self, mode: str = "live") -> PreflightResult:
        # Same verdict run() reaches for the same caches.
        try:
            records, skipped = self._records_with_skips()
            _sites, site_skipped = self._known_sites()
        except CacheUnusable as exc:
            return unusable_cache_preflight(
                exc, source_name=self.name, required=False
            )
        status, reason, _complete = _verdict(
            "local CMS SST site status cache present",
            record_count=len(records),
            skipped_count=skipped + site_skipped,
        )
        return PreflightResult(
            source_name=self.name,
            status=status,
            mode="cache",
            required=False,
            record_count=len(records),
            content_hash=content_hash(self.cache_paths, base=self.base),
            reason=reason,
            checked_at=_checked_at(),
        )

    def run(self, run_id: str, *, mode: str = "cursor") -> ConnectorRun:
        try:
            records, skipped = self._records_with_skips()
            known_sites, site_skipped = self._known_sites()
        except CacheUnusable as exc:
            return unusable_cache_run(exc, mode=mode)
        revision = {
            "run_id": run_id,
            "content_hash": content_hash(self.cache_paths, base=self.base),
            "n_records": len(records),
        }

        def _facts() -> Iterator[Any]:
            for record in records:
                yield from _facts_for_record(record, revision, known_sites)

        status, reason, complete = _verdict(
            "local CMS SST site status cache used",
            record_count=len(records),
            skipped_count=skipped + site_skipped,
        )
        return ConnectorRun(
            facts=_facts(),
            completed_scope=(
                mode in {"scope_complete", "reconcile"} and complete
            ),
            run_mode=mode,
            health=ConnectorHealth(
                status=status,
                mode="cache",
                record_count=len(records),
                content_hash=revision["content_hash"],
                reason=reason,
            ),
        )

    def _known_sites(self) -> tuple[set[str], int]:
        """CRIC site names, or an empty set when no sites cache is set.

        A configured sites cache is read as gocdb reads it: missing,
        empty or error-shaped is refused, never an empty site set.
        """
        if not self.sites_path:
            return set(), 0
        skipped = _Skipped()
        sites = _read_objects(
            self.sites_path, base=self.base, skipped=skipped,
            key_pattern=_CMS_SITE_NAME,
        )
        return set(sites), skipped.total

    def _records_with_skips(self) -> tuple[list[SiteStatusRecord], int]:
        payload = read_cache_json(
            self.records_path, expect=list, base=self.base
        )
        return records_from_payload(payload)


def records_from_payload(
    payload: list[Any],
) -> tuple[list[SiteStatusRecord], int]:
    """(records, skipped item count). A repeated site keeps its first
    record (deduplication, not a skip); a malformed item is a skip."""
    records: list[SiteStatusRecord] = []
    seen: set[str] = set()
    skipped = 0
    for item in payload:
        record = _parse_record(item)
        if record is None:
            skipped += 1
            continue
        if record.site in seen:
            continue
        seen.add(record.site)
        records.append(record)
    return records, skipped


def _parse_record(item: Any) -> SiteStatusRecord | None:
    if not isinstance(item, dict):
        return None
    site = item.get("site")
    if not isinstance(site, str) or not SITE_NAME.match(site):
        return None
    snapshot = item.get("time")
    if not isinstance(snapshot, int) or isinstance(snapshot, bool) or snapshot <= 0:
        return None
    ggus = _parse_ggus(item.get("ggus", []))
    metrics = item.get("metrics")
    if ggus is None or not isinstance(metrics, dict) or not metrics:
        return None
    today: list[tuple[str, str]] = []
    for name, bins in metrics.items():
        if not isinstance(bins, dict):
            return None
        letters = _join_bins(bins.get("today"))
        if letters is None:
            return None
        today.append((str(name), letters))
    return SiteStatusRecord(
        site=site,
        time=snapshot,
        alert=_plain(item.get("alert")),
        msg=_plain(item.get("msg")),
        ggus=ggus,
        today=tuple(today),
    )


def _parse_ggus(value: Any) -> tuple[tuple[int, int], ...] | None:
    if not isinstance(value, list):
        return None
    tickets: list[tuple[int, int]] = []
    for entry in value:
        if (
            not isinstance(entry, list)
            or len(entry) != 2
            or not all(
                isinstance(part, int) and not isinstance(part, bool)
                for part in entry
            )
        ):
            return None
        tickets.append((entry[0], entry[1]))
    return tuple(sorted(tickets))


def _join_bins(value: Any) -> str | None:
    """The page joins each range's list of strings (detail.html)."""
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(part, str) for part in value):
        return "".join(value)
    return None


def _plain(value: Any) -> str:
    text = html.unescape(_TAG.sub(" ", str(value or "")))
    return re.sub(r"\s+", " ", text).replace("\x00", " ").strip()


def decode_letter(letter: str) -> str:
    """Plain words for one status letter; an unknown letter says so."""
    return STATUS_WORDS.get(letter, f"unrecognised status code {letter!r}")


def current_status(record: SiteStatusRecord, metric: str) -> str:
    """One metric's status at the snapshot time, as a sentence fragment."""
    letters = dict(record.today).get(metric)
    if letters is None:
        return "not reported on the page"
    midnight = record.time - record.time % 86400
    now = min(_BINS_PER_DAY - 1, (record.time - midnight) // _BIN_SECONDS)
    if metric == "Downtime" or metric.startswith("man"):
        # u here means "none in effect", not "not yet evaluated".
        letter = letters[now] if now < len(letters) else "u"
        if letter != "u":
            return f"{decode_letter(letter)}, at the snapshot time"
        if metric == "Downtime":
            return "no downtime at the snapshot time"
        return "no manual override set at the snapshot time"
    for index in range(min(now, len(letters) - 1), -1, -1):
        if letters[index] != "u":
            return f"{decode_letter(letters[index])}, latest evaluation"
    return "unknown (no evaluation yet today)"


def page_body(record: SiteStatusRecord) -> str:
    lines = [
        f"{record.site} current site status from the CMS Site Support Team "
        "(CMS SST) site status page.",
    ]
    for metric, label in STATUS_METRICS:
        lines.append(f"{label}: {current_status(record, metric)}.")
    if record.ggus:
        tickets = ", ".join(
            f"{ticket} (opened {_date(opened)}, {GGUS_TICKET_URL}{ticket})"
            for ticket, opened in record.ggus
        )
        lines.append(f"GGUS tickets listed for {record.site}: {tickets}.")
    else:
        lines.append(f"GGUS tickets listed for {record.site}: none.")
    if record.alert:
        lines.append(f"Alert: {record.alert}")
    if record.msg:
        lines.append(f"Message: {record.msg}")
    lines.append(f"Source: {record.url}")
    return "\n".join(lines)


def _facts_for_record(
    record: SiteStatusRecord,
    revision: dict[str, Any],
    known_sites: set[str],
) -> Iterator[NodeFact | EdgeFact]:
    body = page_body(record)
    yield NodeFact(
        node_id=record.node_id,
        subtype="documentation_page",
        attrs={
            "label": record.title,
            "title": record.title,
            "url": record.url,
            "body": body,
            "site_name": "cmssst.web.cern.ch",
            "text": f"{record.title}\n{body}",
            "record_kind": "cmssst_site_status",
            "observed_at": _iso(record.time),
        },
        source_record_id=record.source_record_id,
        source_revision=revision,
    )
    for chunk_index, offset, chunk_text in _chunks(body):
        chunk_hash = _sha256(f"{record.node_id}\0{chunk_index}\0{chunk_text}")
        chunk_id = f"chunk:{chunk_hash[:16]}"
        chunk_record_id = {**record.source_record_id, "chunk_index": chunk_index}
        yield NodeFact(
            node_id=chunk_id,
            subtype="document_chunk",
            attrs={
                "chunk_id": chunk_id,
                "content_sha256": _sha256(chunk_text),
                "text": chunk_text,
                "char_offset": offset,
                "char_length": len(chunk_text),
                "char_end": offset + len(chunk_text),
                "chunker_name": CHUNKER_NAME,
                "heading_path": record.title,
            },
            source_record_id=chunk_record_id,
            source_revision=revision,
        )
        yield EdgeFact(
            src=record.node_id,
            dst=chunk_id,
            edge_type="contains",
            attrs={"chunk_index": chunk_index},
            source_record_id=chunk_record_id,
            source_revision=revision,
        )
    if record.site in known_sites:
        yield EdgeFact(
            src=record.node_id,
            dst=f"site:{record.site}",
            edge_type="references",
            attrs={"match_type": "cmssst_site_status"},
            source_record_id=record.source_record_id,
            source_revision=revision,
        )


def _verdict(
    ok_reason: str,
    *,
    record_count: int,
    skipped_count: int,
) -> tuple[str, str, bool]:
    """(status, reason, may claim a complete scope), shared by
    preflight() and run(). An empty list never gets here."""
    status, reason = skipped_items_status(
        status="ok",
        reason=ok_reason,
        record_count=record_count,
        skipped_count=skipped_count,
    )
    return status, reason, not skipped_count


def _date(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%d")


def _iso(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def _checked_at() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class CMSSSTSiteStatusAdapter(ReaderAdapter):
    """Registry adapter for :class:`CMSSSTSiteStatusSource`.

    The registry-entry template in this module's docstring names this
    class. ``profile`` and ``change_probe_kind`` must be string literals;
    see ``ReaderAdapter``.
    """

    reader_class = CMSSSTSiteStatusSource
    profile = "discovery_crawl"
    change_probe_kind = "content_hash"
