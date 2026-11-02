"""CDS connector — CMS Physics Analysis Summaries, Notes and papers.

CDS is the CERN Document Server: the published and semi-published record
of CMS physics. This reader harvests it over OAI-PMH.

WHY OAI-PMH AND NOT THE SEARCH API (measured 2026-09-25, pact
``cds-connector``): CDS's search and REST endpoints sit behind an
anti-bot interstitial. ``GET /api/records/?q=CMS`` returns HTTP 200 with
an HTML page titled "Making sure you're not a bot!", not JSON. A
connector built on them would appear to work and harvest nothing.
``/oai2d`` answers properly and unauthenticated, advertises
``cerncds:cms-pas`` (2,081 records) and ``cerncds:cms-notes``, and pages
with resumption tokens.

THE REPORT NUMBER IS THE POINT. ``CMS-PAS-HIG-19-001`` shares its
PAG-YY-NNN code with analysis note ``HIG-19-001``, so a deployment
carrying both can answer what was published from a note and which note
backs a summary. The identifier is parsed here, into attributes a
consumer can join on, rather than left as a string for each consumer to
re-parse. On a live page of 500 PAS records, 406 carry a joinable code.

Subtype: ``cds_record``, declared in ``archi/schemas/sources.yaml``. The
substrate's ``literature`` module has a ``Paper`` class, and it was
considered first -- but it is shaped for a frozen abstract corpus
(required ``doc_id`` and ``sentence_count``, children per abstract
sentence), and fitting CDS to it would mean inventing values for fields
that mean something else. Reuse is preferable to a new subtype; reuse by
distortion is not.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterator

import requests

from okg.deployment import (
    ConnectorHealth,
    ConnectorRun,
    ContentHashProbe,
    EdgeFact,
    NodeFact,
    PreflightResult,
)

from archi.sources._sdk_adapter import ReaderAdapter
from archi.sources._cds_parse import (
    CDSOAIError,
    CDSRecord,
    parse_records,
)

DEFAULT_BASE_URL = "https://cds.cern.ch/oai2d"
DEFAULT_SETS: tuple[str, ...] = ("cerncds:cms-pas", "cerncds:cms-notes")
DEFAULT_METADATA_PREFIX = "oai_dc"
DEFAULT_TIMEOUT = 60.0
#: A resumption walk that never ends would hang a build. 200 pages at
#: CDS's 500 records per page is 100k records, far above any CMS set.
MAX_PAGES = 200


class CDSSource:
    """Harvest CDS records for one or more OAI sets."""

    name = "cds"
    profile = "discovery_crawl"
    change_probe_kind = "content_hash"

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        sets: list[str] | tuple[str, ...] = DEFAULT_SETS,
        metadata_prefix: str = DEFAULT_METADATA_PREFIX,
        required: bool = False,
        max_records: int | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        records: list[CDSRecord] | None = None,
        base: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        if not sets:
            raise ValueError(
                "sets must name at least one OAI set (e.g. cerncds:cms-pas)"
            )
        self.sets = tuple(sets)
        self.metadata_prefix = metadata_prefix
        self.required = required
        if max_records is not None and max_records < 1:
            raise ValueError(
                f"max_records must be >= 1 (or None for unlimited), "
                f"got {max_records}"
            )
        self.max_records = max_records
        self.timeout = timeout
        self._records = records
        self.base = base
        #: Counters from the last run, for the operator. This module
        #: logs nothing, matching the rest of archi.sources.
        self.last_harvest_report: dict[str, Any] | None = None
        self.change_probe = ContentHashProbe(
            content_items=self._probe_content_items,
            config={
                "base_url": self.base_url,
                "sets": list(self.sets),
                "metadata_prefix": self.metadata_prefix,
                "max_records": self.max_records,
            },
            emit_targets=CDSSource,
        )

    def _probe_content_items(self) -> list[tuple[str, Any]]:
        """Identity of the current harvest.

        A full re-harvest just to probe would cost as much as the run,
        so the probe asks each set for a single record and uses the
        advertised completeListSize plus the newest datestamp. A set that
        gained, lost or updated a record changes one of the two.
        """
        if self._records is not None:
            return [
                (record.oai_id, json.dumps(
                    {"t": record.title, "d": record.datestamp},
                    sort_keys=True,
                ).encode("utf-8"))
                for record in self._records
            ]
        items: list[tuple[str, Any]] = []
        for set_spec in self.sets:
            try:
                page = self._fetch(
                    {"verb": "ListRecords",
                     "metadataPrefix": self.metadata_prefix,
                     "set": set_spec}
                )
                records, _token, size = parse_records(page)
            except (requests.RequestException, CDSOAIError):
                # A probe that cannot reach the endpoint must not claim
                # "unchanged": leave the set out and let the run decide.
                continue
            newest = max((r.datestamp for r in records), default="")
            items.append((
                set_spec,
                json.dumps({"size": size, "newest": newest}).encode("utf-8"),
            ))
        return items

    # ── lifecycle ──────────────────────────────────────────────────────

    def preflight(self, mode: str = "live") -> PreflightResult:
        if self._records is not None:
            return PreflightResult(
                source_name=self.name, status="ok", mode="fixture",
                required=self.required, record_count=len(self._records),
                reason="fixture records supplied",
            )
        try:
            page = self._fetch({"verb": "Identify"})
        except requests.RequestException as exc:
            return PreflightResult(
                source_name=self.name, status="endpoint_failed", mode="live",
                required=self.required, endpoint=self.base_url,
                reason=f"CDS OAI endpoint unreachable: {exc}",
            )
        if "<Identify>" not in page and "<OAI-PMH" not in page:
            # The anti-bot interstitial answers 200 with HTML. Treat a
            # non-OAI body as a failure rather than parsing it as empty.
            return PreflightResult(
                source_name=self.name, status="endpoint_failed", mode="live",
                required=self.required, endpoint=self.base_url,
                reason=(
                    "CDS returned a non-OAI response to Identify; the "
                    "endpoint may be serving an interstitial"
                ),
            )
        return PreflightResult(
            source_name=self.name, status="ok", mode="live",
            required=self.required, endpoint=self.base_url,
            reason="CDS OAI endpoint reachable",
        )

    def run(self, run_id: str, *, mode: str = "cursor") -> ConnectorRun:
        if self._records is not None:
            records = list(self._records)
            complete = True
            report = {"records": len(records), "sets": list(self.sets),
                      "truncated": False, "incomplete_sets": []}
        else:
            records, complete, report = self._harvest()
        self.last_harvest_report = report

        if report["truncated"] and not report["incomplete_sets"]:
            # The operator's own max_records bound stopped the walk. That is
            # a choice, not a failure: report healthy so admission lets the
            # publish through, but claim no complete scope (a partial harvest
            # must never retract the records it did not fetch).
            return ConnectorRun(
                facts=self._facts(records, run_id),
                completed_scope=False,
                run_mode=mode,
                health=ConnectorHealth(
                    status="ok", mode="live",
                    endpoint=self.base_url, record_count=len(records),
                    content_hash=_records_hash(records),
                    reason=(
                        f"bounded by max_records={self.max_records}: "
                        f"harvested {len(records)} records of "
                        f"{', '.join(self.sets)}; the rest of the set was "
                        f"not fetched, so no complete scope is claimed"
                    ),
                ),
            )
        if not complete:
            # A resumption chain that broke mid-walk: the records that
            # did arrive look fine, so claiming complete scope here would
            # retract every record in the missing tail on the next run.
            return ConnectorRun(
                facts=self._facts(records, run_id),
                completed_scope=False,
                run_mode=mode,
                health=ConnectorHealth(
                    status="endpoint_failed", mode="live",
                    endpoint=self.base_url, record_count=len(records),
                    reason=(
                        f"harvest incomplete for "
                        f"{', '.join(report['incomplete_sets'])}; "
                        f"no complete scope claimed"
                    ),
                ),
            )
        return ConnectorRun(
            facts=self._facts(records, run_id),
            completed_scope=(mode in {"scope_complete", "reconcile"}),
            run_mode=mode,
            health=ConnectorHealth(
                status="ok",
                mode="fixture" if self._records is not None else "live",
                endpoint=self.base_url,
                record_count=len(records),
                content_hash=_records_hash(records),
                reason=f"CDS OAI harvest of {', '.join(self.sets)}",
            ),
        )

    # ── harvest ────────────────────────────────────────────────────────

    def _fetch(self, params: dict[str, str]) -> str:
        response = requests.get(
            self.base_url, params=params, timeout=self.timeout,
            headers={"User-Agent": "archi-okg/cds-connector"},
        )
        response.raise_for_status()
        return response.text

    def _harvest(self) -> tuple[list[CDSRecord], bool, dict[str, Any]]:
        seen: dict[str, CDSRecord] = {}
        incomplete: list[str] = []
        truncated = False
        pages = 0

        for set_spec in self.sets:
            params = {
                "verb": "ListRecords",
                "metadataPrefix": self.metadata_prefix,
                "set": set_spec,
            }
            while True:
                try:
                    page = self._fetch(params)
                    records, token, _size = parse_records(page)
                except (requests.RequestException, CDSOAIError):
                    incomplete.append(set_spec)
                    break
                pages += 1
                for record in records:
                    if record.deleted:
                        continue
                    # A record in several sets arrives more than once.
                    seen.setdefault(record.oai_id, record)
                if self.max_records is not None and len(seen) >= self.max_records:
                    truncated = True
                    break
                if not token:
                    break
                if pages >= MAX_PAGES:
                    incomplete.append(set_spec)
                    break
                params = {"verb": "ListRecords", "resumptionToken": token}
            if truncated:
                break

        records = list(seen.values())
        if self.max_records is not None:
            records = records[:self.max_records]
        report = {
            "records": len(records),
            "sets": list(self.sets),
            "pages": pages,
            "truncated": truncated,
            "incomplete_sets": incomplete,
        }
        # max_records is an operator bound, not a failure -- but it does
        # mean the scope is not the whole set, so it is not "complete".
        return records, (not incomplete and not truncated), report

    def _facts(self, records: list[CDSRecord], run_id: str) -> Iterator[Any]:
        revision = {
            "run_id": run_id,
            "content_hash": _records_hash(records),
            "n_records": len(records),
            "sets": list(self.sets),
        }
        return _facts_for_records(records, revision)


def _records_hash(records: list[CDSRecord]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda r: r.oai_id):
        digest.update(record.oai_id.encode("utf-8"))
        digest.update(record.datestamp.encode("utf-8"))
    return digest.hexdigest()


def _facts_for_records(
    records: list[CDSRecord], revision: dict[str, Any]
) -> Iterator[NodeFact | EdgeFact]:
    for record in records:
        report = record.primary_report_number
        attrs: dict[str, Any] = {
            "label": report.raw if report else record.recid,
            "recid": record.recid,
            "title": record.title,
            "abstract": record.description,
            "authors": "; ".join(record.creators),
            "subjects": "; ".join(record.subjects),
            "date": record.date,
            "url": record.url,
            "oai_id": record.oai_id,
            "sets": "; ".join(record.sets),
            "text": " — ".join(part for part in (
                report.raw if report else "", record.title, record.description,
            ) if part),
        }
        if report is not None:
            attrs["report_number"] = report.raw
            attrs["report_kind"] = report.kind
            # The join key to the analysis-note layer. Empty for report
            # numbers with no PAG token, so a consumer joining on it
            # never matches the wrong thing.
            attrs["analysis_code"] = report.analysis_code
            if report.pag_code:
                attrs["pag_code"] = report.pag_code
        if len(record.report_numbers) > 1:
            attrs["other_report_numbers"] = "; ".join(
                other.raw for other in record.report_numbers
                if other is not report
            )
        yield NodeFact(
            node_id=record.node_id,
            subtype="cds_record",
            attrs=attrs,
            source_record_id={"oai_id": record.oai_id},
            source_revision=revision,
        )


class CDSAdapter(ReaderAdapter):
    """Registry entry point for :class:`CDSSource`.

    ``profile`` and ``change_probe_kind`` are class-level string
    LITERALS on purpose: the substrate reads ``change_probe_kind`` by
    parsing this module's AST before importing anything, and ``profile``
    with ``inspect.getattr_static``. A reference such as
    ``profile = CDSSource.profile`` parses as an attribute and reads as
    absent.
    """

    reader_class = CDSSource
    profile = "discovery_crawl"
    change_probe_kind = "content_hash"
