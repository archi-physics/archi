"""Live Indico connector — events read from the API, not from a cache.

archi's other Indico source (:mod:`archi.sources.indico`) reads a
``records.json`` that something else must produce. Nothing does: not
archi, not okg-deployments' cms deployment, not its cms_tools. This
reader closes that gap by talking to Indico directly.

PORTED FROM ARCHI V2, ``archi-physics/archi`` at ``dev``,
``src/data_manager/collectors/scrapers/integrations/indico_scraper.py``
(1,287 lines). What came across is the part worth keeping: the
``/export/event/<id>.json`` metadata call, the
``?detail=contributions`` call, the **timetable fallback** for events
whose contributions endpoint returns empty, and the category walk.

WHAT DID NOT COME ACROSS is v2's authentication. It logs in with
Playwright, driving a browser through CERN SSO and harvesting cookies
mid-run. v3 does not run an interactive login in-process -- see
``archi/auth/cookies.py`` -- and reads a Netscape cookie file produced
out of band by CERN's Kerberos-backed ``auth-get-sso-cookie``, exactly
as the TWiki crawl and the SSO docs source already do. One consequence
is worth stating: this connector never holds a credential. Only the
NAME of the environment variable holding the cookie file's PATH travels
through configuration.

Public Indico categories need no cookie file at all, and the connector
does not demand one.

EMISSION IS NOT REIMPLEMENTED. The reader builds the same
``IndicoEventRecord`` objects the cache-backed source builds and hands
them to the same ``_meeting_node`` / ``_pdf_document_facts`` functions,
so a deployment can swap one source for the other without a schema
change and neither can drift from the other. Only the acquisition
differs, which is the whole point.

ONE RECORD PER TALK (option `contributions`, off by default). With
`contributions=True` each contribution also becomes its own
``meeting_minutes`` record -- title, speakers, session, time, abstract,
its own URL, attachment names and URLs -- linked from the event record
by a ``contains`` edge, so a search for a talk title returns that talk
with its own citation. The event record then stops carrying every
talk's text inline. Off, the output is exactly what it was.

IMAGES OF ATTACHMENTS (option `attachment_images`, off by default, needs
`contributions`). A talk's PDF, PNG and JPEG attachments become
`okg_image` URL values on the talk record (attr `attachment_images`), so an
agent can open them with okg's `inspect(kind="image")`. Nothing is fetched
here: okg fetches the URL at read time, only from hosts the deployment
allowlists under `agent_surface.images.url.prefixes`. Slide decks and other
types stay names and URLs. See the indico pack's `sources/indico_live.yaml`.

Attachment text extraction (v2's MarkItDown slide conversion) is NOT
ported here: it pulls a document-conversion dependency into the
substrate path for a gain this connector can add later, and the
cache-backed source already carries the ``pdf_texts`` field for when it
does. Attachment URLs are emitted; their contents are not fetched.
"""
from __future__ import annotations

import os

import hashlib
import html
import json
import re
import sys
from dataclasses import dataclass, replace
from typing import Any, Iterator

import requests

from okg.deployment import (
    ConnectorHealth,
    ConnectorRun,
    ContentHashProbe,
    EdgeFact,
    ImageRef,
    NodeFact,
    PreflightResult,
)

from archi.auth.cookies import load_cookie_jar_from_env, looks_like_login_page
from archi.sources._sdk_adapter import ReaderAdapter
from archi.sources.indico import (
    DEFAULT_CHUNKER_NAME,
    IndicoEventRecord,
    _meeting_node,
    _pdf_document_facts,
)

DEFAULT_BASE_URL = "https://indico.cern.ch"
DEFAULT_TIMEOUT = 30.0
#: okg fetches an image URL only up to its `agent_surface.images.url.max_bytes`
#: (10 MiB by default); a larger attachment would only ever read as an error.
DEFAULT_MAX_IMAGE_BYTES = 10 * 1024 * 1024
DEFAULT_MAX_IMAGES_PER_TALK = 3


#: Indico's category export path. It is ``/export/categ/``, NOT
#: ``/export/category/`` -- the spelled-out form 404s:
#:
#:     /export/category/100.json -> 404
#:     /export/categ/100.json    -> 200, count=3
#:
#: Measured against indico.cern.ch on 2026-09-29. The event export
#: path IS spelled out (``/export/event/``), which is where the
#: original mistake came from.
CATEGORY_EXPORT = "/export/categ"


class IndicoLiveSource:
    """Read Indico events and their contributions over the REST API."""

    name = "indico_live"
    profile = "discovery_crawl"
    change_probe_kind = "content_hash"

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        event_ids: list[str] | tuple[str, ...] = (),
        category_ids: list[str] | tuple[str, ...] = (),
        default_event_ids: list[str] | tuple[str, ...] = (),
        cookie_file_env: str | None = None,
        contributions: bool = False,
        attachment_images: bool = False,
        max_images_per_talk: int = DEFAULT_MAX_IMAGES_PER_TALK,
        max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
        token_env: str | None = None,
        since: str | None = None,
        required: bool = False,
        max_events: int | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        chunker_name: str = DEFAULT_CHUNKER_NAME,
        records: list[IndicoEventRecord] | None = None,
        base: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        #: Events read only when the configuration gives neither event_ids
        #: nor category_ids. A named scope of either kind replaces them, so a
        #: category-only configuration does not also read the default event.
        if not event_ids and not category_ids:
            event_ids = default_event_ids
        self.event_ids = tuple(str(e) for e in event_ids)
        self.category_ids = tuple(str(c) for c in category_ids)
        if not self.event_ids and not self.category_ids and records is None:
            raise ValueError(
                "configure event_ids or category_ids: a live Indico source "
                "with neither has no scope to read"
            )
        self.cookie_file_env = cookie_file_env
        #: One ``meeting_minutes`` record per talk, linked from its event.
        #: Off by default: it adds records and shortens the event record's
        #: text, which would change what a current user already has.
        self.contributions = contributions
        #: PDF, PNG and JPEG attachments of a talk as `okg_image` URL values.
        #: Off by default; needs `contributions` (the talk record carries them).
        if attachment_images and not contributions:
            raise ValueError(
                "attachment_images needs contributions: true; the images are "
                "attached to the per-talk records"
            )
        if max_images_per_talk < 1 or max_image_bytes < 1:
            raise ValueError(
                "max_images_per_talk and max_image_bytes must be >= 1"
            )
        self.attachment_images = attachment_images
        self.max_images_per_talk = max_images_per_talk
        self.max_image_bytes = max_image_bytes
        #: Name of the env var holding an Indico personal access token
        #: (``indp_…``), as `siteconf` does for GitLab. A token is
        #: preferred over a cookie jar for an unattended ingest: CERN
        #: SSO cookies expire in hours and someone has to own refreshing
        #: them, while an Indico token is long-lived and scoped
        #: (`read:legacy_api` is enough for `/export/`).
        self.token_env = token_env
        #: Indico's own relative-date window, e.g. ``-365d``. Passed
        #: through to `/export/` as `from=`; without it the walk is
        #: bounded only by `max_events`, which truncates arbitrarily and
        #: forfeits the completed-scope claim.
        self.since = since
        self.required = required
        if max_events is not None and max_events < 1:
            raise ValueError(
                f"max_events must be >= 1 (or None for unlimited), "
                f"got {max_events}"
            )
        self.max_events = max_events
        self.timeout = timeout
        self.chunker_name = chunker_name
        self._records = records
        self.base = base
        #: Counters from the last run. This module logs nothing, matching
        #: the rest of archi.sources.
        self.last_harvest_report: dict[str, Any] | None = None
        #: Configured categories that came back with zero events on the
        #: last walk. Reset per harvest; see `_event_ids_in_scope`.
        self._empty_categories: list[str] = []
        self.change_probe = ContentHashProbe(
            content_items=self._probe_content_items,
            config={
                "base_url": self.base_url,
                "event_ids": list(self.event_ids),
                "category_ids": list(self.category_ids),
                "max_events": self.max_events,
                # Only when on, so an existing user's probe hash is unchanged.
                **({"contributions": True} if self.contributions else {}),
                **(
                    {
                        "attachment_images": True,
                        "max_images_per_talk": self.max_images_per_talk,
                        "max_image_bytes": self.max_image_bytes,
                    }
                    if self.attachment_images else {}
                ),
                "since": self.since,
            },
            # The class alone misses the module-level fact builders
            # (`_event_facts`, `_contribution_node`); fingerprint both.
            emit_targets=[IndicoLiveSource, sys.modules[__name__]],
        )

    # ── session ────────────────────────────────────────────────────────

    def _session(self) -> requests.Session:
        session = requests.Session()
        session.headers["User-Agent"] = "archi-okg/indico-live"
        if self.token_env:
            token = os.environ.get(self.token_env, "").strip()
            if token:
                session.headers["Authorization"] = f"Bearer {token}"
        if self.cookie_file_env:
            jar = load_cookie_jar_from_env(self.cookie_file_env)
            if jar is not None:
                session.cookies = jar
        return session

    def _with_window(self, path: str) -> str:
        """Append Indico's `from=` window to an export path.

        Indico accepts a relative date (`-365d`) on `/export/` paths.
        Bounding the walk by date rather than by `max_events` is what
        lets a run still claim a complete scope: a truncated walk saw
        less than its declared scope, a dated walk saw all of it.
        """
        if not self.since or "/export/" not in path:
            return path
        sep = "&" if "?" in path else "?"
        return f"{path}{sep}from={self.since}"

    def _get_json(self, session: requests.Session, path: str) -> Any:
        response = session.get(
            f"{self.base_url}{path}", timeout=self.timeout
        )
        response.raise_for_status()
        if looks_like_login_page(response.text):
            # CERN answers an unauthenticated request with a login page
            # and HTTP 200. Parsing it as JSON would fail; parsing it as
            # an event would publish the login form as meeting minutes.
            raise _LoginRedirect(path)
        return response.json()

    # ── lifecycle ──────────────────────────────────────────────────────

    def preflight(self, mode: str = "live") -> PreflightResult:
        if self._records is not None:
            return PreflightResult(
                source_name=self.name, status="ok", mode="fixture",
                required=self.required, record_count=len(self._records),
                reason="fixture records supplied",
            )
        probe = self.event_ids[0] if self.event_ids else None
        path = (
            f"/export/event/{probe}.json" if probe
            else self._with_window(f"{CATEGORY_EXPORT}/{self.category_ids[0]}.json")
        )
        try:
            self._get_json(self._session(), path)
        except _LoginRedirect:
            return PreflightResult(
                source_name=self.name, status="auth_failed", mode="live",
                required=self.required, endpoint=self.base_url,
                credential_refs=(
                    (self.cookie_file_env,) if self.cookie_file_env else ()
                ),
                reason=(
                    "Indico answered with a login page: the cookie file is "
                    "missing, expired, or for another host. Refresh it with "
                    "CERN's auth-get-sso-cookie (Kerberos-backed); this "
                    "connector never logs in itself"
                ),
            )
        except requests.RequestException as exc:
            return PreflightResult(
                source_name=self.name, status="endpoint_failed", mode="live",
                required=self.required, endpoint=self.base_url,
                reason=f"Indico unreachable: {exc}",
            )
        return PreflightResult(
            source_name=self.name, status="ok", mode="live",
            required=self.required, endpoint=self.base_url,
            credential_refs=(
                (self.cookie_file_env,) if self.cookie_file_env else ()
            ),
            reason="Indico API reachable",
        )

    def run(self, run_id: str, *, mode: str = "cursor") -> ConnectorRun:
        if self._records is not None:
            records = list(self._records)
            report = {"events": len(records), "failed_events": [],
                      "truncated": False, "timetable_fallbacks": 0,
                      "empty_categories": []}
        else:
            records, report = self._harvest()
        self.last_harvest_report = report

        empty_categories = report.get("empty_categories") or []
        complete = (
            not report["failed_events"]
            and not report["truncated"]
            and not empty_categories
        )
        revision = {
            "run_id": run_id,
            "content_hash": _records_hash(records),
            "n_records": len(records),
            "base_url": self.base_url,
        }

        def _facts() -> Iterator[Any]:
            for record in records:
                yield from _event_facts(record, revision, self.chunker_name)

        if not complete:
            # An event that failed to read, or a bound that truncated the
            # walk, means this run saw less than its declared scope.
            # Claiming completion would retract every event it missed.
            failed = report["failed_events"]
            if failed:
                reason = (
                    f"{len(failed)} event(s) failed to read, e.g. "
                    f"{', '.join(failed[:3])}"
                )
            elif empty_categories:
                reason = (
                    f"category {', '.join(empty_categories[:3])} returned "
                    "no events: on indico.cern.ch a gated category answers "
                    "200 with zero results rather than refusing, so an "
                    "empty walk cannot be told from a missing credential"
                )
            else:
                reason = (
                    f"walk truncated at max_events={self.max_events}"
                )
            return ConnectorRun(
                facts=_facts(), completed_scope=False, run_mode=mode,
                health=ConnectorHealth(
                    status=(
                        "endpoint_failed" if failed
                        else "auth_failed" if empty_categories
                        else "ok"
                    ),
                    mode="live", endpoint=self.base_url,
                    record_count=len(records),
                    reason=f"{reason}; no complete scope claimed",
                ),
            )
        return ConnectorRun(
            facts=_facts(),
            completed_scope=(mode in {"scope_complete", "reconcile"}),
            run_mode=mode,
            health=ConnectorHealth(
                status="ok",
                mode="fixture" if self._records is not None else "live",
                endpoint=self.base_url,
                record_count=len(records),
                content_hash=revision["content_hash"],
                credential_refs=(
                    (self.cookie_file_env,) if self.cookie_file_env else ()
                ),
                reason=f"Indico live read of {self.base_url}",
            ),
        )

    # ── harvest ────────────────────────────────────────────────────────

    def _probe_content_items(self) -> list[tuple[str, Any]]:
        if self._records is not None:
            return [
                (r.event_id, json.dumps({"t": r.title, "d": r.date}).encode())
                for r in self._records
            ]
        # The scope's identity: which events, and when each last changed.
        # Reading the metadata call only (not contributions) keeps the
        # probe far cheaper than the run it guards.
        items: list[tuple[str, Any]] = []
        session = self._session()
        for event_id in self._event_ids_in_scope(session, probe=True):
            try:
                payload = self._get_json(
                    session, f"/export/event/{event_id}.json"
                )
            except (requests.RequestException, _LoginRedirect):
                continue
            event = _first_result(payload)
            items.append((
                event_id,
                json.dumps(
                    {"title": event.get("title", ""),
                     "modified": event.get("modificationDate", "")},
                    sort_keys=True, default=str,
                ).encode("utf-8"),
            ))
        return items

    def _event_ids_in_scope(
        self, session: requests.Session, *, probe: bool = False
    ) -> list[str]:
        ids = list(self.event_ids)
        for category_id in self.category_ids:
            try:
                payload = self._get_json(
                    session, self._with_window(f"{CATEGORY_EXPORT}/{category_id}.json")
                )
            except (requests.RequestException, _LoginRedirect):
                if probe:
                    continue
                raise
            found = 0
            for event in payload.get("results", []) or []:
                event_id = str(event.get("id", "")).strip()
                if event_id and event_id not in ids:
                    ids.append(event_id)
                    found += 1
            if found == 0:
                # A category the caller CONFIGURED returned nothing. On
                # indico.cern.ch an SSO-gated category does not redirect
                # and does not 401 -- it answers HTTP 200 with zero
                # results, which is indistinguishable from a quiet week:
                #
                #     /export/categ/100.json  (public)     -> 200 count=3
                #     /export/categ/6803.json (CMS, gated) -> 200 count=0
                #
                # That matters beyond a missing ingest. Under
                # `missing_from_completed_scope`, a COMPLETE scope with
                # no records is an instruction to delete everything this
                # source ingested before, so an expired cookie would
                # quietly empty the meetings out of the graph rather
                # than failing. Record it and refuse the completeness
                # claim; never guess which of the two it was.
                self._empty_categories.append(str(category_id))
        return ids

    def _harvest(self) -> tuple[list[IndicoEventRecord], dict[str, Any]]:
        session = self._session()
        self._empty_categories = []
        failed: list[str] = []
        fallbacks = 0
        records: list[IndicoEventRecord] = []

        try:
            event_ids = self._event_ids_in_scope(session)
        except (requests.RequestException, _LoginRedirect):
            return [], {"events": 0, "failed_events": list(self.category_ids),
                        "truncated": False, "timetable_fallbacks": 0,
                        "empty_categories": list(self.category_ids)}

        truncated = False
        if self.max_events is not None and len(event_ids) > self.max_events:
            event_ids = event_ids[:self.max_events]
            truncated = True

        for event_id in event_ids:
            try:
                payload = self._get_json(
                    session, f"/export/event/{event_id}.json"
                )
            except (requests.RequestException, _LoginRedirect):
                failed.append(event_id)
                continue
            event = _first_result(payload)
            try:
                contributions, from_timetable = self._contributions(
                    session, event_id
                )
            except (requests.RequestException, _LoginRedirect):
                # Only reachable with `contributions` on (see below).
                failed.append(event_id)
                continue
            fallbacks += int(from_timetable)
            records.append(_record_from_event(
                event, event_id, contributions, self.base_url,
                with_contributions=self.contributions,
                image_limit=(
                    self.max_images_per_talk if self.attachment_images else 0
                ),
                image_max_bytes=self.max_image_bytes,
            ))

        return records, {
            "events": len(records),
            "contributions": sum(
                len(getattr(r, "contributions", ())) for r in records
            ),
            "image_urls_refused": sum(
                t.images_refused
                for r in records for t in getattr(r, "contributions", ())
            ),
            "failed_events": failed,
            "truncated": truncated,
            "timetable_fallbacks": fallbacks,
            "empty_categories": list(self._empty_categories),
        }

    def _contributions(
        self, session: requests.Session, event_id: str
    ) -> tuple[list[dict[str, Any]], bool]:
        """Contributions for an event, falling back to the timetable.

        Some events carry their programme in timetable sessions rather
        than as direct contributions, and the contributions endpoint
        returns an empty list for them. v2 handled this and so does
        this: an empty answer is indistinguishable from a meeting with
        no talks unless you look.

        With `contributions` on, a FAILED request (not an empty answer)
        is raised, so the event counts as failed and the run claims no
        complete scope: a silent fallback would drop abstracts and
        unscheduled talks, and a complete scope would retract them.
        """
        try:
            payload = self._get_json(
                session, f"/export/event/{event_id}.json?detail=contributions"
            )
            contributions = _first_result(payload).get("contributions") or []
        except (requests.RequestException, _LoginRedirect):
            if self.contributions:
                raise
            contributions = []
        if contributions:
            return list(contributions), False

        try:
            payload = self._get_json(
                session, f"/export/timetable/{event_id}.json"
            )
        except (requests.RequestException, _LoginRedirect):
            if self.contributions:
                raise
            return [], False
        return _contributions_from_timetable(payload, event_id), True


class _LoginRedirect(RuntimeError):
    """Indico answered with a login page instead of data."""


def _first_result(payload: Any) -> dict[str, Any]:
    """Indico wraps a single object in a ``results`` list."""
    if isinstance(payload, dict):
        results = payload.get("results")
        if isinstance(results, list) and results:
            first = results[0]
            return first if isinstance(first, dict) else {}
        return payload
    return {}


def _contributions_from_timetable(
    payload: Any, event_id: str
) -> list[dict[str, Any]]:
    """Pull contributions out of a timetable payload.

    v2 scraped the timetable's HTML with BeautifulSoup, and through
    Selenium when the page needed JavaScript. This reads Indico's JSON
    timetable instead: same information, no browser, no HTML parsing,
    and nothing that silently returns zero when a page renders
    differently.
    """
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, dict):
        return []
    day_map = results.get(str(event_id)) or next(
        (v for v in results.values() if isinstance(v, dict)), {}
    )
    contributions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for day in (day_map or {}).values():
        if not isinstance(day, dict):
            continue
        for entry in day.values():
            # The enclosing session block names the session its talks
            # belong to; a bare contribution has none.
            session = (
                (entry.get("title") or "").strip()
                if isinstance(entry, dict)
                and entry.get("entryType") == "Session" else ""
            )
            for item in _timetable_items(entry):
                contribution_id = str(
                    item.get("contributionId") or item.get("id") or ""
                ).strip()
                if not contribution_id or contribution_id in seen:
                    continue
                seen.add(contribution_id)
                contributions.append({
                    "id": contribution_id,
                    "title": item.get("title", ""),
                    "description": item.get("description", ""),
                    "speakers": item.get("speakers", []),
                    "folders": item.get("folders", []),
                    "_from_timetable": True,
                    # Read only by the per-contribution records. The
                    # keys above feed the event record and are unchanged.
                    "db_id": contribution_id,
                    "startDate": item.get("startDate"),
                    "endDate": item.get("endDate"),
                    "session": session,
                    # The endpoint says "40/S2-C01"; the timetable says
                    # "40/S2-C01 - Salle ...". Same talk, same text.
                    "room": (item.get("room") or "").split(" - ")[0].strip(),
                    "url": item.get("url", ""),
                    "presenters": [
                        _presenter_like_speaker(p)
                        for p in item.get("presenters") or []
                    ],
                    "attachments": item.get("attachments") or {},
                })
    return contributions


def _presenter_like_speaker(person: Any) -> Any:
    """Write a timetable presenter the way the contributions endpoint
    writes a speaker (``Family, First``), so one talk reads the same on
    both paths."""
    if not isinstance(person, dict):
        return person
    family = (person.get("familyName") or "").strip()
    first = (person.get("firstName") or "").strip()
    full = f"{family}, {first}" if family and first else (
        family or first or (person.get("name") or "").strip()
    )
    return {"fullName": full, "affiliation": person.get("affiliation") or ""}


def _timetable_items(entry: Any) -> Iterator[dict[str, Any]]:
    """Yield contribution-shaped items from one timetable entry.

    A session block nests its talks under ``entries``; a bare
    contribution is the item itself.
    """
    if not isinstance(entry, dict):
        return
    if entry.get("entryType") in {"Contribution", "Break"} or entry.get("contributionId"):
        if entry.get("entryType") != "Break":
            yield entry
    for nested in (entry.get("entries") or {}).values():
        yield from _timetable_items(nested)


def _record_from_event(
    event: dict[str, Any],
    event_id: str,
    contributions: list[dict[str, Any]],
    base_url: str,
    *,
    with_contributions: bool = False,
    image_limit: int = 0,
    image_max_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
) -> IndicoEventRecord:
    """Build the record shape the cache-backed source already emits.

    With `with_contributions` the result is the subclass that also
    carries one :class:`IndicoContribution` per talk; without it the
    result is exactly what it always was."""
    speakers: list[str] = []
    attachments: list[str] = []
    contribution_text: list[str] = []

    for contribution in contributions:
        title = (contribution.get("title") or "").strip()
        description = (contribution.get("description") or "").strip()
        if title or description:
            contribution_text.append(
                " — ".join(part for part in (title, description) if part)
            )
        for speaker in contribution.get("speakers") or []:
            name = _person_name(speaker)
            if name and name not in speakers:
                speakers.append(name)
        for folder in contribution.get("folders") or []:
            for attachment in folder.get("attachments") or []:
                url = attachment.get("download_url") or attachment.get("url")
                if url:
                    attachments.append(
                        url if url.startswith("http") else f"{base_url}{url}"
                    )

    chairs = [
        name for name in (
            _person_name(person) for person in event.get("chairs") or []
        ) if name
    ]
    category_id = event.get("categoryId")
    record_class = IndicoLiveEventRecord if with_contributions else IndicoEventRecord
    extra = (
        {"contributions": _contribution_records(
            event, str(event_id), contributions, base_url,
            image_limit=image_limit, image_max_bytes=image_max_bytes,
        )}
        if with_contributions else {}
    )
    return record_class(
        event_id=str(event_id),
        title=event.get("title", "") or "",
        url=event.get("url", "") or f"{base_url}/event/{event_id}/",
        description=event.get("description", "") or "",
        date=_stamp(event.get("startDate")),
        end_date=_stamp(event.get("endDate")),
        event_type=event.get("type", "") or "",
        category=event.get("category", "") or "",
        category_id=int(category_id) if str(category_id or "").isdigit() else None,
        speakers=tuple(speakers),
        chairs=tuple(chairs),
        attachment_urls=tuple(attachments),
        contributions_text="\n".join(contribution_text),
        **extra,
    )


@dataclass(frozen=True)
class IndicoContribution:
    """One talk, poster or paper inside an event.

    Attachment NAMES and URLs only: their text is not fetched here.
    """

    event_id: str
    contribution_id: str
    title: str
    url: str
    description: str = ""
    date: str = ""
    end_date: str = ""
    session: str = ""
    room: str = ""
    speakers: tuple[str, ...] = ()
    speaker_affiliations: tuple[str, ...] = ()
    attachment_urls: tuple[str, ...] = ()
    attachment_names: tuple[str, ...] = ()
    #: URLs of attachments okg can render (see `_renderable`), capped.
    #: Empty unless the `attachment_images` option is on.
    image_urls: tuple[str, ...] = ()
    #: Renderable attachments whose URL okg's rules refuse (skipped).
    images_refused: int = 0

    @property
    def node_id(self) -> str:
        return f"meeting_minutes:{self.event_id}:contribution:{self.contribution_id}"

    @property
    def source_path(self) -> str:
        return f"event/{self.event_id}/contribution/{self.contribution_id}"


@dataclass(frozen=True)
class IndicoLiveEventRecord(IndicoEventRecord):
    """An event record that also carries its talks as records."""

    contributions: tuple[IndicoContribution, ...] = ()


_TAG_RE = re.compile(r"</?[A-Za-z!][^>]*>")


def _talk_text(value: Any) -> str:
    """Plain text of an Indico field: drop real tags only, decode
    entities, collapse whitespace. Unlike `_strip_html` this keeps text
    such as "a < b > c" and turns "&amp;" into "&"."""
    text = _TAG_RE.sub(" ", str(value or ""))
    return re.sub(r"\s+", " ", html.unescape(text)).replace("\x00", " ").strip()


def _absolute(url: str, base_url: str) -> str:
    return url if url.startswith("http") else f"{base_url}{url}"


#: What okg can turn into an image: PNG, JPEG, and page 1 of a PDF.
_RENDERABLE_TYPES = frozenset({"image/png", "image/jpeg", "application/pdf"})
_RENDERABLE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".pdf")


URL_REFUSED = "url_refused"


def _renderable(
    attachment: dict[str, Any],
    url: str,
    max_bytes: int,
    *,
    folder_protected: bool = False,
) -> str | None:
    """Whether okg can show this attachment through an anonymous fetch.

    The contributions export names a `content_type`; the timetable export
    does not, so the file name's extension decides there. A protected
    attachment is skipped: okg sends no cookies or credentials, so it
    could only read as an error. So is a link entry, a file reported over
    `max_bytes`, and anything else (slide decks, archives).

    Returns None when it is renderable, else the reason. `URL_REFUSED`
    means okg's own URL rules refuse the URL (a `..` segment, a backslash
    or `%5c`, `@` in the host, a non-https scheme, whitespace): the
    attachment is skipped and counted, never allowed to fail the run."""
    if attachment.get("is_protected") or folder_protected:
        return "protected"
    if attachment.get("type") not in (None, "file"):
        return "not_a_file"
    size = attachment.get("size")
    if isinstance(size, int) and size > max_bytes:
        return "too_large"
    content_type = str(attachment.get("content_type") or "").split(";")[0]
    if content_type:
        if content_type.strip().lower() not in _RENDERABLE_TYPES:
            return "not_renderable"
    else:
        name = str(attachment.get("filename") or attachment.get("title") or "")
        path = url.split("?", 1)[0]
        if not (name or path).lower().endswith(_RENDERABLE_EXTENSIONS):
            return "not_renderable"
    # The value is built again when the fact is emitted; if okg refuses it
    # here it would raise there and fail the whole source run.
    try:
        ImageRef.url(url)
    except Exception:
        return URL_REFUSED
    return None


def _contribution_records(
    event: dict[str, Any],
    event_id: str,
    contributions: list[dict[str, Any]],
    base_url: str,
    *,
    image_limit: int = 0,
    image_max_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
) -> tuple[IndicoContribution, ...]:
    """Turn the contribution dicts into per-talk records.

    The contributions endpoint and the timetable fallback name things
    differently (`speakers` / `presenters`, a `folders` list /
    an `attachments.files` list); both are read here. `id` on the
    contributions endpoint is the per-event number, while the URL uses
    `db_id`, so `db_id` names the record when it is present.
    """
    records: list[IndicoContribution] = []
    seen: set[str] = set()
    for item in contributions:
        contribution_id = str(item.get("db_id") or item.get("id") or "").strip()
        if not contribution_id or contribution_id in seen:
            continue
        description = _talk_text(item.get("description"))
        # An untitled talk keeps its abstract: with the option on the
        # event record no longer carries it.
        title = _talk_text(item.get("title")) or (
            description[:80].rstrip() if description
            else f"Contribution {contribution_id}"
        )
        seen.add(contribution_id)
        people = list(item.get("speakers") or []) + list(
            item.get("presenters") or []
        )
        speakers: list[str] = []
        affiliations: list[str] = []
        for person in people:
            name = _person_name(person)
            if not name or name in speakers:
                continue
            speakers.append(name)
            affiliation = (
                str(person.get("affiliation") or "").strip()
                if isinstance(person, dict) else ""
            )
            affiliations.append(f"{name} ({affiliation})" if affiliation else name)
        files = [
            (attachment, bool(folder.get("is_protected")))
            for folder in item.get("folders") or []
            for attachment in (folder.get("attachments") or [])
        ] + [
            (attachment, False)
            for attachment in (item.get("attachments") or {}).get("files") or []
        ]
        urls: list[str] = []
        names: list[str] = []
        images: list[str] = []
        refused = 0
        for attachment, folder_protected in files:
            url = attachment.get("download_url") or attachment.get("url")
            if url and _absolute(url, base_url) not in urls:
                absolute = _absolute(url, base_url)
                urls.append(absolute)
                names.append(str(
                    attachment.get("filename") or attachment.get("title") or ""
                ))
                if len(images) < image_limit:
                    why = _renderable(
                        attachment, absolute, image_max_bytes,
                        folder_protected=folder_protected,
                    )
                    if why is None:
                        images.append(absolute)
                    elif why == URL_REFUSED:
                        refused += 1
        records.append(IndicoContribution(
            event_id=event_id,
            contribution_id=contribution_id,
            title=title,
            url=_absolute(
                item.get("url")
                or f"/event/{event_id}/contributions/{contribution_id}/",
                base_url,
            ),
            description=description,
            date=_stamp(item.get("startDate")),
            end_date=_stamp(item.get("endDate")),
            session=_talk_text(item.get("session")),
            room=str(item.get("room") or "").strip(),
            speakers=tuple(speakers),
            speaker_affiliations=tuple(affiliations),
            attachment_urls=tuple(urls),
            attachment_names=tuple(names),
            image_urls=tuple(images),
            images_refused=refused,
        ))
    return tuple(records)


def _event_facts(
    record: IndicoEventRecord, revision: dict[str, Any], chunker_name: str
) -> Iterator[Any]:
    """The event's facts, then (option on) one record per talk."""
    contributions = getattr(record, "contributions", ())
    if contributions:
        # The talks are records of their own now; repeating all of their
        # text in the event record would make a search for a talk return
        # the whole meeting first.
        yield _meeting_node(replace(record, contributions_text=""), revision)
    else:
        yield _meeting_node(record, revision)
    yield from _pdf_document_facts(record, revision, chunker_name)
    for contribution in contributions:
        yield _contribution_node(record, contribution, revision)
        yield EdgeFact(
            src=record.node_id,
            dst=contribution.node_id,
            edge_type="contains",
            attrs={"relationship": "indico_contribution"},
            source_record_id={"path": contribution.source_path},
            source_revision=revision,
        )


def _contribution_node(
    record: IndicoEventRecord,
    contribution: IndicoContribution,
    revision: dict[str, Any],
) -> NodeFact:
    text = " ".join(filter(None, [
        contribution.title,
        f"Part of {record.title}." if record.title else "",
        f"Session: {contribution.session}." if contribution.session else "",
        "Speakers: " + "; ".join(contribution.speaker_affiliations)
        if contribution.speaker_affiliations else "",
        contribution.description,
        "Attachments: " + ", ".join(n for n in contribution.attachment_names if n)
        if any(contribution.attachment_names) else "",
    ])).strip()
    return NodeFact(
        node_id=contribution.node_id,
        subtype="meeting_minutes",
        attrs={
            "label": contribution.title,
            "title": contribution.title,
            "url": contribution.url,
            "date": contribution.date,
            "end_date": contribution.end_date,
            "event_type": "contribution",
            "category": record.category,
            "category_id": record.category_id,
            "speakers": list(contribution.speakers),
            "chairs": [],
            "attachment_urls": list(contribution.attachment_urls),
            "attachment_names": list(contribution.attachment_names),
            # okg_image URL values, read by inspect(kind="image"). Only
            # present when the `attachment_images` option found some.
            **(
                {"attachment_images": [
                    ImageRef.url(url) for url in contribution.image_urls
                ]}
                if contribution.image_urls else {}
            ),
            "session": contribution.session,
            "room": contribution.room,
            "parent_event_id": record.event_id,
            "parent_event_title": record.title,
            "parent_event_url": record.url,
            "text": text,
        },
        source_record_id={"path": contribution.source_path},
        source_revision=revision,
    )


def _person_name(person: Any) -> str:
    if isinstance(person, str):
        return person.strip()
    if not isinstance(person, dict):
        return ""
    full = (person.get("fullName") or person.get("name") or "").strip()
    if full:
        return full
    parts = [
        (person.get("first_name") or person.get("firstName") or "").strip(),
        (person.get("last_name") or person.get("familyName") or "").strip(),
    ]
    return " ".join(part for part in parts if part)


def _stamp(value: Any) -> str:
    """Indico dates arrive as {"date": ..., "time": ..., "tz": ...}."""
    if isinstance(value, dict):
        date = (value.get("date") or "").strip()
        time = (value.get("time") or "").strip()
        return f"{date}T{time}" if date and time else date
    return str(value or "")


def _records_hash(records: list[IndicoEventRecord]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda r: r.event_id):
        digest.update(record.event_id.encode("utf-8"))
        digest.update(record.title.encode("utf-8"))
        digest.update(record.contributions_text.encode("utf-8"))
        # Only when per-talk records are on: a session, time or speaker
        # change then changes the hash. Off, the hash is what it was.
        for talk in getattr(record, "contributions", ()):
            digest.update(json.dumps(
                [talk.contribution_id, talk.title, talk.session, talk.date,
                 talk.end_date, talk.room, talk.speakers,
                 talk.attachment_urls,
                 *([talk.image_urls] if talk.image_urls else [])],
                default=str,
            ).encode("utf-8"))
    return digest.hexdigest()


class IndicoLiveAdapter(ReaderAdapter):
    """Registry entry point for :class:`IndicoLiveSource`."""

    reader_class = IndicoLiveSource
    profile = "discovery_crawl"
    change_probe_kind = "content_hash"
