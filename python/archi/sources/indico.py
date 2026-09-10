"""Cache-backed Indico meeting source.

Ported from okg-deployments ``cms/cms_sources/indico.py`` (385 LOC,
``IndicoSource``) at ``main@f33a9c4`` for the archi v3 package
(req.w2.sources-catalogs). Behavior kept verbatim; only changes: cache
helpers come from :mod:`archi.auth.cache` with an explicit ``base``
parameter, the hardcoded ``data/cms/indico/records.json`` path is a
parameter (default keeps the cms layout minus the ``cms/`` segment),
and the ``https://indico.cern.ch/event/<id>/`` URL fallback is built
from a ``base_url`` parameter (default kept at CERN Indico).

v2 note (documented only, deliberately NOT ported): archi v2 also had
an *on-demand* Indico path —
``src/archi/pipelines/agents/tools/indico_ingest.py`` at
``main@9c9e1cb0`` (162 LOC), a LangChain agent tool that POSTs to the
v2 data-manager's ``/document_index/ingest_local_path`` after an
Indico MCP container (holding the bearer credentials) downloads an
event's attachments to a shared volume. That is an agent-runtime
ingestion trigger, not a source adapter: in v3 the equivalent belongs
to the agent/tools port, and this source stays cache-backed (the cache
already carries enriched contribution text and extracted PDF text).

Registry-entry template — same three prerequisites as
``archi/sources/jira.py``'s template; ``meeting_minutes`` ships in
``archi/schemas/operations.yaml``; ``document`` / ``document_chunk``
come from the ``extraction`` module. ::

    indico:
      module: archi.sources.indico
      class: IndicoSource
      ownership_id: <instance>.indico
      admission_policy:
        producer_id: <instance>.indico
        producer_kind: source
        trust_label: implicit_legacy_trusted
        admission_mode: fast_track
        authority_scope:
          source_family: <family>
          source_name: indico
        output_signature:
          nodes:
            - {subtype: meeting_minutes}
            - {subtype: document}
            - {subtype: document_chunk}
          edges:
            - {src_subtype: meeting_minutes, edge_type: contains, dst_subtype: document}
            - {src_subtype: document, edge_type: contains, dst_subtype: document_chunk}
            - {src_subtype: document_chunk, edge_type: member_of, dst_subtype: document}
        output_scope_summary:
          summary: Indico meetings with parsed PDF attachment documents and chunks
          nodes: [meeting_minutes, document, document_chunk]
          edges:
            - meeting_minutes contains document
            - document contains document_chunk
            - document_chunk member_of document
      source_class: discovery_crawl
      record_identity_kind: remote_id
      record_identity_fields: [event_id]
      source_revision_kind: content_hash
      deletion_semantics: missing_from_completed_scope
      publication_mode: published_generation
      required_for_baseline: true
      params:
        # cms default; the cms deployment used data/cms/indico/records.json
        records_path: data/indico/records.json
        base_url: https://indico.cern.ch
      sync:
        triggers: [manual, reconcile]
        default_event_mode: scope_complete
        reconcile_mode: scope_complete
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator

from okg.deployment import (
    EdgeFact,
    NodeFact,
    ConnectorHealth,
    PreflightResult,
)
from okg.substrate.library.sources.base import SourceRun

from archi.auth.cache import (
    content_hash,
    content_hash_change_probe,
    load_json,
    resolve_repo_path,
)
from archi.sources._cache_report import skipped_items_status

_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_CHUNK_SIZE = 4000
_CHUNK_OVERLAP = 200
DEFAULT_INDICO_BASE_URL = "https://indico.cern.ch"
DEFAULT_CHUNKER_NAME = "cms_indico_pdf_window_v1"


@dataclass(frozen=True)
class IndicoPDFText:
    index: int
    text: str
    title: str = ""
    url: str = ""


@dataclass(frozen=True)
class IndicoEventRecord:
    event_id: str
    title: str
    url: str
    description: str
    date: str
    end_date: str
    event_type: str
    category: str
    category_id: int | None
    speakers: tuple[str, ...] = ()
    chairs: tuple[str, ...] = ()
    attachment_urls: tuple[str, ...] = ()
    contributions_text: str = ""
    pdf_texts: tuple[IndicoPDFText, ...] = ()

    @property
    def node_id(self) -> str:
        return f"meeting_minutes:{self.event_id}"


class IndicoSource:
    """Cache-backed Indico meeting source.

    The local cache already carries enriched contribution text and
    extracted PDF text. PDF text is emitted through the substrate
    `document` / `document_chunk` subtypes without a source-level text
    cap; parent meeting nodes keep only concise event metadata in `text`.
    """

    name = "indico"
    profile = "discovery_crawl"
    change_probe_kind = "content_hash"

    def __init__(
        self,
        *,
        records_path: str = "data/indico/records.json",
        base_url: str = DEFAULT_INDICO_BASE_URL,
        chunker_name: str = DEFAULT_CHUNKER_NAME,
        base: str | None = None,
    ) -> None:
        self.records_path = records_path
        self.base_url = base_url.rstrip("/")
        self.chunker_name = chunker_name
        self.base = base
        self.change_probe = content_hash_change_probe(
            cache_paths=self.cache_paths,
            config={"records_path": self.records_path},
            emit_targets=IndicoSource,
            base=base,
        )

    @property
    def cache_paths(self) -> tuple[str, ...]:
        return (self.records_path,)

    def preflight(self, mode: str = "live") -> PreflightResult:
        path = resolve_repo_path(self.records_path, base=self.base)
        if not path.is_file():
            return PreflightResult(
                source_name=self.name,
                status="cache_missing",
                mode="cache",
                required=True,
                cache_path=str(path),
                reason="Indico cache file is missing",
                checked_at=_checked_at(),
            )
        records = self._records()
        return PreflightResult(
            source_name=self.name,
            status="ok",
            mode="cache",
            required=True,
            record_count=len(records),
            content_hash=content_hash(self.cache_paths, base=self.base),
            reason="local Indico cache present",
            checked_at=_checked_at(),
        )

    def run(self, run_id: str, *, mode: str = "cursor") -> SourceRun:
        records, skipped = self._records_with_skips()
        revision = {
            "run_id": run_id,
            "content_hash": content_hash(self.cache_paths, base=self.base),
            "n_records": len(records),
        }

        def _facts() -> Iterator[Any]:
            for record in records:
                yield _meeting_node(record, revision)
                yield from _pdf_document_facts(
                    record, revision, self.chunker_name
                )

        status, reason = skipped_items_status(
            status="ok",
            reason="local Indico cache used",
            record_count=len(records),
            skipped_count=skipped,
        )
        return SourceRun(
            facts=_facts(),
            completed_scope=(
                mode in {"scope_complete", "reconcile"} and not skipped
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

    def _records(self) -> list[IndicoEventRecord]:
        return self._records_with_skips()[0]

    def _records_with_skips(self) -> tuple[list[IndicoEventRecord], int]:
        payload = load_json(self.records_path, base=self.base)
        if not isinstance(payload, list):
            raise ValueError(
                f"{self.records_path}: expected a JSON list of events"
            )
        records: list[IndicoEventRecord] = []
        seen_event_ids: set[str] = set()
        skipped = 0
        for item in payload:
            if not isinstance(item, dict):
                skipped += 1
                continue
            event_id = str(item.get("id") or item.get("event_id") or "")
            if not event_id:
                skipped += 1
                continue
            # Duplicate event ids are a deliberate dedup (the record is
            # still represented by its first occurrence), not drift.
            if event_id in seen_event_ids:
                continue
            seen_event_ids.add(event_id)
            records.append(_parse_event(event_id, item, self.base_url))
        return records, skipped


def _checked_at() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_event(
    event_id: str,
    item: dict[str, Any],
    base_url: str = DEFAULT_INDICO_BASE_URL,
) -> IndicoEventRecord:
    start = item.get("startDate") or {}
    end = item.get("endDate") or {}
    date = start.get("date", "") if isinstance(start, dict) else ""
    end_date = end.get("date", "") if isinstance(end, dict) else ""
    attachment_urls: list[str] = []
    for folder in item.get("folders") or []:
        if not isinstance(folder, dict):
            continue
        for attachment in folder.get("attachments") or []:
            if not isinstance(attachment, dict):
                continue
            url = str(
                attachment.get("download_url")
                or attachment.get("url")
                or ""
            )
            if url:
                attachment_urls.append(url)
    pdf_texts: list[IndicoPDFText] = []
    for idx, pdf in enumerate(item.get("_pdf_texts") or []):
        if not isinstance(pdf, dict):
            continue
        text = _pg_text(str(pdf.get("text") or ""))
        if not text:
            continue
        pdf_texts.append(IndicoPDFText(
            index=idx,
            text=text,
            title=str(
                pdf.get("title")
                or pdf.get("filename")
                or f"Indico event {event_id} PDF {idx + 1}"
            ),
            url=str(pdf.get("url") or pdf.get("download_url") or ""),
        ))
    return IndicoEventRecord(
        event_id=event_id,
        title=_pg_text(str(item.get("title") or "")),
        url=_pg_text(str(
            item.get("url") or f"{base_url}/event/{event_id}/"
        )),
        description=_strip_html(str(item.get("description") or "")),
        date=date,
        end_date=end_date,
        event_type=_pg_text(str(
            item.get("type") or item.get("event_type") or ""
        )),
        category=_pg_text(str(item.get("category") or "")),
        category_id=_maybe_int(item.get("categoryId")),
        speakers=_names(item.get("speakers") or item.get("creators") or ()),
        chairs=_names(item.get("chairs") or ()),
        attachment_urls=tuple(attachment_urls),
        contributions_text=_pg_text(str(
            item.get("_contributions_text") or ""
        )),
        pdf_texts=tuple(pdf_texts),
    )


def _names(items: Any) -> tuple[str, ...]:
    result: list[str] = []
    if not isinstance(items, list):
        return ()
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(
            item.get("fullName")
            or f"{item.get('first_name', '')} {item.get('last_name', '')}"
        ).strip()
        if name:
            result.append(name)
    return tuple(result)


def _strip_html(text: str) -> str:
    return _pg_text(_WHITESPACE_RE.sub(" ", _TAG_RE.sub(" ", text)).strip())


def _pg_text(text: str) -> str:
    return text.replace("\x00", " ")


def _maybe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _meeting_node(
    record: IndicoEventRecord,
    revision: dict[str, Any],
) -> NodeFact:
    text = " ".join(filter(None, [
        record.title,
        record.category,
        record.description,
        record.contributions_text,
        "Speakers: " + ", ".join(record.speakers)
        if record.speakers else "",
        "Chairs: " + ", ".join(record.chairs) if record.chairs else "",
    ])).strip()
    return NodeFact(
        node_id=record.node_id,
        subtype="meeting_minutes",
        attrs={
            "label": record.title or f"Indico event {record.event_id}",
            "title": record.title,
            "url": record.url,
            "date": record.date,
            "end_date": record.end_date,
            "event_type": record.event_type,
            "category": record.category,
            "category_id": record.category_id,
            "speakers": list(record.speakers),
            "chairs": list(record.chairs),
            "attachment_urls": list(record.attachment_urls),
            "pdf_text_count": len(record.pdf_texts),
            "pdf_text_char_count": sum(len(p.text) for p in record.pdf_texts),
            "text": text,
        },
        source_record_id={"event_id": record.event_id},
        source_revision=revision,
    )


def _pdf_document_facts(
    record: IndicoEventRecord,
    revision: dict[str, Any],
    chunker_name: str = DEFAULT_CHUNKER_NAME,
) -> Iterator[NodeFact | EdgeFact]:
    for pdf in record.pdf_texts:
        doc_id = f"doc:indico:event/{record.event_id}/pdf/{pdf.index}"
        doc_hash = _sha256(pdf.text)
        path = f"event/{record.event_id}/pdf/{pdf.index}.pdf"
        source_record_id = {
            "event_id": record.event_id,
            "pdf_index": pdf.index,
        }
        yield NodeFact(
            node_id=doc_id,
            subtype="document",
            attrs={
                "doc_id": doc_id,
                "repo": "indico",
                "path": path,
                "format": "pdf",
                "byte_count": len(pdf.text.encode("utf-8")),
                "content_sha256": doc_hash,
                "title": pdf.title,
                "source_url": pdf.url,
                "parse_warnings": [],
                "text": pdf.title,
            },
            source_record_id=source_record_id,
            source_revision=revision,
        )
        yield EdgeFact(
            src=record.node_id,
            dst=doc_id,
            edge_type="contains",
            attrs={"relationship": "indico_pdf_attachment"},
            source_record_id=source_record_id,
            source_revision=revision,
        )
        for chunk_index, offset, chunk_text in _chunks(pdf.text):
            chunk_hash = _sha256(chunk_text)
            # Salt the chunk id with the parent doc id + index (the
            # hypernews pattern): a bare content hash collides identical
            # boilerplate from different events onto one node with
            # contradictory parents.
            chunk_id = (
                "chunk:"
                f"{_sha256(f'{doc_id}\\0{chunk_index}\\0{chunk_text}')[:16]}"
            )
            chunk_record_id = {
                **source_record_id,
                "chunk_index": chunk_index,
            }
            yield NodeFact(
                node_id=chunk_id,
                subtype="document_chunk",
                attrs={
                    "chunk_id": chunk_id,
                    "content_sha256": chunk_hash,
                    "text": chunk_text,
                    "char_offset": offset,
                    "char_length": len(chunk_text),
                    "chunker_name": chunker_name,
                    "heading_path": record.title,
                },
                source_record_id=chunk_record_id,
                source_revision=revision,
            )
            yield EdgeFact(
                src=doc_id,
                dst=chunk_id,
                edge_type="contains",
                attrs={"chunk_index": chunk_index},
                source_record_id=chunk_record_id,
                source_revision=revision,
            )
            yield EdgeFact(
                src=chunk_id,
                dst=doc_id,
                edge_type="member_of",
                attrs={"chunk_index": chunk_index},
                source_record_id=chunk_record_id,
                source_revision=revision,
            )


def _chunks(text: str) -> Iterator[tuple[int, int, str]]:
    if not text:
        return
    step = max(1, _CHUNK_SIZE - _CHUNK_OVERLAP)
    idx = 0
    offset = 0
    while offset < len(text):
        chunk = _pg_text(text[offset:offset + _CHUNK_SIZE].strip())
        if chunk:
            yield idx, offset, chunk
            idx += 1
        if offset + _CHUNK_SIZE >= len(text):
            break
        offset += step


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
