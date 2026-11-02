"""CDS OAI-PMH parsing: records, resumption, and CMS report numbers.

Split from the connector so the parsing is testable without a network
and without the SDK. Everything here is a pure function over XML text.

CDS serves Dublin Core over OAI-PMH. A record's useful identity is
spread across repeated ``dc:identifier`` elements, which carry three
different things with no attribute distinguishing them:

    <dc:identifier>http://cds.cern.ch/record/1149915</dc:identifier>
    <dc:identifier>CMS-PAS-SUS-08-005</dc:identifier>
    <dc:identifier>oai:cds.cern.ch:1149915</dc:identifier>

So they are classified by shape: a URL is the record link, an
``oai:`` string is the OAI id, and what remains is a report number.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from xml.etree import ElementTree

OAI_NS = "{http://www.openarchives.org/OAI/2.0/}"
DC_NS = "{http://purl.org/dc/elements/1.1/}"

#: CMS report numbers: CMS-<KIND>-<PAG>-<YY>-<NNN>, e.g. CMS-PAS-SUS-08-005,
#: and the two-part forms CDS also carries (CMS-NOTE-2019-001).
_REPORT_RE = re.compile(
    r"^(?P<prefix>CMS)-(?P<kind>PAS|NOTE|AN|IN|DP|PAPER|CR)-"
    r"(?:(?P<pag>[A-Z0-9]{2,4})-)?(?P<year>\d{2,4})-(?P<number>\d{3,4})$"
)

#: A PAG-YY-NNN code as an analysis note spells it, e.g. SUS-08-005. This
#: is the join key to the analysis-note layer.
_PAG_CODES = frozenset({
    "HIG", "TOP", "SUS", "SMP", "B2G", "BPH", "BTV", "EXO", "FTR",
    "HIN", "JME", "MUO", "TAU", "TRK", "EGM", "PPS", "FSQ",
})


@dataclass(frozen=True)
class ReportNumber:
    """A parsed CMS report number."""

    raw: str
    kind: str            # PAS | NOTE | AN | IN | DP | PAPER | CR
    pag_code: str = ""   # HIG, SUS, ... empty when the form carries none
    year: str = ""
    number: str = ""

    @property
    def analysis_code(self) -> str:
        """``SUS-08-005`` — the code an analysis note is named by.

        Empty when the report number has no PAG token (``CMS-NOTE-2019-001``
        identifies a note series, not an analysis), so a consumer joining on
        this never matches the wrong thing.
        """
        if not self.pag_code or self.pag_code not in _PAG_CODES:
            return ""
        return f"{self.pag_code}-{self.year}-{self.number}"


def parse_report_number(raw: str) -> ReportNumber | None:
    """Parse a CMS report number, or None when it is not one."""
    match = _REPORT_RE.match(raw.strip())
    if match is None:
        return None
    return ReportNumber(
        raw=raw.strip(),
        kind=match.group("kind"),
        pag_code=match.group("pag") or "",
        year=match.group("year"),
        number=match.group("number"),
    )


@dataclass(frozen=True)
class CDSRecord:
    """One CDS record, as Dublin Core carries it."""

    oai_id: str
    datestamp: str
    sets: tuple[str, ...]
    title: str
    description: str
    creators: tuple[str, ...]
    subjects: tuple[str, ...]
    date: str
    url: str
    report_numbers: tuple[ReportNumber, ...] = field(default_factory=tuple)
    deleted: bool = False

    @property
    def recid(self) -> str:
        """The numeric CDS record id, from the OAI identifier."""
        return self.oai_id.rsplit(":", 1)[-1] if self.oai_id else ""

    @property
    def node_id(self) -> str:
        return f"cds:{self.recid}"

    @property
    def primary_report_number(self) -> ReportNumber | None:
        """The report number to join on when a record carries several.

        Preference order, most specific first: one that yields an
        analysis code, then any. A record with several is not silently
        reduced to "the first one the server happened to emit".
        """
        for report in self.report_numbers:
            if report.analysis_code:
                return report
        return self.report_numbers[0] if self.report_numbers else None


def _text(element, path: str) -> str:
    found = element.find(path)
    return (found.text or "").strip() if found is not None else ""


def _all_text(element, path: str) -> list[str]:
    return [
        (node.text or "").strip()
        for node in element.findall(path)
        if (node.text or "").strip()
    ]


def parse_records(xml_text: str) -> tuple[list[CDSRecord], str | None, int | None]:
    """Parse one OAI ``ListRecords`` page.

    Returns ``(records, resumption_token, complete_list_size)``. The
    token is None on the last page; the size is None when the server
    does not advertise one.
    """
    root = ElementTree.fromstring(xml_text)

    error = root.find(f"{OAI_NS}error")
    if error is not None:
        raise CDSOAIError(
            f"{error.get('code', 'unknown')}: {(error.text or '').strip()}"
        )

    records: list[CDSRecord] = []
    list_records = root.find(f"{OAI_NS}ListRecords")
    if list_records is None:
        return [], None, None

    for node in list_records.findall(f"{OAI_NS}record"):
        header = node.find(f"{OAI_NS}header")
        if header is None:
            continue
        deleted = header.get("status") == "deleted"
        oai_id = _text(header, f"{OAI_NS}identifier")
        sets = tuple(_all_text(header, f"{OAI_NS}setSpec"))
        datestamp = _text(header, f"{OAI_NS}datestamp")

        metadata = node.find(f"{OAI_NS}metadata")
        dc = None if metadata is None else metadata.find(
            "{http://www.openarchives.org/OAI/2.0/oai_dc/}dc"
        )
        if dc is None:
            # A deleted record carries a header and no metadata. Emit it
            # so the caller can see the deletion rather than silently
            # treating the id as absent from the set.
            records.append(CDSRecord(
                oai_id=oai_id, datestamp=datestamp, sets=sets, title="",
                description="", creators=(), subjects=(), date="", url="",
                deleted=True,
            ))
            continue

        url = ""
        reports: list[ReportNumber] = []
        for identifier in _all_text(dc, f"{DC_NS}identifier"):
            if identifier.startswith(("http://", "https://")):
                if not url:
                    url = identifier
            elif identifier.startswith("oai:"):
                continue
            else:
                parsed = parse_report_number(identifier)
                if parsed is not None:
                    reports.append(parsed)

        records.append(CDSRecord(
            oai_id=oai_id,
            datestamp=datestamp,
            sets=sets,
            title=_text(dc, f"{DC_NS}title"),
            description=_text(dc, f"{DC_NS}description"),
            creators=tuple(_all_text(dc, f"{DC_NS}creator")),
            subjects=tuple(_all_text(dc, f"{DC_NS}subject")),
            date=_text(dc, f"{DC_NS}date"),
            url=url,
            report_numbers=tuple(reports),
            deleted=deleted,
        ))

    token_node = list_records.find(f"{OAI_NS}resumptionToken")
    token = (token_node.text or "").strip() if token_node is not None else ""
    size = None
    if token_node is not None:
        raw_size = token_node.get("completeListSize")
        if raw_size and raw_size.isdigit():
            size = int(raw_size)
    return records, (token or None), size


class CDSOAIError(RuntimeError):
    """The OAI endpoint returned an error element."""
