"""Physics scoping for the TWiki EOS connector — opt-in, never default.

Lifted from cms-kb's ``src/cms_kb/twiki.py`` (``PhysicsTwikiSource``),
itself a fork of okg-deployments' ``cms/cms_sources/twiki_eos.py`` at
``d259d0df``. cms-kb forked the whole connector to add four filter
stages; this module carries only the stages, so the fork can be
deleted and one connector serves both consumers.

The CMS TWiki snapshot is ~43.5k topics, the large majority of them
computing-operations pages. A physics deployment wants the ~8k that
are physics. The filter runs in four stages:

  1. **Allow-list** — topic names that are unambiguously physics
     (PAG-coded paper pages, POG/DPG name roots, physics workbook and
     SWGuide roots). These are the *seeds*.
  2. **Blacklist** — CompOps/WMCore/DataOps topic roots, dropped
     before anything else looks at them.
  3. **Parent-topic closure** — any topic whose ``%META:TOPICPARENT``
     chain reaches a seed. This is what pulls in a physics page whose
     own name says nothing (e.g. ``StatisticsCommittee``).
  4. **Page type** — a topic that survived 1–3 without being kept is
     still kept when its name classifies as documentation worth having
     (howto, faq, tutorial, glossary, ...).

A parent may be recorded web-qualified (``%META:TOPICPARENT{name=
"CMS.HiggsPhysics"}%``) while topics are keyed by bare name. Stage 3
strips a leading ``<Web>.`` before the lookup, but only when every
dotted part before the topic names a web the snapshot actually has
(its web root or one of its web directories), so a dotted name that is
not web-qualified is left alone. A parent given as a TWiki view URL
(``.../twiki/bin/view/<Web>[/<SubWeb>]/<Topic>``) is read the same way.
cms-kb matched the raw string, so a web-qualified parent stopped the
chain and dropped its subtree.

Nothing here touches the ontology: the filter only ever *selects*
records the connector already produces. Physics-specific vocabulary
(PAG/POG hub nodes and their subtypes) stays with the deployment that
wants it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Sequence

if TYPE_CHECKING:  # pragma: no cover - import cycle at runtime only
    from archi.sources.twiki import TwikiRecord


# ── Stage 1: physics allow-list (validated against a 43,537-file snapshot) ──

# PAG codes with digit follow-through — paper-tracking pages.
PAG_CODE_RE = re.compile(
    r"^(HIG|TOP|SUS|SMP|B2G|BPH|BTV|EXO|FTR|HIN|JME|MUO|TAU|TRK|EGM|PPS|FSQ)"
    r"[-_]?[0-9]"
)

# POG / DPG / coordination name roots — physicists prefix filenames with
# the POG name far more often than with the three-letter code.
POG_NAME_RE = re.compile(r"^(Muon|Egamma|JetMET|BTag|Tau|Tracking|Trigger|AlCa)")

# Physics workbook / SWGuide / PhysicsResults roots. ``Top`` is gated on
# ``[A-Z0-9]`` so it cannot collide with TWiki UI stubs.
PHYSICS_ROOT_RE = re.compile(
    r"^(Higgs|SUSY|Exotica|WorkBook|PhysicsResults|StandardModel|HeavyIons"
    r"|HeavyFlavor|SWGuidePhysics|Top[A-Z0-9])"
)


def passes_allow_list(topic: str) -> bool:
    """True when the topic name alone identifies the page as physics."""
    return bool(
        PAG_CODE_RE.match(topic)
        or POG_NAME_RE.match(topic)
        or PHYSICS_ROOT_RE.match(topic)
    )


# ── Stage 2: computing-operations blacklist ─────────────────────────────

COMPOPS_BLACKLIST_RE = re.compile(
    r"^(CompOps|WMCore|DataOps|SiteReadiness|SiteStatus)"
)


def is_compops_topic(topic: str) -> bool:
    """True for the operations topic roots a physics scope excludes."""
    return bool(COMPOPS_BLACKLIST_RE.match(topic))


# ── Stage 4: page-type classification ───────────────────────────────────

_PAGE_TYPE_TITLE_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^HowTo[A-Z0-9]"), "howto"),
    (re.compile(r"(FAQ|FrequentlyAsked)", re.I), "faq"),
    (re.compile(r"(Glossary|Acronym|Abbreviat|Terminolog)", re.I), "glossary"),
    (re.compile(r"^Tutorial", re.I), "tutorial"),
    (re.compile(r"^(Overview|Introduction)[A-Z0-9]?"), "overview"),
    (re.compile(r"(SiteMap|TopicMap|TopicTree)", re.I), "topicmap"),
    (re.compile(r"Reference$", re.I), "reference"),
    (
        re.compile(
            r"^(HIG|TOP|SUS|SMP|B2G|BPH|BTV|EXO|FTR|HIN|JME|MUO|TAU|TRK|EGM"
            r"|PPS|FSQ)[\-_]?\d{2,}([\-_]?\d{2,})?"
        ),
        "analysis",
    ),
    (re.compile(r"(Minutes$|Meeting|Weekly|Daily)", re.I), "meeting_minutes"),
    (re.compile(r"^Web(Home|Index|TopicList)$"), "web_page"),
)

KEPT_PAGE_TYPES: frozenset[str] = frozenset({
    "analysis",
    "howto",
    "faq",
    "tutorial",
    "glossary",
    "overview",
    "reference",
    "meeting_minutes",
})


def classify_page_type(title: str, parent_topic: str) -> str:
    """Name-and-parent heuristic for what kind of page this is."""
    for pattern, label in _PAGE_TYPE_TITLE_RULES:
        if pattern.search(title):
            return label
    if parent_topic and re.search(
        r"(Weekly|Daily|Minutes|Agenda)", parent_topic, re.I
    ):
        return "meeting_minutes"
    return "other"


# ── Stage 3: parent-topic transitive closure ────────────────────────────

# A parent given as a view URL. TWiki itself often rewrites such a value,
# turning the dots before the last slash into slashes and that slash into
# a dot (``https://twiki/cern/ch/twiki/bin/view/CMS.Page``), so both the
# raw and the rewritten form end in ``<Web>[/<SubWeb>](/|.)<Topic>``.
_VIEW_URL_RE = re.compile(r"/twiki/bin/view(?:auth)?/([^?#]+)$")


def bare_parent_topic(parent: str, web_names: frozenset[str]) -> str:
    """``parent`` without a leading ``<Web>.``, when that prefix is a web.

    ``CMS.HiggsPhysics`` becomes ``HiggsPhysics`` when ``CMS`` is in
    ``web_names``; ``CMS.HiggsWG.Page`` needs both ``CMS`` and
    ``HiggsWG``. A view URL (``.../twiki/bin/view/CMS/HiggsPhysics``) is
    read the same way. Anything else is returned unchanged.
    """
    url = _VIEW_URL_RE.search(parent)
    qualified = url.group(1).replace("/", ".") if url else parent
    web, dot, topic = qualified.rpartition(".")
    if dot and web and topic and all(
        part in web_names for part in web.split(".")
    ):
        return topic
    return parent


def compute_keep_set(
    topics: Sequence[str],
    parent_topic_map: dict[str, str],
    web_names: Iterable[str] = (),
) -> tuple[set[str], set[str]]:
    """Return ``(seed_set, kept_set)``.

    ``seed_set`` is stage 1. ``kept_set`` is ``seed_set`` plus every
    topic whose parent chain reaches a seed. A parent qualified by one
    of ``web_names`` is looked up by its bare topic name. Cycles are
    tolerated: each topic's answer is memoised, so an ancestor walk
    visits a topic once.
    """
    webs = frozenset(web_names)
    seed_set: set[str] = {topic for topic in topics if passes_allow_list(topic)}
    resolved: dict[str, bool] = {}

    def ancestry_hits_seed(topic: str) -> bool:
        walked: list[str] = []
        current: str | None = topic
        while current is not None:
            if current in seed_set:
                outcome = True
                break
            if current in resolved:
                outcome = resolved[current]
                break
            if current in walked:  # cycle
                outcome = False
                break
            walked.append(current)
            parent = parent_topic_map.get(current) or ""
            current = bare_parent_topic(parent, webs) or None
        else:
            outcome = False
        for topic_name in walked:
            resolved[topic_name] = outcome
        return outcome

    kept_set = set(seed_set)
    for topic in topics:
        if topic in kept_set:
            continue
        if ancestry_hits_seed(topic):
            kept_set.add(topic)
    return seed_set, kept_set


# ── The filter as one pass over parsed records ──────────────────────────

@dataclass(frozen=True)
class PhysicsFilterReport:
    """What the filter did, for the run's counters and the operator."""

    input_total: int
    blacklist_dropped: int
    seed_count: int
    closure_count: int
    kept_count: int
    dropped_topics: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "input_total": self.input_total,
            "blacklist_dropped": self.blacklist_dropped,
            "seed_count": self.seed_count,
            "closure_count": self.closure_count,
            "kept_count": self.kept_count,
            "dropped_count": len(self.dropped_topics),
        }


def filter_records(
    records: Iterable["TwikiRecord"],
) -> tuple[list["TwikiRecord"], PhysicsFilterReport]:
    """Apply stages 2-4 to already-parsed records; stage 1 seeds them.

    Record order is preserved. The returned report carries the stage
    counts the connector logs and the ingest run reports.
    """
    all_records = list(records)
    survivors = [
        record for record in all_records
        if not is_compops_topic(record.title)
    ]
    blacklist_dropped = len(all_records) - len(survivors)

    topics = [record.title for record in survivors]
    parent_map = {record.title: record.parent_topic for record in survivors}
    # The webs are the directories a page id sits under, plus the web
    # root; the last page-id part is the topic itself, never a web.
    web_names = {
        part
        for record in survivors
        for part in [*record.web_root.split("/"), *record.page_id.split("/")[:-1]]
        if part
    }
    seed_set, closure_kept = compute_keep_set(topics, parent_map, web_names)

    kept: list["TwikiRecord"] = []
    dropped: list[str] = []
    for record in survivors:
        if record.title in closure_kept:
            kept.append(record)
            continue
        page_type = classify_page_type(record.title, record.parent_topic)
        if page_type in KEPT_PAGE_TYPES:
            kept.append(record)
        else:
            dropped.append(record.title)

    report = PhysicsFilterReport(
        input_total=len(all_records),
        blacklist_dropped=blacklist_dropped,
        seed_count=len(seed_set),
        closure_count=len(closure_kept),
        kept_count=len(kept),
        dropped_topics=tuple(dropped),
    )
    return kept, report
