"""The CMS physics organisational vocabulary, as graph nodes.

A Physics Analysis Group and a Physics Object Group are facts about how
CMS is organised, not records in any upstream system. No connector
harvests them because nothing publishes them as data: they are a closed,
slow-moving list that every other source refers to.

Left to the analysis-note reader, PAG hubs would exist only if some note
happened to carry their code. That makes the spine a function of the
corpus rather than of CMS, and it leaves POGs with no home at all, so
TWiki pages naming a POG root could not attach to anything.

Declaring them here instead means:

* all 17 PAGs and all 8 POGs exist whether or not a note mentions them,
* the TWiki layer can link to the same hubs the notes do, and
* the vocabulary is reviewable in one place rather than inferred.

The list is deliberately static. If CMS adds a group, that is a commit,
not a silent change in what a corpus happened to contain.
"""
from __future__ import annotations

import hashlib
from typing import Any, Iterator

from okg.deployment import (
    ConnectorHealth,
    ConnectorRun,
    ContentHashProbe,
    NodeFact,
    PreflightResult,
)

from archi.sources._run_modes import COMPLETED_SCOPE_RUN_MODES
from archi.sources._sdk_adapter import ReaderAdapter

#: Physics Analysis Groups. Codes are the ones that appear in analysis
#: note ids, CDS report numbers and TWiki topic names.
PAGS: tuple[tuple[str, str], ...] = (
    ("HIG", "Higgs"),
    ("TOP", "Top quark"),
    ("SUS", "Supersymmetry"),
    ("SMP", "Standard Model physics"),
    ("B2G", "Beyond two generations"),
    ("BPH", "B physics and quarkonia"),
    ("BTV", "b tagging and vertexing"),
    ("EXO", "Exotica"),
    ("FTR", "Future studies"),
    ("HIN", "Heavy ions"),
    ("JME", "Jets and missing transverse energy"),
    ("MUO", "Muons"),
    ("TAU", "Tau leptons"),
    ("TRK", "Tracking"),
    ("EGM", "Electrons and photons"),
    ("PPS", "Precision proton spectrometer"),
    ("FSQ", "Forward and small-x QCD"),
)

#: Physics Object Groups and the detector-performance groups physicists
#: name interchangeably with them in page titles. The `roots` are the
#: title prefixes TWiki pages actually use, which is what the page-to-hub
#: rule matches on -- physicists write "MuonPOGRecipes", not "MUO".
#: `pag_code` is the token the SAME group carries in the analysis-code
#: namespace, where `PAGS` declares it. Six groups exist in both: an
#: analysis is coded BTV-99-002 while its documentation is titled
#: BTagRecipes, so the pages reached one hub and the notes the other,
#: and the neighbourhood of each group split in two. Carrying the code here
#: lets one rule join the pair, without collapsing either namespace:
#: the analysis-code token and the documentation spelling are both real,
#: and neither is the other's alias.
POGS: tuple[tuple[str, str, tuple[str, ...], str | None], ...] = (
    ("MUON", "Muon POG", ("Muon",), "MUO"),
    ("EGAMMA", "Egamma POG", ("Egamma",), "EGM"),
    ("JETMET", "JetMET POG", ("JetMET", "JERC"), "JME"),
    ("BTAG", "b tagging POG", ("BTag", "BTV"), "BTV"),
    ("TAUPOG", "Tau POG", ("Tau",), "TAU"),
    ("TRACKING", "Tracking POG", ("Tracking",), "TRK"),
    # No analysis-code twin: neither is a PAG token.
    ("TRIGGER", "Trigger", ("Trigger",), None),
    ("ALCA", "Alignment and calibration", ("AlCa",), None),
)


def pag_hub_id(code: str) -> str:
    return f"pag:{code}"


def pog_hub_id(code: str) -> str:
    return f"pog:{code}"


class PhysicsVocabularySource:
    """Emit the PAG and POG hubs the rest of the graph links to."""

    name = "physics_vocabulary"
    profile = "reference_catalog"
    change_probe_kind = "content_hash"

    def __init__(self, *, required: bool = True, base: str | None = None) -> None:
        self.required = required
        self.base = base
        self.change_probe = ContentHashProbe(
            content_items=self._probe_content_items,
            config={"pags": len(PAGS), "pogs": len(POGS)},
            emit_targets=PhysicsVocabularySource,
        )

    def _probe_content_items(self) -> list[tuple[str, Any]]:
        # The vocabulary changes only when this file changes, so its
        # identity IS the list.
        # The roots are part of the identity: they are emitted as node
        # attributes, so a change to them has to invalidate the cached
        # state or the hubs keep whatever the last run wrote.
        payload = ";".join(
            [f"{code}:{label}" for code, label in PAGS]
            + [
                f"{code}:{label}:{'|'.join(roots)}:{pag or ''}"
                for code, label, roots, pag in POGS
            ]
        )
        return [("vocabulary", payload.encode("utf-8"))]

    def preflight(self, mode: str = "live") -> PreflightResult:
        return PreflightResult(
            source_name=self.name,
            status="ok",
            mode="static",
            required=self.required,
            record_count=len(PAGS) + len(POGS),
            reason="CMS physics vocabulary is declared in code",
        )

    def run(self, run_id: str, *, mode: str = "cursor") -> ConnectorRun:
        revision = {
            "run_id": run_id,
            "content_hash": hashlib.sha256(
                self._probe_content_items()[0][1]
            ).hexdigest(),
            "n_records": len(PAGS) + len(POGS),
        }

        def _facts() -> Iterator[Any]:
            for code, label in PAGS:
                yield NodeFact(
                    node_id=pag_hub_id(code),
                    subtype="physics_analysis_group",
                    attrs={
                        "label": code,
                        "code": code,
                        "name": label,
                        "text": f"{code} {label} physics analysis group",
                    },
                    source_record_id={"pag": code},
                    source_revision=revision,
                )
            for code, label, roots, pag_code in POGS:
                yield NodeFact(
                    node_id=pog_hub_id(code),
                    subtype="physics_object_group",
                    attrs={
                        "label": code,
                        "code": code,
                        "name": label,
                        # The title prefixes TWiki pages use, carried so
                        # the page-to-hub rule can match what physicists
                        # actually write.
                        #
                        # Also carried as SCALARS. The declarative
                        # linker's derive_from_path matcher compares a
                        # captured string to one field value by exact
                        # lookup and does not expand a list field, so a
                        # list alone matches nothing. `title_root` is the primary
                        # spelling and `title_root_alt` the second one
                        # where a group has two (BTag/BTV); each gets
                        # its own rule.
                        "title_roots": list(roots),
                        "title_root": roots[0],
                        **({"title_root_alt": roots[1]} if len(roots) > 1 else {}),
                        # The analysis-code token for the same group, where
                        # one exists. Joined to the PAG hub by
                        # cms_analyses_pag_pog_same_group.
                        **({"pag_code": pag_code} if pag_code else {}),
                        "text": f"{code} {label}",
                    },
                    source_record_id={"pog": code},
                    source_revision=revision,
                )

        return ConnectorRun(
            facts=_facts(),
            # See physics_entities for why this reads okg's table rather
            # than a hand-written set: this source is a
            # `reference_catalog`, so okg runs it in a release mode. The
            # vocabulary is compiled in, so there is always something to
            # claim a scope over -- asserted rather than assumed.
            completed_scope=(
                mode in COMPLETED_SCOPE_RUN_MODES and bool(PAGS or POGS)
            ),
            run_mode=mode,
            health=ConnectorHealth(
                status="ok",
                mode="static",
                record_count=len(PAGS) + len(POGS),
                content_hash=revision["content_hash"],
                reason="CMS physics vocabulary",
            ),
        )


class PhysicsVocabularyAdapter(ReaderAdapter):
    """What ``bundles/packs/cms-analyses/sources/physics_vocabulary.yaml`` names.

    ``profile`` and ``change_probe_kind`` are string LITERALS: the
    substrate reads the first with ``inspect.getattr_static`` and the
    second by parsing this module's AST, both before importing anything.
    """

    reader_class = PhysicsVocabularySource
    profile = "reference_catalog"
    change_probe_kind = "content_hash"
