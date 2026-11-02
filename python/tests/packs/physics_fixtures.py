"""Synthetic inputs for the analysis-notes and cms-analyses pack tests.

Everything here is invented: note ids in the 2099 and 99 year ranges, made-up
titles and captions, and no real people, analyses or numbers. The shapes are
real (the LaTeX conventions the walker reads, the BibTeX fields, the TWiki
markup) so the tests exercise the same code paths a real mirror does.
"""
from __future__ import annotations

from pathlib import Path

#: A valid 1x1 PNG. The reader only stats image files, but a real image keeps
#: the fixture honest for okg's ingest, which renders them.
TINY_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c6360f8ffff3f0005fe02fea7c8a3720000000049454e44ae426082"
)

#: A bibliography URL naming a TWiki topic; `write_twiki_dump` creates it.
BRIDGE_TOPIC = "TopicTwo"
BRIDGE_PAGE_ID = f"twiki:CMS:{BRIDGE_TOPIC}"

LEGACY_NOTE = "AN-2099-001"
PAG_NOTE = "HIG-99-001"
#: The analysis code a bibliography entry cites; `synthetic_cds_records`
#: holds the summary that carries it.
CITED_CODE = "TOP-99-777"

_MAIN_TEX = r"""\documentclass{article}
\graphicspath{{figs/}}
\title{Synthetic search for a fictional resonance}
\begin{document}
\maketitle
\input{chapters/results}
\end{document}
"""

_INTRO_TEX = r"""\section{Introduction}
This invented note applies b tagging to a fictional dijet resonance. The
invariant mass distribution is shown in Figure~\ref{fig:massplot}.
"""

_SELECTION_TEX = r"""\section{Selection}
Efficiencies come from the tag-and-probe method. % a comment to strip
Events with a Higgs boson candidate are rejected.
"""

_RESULTS_TEX = r"""\section{Results}
\begin{figure}
  \includegraphics[width=0.8\textwidth]{masspeak}
  \caption{Fictional dijet mass spectrum with a toy signal.}
  \label{fig:massplot}
\end{figure}
"""

_BIB = r"""@article{synth_search,
  title = {{A fictional resonance search}},
  author = {{Synthetic Collaboration}},
  journal = {Journal of Fictional Physics},
  year = {2099},
  reportnumber = {CMS-TOP-99-777, CERN-EP-2099-001},
}

@misc{synth_recipe,
  title = {{A fictional recipe page}},
  howpublished = {\url{https://twiki.cern.ch/twiki/bin/view/CMS/TopicTwo}},
  year = {2099},
}

@misc{synth_missing_page,
  title = {{A page that was filtered out}},
  howpublished = {\url{https://twiki.cern.ch/twiki/bin/view/CMS/NotInTheDump}},
  year = {2099},
}

@techreport{synth_conference,
  title = {{A fictional conference record}},
  year = {2099},
  reportnumber = {CMS-CR-2099-001},
}
"""

_PAG_TEX = r"""\documentclass{article}
\title{Synthetic measurement of a toy Higgs boson property}
\begin{document}
\section{Introduction}
A toy W boson control region is used. The jet energy scale is
applied as described in Figure~\ref{fig:missing}.
\begin{figure}
  \includegraphics{not-in-the-mirror}
  \caption{A figure whose image was never committed.}
  \label{fig:missing}
\end{figure}
\end{document}
"""


def write_note_mirror(root: Path) -> Path:
    """A mirror of two invented notes and one directory the walker ignores.

    ``AN-2099-001`` is a legacy note with numbered sections, a chapter below
    the top level reached through ``\\input``, a macros file the walker must
    skip, a figure with an image on disk, and a bibliography. ``HIG-99-001``
    is a single-file PAG-coded note whose figure image is missing.
    """
    root.mkdir(parents=True, exist_ok=True)
    note = root / LEGACY_NOTE
    (note / "chapters").mkdir(parents=True, exist_ok=True)
    (note / "figs").mkdir(exist_ok=True)
    (note / f"{LEGACY_NOTE}.tex").write_text(_MAIN_TEX, encoding="utf-8")
    (note / "01-Introduction.tex").write_text(_INTRO_TEX, encoding="utf-8")
    (note / "02-Selection.tex").write_text(_SELECTION_TEX, encoding="utf-8")
    (note / "mydefs.tex").write_text(r"\newcommand{\toy}{toy}", encoding="utf-8")
    (note / "chapters" / "results.tex").write_text(_RESULTS_TEX, encoding="utf-8")
    (note / "figs" / "masspeak.png").write_bytes(TINY_PNG)
    (note / f"{LEGACY_NOTE}.bib").write_text(_BIB, encoding="utf-8")

    pag = root / PAG_NOTE
    pag.mkdir(exist_ok=True)
    (pag / f"{PAG_NOTE}.tex").write_text(_PAG_TEX, encoding="utf-8")

    (root / "README-not-a-note").mkdir(exist_ok=True)
    (root / "README-not-a-note" / "x.tex").write_text("ignored", encoding="utf-8")
    return root


def write_twiki_dump(root: Path) -> Path:
    """A tiny EOS-style dump: an analysis page, a POG page and the bridge target.

    Topic names follow the CMS convention the cross-link rules read: an
    analysis id with the hyphens dropped, and a POG root in the title.
    """
    root.mkdir(parents=True, exist_ok=True)
    info = '%META:TOPICINFO{author="jdoe" date="1700000000" version="3"}%\n'
    (root / "HIG99001Review.txt").write_text(
        info + "---+ Review page\nThe review of the invented analysis covers b tagging "
        "and the tag-and-probe method.\n",
        encoding="utf-8",
    )
    (root / "MuonPOGRecipes.txt").write_text(
        info + "---+ Muon recipes\nMuon identification and isolation for the toy POG.\n",
        encoding="utf-8",
    )
    (root / f"{BRIDGE_TOPIC}.txt").write_text(
        info + "---+ Topic two\nA page the invented note cites.\n", encoding="utf-8"
    )
    return root


def synthetic_cds_records():
    """Two invented CDS records: the summary behind ``HIG-99-001`` and the
    summary the legacy note cites (``TOP-99-777``)."""
    from archi.sources._cds_parse import CDSRecord, parse_report_number

    def record(recid: str, report: str, title: str) -> CDSRecord:
        number = parse_report_number(report)
        assert number is not None and number.analysis_code
        return CDSRecord(
            oai_id=f"oai:cds.cern.ch:{recid}",
            datestamp="2099-01-01T00:00:00Z",
            sets=("cerncds:cms-pas",),
            title=title,
            description="An invented summary.",
            creators=("Synthetic Collaboration",),
            subjects=("Particle Physics - Experiment",),
            date="2099",
            url=f"http://cds.cern.ch/record/{recid}",
            report_numbers=(number,),
        )

    return [
        record("9990001", "CMS-PAS-HIG-99-001", "Synthetic Higgs summary"),
        record("9990002", "CMS-PAS-TOP-99-777", "Synthetic top summary"),
    ]
