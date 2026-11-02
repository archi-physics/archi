"""CMS analysis ids in TWiki topic names.

CMS analysis TWiki topics are named by the analysis id with the hyphens
dropped, because a TWiki topic name is a WikiWord: ``HIG29991`` is
HIG-29-991, ``B2G28992Review`` is B2G-28-992, ``ReviewHIG27993QA`` is
HIG-27-993. People and agents write the id with hyphens (``HIG-29-991``)
or with the ``CMS-`` prefix (``CMS-HIG-29-991``). A lexical index sees
``HIG29991`` and ``HIG-29-991`` as different tokens, so a search for the
written form misses the page. :func:`analysis_ids_from_name` recovers the
id from the topic name; the reader records it and puts both written
forms in the searchable text.

The pattern (all of it must hold, matched on the topic's own name, case
sensitive)::

    <CODE> [-_]? <YY> [-_]? <NNN>      not followed by another digit

* ``CODE`` is one of the codes in :data:`ANALYSIS_CODES`: the current
  CMS physics-analysis group codes plus the older codes the dump still
  carries (``QCD``, ``EWK``, ``FWD``, ``PFT``), ``CFT`` (2009 CRAFT papers) and ``TDR``.
* ``YY`` is a two-digit year, 07 to 29.
* ``NNN`` is a three-digit number, 001 to 999.
* Anything may come before ``CODE`` (``Review``, ``CWR``, ``CMS``, ``Twiki``,
  even a digit: ``Survey13SUS29981``) and anything but a digit may follow (``Review``, ``QA``, ``CWR``, ``ARC``, ``_2``).

Examples, topic name -> ids::

    HIG29991              -> HIG-29-991
    ReviewHIG27993QA      -> HIG-27-993
    B2G28992Review        -> B2G-28-992
    SUS25995CWR           -> SUS-25-995
    CWRFWD29982COMMENT099 -> FWD-29-982
    HIN-21-999Comments    -> HIN-21-999
    SurveySUS29981        -> SUS-29-981
    HIG29991andHIG27993   -> HIG-29-991, HIG-27-993

Not matched::

    NoDigitsTopic           (no digits)
    SUS20990305           (eight digits: a date; the digit rule rejects it)
    HIG299910             (six digits)
    SMPJS21999FR          (letters between the code and the digits)
    B2GAN99310            (an analysis note number, not a paper id)
    HWW22998              (HWW is not an analysis group code)
    HIG29000              (number 000)
    TOP35001              (year 35)
    hig18026              (case sensitive)
"""
from __future__ import annotations

import re

# Current CMS physics-analysis group codes (the PAG codes): B2G BPH BTV
# EGM EXO FSQ FTR HCA HIG HIN JME LUM MUO PRF SMP SUS TAU TOP TRK.
# Added because the TWiki dump names review pages with them: GEN MLG
# (generator and machine-learning groups), PFT (particle flow, old),
# QCD EWK FWD (2010-2012 groups, replaced by SMP and FSQ), CAT NPS TRG
# (recent review pages), CFT (2009 CRAFT papers), TDR (technical design
# reports, same id shape).
ANALYSIS_CODES = (
    "B2G", "BPH", "BTV", "CAT", "CFT", "EGM", "EWK", "EXO", "FSQ", "FTR",
    "FWD", "GEN", "HCA", "HIG", "HIN", "JME", "LUM", "MLG", "MUO", "NPS", "PFT",
    "PRF", "QCD", "SMP", "SUS", "TAU", "TDR", "TOP", "TRG", "TRK",
)

_FIRST_YEAR = 7
_LAST_YEAR = 29

_PATTERN = re.compile(
    r"(" + "|".join(ANALYSIS_CODES) + r")"
    r"(?:[-_]?)([0-9]{2})(?:[-_]?)([0-9]{3})(?![0-9])"
)


def analysis_ids_from_name(name: str) -> tuple[str, ...]:
    """The hyphenated analysis ids in a topic name, in order, no repeats."""
    found: list[str] = []
    for match in _PATTERN.finditer(name):
        code, year, number = match.groups()
        if not _FIRST_YEAR <= int(year) <= _LAST_YEAR or number == "000":
            continue
        analysis_id = f"{code}-{year}-{number}"
        if analysis_id not in found:
            found.append(analysis_id)
    return tuple(found)


def analysis_id_forms(analysis_ids: tuple[str, ...]) -> str:
    """The written forms of each id, for the searchable text.

    ``("HIG-29-991",)`` gives ``"HIG-29-991 CMS-HIG-29-991"``.
    """
    return " ".join(
        form for i in analysis_ids for form in (i, f"CMS-{i}")
    )
