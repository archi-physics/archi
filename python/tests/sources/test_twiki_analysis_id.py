"""pact twiki-analysis-id -- CMS analysis ids from TWiki topic names."""
import pytest
from okg.deployment import NodeFact

from archi.sources._twiki_analysis_id import (
    analysis_id_forms,
    analysis_ids_from_name,
)
from archi.sources.twiki import TwikiEOSSource


@pytest.mark.parametrize(
    "name, expected",
    [
        ("HIG29991", ("HIG-29-991",)),
        ("B2G28992Review", ("B2G-28-992",)),
        ("ReviewHIG27993QA", ("HIG-27-993",)),
        ("EXO26994", ("EXO-26-994",)),
        ("SUS25995CWR", ("SUS-25-995",)),
        ("TWikiSMP12019QA", ("SMP-12-019",)),
        ("CMSHIG24996Comments", ("HIG-24-996",)),
        ("CWRFWD29982COMMENT099", ("FWD-29-982",)),
        ("CFT-09-998-Note", ("CFT-09-998",)),
        # A digit before the code is allowed.
        ("Survey13SUS29981", ("SUS-29-981",)),
        ("2010HIG29991", ("HIG-29-991",)),
        ("ReviewOfNPS23997", ("NPS-23-997",)),
        # Already hyphenated or underscored.
        ("HIN-21-999CommentsFromARC20990101", ("HIN-21-999",)),
        ("SUS_25_995", ("SUS-25-995",)),
        # Two ids, in order, and a repeat counted once.
        ("HIG29991andHIG27993", ("HIG-29-991", "HIG-27-993")),
        ("HIG29991vsHIG29991", ("HIG-29-991",)),
    ],
)
def test_names_with_an_analysis_id(name, expected):
    assert analysis_ids_from_name(name) == expected


@pytest.mark.parametrize(
    "name",
    [
        "NoDigitsTopic",  # no digits
        "WebHome",
        "SUS20990305",  # a date: eight digits
        "HIG299910",  # six digits
        "HIG2999",  # four digits
        "HIG29991011",  # nine digits
        "PPS18026",  # not a listed code
        "SMPJS21999FR",  # letters between code and digits
        "B2GAN99310",  # an analysis note number, not a paper id
        "TOPAN99305",
        "HWW22998",  # not an analysis group code
        "HIG29000",  # number 000
        "TOP35001",  # year out of range
        "SUS05001",
        "hig18026",  # case sensitive
        "Hig18026",
        "Run99359",
        "PROD20990829",
        "",
    ],
)
def test_names_without_an_analysis_id(name):
    assert analysis_ids_from_name(name) == ()


def test_forms_carry_the_hyphenated_and_cms_prefixed_spelling():
    assert analysis_id_forms(("HIG-29-991",)) == "HIG-29-991 CMS-HIG-29-991"
    assert analysis_id_forms(("HIG-29-991", "SUS-25-995")) == (
        "HIG-29-991 CMS-HIG-29-991 SUS-25-995 CMS-SUS-25-995"
    )
    assert analysis_id_forms(()) == ""


TOPIC = (
    '%META:TOPICINFO{author="jdoe" date="1700000000" version="3"}%\n'
    "---+ Review page\nThe signal region is defined here.\n"
)


def _facts(tmp_path, names):
    root = tmp_path / "snapshot"
    root.mkdir()
    for name in names:
        (root / f"{name}.txt").write_text(TOPIC)
    run = TwikiEOSSource(eos_root=str(root)).run("r", mode="scope_complete")
    return list(run.facts)


def _pages(facts):
    return {
        f.attrs["title"]: f
        for f in facts
        if isinstance(f, NodeFact) and f.subtype == "documentation_page"
    }


def _chunk_texts(facts):
    return " ".join(
        f.attrs["text"]
        for f in facts
        if isinstance(f, NodeFact) and f.subtype == "document_chunk"
    )


def test_reader_page_carries_attribute_and_searchable_text(tmp_path):
    facts = _facts(tmp_path, ["HIG29991"])
    page = _pages(facts)["HIG29991"]
    assert page.attrs["analysis_id"] == "HIG-29-991"
    assert "HIG-29-991" in page.attrs["text"].split()
    assert "CMS-HIG-29-991" in page.attrs["text"].split()
    assert page.attrs["text"].endswith(" HIG-29-991 CMS-HIG-29-991")
    assert "Analysis" not in page.attrs["text"]
    # Chunks are untouched: a changed first window would move every chunk
    # id of the page.
    chunks = _chunk_texts(facts)
    assert "HIG-29-991" not in chunks
    assert chunks.startswith("HIG29991 ")
    assert "The signal region is defined here." in chunks


def test_reader_several_ids_are_comma_separated(tmp_path):
    page = _pages(_facts(tmp_path, ["HIG29991andSUS25995"]))[
        "HIG29991andSUS25995"
    ]
    assert page.attrs["analysis_id"] == "HIG-29-991,SUS-25-995"
    assert "CMS-SUS-25-995" in page.attrs["text"].split()


def test_reader_topic_without_an_id_is_unchanged(tmp_path):
    facts = _facts(tmp_path, ["NoIdTopicName", "SUS20990305"])
    pages = _pages(facts)
    for title in ("NoIdTopicName", "SUS20990305"):
        assert "analysis_id" not in pages[title].attrs
        assert "CMS-" not in pages[title].attrs["text"]
    assert "CMS-" not in _chunk_texts(facts)


def test_reader_node_ids_do_not_change(tmp_path):
    facts = _facts(tmp_path, ["HIG29991"])
    ids = {f.node_id for f in facts if isinstance(f, NodeFact)}
    assert "twiki:CMS:HIG29991" in ids
