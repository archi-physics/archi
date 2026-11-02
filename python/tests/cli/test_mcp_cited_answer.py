"""The golden path's last check reads a cited answer from okg's search text."""
import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "mcp_cited_answer", Path(__file__).with_name("mcp_cited_answer.py"))
mca = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mca)


def _sources(text):
    header = mca.HEADER.search(text)
    return [(node, src or (header.group(1) if header else None))
            for _, node, _, src in mca.ROW.findall(text)]


def test_one_source_is_named_in_the_header():
    text = ('search ok · "q" · source gp.cmssw-releases · rows 1-2\n\n'
            '[1] cmssw_release:A (cmssw_release) · "A"\n'
            '[2] cmssw_release:B (cmssw_release) · "B"\n')
    assert _sources(text) == [("cmssw_release:A", "gp.cmssw-releases"),
                              ("cmssw_release:B", "gp.cmssw-releases")]


def test_several_sources_are_named_on_each_row():
    text = ('search ok · "q" · rows 1-2 of count unknown\n\n'
            '[1] meeting_minutes:1716863 (meeting_minutes) · "K" · source gp.indico-live\n'
            '    an excerpt\n'
            '[2] documentation_page:22 (documentation_page) · "Intro · FTS3" · source gp.docsite\n')
    assert _sources(text) == [("meeting_minutes:1716863", "gp.indico-live"),
                              ("documentation_page:22", "gp.docsite")]


def test_a_row_with_no_source_is_not_a_cited_answer():
    text = 'search ok · "q" · rows 1-1\n\n[1] doc:1 (document) · "x"\n'
    assert _sources(text) == [("doc:1", None)]
