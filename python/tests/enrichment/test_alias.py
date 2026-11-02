"""task.w2.enrichment — projection alias backend on a minimal fixture.

The backend loads its indexes through ``_chronos.query``; a fake
returning canned node rows exercises the type-aware matching without a
database.
"""
import archi.enrichment.alias as alias_mod
from archi.enrichment.alias import ProjectionAliasBackend


NODE_ROWS = [
    {"node_id": "site:T2_US_MIT", "subtype": "site", "attrs": {}},
    {
        "node_id": "cmssw_release:CMSSW_15_0_15",
        "subtype": "cmssw_release",
        "attrs": {},
    },
    {"node_id": "jira:CMSCOMPPR-67026", "subtype": "jira_issue", "attrs": {}},
    {
        "node_id": "service:eoscms",
        "subtype": "infrastructure_service",
        "attrs": {"endpoint": "eoscms.cern.ch:1094"},
    },
    # Wrong prefix for its subtype: must not be indexed.
    {"node_id": "weird:thing", "subtype": "site", "attrs": {}},
]
DATASET_ROWS = [
    {"node_id": "dataset:/A/B/RAW"},
    # Case-distinct twin: DBS dataset paths are case-sensitive, so both
    # must stay individually resolvable.
    {"node_id": "dataset:/a/b/raw"},
]


class FakeCursor:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeConn:
    def cursor(self):
        return FakeCursor()


class FakeChronos:
    def __init__(self):
        self.dataset_queries = 0

    def query(self, cur, sql, params=None):
        if "subtype = 'dataset'" in sql:
            self.dataset_queries += 1
            return list(DATASET_ROWS)
        return list(NODE_ROWS)


def _backend(monkeypatch):
    fake = FakeChronos()
    monkeypatch.setattr(alias_mod, "_chronos", fake)
    return ProjectionAliasBackend(FakeConn()), fake


def test_direct_type_scoped_matches(monkeypatch):
    backend, _ = _backend(monkeypatch)
    (match,) = backend.match("T2_US_MIT", entity_type="cms_site")
    assert match.canonical == "site:T2_US_MIT"
    assert match.similarity == 1.0
    assert match.method == "cms_site"
    (match,) = backend.match("cmssw_15_0_15", entity_type="cmssw_release")
    assert match.canonical == "cmssw_release:CMSSW_15_0_15"
    (match,) = backend.match("CMSCOMPPR-67026", entity_type="cms_jira_key")
    assert match.canonical == "jira:CMSCOMPPR-67026"


def test_type_mismatch_and_unknowns_return_empty(monkeypatch):
    backend, _ = _backend(monkeypatch)
    # A site-like value asked for as a release must not cross types.
    assert backend.match("T2_US_MIT", entity_type="cmssw_release") == []
    assert backend.match("T2_US_MIT", entity_type=None) == []
    assert backend.match("", entity_type="cms_site") == []
    assert backend.match("nope", entity_type="not_a_type") == []
    # Wrong node-id prefix for the subtype was never indexed.
    assert backend.match("thing", entity_type="cms_site") == []


def test_hostname_resolves_endpoint_with_and_without_port(monkeypatch):
    backend, _ = _backend(monkeypatch)
    (match,) = backend.match("eoscms.cern.ch:1094", entity_type="cms_hostname")
    assert match.canonical == "service:eoscms"
    (match,) = backend.match("eoscms.cern.ch", entity_type="cms_hostname")
    assert match.canonical == "service:eoscms"
    assert backend.match("other.cern.ch", entity_type="cms_hostname") == []


def test_dataset_index_is_lazy_and_cached(monkeypatch):
    backend, fake = _backend(monkeypatch)
    assert fake.dataset_queries == 0
    (match,) = backend.match("/A/B/RAW", entity_type="cms_dataset")
    assert match.canonical == "dataset:/A/B/RAW"
    assert backend.match("/X/Y/MISS", entity_type="cms_dataset") == []
    assert fake.dataset_queries == 1  # loaded once, then cached


def test_dataset_alias_is_case_sensitive(monkeypatch):
    # circleback finding: _norm case-folding collapsed case-distinct
    # dataset ids into whichever loaded last. DBS paths are
    # case-sensitive, so each case-distinct twin resolves to itself and
    # a case-mismatched query resolves to nothing.
    backend, _ = _backend(monkeypatch)
    (match,) = backend.match("/A/B/RAW", entity_type="cms_dataset")
    assert match.canonical == "dataset:/A/B/RAW"
    (match,) = backend.match("/a/b/raw", entity_type="cms_dataset")
    assert match.canonical == "dataset:/a/b/raw"
    assert backend.match("/A/b/RAW", entity_type="cms_dataset") == []
    # "dataset:"-prefixed needles still resolve (exact case).
    (match,) = backend.match("dataset:/A/B/RAW", entity_type="cms_dataset")
    assert match.canonical == "dataset:/A/B/RAW"


def test_non_dataset_matching_stays_case_insensitive(monkeypatch):
    # Case is transcription noise for sites/releases/hostnames; only
    # dataset paths carry case-sensitive identity.
    backend, _ = _backend(monkeypatch)
    (match,) = backend.match("t2_us_mit", entity_type="cms_site")
    assert match.canonical == "site:T2_US_MIT"
    (match,) = backend.match("EOSCMS.cern.ch", entity_type="cms_hostname")
    assert match.canonical == "service:eoscms"


def test_dataset_misses_are_not_cached(monkeypatch):
    # circleback finding: every unresolved mention used to be cached as
    # a negative entry, growing the index without bound.
    backend, _ = _backend(monkeypatch)
    backend.match("/A/B/RAW", entity_type="cms_dataset")
    size_before = len(backend._dataset_by_value)
    for i in range(50):
        assert backend.match(f"/X/Y{i}/MISS", entity_type="cms_dataset") == []
    assert len(backend._dataset_by_value) == size_before
