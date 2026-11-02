"""req.r3-context-quality.gate: a context gate evaluated against the current
generation passes, or fails with each failing check named
(acc.r3-context-quality.gate)."""
import pytest

from archi.ops.context_gate import (
    GateFacts,
    GateSpec,
    GateSpecError,
    evaluate_gate,
    facts_from_doctor,
)

H = 3600


def row(source, age_hours, limit_hours=24):
    return {"source": source, "age_seconds": None if age_hours is None else age_hours * H,
            "max_staleness_seconds": limit_hours * H, "reason": "no completed run recorded in okg's watermarks"
            if age_hours is None else ""}


SPEC = GateSpec.from_mapping({
    "sources": ["cms-catalogs.cric", "monit-live.monit_sam"],
    "min_context_eval_score": 0.8,
    "invariants": ["cms_catalogs_coverage", "cms_catalogs_link_site_config_site"],
})


def healthy(**changes):
    facts = dict(
        generation="okg:myteam:gen:7",
        freshness=(row("cms-catalogs.cric", 3), row("monit-live.monit_sam", 30, 36)),
        invariants={"cms_catalogs_coverage": "pass", "cms_catalogs_link_site_config_site": True},
        context_eval_score=0.91,
    )
    facts.update(changes)
    return GateFacts(**facts)


def test_a_gate_with_every_fact_in_order_passes():
    result = evaluate_gate(SPEC, healthy())
    assert result.ok and result.status == "pass"
    assert result.failing == ()
    assert result.generation == "okg:myteam:gen:7"
    assert [c.name for c in result.checks] == [
        "freshness:cms-catalogs.cric", "freshness:monit-live.monit_sam", "context_eval",
        "invariant:cms_catalogs_coverage", "invariant:cms_catalogs_link_site_config_site"]


def test_a_stale_source_fails_and_is_named():
    result = evaluate_gate(SPEC, healthy(freshness=(row("cms-catalogs.cric", 50), row("monit-live.monit_sam", 30, 36))))
    assert not result.ok
    assert result.failing == ("freshness:cms-catalogs.cric",)
    assert "2d 2h" in result.message and "24h" in result.message


def test_a_source_that_never_ran_or_is_unreported_fails():
    result = evaluate_gate(SPEC, healthy(freshness=(row("cms-catalogs.cric", None),)))
    assert result.failing == ("freshness:cms-catalogs.cric", "freshness:monit-live.monit_sam")


def test_the_gates_own_max_staleness_overrides_the_pack_slo():
    spec = GateSpec.from_mapping({"sources": ["cms-catalogs.cric"], "max_staleness": "2h"})
    assert evaluate_gate(spec, healthy()).failing == ("freshness:cms-catalogs.cric",)  # 3h old
    spec = GateSpec.from_mapping({"sources": ["cms-catalogs.cric"], "max_staleness": "4h"})
    assert evaluate_gate(spec, healthy()).ok


def test_a_low_context_eval_score_fails():
    result = evaluate_gate(SPEC, healthy(context_eval_score=0.5))
    assert result.failing == ("context_eval",)
    assert "below the minimum 0.8" in result.message


def test_a_missing_context_eval_score_is_a_named_failure():
    result = evaluate_gate(SPEC, healthy(context_eval_score=None))
    assert result.failing == ("context_eval",)
    assert "no context-eval score recorded" in result.message


def test_no_minimum_score_means_no_score_is_needed():
    spec = GateSpec.from_mapping({"sources": ["cms-catalogs.cric"]})
    assert evaluate_gate(spec, healthy(context_eval_score=None)).ok


@pytest.mark.parametrize("value", [False, "fail", "error", "warn", 1, 0, ["pass"], None])
def test_a_violated_invariant_fails_and_is_named(value):
    invariants = {"cms_catalogs_coverage": value, "cms_catalogs_link_site_config_site": "pass"}
    assert evaluate_gate(SPEC, healthy(invariants=invariants)).failing == ("invariant:cms_catalogs_coverage",)


def test_an_invariant_with_no_recorded_result_fails():
    result = evaluate_gate(SPEC, healthy(invariants={"cms_catalogs_coverage": "pass"}))
    assert result.failing == ("invariant:cms_catalogs_link_site_config_site",)


def test_every_failing_check_is_named_together():
    facts = healthy(freshness=(row("cms-catalogs.cric", 99),), context_eval_score=None, invariants={})
    result = evaluate_gate(SPEC, facts)
    assert result.failing == (
        "freshness:cms-catalogs.cric", "freshness:monit-live.monit_sam", "context_eval",
        "invariant:cms_catalogs_coverage", "invariant:cms_catalogs_link_site_config_site")
    assert result.as_dict()["status"] == "fail" and len(result.as_dict()["checks"]) == 5


def test_an_empty_gate_passes_vacuously():
    assert evaluate_gate(GateSpec(), GateFacts()).ok


def test_facts_come_from_a_doctor_report():
    report = {"checks": [
        {"id": "published", "message": "published okg:myteam:gen:7"},
        {"id": "source-freshness", "status": "fail", "data": {"sources": [
            {"source": "cms-catalogs.cric", "age_seconds": 3 * H, "max_staleness_seconds": 24 * H},
            {"source": "monit-live.monit_sam", "age_seconds": 80 * H, "max_staleness_seconds": 36 * H}]}},
    ]}
    facts = facts_from_doctor(report, invariants={"cms_catalogs_coverage": "pass"}, context_eval_score=0.9)
    assert facts.generation == "okg:myteam:gen:7"
    result = evaluate_gate(SPEC, facts)
    assert result.failing == ("freshness:monit-live.monit_sam", "invariant:cms_catalogs_link_site_config_site")


def test_a_doctor_report_without_freshness_cannot_pass_a_source_gate():
    facts = facts_from_doctor({"checks": [{"id": "source-freshness", "status": "skip", "data": {}}]})
    assert evaluate_gate(GateSpec(sources=("a.b",)), facts).failing == ("freshness:a.b",)


@pytest.mark.parametrize("bad", [
    {"sourcez": []},
    {"sources": "cms-catalogs.cric"},
    {"sources": [1]},
    {"max_staleness": "soon"},
    {"min_context_eval_score": 1.5},
    {"min_context_eval_score": True},
    {"invariants": "x"},
])
def test_a_malformed_spec_is_refused(bad):
    with pytest.raises(GateSpecError):
        GateSpec.from_mapping(bad)
