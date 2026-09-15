"""The guard over ``reports/path-a-vs-path-b.json`` (PACT archi-agent-pipe-bridge).

Why a shape check is not enough
-------------------------------

``req.archi-agent-pipe-bridge.benchmark-gates-cutover`` is explicit that the
report is NOT self-attesting. A guard that validates only field presence and
cross-field consistency accepts a hand-written file carrying plausible
numbers, and that file then becomes the evidence a cutover cites. So the
report must be accompanied by RAW per-question transcripts, and the guard must
RE-DERIVE the report's numbers from them:

* every digest hash resolves to a file that exists and re-hashes to the
  recorded value;
* every question id in a path's answers has a transcript, and every
  transcript has a question id;
* each path's ``completed_runs`` EQUALS that path's transcript count;
* ``observed_generation_ids`` is RECOMPUTED from the transcripts' raw tool
  responses and asserted equal to the set the report declares.

And two declaration-only escapes are closed. A non-OKG inventory delta must
name each differing tool with a reason, and fails UNCONDITIONALLY — declared
or not — when a differing tool is a retrieval, search or grep tool, because
equalizing the arms is the entire reason path (b) is bound to the OKG-only
agent spec. Path (a)'s "unobservable" tool-call count must ESTABLISH itself:
the specific mechanism checked, the specific reason it yields nothing, and
the census clause it depends on. A bare "unobservable" is
absence-dressed-as-agreement and fails.

Anti-vacuity
------------

Every refusal arm below has the same well-formed report as its positive
control: :func:`_well_formed` builds a report and its transcripts that the
guard passes with ZERO failures, and each arm doctors exactly one thing. An
arm that fails for two reasons at once is not evidence that the check it
names works, so the doctoring is kept isolated and the arms assert on the
specific message.

The derived required set is NOT re-typed here: it is recomputed by
``archi.compat.agent_pipe.derive_required_tools`` from the
deployment manifest excerpt the report carries, so a report declaring the
hardcoded seven on a masked deployment fails against the deployment rather
than against a literal in this file.

Hermetic: transcripts are written into ``tmp_path``. No database, no network,
no live instance.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from archi.compat.agent_pipe import (
    HARDCODED_SEVEN,
    POSTURE_EXCLUDED_TOOL,
    derive_required_tools,
)
from okg.chat import CHAT_DEFAULT_TOOLS

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = (
    REPO_ROOT
    / "pact/changes/archi-agent-pipe-bridge/reports/path-a-vs-path-b.json"
)
TRANSCRIPTS_DIR = (
    REPO_ROOT / "pact/changes/archi-agent-pipe-bridge/evidence/transcripts"
)

PATH_IDS = ("a", "b")

REQUIRED_TOP_LEVEL = (
    "question_set",
    "model",
    "openwebui_version",
    "comparison_predicate",
    "grader",
    "deployment",
    "paths",
    "digest",
    "non_okg_inventory_delta",
)

REQUIRED_PER_PATH = (
    "model",
    "completed_runs",
    "tool_inventory",
    "derived_required_tools",
    "non_okg_tool_inventory",
    "tool_calls_attempted",
    "tool_calls_succeeded",
    "observed_generation_ids",
    "answers",
)

#: A differing tool whose name carries any of these could plausibly explain a
#: quality gap on its own, so a delta containing one fails whether or not it
#: was declared. Substring matching is deliberate: ``search_metadata_index``,
#: ``search_vectorstore_hybrid`` and ``grep`` must all be caught.
RISKY_TOOL_MARKERS = (
    "retriev",
    "search",
    "grep",
    "vectorstore",
    "rag",
    "index",
    "lookup",
)


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------

class _Catalog:
    def __init__(self, subtype_metadata: dict[str, Any]) -> None:
        self.subtype_metadata = subtype_metadata


def _catalog_loader_for(posture: Mapping[str, Any], enforcement: Any):
    """A loader that reproduces the report's DECLARED posture.

    The nomos half of the masked predicate short-circuits, so when the
    manifest excerpt settles the posture the catalog is irrelevant. When it
    does not — enforcement absent or ``off`` — the declared posture has to
    rest on ``pii_classes``, and this supplies exactly that.
    """
    if posture.get("masked") and enforcement in (None, "off"):
        return lambda: _Catalog({"okg.person": {"pii_classes": ["name"]}})
    return lambda: _Catalog({"okg.file": {"description": "a file"}})


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_benchmark_report(
    report: Mapping[str, Any], *, transcripts_root: Path,
) -> list[str]:
    """Return every reason the report is not a result. Empty means admissible."""
    failures: list[str] = []

    for key in REQUIRED_TOP_LEVEL:
        if key not in report:
            failures.append(f"report is missing required field {key!r}")
    if failures:
        return failures

    paths = report["paths"]
    for arm in PATH_IDS:
        if arm not in paths:
            failures.append(f"report declares no path {arm!r}")
        else:
            for key in REQUIRED_PER_PATH:
                if key not in paths[arm]:
                    failures.append(f"path {arm}: missing required field {key!r}")
    if failures:
        return failures

    failures.extend(_check_models(report, paths))
    failures.extend(_check_derivation(report, paths))
    failures.extend(_check_completed_runs(paths))
    failures.extend(_check_inventories(paths))
    failures.extend(_check_non_okg_delta(report, paths))
    failures.extend(_check_observability(paths))
    failures.extend(_check_generation_singleton(paths))
    failures.extend(
        _check_digest_linkage(report, paths, transcripts_root=transcripts_root)
    )
    return failures


def _check_models(
    report: Mapping[str, Any], paths: Mapping[str, Any],
) -> list[str]:
    declared = report["model"]
    out = []
    for arm in PATH_IDS:
        if paths[arm]["model"] != declared:
            out.append(
                f"path {arm}: model {paths[arm]['model']!r} differs from the "
                f"report's model {declared!r}; a comparison across two models "
                "answers a different question than the cutover asks"
            )
    return out


def _check_derivation(
    report: Mapping[str, Any], paths: Mapping[str, Any],
) -> list[str]:
    out: list[str] = []
    deployment = report["deployment"]
    for key in ("id", "posture", "raw_manifest_excerpt"):
        if key not in deployment:
            out.append(f"report.deployment is missing {key!r}")
    if out:
        return out

    posture = deployment["posture"]
    manifest = deployment["raw_manifest_excerpt"]
    nomos = manifest.get("nomos") or {}
    enforcement = nomos.get("runtime_enforcement") if isinstance(nomos, Mapping) else None

    if enforcement is not None and enforcement != "off" and not posture.get("masked"):
        out.append(
            f"report.deployment.posture.masked is false, but the manifest "
            f"excerpt declares nomos.runtime_enforcement={enforcement!r}; any "
            "value other than 'off' is masked"
        )
    if (
        posture.get("masked")
        and enforcement in (None, "off")
        and "pii_classes" not in str(posture.get("reason", ""))
    ):
        out.append(
            "report.deployment.posture claims masked with nomos enforcement "
            "off and a reason that does not name pii_classes; the masked "
            "predicate is not satisfied by the excerpt"
        )

    expected = derive_required_tools(
        deployment=str(deployment["id"]),
        raw_manifest=manifest,
        catalog_loader=_catalog_loader_for(posture, enforcement),
    ).required

    for arm in PATH_IDS:
        derived = tuple(paths[arm]["derived_required_tools"])
        if not derived:
            out.append(
                f"path {arm}: derived_required_tools is EMPTY. The empty set "
                "is a subset of every inventory, so this path measured an "
                "agent with no required graph access at all"
            )
            continue
        if set(derived) == HARDCODED_SEVEN and posture.get("masked"):
            out.append(
                f"path {arm}: derived_required_tools is the hardcoded seven "
                f"{sorted(HARDCODED_SEVEN)} on a MASKED deployment, which "
                f"ignores the posture exclusion of {POSTURE_EXCLUDED_TOOL!r}"
            )
            continue
        if set(derived) != set(expected):
            out.append(
                f"path {arm}: derived_required_tools {sorted(derived)} is not "
                f"the set derived from the deployment {sorted(expected)} "
                "(declared tools minus posture exclusions)"
            )
    return out


def _check_completed_runs(paths: Mapping[str, Any]) -> list[str]:
    out = []
    for arm in PATH_IDS:
        runs = paths[arm]["completed_runs"]
        if not isinstance(runs, int) or isinstance(runs, bool) or runs <= 0:
            out.append(
                f"path {arm}: completed_runs is {runs!r}. A path that ran "
                "nothing is not a tie and not a result"
            )
    return out


def _check_inventories(paths: Mapping[str, Any]) -> list[str]:
    out = []
    for arm in PATH_IDS:
        inventory = set(paths[arm]["tool_inventory"])
        missing = [
            tool
            for tool in paths[arm]["derived_required_tools"]
            if tool not in inventory
        ]
        if missing:
            out.append(
                f"path {arm}: tool_inventory lacks {missing} from that path's "
                "derived_required_tools; the arm ran without the graph "
                "operators the deployment requires"
            )
    return out


def _check_non_okg_delta(
    report: Mapping[str, Any], paths: Mapping[str, Any],
) -> list[str]:
    out: list[str] = []
    a_non = set(paths["a"]["non_okg_tool_inventory"])
    b_non = set(paths["b"]["non_okg_tool_inventory"])
    difference = sorted(a_non ^ b_non)
    if not difference:
        return out

    delta = report["non_okg_inventory_delta"]
    for tool in difference:
        lowered = tool.lower()
        if any(marker in lowered for marker in RISKY_TOOL_MARKERS):
            out.append(
                f"non-OKG inventory delta includes {tool!r}, a retrieval / "
                "search / grep tool whose absence could plausibly explain a "
                "quality gap. This fails whether or not the delta was "
                "declared: equalizing the arms is why path (b) is bound to "
                "the OKG-only agent spec"
            )
    if not delta.get("declared"):
        out.append(
            f"the two paths' non_okg_tool_inventory differ on {difference} "
            "with no declared delta"
        )
        return out
    named = {
        str(entry.get("tool"))
        for entry in delta.get("tools", [])
        if str(entry.get("reason", "")).strip()
    }
    unnamed = [tool for tool in difference if tool not in named]
    if unnamed:
        out.append(
            f"the declared non-OKG delta does not name {unnamed} with a "
            "reason it could not be equalized; a declaration is not a "
            "justification"
        )
    return out


def _check_observability(paths: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    for arm in PATH_IDS:
        attempted = paths[arm]["tool_calls_attempted"]
        if isinstance(attempted, bool):
            out.append(f"path {arm}: tool_calls_attempted is a boolean")
            continue
        if isinstance(attempted, int):
            if attempted < 0:
                out.append(f"path {arm}: tool_calls_attempted is negative")
            continue
        if arm != "a":
            out.append(
                f"path {arm}: tool_calls_attempted must be a count; only path "
                "a may record an established observability limitation"
            )
            continue
        if not isinstance(attempted, Mapping):
            out.append(
                f"path {arm}: tool_calls_attempted is neither a count nor an "
                "established observability limitation"
            )
            continue
        if attempted.get("observability") != "established":
            out.append(
                "path a: tool_calls_attempted claims an observability "
                "limitation that is not marked established. A bare "
                "'unobservable' is absence dressed as agreement"
            )
        for field, description in (
            ("mechanism", "the specific mechanism checked"),
            ("reason", "the specific reason it yields nothing"),
            ("census_clause", "the census question it depends on"),
        ):
            if not str(attempted.get(field, "")).strip():
                out.append(
                    f"path a: the observability limitation names no "
                    f"{field} ({description})"
                )
    return out


def _check_generation_singleton(paths: Mapping[str, Any]) -> list[str]:
    observed: set[str] = set()
    for arm in PATH_IDS:
        observed |= set(paths[arm]["observed_generation_ids"])
    if len(observed) != 1:
        return [
            f"observed_generation_ids across both paths is {sorted(observed)}, "
            "which is not a singleton. Generation pinning is an "
            "observed-singleton assertion; a prompt instruction is not a pin"
        ]
    return []


def _check_digest_linkage(
    report: Mapping[str, Any],
    paths: Mapping[str, Any],
    *,
    transcripts_root: Path,
) -> list[str]:
    out: list[str] = []
    digest = report["digest"]
    for key in ("transcripts", "counts"):
        if key not in digest:
            out.append(f"report.digest is missing {key!r}")
    if out:
        return out

    by_arm: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = {
        arm: [] for arm in PATH_IDS
    }
    for entry in digest["transcripts"]:
        relative = str(entry.get("path", ""))
        target = transcripts_root / relative
        if not relative or not target.is_file():
            out.append(
                f"digest names transcript {relative!r}, which does not exist "
                f"under {transcripts_root}"
            )
            continue
        actual = _sha256(target)
        if actual != entry.get("sha256"):
            out.append(
                f"digest hash for {relative!r} is {entry.get('sha256')!r} but "
                f"the file re-hashes to {actual!r}; the report's numbers "
                "cannot be re-derived from it"
            )
            continue
        try:
            data = json.loads(target.read_text())
        except json.JSONDecodeError as exc:
            out.append(f"transcript {relative!r} is not valid JSON ({exc})")
            continue
        question_id = data.get("question_id")
        if not question_id:
            out.append(
                f"transcript {relative!r} carries no question id, so no "
                "answer in the report can be traced to it"
            )
            continue
        if question_id != entry.get("question_id"):
            out.append(
                f"transcript {relative!r} carries question id "
                f"{question_id!r} but the digest records "
                f"{entry.get('question_id')!r}"
            )
            continue
        arm = data.get("path_arm")
        if arm != entry.get("path_arm"):
            out.append(
                f"transcript {relative!r} carries path_arm {arm!r} but the "
                f"digest records {entry.get('path_arm')!r}"
            )
            continue
        if arm not in by_arm:
            out.append(f"transcript {relative!r} names unknown path {arm!r}")
            continue
        by_arm[arm].append((entry, data))

    for arm in PATH_IDS:
        entries = by_arm[arm]
        transcript_ids = {str(data["question_id"]) for _entry, data in entries}
        answer_ids = {
            str(answer.get("question_id")) for answer in paths[arm]["answers"]
        }
        orphan_answers = sorted(answer_ids - transcript_ids)
        if orphan_answers:
            out.append(
                f"path {arm}: question id(s) {orphan_answers} have an answer "
                "but no transcript"
            )
        orphan_transcripts = sorted(transcript_ids - answer_ids)
        if orphan_transcripts:
            out.append(
                f"path {arm}: transcript(s) for {orphan_transcripts} have no "
                "answer in the report"
            )

        declared_count = digest["counts"].get(arm)
        if declared_count != len(entries):
            out.append(
                f"path {arm}: digest.counts records {declared_count!r} "
                f"transcript(s) but {len(entries)} resolved"
            )
        if paths[arm]["completed_runs"] != len(entries):
            out.append(
                f"path {arm}: completed_runs is "
                f"{paths[arm]['completed_runs']} but {len(entries)} "
                "transcript(s) resolved; a completed run with no transcript "
                "is a number with nothing behind it"
            )

        hashes = {str(entry["sha256"]) for entry, _data in entries}
        for answer in paths[arm]["answers"]:
            if str(answer.get("transcript_sha256")) not in hashes:
                out.append(
                    f"path {arm}: answer for "
                    f"{answer.get('question_id')!r} cites transcript hash "
                    f"{answer.get('transcript_sha256')!r}, which is not a "
                    "digest hash for this path"
                )

        recomputed: set[str] = set()
        for _entry, data in entries:
            for call in data.get("tool_calls", []):
                response = call.get("response") or {}
                generation = response.get("generation_id")
                if not generation:
                    out.append(
                        f"path {arm}: a tool response in transcript for "
                        f"{data['question_id']!r} carries no generation_id"
                    )
                else:
                    recomputed.add(str(generation))
        declared_generations = set(paths[arm]["observed_generation_ids"])
        if recomputed != declared_generations:
            out.append(
                f"path {arm}: observed_generation_ids {sorted(declared_generations)} "
                f"does not match the set recomputed from the transcripts "
                f"{sorted(recomputed)}"
            )
    return out


# ---------------------------------------------------------------------------
# A well-formed report and its transcripts — the positive control
# ---------------------------------------------------------------------------

GENERATION = "g-2026-08-11-0007"
QUESTION_IDS = ("human_ranked_top50_01", "human_ranked_top50_02")

#: The fixture deployment is MASKED, so the derived set is the six-operator
#: default and `query` is excluded. That is the configuration a hardcoded
#: seven would refuse forever, which is why it is the default fixture here.
MANIFEST_EXCERPT: dict[str, Any] = {
    "nomos": {"runtime_enforcement": "enforce"},
    "chat": {
        "enabled": True,
        "mcp": {"port": 8100, "tools": [*CHAT_DEFAULT_TOOLS, POSTURE_EXCLUDED_TOOL]},
    },
}

SHARED_NON_OKG_TOOLS = ["grep", "search_vectorstore_hybrid"]


def _transcript(arm: str, question_id: str) -> dict[str, Any]:
    return {
        "path_arm": arm,
        "question_id": question_id,
        "captured_at": "2026-08-11T18:00:00Z",
        "request": {
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": f"question {question_id}"}],
        },
        "response": {"content": f"answer for {question_id} on path {arm}"},
        "tool_calls": [
            {
                "name": "search",
                "arguments": {"query": "live nodes"},
                "response": {
                    "generation_id": GENERATION,
                    "rows": [{"node_id": "n-1"}],
                },
            }
        ],
    }


def _well_formed(tmp_path: Path) -> tuple[dict[str, Any], Path]:
    """Write transcripts and build the report the guard must pass cleanly."""
    root = tmp_path / "transcripts"
    root.mkdir(parents=True, exist_ok=True)

    digest_entries: list[dict[str, Any]] = []
    answers: dict[str, list[dict[str, Any]]] = {"a": [], "b": []}
    for arm in PATH_IDS:
        for question_id in QUESTION_IDS:
            name = f"{arm}__{question_id}.json"
            target = root / name
            target.write_text(
                json.dumps(_transcript(arm, question_id), indent=2, sort_keys=True)
            )
            digest = _sha256(target)
            digest_entries.append({
                "path": name,
                "sha256": digest,
                "path_arm": arm,
                "question_id": question_id,
            })
            answers[arm].append({
                "question_id": question_id,
                "answer": f"answer for {question_id} on path {arm}",
                "quality": {"grade": 4, "grader": "cms-operator-rubric-v2"},
                "transcript_sha256": digest,
            })

    derived = list(CHAT_DEFAULT_TOOLS)
    report: dict[str, Any] = {
        "question_set": {"id": "human_ranked_top50", "size": len(QUESTION_IDS)},
        "model": "gpt-5.5",
        "openwebui_version": "0.11.0",
        "comparison_predicate": (
            "per-question graded quality, path a >= path b on the agreed grade"
        ),
        "grader": "cms-operator-rubric-v2",
        "deployment": {
            "id": "cms",
            "raw_manifest_excerpt": MANIFEST_EXCERPT,
            "posture": {
                "masked": True,
                "indeterminate": False,
                "reason": "nomos.runtime_enforcement is 'enforce'",
            },
        },
        "paths": {
            "a": {
                "model": "gpt-5.5",
                "completed_runs": len(QUESTION_IDS),
                "tool_inventory": derived,
                "derived_required_tools": derived,
                "non_okg_tool_inventory": list(SHARED_NON_OKG_TOOLS),
                "tool_calls_attempted": {
                    "observability": "established",
                    "mechanism": (
                        "Open WebUI v0.11.0 chat completion response payload and "
                        "the per-chat message record; both were inspected"
                    ),
                    "reason": (
                        "middleware records executed tool results only; a model "
                        "tool_call the server rejects before execution leaves no "
                        "row, so attempts cannot be counted from either surface"
                    ),
                    "census_clause": (
                        "openwebui-chat-frontend ui-knob-census.json: no knob "
                        "exposes attempted tool calls; recorded as a named "
                        "unsatisfiable clause"
                    ),
                },
                "tool_calls_succeeded": 2,
                "observed_generation_ids": [GENERATION],
                "answers": answers["a"],
            },
            "b": {
                "model": "gpt-5.5",
                "completed_runs": len(QUESTION_IDS),
                "tool_inventory": derived,
                "derived_required_tools": derived,
                "non_okg_tool_inventory": list(SHARED_NON_OKG_TOOLS),
                "tool_calls_attempted": 2,
                "tool_calls_succeeded": 2,
                "observed_generation_ids": [GENERATION],
                "answers": answers["b"],
            },
        },
        "non_okg_inventory_delta": {"declared": False, "tools": []},
        "digest": {
            "transcripts": digest_entries,
            "counts": {"a": len(QUESTION_IDS), "b": len(QUESTION_IDS)},
        },
    }
    return report, root


def _run(report: Mapping[str, Any], root: Path) -> list[str]:
    return validate_benchmark_report(report, transcripts_root=root)


def _named(failures: Sequence[str], fragment: str) -> bool:
    return any(fragment in failure for failure in failures)


# ---------------------------------------------------------------------------
# The positive control
# ---------------------------------------------------------------------------

def test_a_well_formed_report_passes_the_guard(tmp_path: Path) -> None:
    """The control every doctored arm below is measured against.

    Without it, an arm that "fails" proves nothing: a guard that rejects
    everything rejects the doctored copies too.
    """
    report, root = _well_formed(tmp_path)
    assert _run(report, root) == []


def test_the_fixture_deployment_is_masked_so_query_is_excluded(
    tmp_path: Path,
) -> None:
    """The fixture is the configuration a hardcoded seven refuses forever.

    Stated as its own assertion so a later edit that quietly unmasks the
    fixture — and thereby makes the hardcoded-seven arm below unreachable —
    reds here.
    """
    report, _root = _well_formed(tmp_path)
    assert report["deployment"]["posture"]["masked"] is True
    derived = set(report["paths"]["a"]["derived_required_tools"])
    assert POSTURE_EXCLUDED_TOOL not in derived
    assert derived == set(CHAT_DEFAULT_TOOLS)
    assert derived != HARDCODED_SEVEN


# ---------------------------------------------------------------------------
# The refusal arms
# ---------------------------------------------------------------------------

def test_zero_completed_runs_fails_and_names_the_path(tmp_path: Path) -> None:
    report, root = _well_formed(tmp_path)
    report["paths"]["a"]["completed_runs"] = 0
    failures = _run(report, root)
    assert _named(failures, "path a: completed_runs is 0")
    assert not _named(failures, "path b: completed_runs")


def test_an_inventory_missing_search_fails(tmp_path: Path) -> None:
    report, root = _well_formed(tmp_path)
    report["paths"]["b"]["tool_inventory"] = [
        tool for tool in CHAT_DEFAULT_TOOLS if tool != "search"
    ]
    failures = _run(report, root)
    assert _named(failures, "path b: tool_inventory lacks ['search']")


def test_a_two_element_observed_generation_set_fails(tmp_path: Path) -> None:
    report, root = _well_formed(tmp_path)
    second = "g-2026-08-11-0008"
    report["paths"]["b"]["observed_generation_ids"] = [GENERATION, second]
    failures = _run(report, root)
    assert _named(failures, "is not a singleton")


def test_a_digest_hash_that_does_not_rehash_fails(tmp_path: Path) -> None:
    report, root = _well_formed(tmp_path)
    target = root / "a__human_ranked_top50_01.json"
    data = json.loads(target.read_text())
    data["response"]["content"] = "a different answer than the one recorded"
    target.write_text(json.dumps(data, indent=2, sort_keys=True))
    failures = _run(report, root)
    assert _named(failures, "re-hashes to")
    assert _named(failures, "cannot be re-derived from it")


def test_a_missing_transcript_file_fails(tmp_path: Path) -> None:
    report, root = _well_formed(tmp_path)
    (root / "b__human_ranked_top50_02.json").unlink()
    failures = _run(report, root)
    assert _named(failures, "which does not exist under")


def test_a_question_id_with_no_transcript_fails(tmp_path: Path) -> None:
    """An extra answer, with the run counts left consistent, so the arm is
    isolated to the linkage check it names."""
    report, root = _well_formed(tmp_path)
    orphan_hash = report["digest"]["transcripts"][0]["sha256"]
    report["paths"]["a"]["answers"].append({
        "question_id": "human_ranked_top50_99",
        "answer": "an answer with nothing behind it",
        "quality": {"grade": 5, "grader": "cms-operator-rubric-v2"},
        "transcript_sha256": orphan_hash,
    })
    failures = _run(report, root)
    assert _named(
        failures,
        "question id(s) ['human_ranked_top50_99'] have an answer but no "
        "transcript",
    )


def test_a_transcript_with_no_question_id_fails(tmp_path: Path) -> None:
    """The hash is UPDATED after doctoring, so the re-hash check passes and
    the missing-question-id check is the one that fires."""
    report, root = _well_formed(tmp_path)
    target = root / "a__human_ranked_top50_01.json"
    data = json.loads(target.read_text())
    del data["question_id"]
    target.write_text(json.dumps(data, indent=2, sort_keys=True))
    for entry in report["digest"]["transcripts"]:
        if entry["path"] == target.name:
            entry["sha256"] = _sha256(target)
    failures = _run(report, root)
    assert _named(failures, "carries no question id")
    assert not _named(failures, "re-hashes to")


def test_completed_runs_not_equal_to_the_transcript_count_fails(
    tmp_path: Path,
) -> None:
    report, root = _well_formed(tmp_path)
    report["paths"]["b"]["completed_runs"] = 50
    report["digest"]["counts"]["b"] = 50
    failures = _run(report, root)
    assert _named(failures, "path b: completed_runs is 50 but 2 transcript(s)")
    assert _named(failures, "digest.counts records 50")


def test_generation_ids_that_do_not_recompute_from_the_transcripts_fail(
    tmp_path: Path,
) -> None:
    """The report declares one generation; the transcripts carry another.

    This is the arm that makes the singleton assertion mean something: a
    hand-written report could declare a singleton with nothing behind it.
    """
    report, root = _well_formed(tmp_path)
    for question_id in QUESTION_IDS:
        target = root / f"b__{question_id}.json"
        data = json.loads(target.read_text())
        data["tool_calls"][0]["response"]["generation_id"] = "g-something-else"
        target.write_text(json.dumps(data, indent=2, sort_keys=True))
        for entry in report["digest"]["transcripts"]:
            if entry["path"] == target.name:
                entry["sha256"] = _sha256(target)
    failures = _run(report, root)
    assert _named(failures, "does not match the set recomputed from the transcripts")


def test_a_tool_response_with_no_generation_id_fails(tmp_path: Path) -> None:
    report, root = _well_formed(tmp_path)
    target = root / "a__human_ranked_top50_01.json"
    data = json.loads(target.read_text())
    del data["tool_calls"][0]["response"]["generation_id"]
    target.write_text(json.dumps(data, indent=2, sort_keys=True))
    for entry in report["digest"]["transcripts"]:
        if entry["path"] == target.name:
            entry["sha256"] = _sha256(target)
    failures = _run(report, root)
    assert _named(failures, "carries no generation_id")


def test_an_undeclared_non_okg_delta_fails(tmp_path: Path) -> None:
    report, root = _well_formed(tmp_path)
    report["paths"]["b"]["non_okg_tool_inventory"] = [
        *SHARED_NON_OKG_TOOLS, "monit_fetch_rucio_document",
    ]
    failures = _run(report, root)
    assert _named(failures, "with no declared delta")


def test_a_declared_non_okg_delta_without_per_tool_justification_fails(
    tmp_path: Path,
) -> None:
    """A declaration is not a justification."""
    report, root = _well_formed(tmp_path)
    report["paths"]["b"]["non_okg_tool_inventory"] = [
        *SHARED_NON_OKG_TOOLS, "monit_fetch_rucio_document",
    ]
    report["non_okg_inventory_delta"] = {"declared": True, "tools": []}
    failures = _run(report, root)
    assert _named(failures, "does not name ['monit_fetch_rucio_document']")
    assert _named(failures, "a declaration is not a justification")


def test_a_declared_delta_naming_a_tool_without_a_reason_fails(
    tmp_path: Path,
) -> None:
    report, root = _well_formed(tmp_path)
    report["paths"]["b"]["non_okg_tool_inventory"] = [
        *SHARED_NON_OKG_TOOLS, "monit_fetch_rucio_document",
    ]
    report["non_okg_inventory_delta"] = {
        "declared": True,
        "tools": [{"tool": "monit_fetch_rucio_document", "reason": "   "}],
    }
    failures = _run(report, root)
    assert _named(failures, "does not name ['monit_fetch_rucio_document']")


def test_a_justified_non_retrieval_delta_passes(tmp_path: Path) -> None:
    """The positive control for the delta arms.

    Without it, "a delta fails" could be true because EVERY delta fails, and
    the retrieval-tool arm below would prove nothing extra.
    """
    report, root = _well_formed(tmp_path)
    report["paths"]["b"]["non_okg_tool_inventory"] = [
        *SHARED_NON_OKG_TOOLS, "monit_fetch_rucio_document",
    ]
    report["non_okg_inventory_delta"] = {
        "declared": True,
        "tools": [{
            "tool": "monit_fetch_rucio_document",
            "reason": (
                "MONIT tools are gated on MONIT_GRAFANA_TOKEN and cannot be "
                "supplied to the native arm, which has no secret channel; the "
                "tool fetches Rucio transfer documents and no benchmark "
                "question is answered from them"
            ),
        }],
    }
    assert _run(report, root) == []


def test_a_retrieval_tool_in_the_delta_fails_even_when_justified(
    tmp_path: Path,
) -> None:
    report, root = _well_formed(tmp_path)
    report["paths"]["b"]["non_okg_tool_inventory"] = [
        *SHARED_NON_OKG_TOOLS, "search_metadata_index",
    ]
    report["non_okg_inventory_delta"] = {
        "declared": True,
        "tools": [{
            "tool": "search_metadata_index",
            "reason": "the native arm has no metadata index to search",
        }],
    }
    failures = _run(report, root)
    assert _named(failures, "whose absence could plausibly explain a quality gap")
    assert _named(failures, "This fails whether or not the delta was declared")


def test_a_bare_unobservable_without_a_mechanism_fails(tmp_path: Path) -> None:
    report, root = _well_formed(tmp_path)
    report["paths"]["a"]["tool_calls_attempted"] = {
        "observability": "unobservable",
    }
    failures = _run(report, root)
    assert _named(failures, "absence dressed as agreement")
    assert _named(failures, "names no mechanism")
    assert _named(failures, "names no reason")
    assert _named(failures, "names no census_clause")


def test_an_observability_limitation_missing_only_the_census_clause_fails(
    tmp_path: Path,
) -> None:
    """One field at a time, so each is proven load-bearing on its own."""
    report, root = _well_formed(tmp_path)
    report["paths"]["a"]["tool_calls_attempted"] = dict(
        report["paths"]["a"]["tool_calls_attempted"]
    )
    report["paths"]["a"]["tool_calls_attempted"]["census_clause"] = ""
    failures = _run(report, root)
    assert _named(failures, "names no census_clause")
    assert not _named(failures, "names no mechanism")


def test_path_b_may_not_claim_an_observability_limitation(
    tmp_path: Path,
) -> None:
    report, root = _well_formed(tmp_path)
    report["paths"]["b"]["tool_calls_attempted"] = copy.deepcopy(
        report["paths"]["a"]["tool_calls_attempted"]
    )
    failures = _run(report, root)
    assert _named(failures, "only path a may record an established")


def test_an_empty_derived_required_set_fails(tmp_path: Path) -> None:
    report, root = _well_formed(tmp_path)
    report["paths"]["a"]["derived_required_tools"] = []
    failures = _run(report, root)
    assert _named(failures, "derived_required_tools is EMPTY")
    assert _named(failures, "no required graph access at all")


def test_the_hardcoded_seven_on_a_masked_deployment_fails(
    tmp_path: Path,
) -> None:
    """The correction this PACT already applied, pinned so it cannot regress."""
    report, root = _well_formed(tmp_path)
    for arm in PATH_IDS:
        report["paths"][arm]["derived_required_tools"] = sorted(HARDCODED_SEVEN)
        report["paths"][arm]["tool_inventory"] = sorted(HARDCODED_SEVEN)
    failures = _run(report, root)
    assert _named(failures, "is the hardcoded seven")
    assert _named(failures, f"ignores the posture exclusion of {POSTURE_EXCLUDED_TOOL!r}")


def test_the_seven_are_admissible_on_an_UNMASKED_deployment(
    tmp_path: Path,
) -> None:
    """The positive control for the arm above: seven is not banned, HARDCODING
    it is. A deployment that declares all seven and is not masked derives all
    seven, and the guard passes."""
    report, root = _well_formed(tmp_path)
    report["deployment"]["raw_manifest_excerpt"] = {
        "nomos": {"runtime_enforcement": "off"},
        "chat": {
            "enabled": True,
            "mcp": {"tools": [*CHAT_DEFAULT_TOOLS, POSTURE_EXCLUDED_TOOL]},
        },
    }
    report["deployment"]["posture"] = {
        "masked": False,
        "indeterminate": False,
        "reason": "nomos.runtime_enforcement is 'off' and no pii_classes",
    }
    for arm in PATH_IDS:
        report["paths"][arm]["derived_required_tools"] = sorted(HARDCODED_SEVEN)
        report["paths"][arm]["tool_inventory"] = sorted(HARDCODED_SEVEN)
    assert _run(report, root) == []


def test_a_derived_set_that_does_not_match_the_deployment_fails(
    tmp_path: Path,
) -> None:
    """A silently narrowed required set is caught against the manifest."""
    report, root = _well_formed(tmp_path)
    report["paths"]["b"]["derived_required_tools"] = ["inspect"]
    failures = _run(report, root)
    assert _named(failures, "is not the set derived from the deployment")


def test_a_posture_contradicted_by_the_manifest_excerpt_fails(
    tmp_path: Path,
) -> None:
    """The report cannot declare itself unmasked out of an enforcing manifest."""
    report, root = _well_formed(tmp_path)
    report["deployment"]["posture"] = {
        "masked": False,
        "indeterminate": False,
        "reason": "we would rather it were unmasked",
    }
    failures = _run(report, root)
    assert _named(failures, "any value other than 'off' is masked")


def test_a_masked_claim_with_no_basis_in_the_excerpt_fails(
    tmp_path: Path,
) -> None:
    report, root = _well_formed(tmp_path)
    report["deployment"]["raw_manifest_excerpt"] = {
        "nomos": {"runtime_enforcement": "off"},
        "chat": {"enabled": True, "mcp": {"tools": list(CHAT_DEFAULT_TOOLS)}},
    }
    report["deployment"]["posture"] = {
        "masked": True,
        "indeterminate": False,
        "reason": "it feels masked",
    }
    failures = _run(report, root)
    assert _named(failures, "the masked predicate is not satisfied by the excerpt")


def test_differing_models_fail(tmp_path: Path) -> None:
    report, root = _well_formed(tmp_path)
    report["paths"]["b"]["model"] = "gpt-5.1"
    failures = _run(report, root)
    assert _named(failures, "answers a different question than the cutover asks")


def test_an_answer_citing_a_foreign_transcript_hash_fails(
    tmp_path: Path,
) -> None:
    report, root = _well_formed(tmp_path)
    report["paths"]["a"]["answers"][0]["transcript_sha256"] = "0" * 64
    failures = _run(report, root)
    assert _named(failures, "which is not a digest hash for this path")


def test_a_missing_top_level_field_fails(tmp_path: Path) -> None:
    report, root = _well_formed(tmp_path)
    del report["grader"]
    failures = _run(report, root)
    assert failures == ["report is missing required field 'grader'"]


def test_a_missing_per_path_field_fails(tmp_path: Path) -> None:
    report, root = _well_formed(tmp_path)
    del report["paths"]["b"]["observed_generation_ids"]
    failures = _run(report, root)
    assert failures == [
        "path b: missing required field 'observed_generation_ids'"
    ]


# ---------------------------------------------------------------------------
# The real artifact
# ---------------------------------------------------------------------------

def test_the_real_report_artifact_state() -> None:
    """Dispatch on the artifact, in both directions.

    The benchmark run is EXTERNAL work (a live Open WebUI instance, a pinned
    generation with the publisher quiesced, and the CMS question set), so the
    artifact does not exist in this repo yet. This test does not pass by
    saying nothing about it: when the report lands, this asserts the guard
    accepts it against the landed transcripts; until then it asserts the
    absence explicitly, including that no half-landed state — a report
    without transcripts, or transcripts without a report — is sitting in the
    tree unnoticed.
    """
    if REPORT_PATH.is_file():
        report = json.loads(REPORT_PATH.read_text())
        assert TRANSCRIPTS_DIR.is_dir(), (
            f"{REPORT_PATH} exists but {TRANSCRIPTS_DIR} does not; the report "
            "is not admissible without its raw per-question transcripts"
        )
        assert validate_benchmark_report(
            report, transcripts_root=TRANSCRIPTS_DIR,
        ) == []
        return

    landed = sorted(TRANSCRIPTS_DIR.glob("*.json")) if TRANSCRIPTS_DIR.is_dir() else []
    assert landed == [], (
        f"transcripts exist under {TRANSCRIPTS_DIR} but {REPORT_PATH} does "
        "not; transcripts with no report cannot be checked for linkage"
    )


def test_the_guard_reads_a_report_off_disk(tmp_path: Path) -> None:
    """The on-disk path is exercised, not only in-memory dicts.

    The real-artifact test above reads JSON from a file; without this arm
    that code path would be unexercised until the day the report lands.
    """
    report, root = _well_formed(tmp_path)
    written = tmp_path / "path-a-vs-path-b.json"
    written.write_text(json.dumps(report, indent=2, sort_keys=True))
    assert validate_benchmark_report(
        json.loads(written.read_text()), transcripts_root=root,
    ) == []


@pytest.mark.parametrize(
    "risky",
    [
        "hybrid_retriever",       # retriev
        "search_metadata_index",  # search, index
        "grep_local_files",       # grep
        "vectorstore_fetch",      # vectorstore
        "rag_context",            # rag
        "alias_lookup",           # lookup
    ],
)
def test_every_risky_marker_is_reachable(tmp_path: Path, risky: str) -> None:
    """Each marker in :data:`RISKY_TOOL_MARKERS` catches a real tool name.

    A marker list nothing matches is a rule that never fires, so every marker
    gets a plausible tool name that it, and it alone in some cases, catches.
    """
    report, root = _well_formed(tmp_path)
    report["paths"]["b"]["non_okg_tool_inventory"] = [*SHARED_NON_OKG_TOOLS, risky]
    report["non_okg_inventory_delta"] = {
        "declared": True,
        "tools": [{"tool": risky, "reason": "could not be equalized"}],
    }
    failures = _run(report, root)
    assert _named(failures, "This fails whether or not the delta was declared")
