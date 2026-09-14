"""The bridge machinery behind an Open WebUI Pipe (PACT archi-agent-pipe-bridge).

Three anti-vacuity rules this file follows, because each one has a defect
class behind it.

**Every refusal arm has a positive control in this same file.** A suite that
only shows a turn being refused cannot tell a working gate from a gate that
refuses everything — and refusing everything is the exact failure a hardcoded
seven-operator required set would produce on a masked deployment.

**The traps are DEMONSTRATED, not asserted away.** "A system message is not
demoted to a human turn" is worth nothing unless the demotion is shown to be
real: :func:`a2rchi_infer_speaker_kind` replicates A2rchi's own default and
the test calls it. Likewise the empty-required-set floor is exercised against
a NON-EMPTY loaded inventory, so the subset test it protects would otherwise
pass.

**The two restated substrate constants are pinned by import.** This file
imports both ``okg.chat.CHAT_DEFAULT_TOOLS`` and
``okg.chat.QUERY_TOOL`` and asserts the bridge's copies equal
them, and it runs the bridge's posture predicate DIFFERENTIALLY against
``sync.resolve_posture`` over a table of manifests. The bridge module cannot
import ``okg`` (it runs inside the Open WebUI process); this file can, and
that is where the drift is caught.

Hermetic: no database, no network, no container, no live instance.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

import pytest

from archi.compat.agent_pipe import (
    HARDCODED_SEVEN,
    POSTURE_EXCLUDED_TOOL,
    REFUSAL_EMPTY_REQUIRED_SET,
    REFUSAL_MISSING_REQUIRED_TOOLS,
    STICKY_REFUSAL_STATEMENT,
    SUBSTRATE_DEFAULT_TOOLS,
    AdaptedTurn,
    BridgeRefusal,
    Posture,
    ToolSetDerivation,
    UnmappedRole,
    a2rchi_infer_speaker_kind,
    adapt_messages,
    agent_failure_turn,
    answer_turn,
    derive_required_tools,
    preflight,
    refusal_turn,
    resolve_posture,
    run_off_event_loop,
)
from okg.chat import QUERY_TOOL
from okg.chat import resolve_posture as substrate_resolve_posture
from okg.chat import CHAT_DEFAULT_TOOLS

DEPLOYMENT = "fixture-cms"

#: What the A2rchi CMS agent loads when its OKG MCP server is DOWN but the
#: rest of its tools build. Taken from the agent's own registry
#: (cms_comp_ops_agent.py:124-178 at A2rchi commit a0e86aa0): grep, the
#: metadata tools, the catalog fetch, and the hybrid retriever. Not one of
#: these is an OKG read operator, and the inventory is emphatically NOT empty
#: — which is why a non-empty-inventory check is not the gate.
NON_GRAPH_TOOLS = (
    "grep",
    "search_metadata_index",
    "list_metadata_schema",
    "fetch_catalog_document",
    "search_vectorstore_hybrid",
)

#: A reply that reads like a real answer. The error and refusal turns are
#: asserted NOT to match it: an apology scores as an answer.
GENERIC_ANSWER = (
    "I'm sorry, I couldn't find anything relevant to that question. "
    "Please try rephrasing it."
)


# ---------------------------------------------------------------------------
# Fixture deployments and catalogs
# ---------------------------------------------------------------------------

class _Catalog:
    def __init__(self, subtype_metadata: dict[str, Any] | None = None) -> None:
        self.subtype_metadata = subtype_metadata or {}


def _plain_catalog() -> _Catalog:
    return _Catalog({"okg.file": {"description": "a file"}})


def _pii_catalog() -> _Catalog:
    return _Catalog({"okg.person": {"pii_classes": ["name", "email"]}})


def _boom_catalog() -> _Catalog:
    raise RuntimeError("catalog module missing")


def _manifest(
    *,
    tools: tuple[str, ...] | None = None,
    declare_mcp: bool = True,
    declare_chat: bool = True,
    nomos: Any = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {"name": DEPLOYMENT}
    if nomos is not None:
        manifest["nomos"] = nomos
    if declare_chat:
        chat: dict[str, Any] = {"enabled": True}
        if declare_mcp:
            chat["mcp"] = {"port": 8100, "tools": list(tools or ())}
        manifest["chat"] = chat
    return manifest


def _derivation(
    *,
    tools: tuple[str, ...] | None = None,
    declare_mcp: bool = True,
    declare_chat: bool = True,
    nomos: Any = None,
    catalog: Any = _plain_catalog,
) -> ToolSetDerivation:
    return derive_required_tools(
        deployment=DEPLOYMENT,
        raw_manifest=_manifest(
            tools=tools,
            declare_mcp=declare_mcp,
            declare_chat=declare_chat,
            nomos=nomos,
        ),
        catalog_loader=catalog,
    )


# ---------------------------------------------------------------------------
# The restated substrate constants are PINNED, not typed and forgotten
# ---------------------------------------------------------------------------

def test_substrate_default_tools_is_pinned_to_the_substrates_own() -> None:
    """The bridge's six-operator default IS ``CHAT_DEFAULT_TOOLS``.

    Read off the substrate rather than compared to a literal, so a rename or
    an addition in ``okg.chat`` reds here instead of
    quietly leaving the Pipe requiring a set the deployment never gets.
    """
    assert SUBSTRATE_DEFAULT_TOOLS == tuple(CHAT_DEFAULT_TOOLS)
    assert QUERY_TOOL not in SUBSTRATE_DEFAULT_TOOLS


def test_posture_excluded_tool_is_pinned_to_the_substrates_own() -> None:
    assert POSTURE_EXCLUDED_TOOL == QUERY_TOOL


def test_the_hardcoded_seven_is_never_a_derived_required_set() -> None:
    """The forbidden set exists only as a tripwire.

    ``HARDCODED_SEVEN`` is the default six plus ``query``. On a masked
    deployment the derivation must never produce it — that is the set that
    would refuse every turn forever.
    """
    assert HARDCODED_SEVEN == frozenset((*CHAT_DEFAULT_TOOLS, QUERY_TOOL))
    assert len(HARDCODED_SEVEN) == 7
    masked = _derivation(
        tools=(*CHAT_DEFAULT_TOOLS, QUERY_TOOL),
        nomos={"runtime_enforcement": "enforce"},
    )
    assert set(masked.required) != HARDCODED_SEVEN


# ---------------------------------------------------------------------------
# The posture predicate, run DIFFERENTIALLY against the substrate's own
# ---------------------------------------------------------------------------

_POSTURE_TABLE = [
    ("unmasked", {}, _plain_catalog),
    ("enforcement-off", {"nomos": {"runtime_enforcement": "off"}}, _plain_catalog),
    ("enforcement-on", {"nomos": {"runtime_enforcement": "enforce"}}, _plain_catalog),
    ("enforcement-audit", {"nomos": {"runtime_enforcement": "audit"}}, _plain_catalog),
    ("nomos-unreadable", {"nomos": "not-a-mapping"}, _plain_catalog),
    ("pii-declared", {}, _pii_catalog),
    ("catalog-unreadable", {}, _boom_catalog),
]


@pytest.mark.parametrize(
    ("label", "manifest", "catalog"),
    _POSTURE_TABLE,
    ids=[row[0] for row in _POSTURE_TABLE],
)
def test_posture_replica_agrees_with_the_substrate_predicate(
    label: str, manifest: dict[str, Any], catalog: Any,
) -> None:
    """Same inputs, same verdict — masked AND indeterminate.

    The bridge restates the predicate because the Pipe process cannot import
    ``okg``. A restatement that drifts would decide whether ``query`` is
    required using a different rule than the one that decides whether
    ``query`` is served, and the two disagreeing is a permanent refusal.
    """
    mine = resolve_posture(DEPLOYMENT, manifest, catalog_loader=catalog)
    theirs = substrate_resolve_posture(
        DEPLOYMENT, manifest, catalog_loader=catalog,
    )
    assert (mine.masked, mine.indeterminate) == (
        theirs.masked,
        theirs.indeterminate,
    ), label


def test_the_posture_table_covers_both_verdicts() -> None:
    """The differential table is not all-masked or all-unmasked.

    Without this, a replica that returned a constant would agree with the
    substrate on every row of a single-verdict table.
    """
    verdicts = {
        substrate_resolve_posture(
            DEPLOYMENT, manifest, catalog_loader=catalog,
        ).masked
        for _label, manifest, catalog in _POSTURE_TABLE
    }
    assert verdicts == {True, False}


def test_posture_without_a_catalog_loader_is_masked_and_indeterminate() -> None:
    """The one named divergence from the substrate, and it is fail-closed.

    ``sync.resolve_posture`` falls back to composing the catalog from disk.
    The Pipe process has no deployment directory, so the bridge treats a
    missing loader as UNKNOWN, and unknown excludes. Returning "unmasked"
    here would be a default-value fallback in the one place that decides
    whether ``query`` is required.
    """
    posture = resolve_posture(DEPLOYMENT, {}, catalog_loader=None)
    assert posture.masked is True
    assert posture.indeterminate is True
    assert "no catalog loader" in posture.reason


# ---------------------------------------------------------------------------
# The derivation
# ---------------------------------------------------------------------------

def test_required_set_is_derived_from_the_deployment_not_hardcoded() -> None:
    """A deployment that declares two operators requires exactly those two."""
    derivation = _derivation(tools=("inspect", "search"))
    assert derivation.required == ("inspect", "search")
    assert derivation.declared == ("inspect", "search")
    assert derivation.declaration_source == "manifest"
    assert set(derivation.required) != HARDCODED_SEVEN
    # Positive control: a DIFFERENT declaration yields a DIFFERENT required
    # set, so the assertion above is reading the deployment rather than a
    # constant that happens to match.
    other = _derivation(tools=("expand", "map", "aggregate"))
    assert other.required == ("expand", "map", "aggregate")


def test_undeclared_deployment_gets_the_substrate_default_six() -> None:
    derivation = _derivation(declare_mcp=False)
    assert derivation.required == tuple(CHAT_DEFAULT_TOOLS)
    assert derivation.declaration_source == "substrate_default"
    assert QUERY_TOOL not in derivation.required


def test_an_empty_declaration_is_not_read_as_unstated() -> None:
    """``tools: []`` means NONE, and must not fall through to the default six.

    Keyed on the PRESENCE of the ``chat.mcp`` block, exactly as
    ``sync.plan_tool_curation`` keys it. Reading "declared nothing" as
    "unstated" is how an empty set turns into six operators.
    """
    declared_empty = _derivation(tools=())
    assert declared_empty.declared == ()
    assert declared_empty.required == ()
    assert declared_empty.declaration_source == "manifest"
    # Positive control, one keystroke away: the block ABSENT does default.
    assert _derivation(declare_mcp=False).required == tuple(CHAT_DEFAULT_TOOLS)


def test_masked_deployment_does_not_require_query() -> None:
    """The clause that a hardcoded seven would break, in both directions.

    On a masked deployment ``chat-sync-projection`` permanently drops
    ``query`` from the bound tools, so ``query`` is never in the loaded
    inventory. The derived set must drop it too — and the turn must then be
    SERVED against an inventory without it.
    """
    derivation = _derivation(
        tools=(*CHAT_DEFAULT_TOOLS, QUERY_TOOL),
        nomos={"runtime_enforcement": "enforce"},
    )
    assert derivation.posture.masked is True
    assert QUERY_TOOL not in derivation.required
    assert derivation.excluded == (QUERY_TOOL,)
    assert set(derivation.required) == set(CHAT_DEFAULT_TOOLS)

    served_inventory = (*CHAT_DEFAULT_TOOLS, *NON_GRAPH_TOOLS)
    preflight(derivation, served_inventory)  # serves; no refusal

    # The trap, DEMONSTRATED: had the required set been the hardcoded seven,
    # this same correctly configured deployment would refuse — forever.
    hardcoded = ToolSetDerivation(
        deployment=DEPLOYMENT,
        declared=tuple(sorted(HARDCODED_SEVEN)),
        declaration_source="manifest",
        posture=derivation.posture,
        excluded=(),
        required=tuple(sorted(HARDCODED_SEVEN)),
    )
    with pytest.raises(BridgeRefusal) as caught:
        preflight(hardcoded, served_inventory)
    assert caught.value.missing == (QUERY_TOOL,)


def test_masked_by_pii_classes_also_drops_query() -> None:
    """The second half of the masked predicate, not just the nomos half."""
    derivation = _derivation(
        tools=(*CHAT_DEFAULT_TOOLS, QUERY_TOOL),
        nomos={"runtime_enforcement": "off"},
        catalog=_pii_catalog,
    )
    assert derivation.posture.masked is True
    assert "pii_classes" in derivation.posture.reason
    assert QUERY_TOOL not in derivation.required


def test_unmasked_deployment_keeps_query_when_it_is_declared() -> None:
    """The exclusion is CONDITIONAL. Without this, dropping ``query``
    unconditionally would pass every masked arm above."""
    derivation = _derivation(
        tools=(*CHAT_DEFAULT_TOOLS, QUERY_TOOL),
        nomos={"runtime_enforcement": "off"},
    )
    assert derivation.posture.masked is False
    assert QUERY_TOOL in derivation.required
    assert derivation.excluded == ()


def test_the_derivation_is_reported_for_an_operator() -> None:
    """The derived set is REPORTED — required, declared, posture, and why."""
    derivation = _derivation(
        tools=(*CHAT_DEFAULT_TOOLS, QUERY_TOOL),
        nomos={"runtime_enforcement": "enforce"},
    )
    record = derivation.to_dict()
    assert record["required"] == list(derivation.required)
    assert record["declared"] == list(derivation.declared)
    assert record["posture_excluded"] == [QUERY_TOOL]
    assert record["posture"]["masked"] is True
    assert record["substrate_default_set"] == list(CHAT_DEFAULT_TOOLS)
    assert record["exclusion_reasons"][0]["tool"] == QUERY_TOOL

    explained = derivation.explain()
    assert "inspect" in explained
    assert "MASKED" in explained
    assert QUERY_TOOL in explained


# ---------------------------------------------------------------------------
# The gate: the non-empty floor, then inclusion
# ---------------------------------------------------------------------------

def test_empty_derived_required_set_refuses_rather_than_passing_vacuously() -> None:
    """The floor, exercised against a NON-EMPTY inventory.

    The empty set is a subset of every inventory, so without the floor this
    call passes the inclusion test and hands back an agent with no graph
    access at all. The inventory here is deliberately rich — a check that
    only refused on an empty inventory would pass this test for the wrong
    reason.
    """
    derivation = _derivation(tools=())
    assert derivation.required == ()
    with pytest.raises(BridgeRefusal) as caught:
        preflight(derivation, (*CHAT_DEFAULT_TOOLS, *NON_GRAPH_TOOLS))
    assert caught.value.reason == REFUSAL_EMPTY_REQUIRED_SET
    assert "vacuously" in refusal_turn(caught.value)

    # Positive control: the SAME inventory, with one operator required,
    # serves. The refusal above is the empty set, not the inventory.
    served = _derivation(tools=("search",))
    preflight(served, (*CHAT_DEFAULT_TOOLS, *NON_GRAPH_TOOLS))


def test_the_floor_is_checked_before_the_subset_test() -> None:
    """Order is load-bearing: an empty required set must never reach the
    subset test, where it would be reported as "nothing missing"."""
    derivation = _derivation(tools=())
    with pytest.raises(BridgeRefusal) as caught:
        preflight(derivation, ("grep",))
    assert caught.value.reason == REFUSAL_EMPTY_REQUIRED_SET
    assert caught.value.missing == ()


def test_a_deployment_with_no_chat_block_refuses_and_says_which() -> None:
    """"Nothing to derive from" is named, not silently the same as "declared
    nothing"."""
    derivation = _derivation(declare_chat=True, declare_mcp=True, tools=())
    absent = _derivation(declare_chat=False)
    assert absent.declaration_source == "absent"
    assert derivation.declaration_source == "manifest"
    with pytest.raises(BridgeRefusal) as caught:
        preflight(absent, NON_GRAPH_TOOLS)
    assert caught.value.reason == REFUSAL_EMPTY_REQUIRED_SET
    assert "no chat block at all" in refusal_turn(caught.value)


def test_okg_server_down_but_non_graph_tools_loaded() -> None:
    """The non-vacuous refusal arm.

    A2rchi keeps the tools that DID load when a server fails
    (``tools/mcp.py:26-41``), so the OKG server being unreachable leaves a
    healthy-looking inventory of grep, retrieval and metadata tools. The
    refusal must fire on REQUIRED-SET INCLUSION, not on emptiness.
    """
    derivation = _derivation(declare_mcp=False)
    with pytest.raises(BridgeRefusal) as caught:
        preflight(derivation, NON_GRAPH_TOOLS)
    refusal = caught.value
    assert refusal.reason == REFUSAL_MISSING_REQUIRED_TOOLS
    assert set(refusal.missing) == set(CHAT_DEFAULT_TOOLS)
    assert refusal.loaded == NON_GRAPH_TOOLS
    # The inventory is emphatically not empty; a non-empty check would pass.
    assert len(NON_GRAPH_TOOLS) >= 5

    # Positive control: the same non-graph tools PLUS the OKG operators
    # serve. The refusal is about the missing operators, not about the
    # presence of A2rchi's own tools.
    preflight(derivation, (*NON_GRAPH_TOOLS, *CHAT_DEFAULT_TOOLS))


def test_a_partial_okg_inventory_still_refuses() -> None:
    """Five of six is not a subset. Named separately because "some graph
    tools loaded" is the case a presence check would wave through."""
    derivation = _derivation(declare_mcp=False)
    partial = tuple(t for t in CHAT_DEFAULT_TOOLS if t != "expand")
    with pytest.raises(BridgeRefusal) as caught:
        preflight(derivation, (*partial, *NON_GRAPH_TOOLS))
    assert caught.value.missing == ("expand",)


def test_a_similarly_named_tool_does_not_satisfy_a_required_operator() -> None:
    """``search_vectorstore_hybrid`` is not ``search``.

    The inclusion test is exact membership, not a substring or suffix match —
    Open WebUI's own filter matcher is ``endswith``
    (``utils/misc.py:71``), and inheriting that here would let the agent's
    hybrid retriever stand in for the graph's ``search`` operator.
    """
    derivation = _derivation(tools=("search",))
    with pytest.raises(BridgeRefusal) as caught:
        preflight(derivation, ("search_vectorstore_hybrid", "search_metadata_index"))
    assert caught.value.missing == ("search",)
    preflight(derivation, ("search",))


def test_the_refusal_text_reports_the_derived_set_and_is_not_an_apology() -> None:
    derivation = _derivation(
        tools=(*CHAT_DEFAULT_TOOLS, QUERY_TOOL),
        nomos={"runtime_enforcement": "enforce"},
    )
    with pytest.raises(BridgeRefusal) as caught:
        preflight(derivation, NON_GRAPH_TOOLS)
    text = refusal_turn(caught.value)
    assert text.strip()
    assert text != GENERIC_ANSWER
    assert "refused this turn" in text
    # The derived set and its provenance are IN the text.
    for operator in CHAT_DEFAULT_TOOLS:
        assert operator in text
    assert "MASKED" in text
    assert f"excluded {QUERY_TOOL!r}" in text
    assert "MISSING:" in text
    # And the stickiness is stated rather than implied away.
    assert STICKY_REFUSAL_STATEMENT in text
    assert "Retrying this turn will produce the same refusal." in text


# ---------------------------------------------------------------------------
# The message adapter
# ---------------------------------------------------------------------------

def test_a2rchi_really_does_demote_an_unknown_role_to_a_human_turn() -> None:
    """The trap, demonstrated. Without this, "system is not demoted" could be
    passing because nothing demotes anything."""
    assert a2rchi_infer_speaker_kind("system") == "human_by_default"
    assert a2rchi_infer_speaker_kind("tool") == "human_by_default"
    assert a2rchi_infer_speaker_kind("user") == "human"
    assert a2rchi_infer_speaker_kind("assistant") == "ai"


def test_system_message_not_demoted_to_human() -> None:
    """The preset's system prompt is routed OUT of the history, not through it.

    Open WebUI prepends a system message to every turn, so raising would
    refuse every request; passing it through would make A2rchi turn it into a
    user turn (``history_utils.py:13``).
    """
    adapted = adapt_messages([
        {"role": "system", "content": "You are the CMS computing operator."},
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
    ])
    assert adapted.system == ("You are the CMS computing operator.",)
    speakers = [speaker for speaker, _ in adapted.history]
    assert "system" not in speakers
    contents = [content for _, content in adapted.history]
    assert "You are the CMS computing operator." not in contents
    assert adapted.prompt == "second question"
    assert adapted.prompt != "You are the CMS computing operator."


def test_every_speaker_the_adapter_emits_is_explicitly_recognized() -> None:
    """No emitted token may rely on A2rchi's warning default.

    A token that only "works" because the default happens to be HumanMessage
    is a demotion wearing a passing test.
    """
    adapted = adapt_messages([
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ])
    emitted = {speaker for speaker, _ in adapted.history}
    assert emitted
    for speaker in emitted:
        assert a2rchi_infer_speaker_kind(speaker) != "human_by_default", speaker


def test_an_unmapped_role_raises_rather_than_becoming_a_human_turn() -> None:
    for role in ("tool", "function", "developer", "moderator"):
        with pytest.raises(UnmappedRole) as caught:
            adapt_messages([
                {"role": "user", "content": "q"},
                {"role": role, "content": "payload"},
            ])
        assert caught.value.role == role
        assert "no A2rchi speaker" in str(caught.value)
    # Positive control: the mapped roles do NOT raise.
    adapt_messages([
        {"role": "system", "content": "s"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
        {"role": "human", "content": "q2"},
    ])


def test_the_last_user_message_is_the_prompt_and_is_not_repeated() -> None:
    adapted = adapt_messages([
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ])
    assert adapted.prompt == "q2"
    assert adapted.history == (("user", "q1"), ("assistant", "a1"))


# ---------------------------------------------------------------------------
# The sync/async boundary
# ---------------------------------------------------------------------------

def _blocking_agent(record: dict[str, Any]) -> str:
    """A stand-in for A2rchi's synchronous ``invoke``. Blocks its thread."""
    record["thread"] = threading.current_thread().name
    threading.Event().wait(0.20)
    return "agent answer"


def test_pipe_call_runs_agent_off_event_loop() -> None:
    """Two observable end states: a different thread, and a live loop.

    ``functions.py:154-157`` calls a synchronous ``pipe()`` directly, so a
    blocking agent turn would stall the whole instance. Asserted on the
    thread the agent actually ran on and on whether the loop kept scheduling,
    not on the presence of an ``await``.
    """
    record: dict[str, Any] = {}
    ticks = {"n": 0}

    async def ticker() -> None:
        while True:
            ticks["n"] += 1
            await asyncio.sleep(0.01)

    async def scenario() -> str:
        task = asyncio.create_task(ticker())
        loop_thread = threading.current_thread().name
        try:
            answer = await run_off_event_loop(lambda: _blocking_agent(record))
        finally:
            task.cancel()
        record["loop_thread"] = loop_thread
        return answer

    answer = asyncio.run(scenario())
    assert answer == "agent answer"
    assert record["thread"] != record["loop_thread"]
    assert ticks["n"] > 1, (
        "the event loop did not schedule anything while the agent ran"
    )


def test_the_off_loop_assertions_can_fail(
) -> None:
    """The positive control for the test above.

    Calling the same blocking agent DIRECTLY on the loop pins both assertions
    to the opposite verdict: same thread, and a ticker that never advanced
    past its first pass. Without this, ``ticks > 1`` could be passing because
    the ticker is fast rather than because the loop was free.
    """
    record: dict[str, Any] = {}
    ticks = {"n": 0}

    async def ticker() -> None:
        while True:
            ticks["n"] += 1
            await asyncio.sleep(0.01)

    async def scenario() -> str:
        task = asyncio.create_task(ticker())
        await asyncio.sleep(0)  # let the ticker take its first pass
        before = ticks["n"]
        loop_thread = threading.current_thread().name
        try:
            answer = _blocking_agent(record)  # ON the loop, deliberately
        finally:
            task.cancel()
        record["loop_thread"] = loop_thread
        record["ticks_during"] = ticks["n"] - before
        return answer

    asyncio.run(scenario())
    assert record["thread"] == record["loop_thread"]
    assert record["ticks_during"] == 0


# ---------------------------------------------------------------------------
# The composed turn
# ---------------------------------------------------------------------------

def _turn(
    *,
    derivation: ToolSetDerivation,
    inventory: tuple[str, ...],
    agent: Any,
    messages: list[dict[str, Any]] | None = None,
) -> str:
    return asyncio.run(
        answer_turn(
            deployment=DEPLOYMENT,
            messages=messages or [
                {"role": "system", "content": "preset prompt"},
                {"role": "user", "content": "how many nodes are live?"},
            ],
            derivation=derivation,
            load_tool_inventory=lambda: inventory,
            invoke_agent=agent,
        )
    )


def test_a_stubbed_agent_turn_round_trips_without_rewriting_the_output() -> None:
    """The Pipe returns the agent's own bytes. Exact equality, markdown and
    trailing whitespace included — a reformatted summary is a rewritten
    answer."""
    answer = "## Live nodes\n\n| gen | nodes |\n|---|---|\n| g-9 | 41,203 |\n"
    seen: dict[str, Any] = {}

    def agent(adapted: AdaptedTurn) -> str:
        seen["adapted"] = adapted
        return answer

    got = _turn(
        derivation=_derivation(declare_mcp=False),
        inventory=(*CHAT_DEFAULT_TOOLS, *NON_GRAPH_TOOLS),
        agent=agent,
    )
    assert got == answer
    assert seen["adapted"].prompt == "how many nodes are live?"
    assert seen["adapted"].system == ("preset prompt",)


def test_a_refused_turn_never_reaches_the_agent() -> None:
    """Observable end state: the agent recorded zero calls.

    "No answer is produced from the remaining non-graph tools" is only true
    if the agent is not run at all.
    """
    calls: list[AdaptedTurn] = []

    def agent(adapted: AdaptedTurn) -> str:
        calls.append(adapted)
        return "an answer from grep alone"

    text = _turn(
        derivation=_derivation(declare_mcp=False),
        inventory=NON_GRAPH_TOOLS,
        agent=agent,
    )
    assert calls == []
    assert "refused this turn" in text
    assert "an answer from grep alone" not in text

    # Positive control: with the operators loaded, the same agent IS called.
    text = _turn(
        derivation=_derivation(declare_mcp=False),
        inventory=(*NON_GRAPH_TOOLS, *CHAT_DEFAULT_TOOLS),
        agent=agent,
    )
    assert len(calls) == 1
    assert text == "an answer from grep alone"


def test_an_agent_exception_surfaces_as_a_visible_error_turn() -> None:
    """Named failure, non-empty, and not mistakable for a real reply."""

    def agent(_adapted: AdaptedTurn) -> str:
        raise RuntimeError("MCP session closed mid-turn")

    text = _turn(
        derivation=_derivation(declare_mcp=False),
        inventory=(*CHAT_DEFAULT_TOOLS, *NON_GRAPH_TOOLS),
        agent=agent,
    )
    assert text.strip()
    assert text != GENERIC_ANSWER
    assert "RuntimeError" in text
    assert "MCP session closed mid-turn" in text
    assert "produced no answer" in text
    # Positive control: the success path returns the answer, so the arm above
    # is not passing because every turn errors.
    assert _turn(
        derivation=_derivation(declare_mcp=False),
        inventory=(*CHAT_DEFAULT_TOOLS, *NON_GRAPH_TOOLS),
        agent=lambda _a: "a real answer",
    ) == "a real answer"


def test_an_inventory_loader_failure_is_a_visible_turn_not_an_empty_one() -> None:
    """A loader that raises must not read as "no tools loaded"."""

    def boom() -> tuple[str, ...]:
        raise ConnectionRefusedError("okg mcp-serve refused the connection")

    text = asyncio.run(
        answer_turn(
            deployment=DEPLOYMENT,
            messages=[{"role": "user", "content": "q"}],
            derivation=_derivation(declare_mcp=False),
            load_tool_inventory=boom,
            invoke_agent=lambda _a: "should not run",
        )
    )
    assert "ConnectionRefusedError" in text
    assert "should not run" not in text


def test_an_unmapped_role_is_a_visible_turn_and_stops_the_agent() -> None:
    calls: list[AdaptedTurn] = []

    text = _turn(
        derivation=_derivation(declare_mcp=False),
        inventory=(*CHAT_DEFAULT_TOOLS, *NON_GRAPH_TOOLS),
        agent=lambda adapted: calls.append(adapted) or "answered anyway",
        messages=[
            {"role": "user", "content": "q"},
            {"role": "tool", "content": "{}"},
        ],
    )
    assert calls == []
    assert "UnmappedRole" in text
    assert "answered anyway" not in text


def test_agent_failure_turn_carries_the_derived_set() -> None:
    text = agent_failure_turn(
        RuntimeError("boom"),
        deployment=DEPLOYMENT,
        derivation=_derivation(declare_mcp=False),
    )
    assert "Derived required OKG operators" in text
    assert "inspect" in text


def test_the_bridge_module_imports_nothing_from_okg() -> None:
    """The self-contained contract, asserted rather than stated.

    The Open WebUI process does not import ``okg``, so an ``okg.*`` import
    that slipped into this module would work in this test suite and fail at
    load time inside the container — the worst place to discover it. Read off
    the module's AST, not off a substring search, so an import inside a
    function body is caught too.
    """
    import ast

    from archi.compat import agent_pipe as agent_pipe_bridge

    source = Path(agent_pipe_bridge.__file__).read_text()
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # a relative import is an okg-package import
                imported.add("okg")
            elif node.module:
                imported.add(node.module.split(".")[0])
    assert imported, "the AST walk found no imports at all"
    assert "okg" not in imported, sorted(imported)
    # Positive control: the walk really does see module names, so the
    # assertion above is not passing on an empty set.
    assert "asyncio" in imported


def test_posture_dataclass_round_trips() -> None:
    posture = Posture(masked=True, indeterminate=False, reason="because")
    assert posture.to_dict() == {
        "masked": True,
        "indeterminate": False,
        "reason": "because",
    }
