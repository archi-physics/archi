"""Bridge machinery for hosting an EXTERNAL agent behind an Open WebUI Pipe
(PACT ``archi-agent-pipe-bridge``).

Consumer ownership
------------------

Archi owns these message/protocol compatibility helpers. A deployment-specific
Open WebUI Pipe still supplies its valves, agent class, agent-spec binding and
MCP endpoint bootstrap; this module deliberately provides no Pipe class or
pipes() entry point. It contains the required-tool derivation, fail-closed
message adaptation, event-loop boundary and visible failure turns used by that
composition. The former framework location is historical, not a second runtime
home. Import these helpers from ``archi.compat.agent_pipe``.

Self-contained by construction
------------------------------

The Open WebUI process does not import ``okg`` (the same contract
``review_queue_pipe`` states). This module therefore imports only the
standard library. The two substrate constants it restates —
:data:`SUBSTRATE_DEFAULT_TOOLS` and :data:`POSTURE_EXCLUDED_TOOL` — are
pinned to the substrate's own by
``python/tests/test_agent_pipe_bridge.py``, which imports both sides
and asserts equality; a substrate change that is not mirrored here reds that
test rather than silently widening a required set.

Vendor facts, re-verified against the running
``ghcr.io/open-webui/open-webui:v0.11.0`` container on 2026-08-11
-----------------------------------------------------------------

* A Pipe is registered as a MODEL and handles the turn itself:
  ``functions.py:150 generate_function_chat_completion`` dispatches to the
  module's ``pipe()`` instead of a provider.
* ``functions.py:154-157 execute_pipe``::

      if inspect.iscoroutinefunction(pipe):
          return await pipe(**params)
      else:
          return pipe(**params)

  A SYNCHRONOUS ``pipe()`` therefore runs directly on the event loop. A2rchi's
  ``invoke``/``stream`` are synchronous (``base_react.py:256``, ``:306``) and
  drive MCP through a process-wide background loop thread
  (``base_react.py:1088-1092``, ``src/archi/utils/async_loop.py:17``), so a
  synchronous Pipe would stall the whole instance for the length of an agent
  turn. Hence :func:`run_off_event_loop`, and hence :func:`answer_turn` is a
  coroutine.
* A manifold registers its models as ``f'{pipe.id}.{p["id"]}'``
  (``functions.py:106``), where ``pipe.id`` is the FUNCTION id — a bare id
  matches zero rows.
* A Pipe REPLACES Open WebUI's own tool loop
  (``utils/middleware.py:4892-4917``, bounded by
  ``CHAT_RESPONSE_MAX_TOOL_CALL_ITERATIONS``, default ``256`` at
  ``env.py:1015``). Inside a Pipe none of that is inherited: not the MCP tool
  registration, not the concurrent execution, not the citations. The wrapped
  agent brings its own graph access.

A2rchi facts, verified against checkout ``/Users/jason/projects/A2rchi`` at
commit ``a0e86aa0fecaaad6f3fc4de55516fbf59f744037``
-------------------------------------------------------------------------

These are the failure behaviours this module exists to refuse to inherit.

* ``src/archi/pipelines/agents/tools/mcp.py:28-41`` —
  ``initialize_mcp_client`` wraps EACH server in ``try/except``, logs the
  failure into a LOCAL ``failed_servers`` dict (declared at ``:26``), and
  returns ``client, all_tools``. The failure map is never returned, so a
  caller cannot distinguish "the OKG server is down" from "the OKG server
  advertised nothing" except by reading logs. (Note: there is no literal
  ``continue`` statement; the except block simply falls off the end of the
  loop body. The behaviour is as described, the keyword is not there.)
* ``src/archi/pipelines/agents/base_react.py:1031-1035`` — ``refresh_agent``
  caches the built MCP tool list under ``if self._mcp_tools is None``, storing
  ``list(built or [])``. ``_build_mcp_tools`` (``:1085``) returns ``None`` on
  its exception path (``:1127-1128``) and on an empty result (``:1122``), so
  a failure caches ``[]`` — which is not ``None``, so the guard never re-fires
  and nothing else resets it. ``refresh_agent(force=True)`` does not help;
  ``force`` only reaches the LangGraph rebuild at ``:1042-1047``. A refusal is
  therefore STICKY until the process restarts, and
  :data:`STICKY_REFUSAL_STATEMENT` says so instead of implying a retry heals
  it.
* ``src/archi/pipelines/agents/utils/history_utils.py:7-14`` —
  ``infer_speaker`` recognises ``user``/``human`` and
  ``agent``/``ai``/``assistant``/``archi``, and DEFAULTS everything else to
  ``HumanMessage`` with only ``logger.warning``. ``SystemMessage`` is not even
  imported. An OpenAI-style ``system`` message passed through unadapted is
  silently demoted to a user turn. :func:`adapt_messages` routes ``system``
  out of the history entirely and raises on any other unmapped role.

UNVERIFIABLE from this repo, and named rather than asserted
-----------------------------------------------------------

Whether the okg-deployments Pipe can bind the agent's MCP endpoint from
valves is EXTERNAL work and is not settled here. Measured fact:
``get_mcp_servers_config()`` (``A2rchi src/utils/config_access.py:67``)
resolves through ``ConfigService`` → ``PostgresServiceFactory`` → a static
config seeded into Postgres (``src/cli/tools/config_seed.py:33,63``), and
neither ``initialize_mcp_client()`` nor ``_build_mcp_tools(self)`` takes a
servers argument — so valves alone cannot bind the endpoint. This module
therefore takes the loaded inventory through an INJECTED callable and makes
no claim about how the endpoint was resolved; the behavioural binding proof
belongs to the okg-deployments test named in
``task.archi-agent-pipe-bridge.bridge``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

__all__ = [
    "A2RCHI_AI_SPEAKERS",
    "A2RCHI_HUMAN_SPEAKERS",
    "DECLARATION_ABSENT",
    "DECLARATION_MANIFEST",
    "DECLARATION_SUBSTRATE_DEFAULT",
    "HARDCODED_SEVEN",
    "POSTURE_EXCLUDED_TOOL",
    "REFUSAL_EMPTY_REQUIRED_SET",
    "REFUSAL_MISSING_REQUIRED_TOOLS",
    "STICKY_REFUSAL_STATEMENT",
    "SUBSTRATE_DEFAULT_TOOLS",
    "AdaptedTurn",
    "BridgeRefusal",
    "Posture",
    "ToolSetDerivation",
    "UnmappedRole",
    "adapt_messages",
    "a2rchi_infer_speaker_kind",
    "agent_failure_turn",
    "answer_turn",
    "derive_required_tools",
    "preflight",
    "refusal_turn",
    "resolve_posture",
    "run_off_event_loop",
]

T = TypeVar("T")


# ---------------------------------------------------------------------------
# The two substrate constants, restated and PINNED by test
# ---------------------------------------------------------------------------

#: ``okg.substrate.mcp.operators.CHAT_DEFAULT_TOOLS`` — what a deployment gets
#: when it declares no ``chat.mcp.tools``. SIX operators; ``query`` is not one
#: of them. Restated here only because the Pipe process cannot import ``okg``;
#: the test suite asserts this tuple equals the substrate's own.
SUBSTRATE_DEFAULT_TOOLS: tuple[str, ...] = (
    "inspect",
    "search",
    "expand",
    "filter",
    "map",
    "aggregate",
)

#: ``okg.substrate.chat.sync.QUERY_TOOL`` — the one operator a MASKED
#: deployment's preset drops. The exclusion is PERMANENT: ``mcp-http-auth``
#: measured masking applied to returned rows in ZERO cells, so it stands in
#: every cell.
POSTURE_EXCLUDED_TOOL = "query"

#: The set this module exists to refuse to hardcode. Named ONLY so a report
#: or a caller asserting it can be caught; never used as a required set. On a
#: masked deployment ``query`` is never in the loaded inventory, so requiring
#: these seven refuses every turn, forever, on a correctly configured
#: deployment — which is the bug this constant is a tripwire for.
HARDCODED_SEVEN: frozenset[str] = frozenset(
    (*SUBSTRATE_DEFAULT_TOOLS, POSTURE_EXCLUDED_TOOL)
)

#: The deployment's ``chat.mcp`` block named the set.
DECLARATION_MANIFEST = "manifest"
#: The deployment has a chat block but no ``chat.mcp``, so the substrate
#: default applies.
DECLARATION_SUBSTRATE_DEFAULT = "substrate_default"
#: The deployment has no chat block at all. There is nothing to derive FROM,
#: which is not the same as deriving an empty set on purpose — it is named so
#: the floor's refusal can say which of the two happened.
DECLARATION_ABSENT = "absent"

REFUSAL_EMPTY_REQUIRED_SET = "empty_derived_required_set"
REFUSAL_MISSING_REQUIRED_TOOLS = "required_tools_missing_from_loaded_inventory"

STICKY_REFUSAL_STATEMENT = (
    "This refusal is STICKY. A2rchi caches the MCP tool list on first build "
    "and stores an empty list when the build fails "
    "(base_react.py:1031-1035 with _build_mcp_tools returning None at "
    ":1122/:1127-1128), and nothing resets that cache — refresh_agent(force=True) "
    "only rebuilds the LangGraph agent at :1042-1047. Retrying this turn will "
    "produce the same refusal. Restart the process after the MCP server is "
    "reachable again."
)


# ---------------------------------------------------------------------------
# Masking posture — the predicate, replicated fail-closed
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Posture:
    """The deployment's masking posture, and why.

    Mirrors ``okg.substrate.chat.sync.Posture``. ``indeterminate`` is a THIRD
    state that resolves to masked: a posture that cannot be evaluated is not
    "probably unmasked", it is unknown, and unknown excludes.
    """

    masked: bool
    indeterminate: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "masked": self.masked,
            "indeterminate": self.indeterminate,
            "reason": self.reason,
        }


def resolve_posture(
    deployment: str,
    raw_manifest: Mapping[str, Any],
    *,
    catalog_loader: Callable[[], Any] | None = None,
) -> Posture:
    """The masked predicate, EXACTLY as ``chat-sync-projection`` states it.

    Masked when ``nomos.runtime_enforcement`` is any value other than ``off``,
    OR when any subtype in the composed catalog declares
    ``okg.subtypes.metadata.pii_classes``.

    One deliberate divergence from ``sync.resolve_posture``, named rather than
    hidden: that function falls back to composing the catalog from disk when
    no loader is given. The Pipe process has no deployment directory and
    cannot import the composer, so **no loader here means INDETERMINATE, which
    means masked**. Silently returning "unmasked" because the catalog half of
    the predicate could not be evaluated would be a default-value fallback in
    the one place it decides whether ``query`` is required.
    """
    nomos = raw_manifest.get("nomos")
    if nomos is not None and not isinstance(nomos, Mapping):
        return Posture(
            masked=True,
            indeterminate=True,
            reason=(
                f"deployment {deployment!r} has an unreadable nomos block "
                f"({type(nomos).__name__}), so the masking predicate cannot "
                "be evaluated; an indeterminate posture is treated as masked"
            ),
        )
    enforcement = (nomos or {}).get("runtime_enforcement")
    if enforcement is not None and enforcement != "off":
        return Posture(
            masked=True,
            indeterminate=False,
            reason=(
                f"nomos.runtime_enforcement is {enforcement!r} (any value "
                "other than 'off' is masked)"
            ),
        )

    if catalog_loader is None:
        return Posture(
            masked=True,
            indeterminate=True,
            reason=(
                f"no catalog loader was supplied for {deployment!r}, so the "
                "pii_classes half of the masking predicate cannot be "
                "evaluated; an indeterminate posture is treated as masked"
            ),
        )
    try:
        catalog = catalog_loader()
    except Exception as exc:  # noqa: BLE001 — every compose failure is UNKNOWN
        return Posture(
            masked=True,
            indeterminate=True,
            reason=(
                f"the composed catalog for {deployment!r} could not be read "
                f"({type(exc).__name__}: {exc}), so the pii_classes half of "
                "the predicate cannot be evaluated; an indeterminate posture "
                "is treated as masked"
            ),
        )
    metadata = getattr(catalog, "subtype_metadata", None) or {}
    declaring = sorted(
        subtype
        for subtype, meta in metadata.items()
        if isinstance(meta, Mapping) and "pii_classes" in meta
    )
    if declaring:
        return Posture(
            masked=True,
            indeterminate=False,
            reason=(
                f"{len(declaring)} composed subtype(s) declare "
                f"okg.subtypes.metadata.pii_classes (first: {declaring[0]})"
            ),
        )
    return Posture(
        masked=False,
        indeterminate=False,
        reason=(
            "nomos.runtime_enforcement is 'off' and no composed subtype "
            "declares pii_classes"
        ),
    )


# ---------------------------------------------------------------------------
# The derivation: projected set MINUS posture exclusions, with a floor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolSetDerivation:
    """What the deployment requires of its OKG MCP server, and why.

    ``required`` is the answer. Everything else exists so a refusal can SAY
    why, and so a benchmark record can carry the derivation rather than a
    number a reader has to trust.
    """

    deployment: str
    declared: tuple[str, ...]
    declaration_source: str
    posture: Posture
    excluded: tuple[str, ...]
    required: tuple[str, ...]
    exclusion_reasons: tuple[tuple[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "deployment": self.deployment,
            "declared": list(self.declared),
            "declaration_source": self.declaration_source,
            "substrate_default_set": list(SUBSTRATE_DEFAULT_TOOLS),
            "posture": self.posture.to_dict(),
            "posture_excluded": list(self.excluded),
            "exclusion_reasons": [
                {"tool": tool, "reason": reason}
                for tool, reason in self.exclusion_reasons
            ],
            "required": list(self.required),
        }

    def explain(self) -> str:
        """One human paragraph naming the derived set and every step to it.

        Printed in the refusal turn. An operator reading it can tell a
        narrowed required set from a mis-declared deployment without opening a
        log.
        """
        lines = [
            f"Derived required OKG operators for deployment "
            f"{self.deployment!r}: {list(self.required)}.",
            f"  declared ({self.declaration_source}): {list(self.declared)}"
            + (
                f" (substrate default {list(SUBSTRATE_DEFAULT_TOOLS)} applies "
                "when the deployment declares no chat.mcp block)"
                if self.declaration_source == DECLARATION_SUBSTRATE_DEFAULT
                else ""
            ),
            f"  posture: {'MASKED' if self.posture.masked else 'unmasked'}"
            + (" [indeterminate]" if self.posture.indeterminate else "")
            + f" — {self.posture.reason}",
        ]
        if self.exclusion_reasons:
            for tool, reason in self.exclusion_reasons:
                lines.append(f"  excluded {tool!r}: {reason}")
        else:
            lines.append("  excluded: nothing")
        return "\n".join(lines)


def derive_required_tools(
    *,
    deployment: str,
    raw_manifest: Mapping[str, Any],
    catalog_loader: Callable[[], Any] | None = None,
) -> ToolSetDerivation:
    """Read the required set OFF THE DEPLOYMENT. Never a hardcoded seven.

    ``required := the deployment's projected tool set (chat.mcp.tools, or the
    substrate's six-operator default when the deployment declares no
    chat.mcp block) MINUS the posture exclusions (query on a masked
    deployment)``.

    Two siblings make a hardcoded set wrong rather than merely inelegant.
    ``chat-tool-surface`` lets a deployment declare an ARBITRARY admitted
    subset, and ``chat-sync-projection`` PERMANENTLY drops ``query`` on any
    masked deployment. Requiring seven operators against either would make
    ``required`` a non-subset of ``loaded`` forever, so every turn would be
    refused on a deployment configured exactly as this program's own siblings
    configure it.

    The presence of the ``chat.mcp`` block — not the emptiness of its list —
    keys the source, matching ``sync.plan_tool_curation``: a deployment that
    declares ``tools: []`` means NONE, and reading that as "unstated" would
    turn an empty declaration into six operators.
    """
    chat = raw_manifest.get("chat")
    if not isinstance(chat, Mapping):
        declared: tuple[str, ...] = ()
        source = DECLARATION_ABSENT
    elif "mcp" not in chat or not isinstance(chat.get("mcp"), Mapping):
        declared = tuple(SUBSTRATE_DEFAULT_TOOLS)
        source = DECLARATION_SUBSTRATE_DEFAULT
    else:
        declared = tuple(chat["mcp"].get("tools") or ())
        source = DECLARATION_MANIFEST

    posture = resolve_posture(
        deployment, raw_manifest, catalog_loader=catalog_loader,
    )

    excluded: list[str] = []
    exclusion_reasons: list[tuple[str, str]] = []
    if posture.masked and POSTURE_EXCLUDED_TOOL in declared:
        excluded.append(POSTURE_EXCLUDED_TOOL)
        exclusion_reasons.append((
            POSTURE_EXCLUDED_TOOL,
            (
                f"deployment {deployment!r} resolves MASKED ({posture.reason})"
                + (" [posture indeterminate]" if posture.indeterminate else "")
                + ". chat-sync-projection drops this operator from the "
                "preset's bound tools on any masked deployment, so it will "
                "not be in the loaded inventory and must not be required. The "
                "exclusion is permanent: mcp-http-auth measured masking "
                "applied to returned rows in ZERO cells."
            ),
        ))

    required = tuple(t for t in declared if t not in set(excluded))
    # Structural: the derivation may narrow, never widen. A widened required
    # set would refuse turns the deployment configured correctly.
    assert set(required) <= set(declared), (
        "the derived required set widened beyond the declared set"
    )
    return ToolSetDerivation(
        deployment=deployment,
        declared=declared,
        declaration_source=source,
        posture=posture,
        excluded=tuple(excluded),
        required=required,
        exclusion_reasons=tuple(exclusion_reasons),
    )


# ---------------------------------------------------------------------------
# The gate: floor first, then inclusion
# ---------------------------------------------------------------------------

class BridgeRefusal(Exception):
    """The turn is refused, visibly, with the derived set reported."""

    def __init__(
        self,
        reason: str,
        *,
        derivation: ToolSetDerivation,
        loaded: tuple[str, ...],
        missing: tuple[str, ...] = (),
        detail: str = "",
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.derivation = derivation
        self.loaded = loaded
        self.missing = missing
        self.detail = detail


def preflight(
    derivation: ToolSetDerivation, loaded_inventory: Iterable[str],
) -> None:
    """Refuse before serving, on either of two grounds.

    (a) NON-EMPTY FLOOR, checked FIRST. The empty set is a subset of every
    inventory, so an empty derivation would make the inclusion test below a
    vacuous pass and hand back exactly the graph-less agent this gate exists
    to stop. Order matters: an empty required set must never reach the
    subset test.

    (b) REQUIRED-SET INCLUSION. ``required`` must be a subset of ``loaded``.
    A bare non-empty-inventory check is NOT this test: A2rchi returns the
    tools that did load when a server fails (``mcp.py:28-41``), so an agent
    that lost only its OKG server still presents a non-empty inventory of
    grep, retrieval and metadata tools.
    """
    loaded = tuple(loaded_inventory)
    if not derivation.required:
        raise BridgeRefusal(
            REFUSAL_EMPTY_REQUIRED_SET,
            derivation=derivation,
            loaded=loaded,
            detail=(
                "The derived required operator set is EMPTY, so no graph "
                "access was demanded of the agent. The empty set is a subset "
                "of every inventory, so serving this turn would pass the "
                "inclusion check vacuously and answer from whatever non-graph "
                "tools happen to be loaded."
                + (
                    " The deployment declares no chat block at all, so there "
                    "was nothing to derive from."
                    if derivation.declaration_source == DECLARATION_ABSENT
                    else ""
                )
            ),
        )
    missing = tuple(t for t in derivation.required if t not in set(loaded))
    if missing:
        raise BridgeRefusal(
            REFUSAL_MISSING_REQUIRED_TOOLS,
            derivation=derivation,
            loaded=loaded,
            missing=missing,
            detail=(
                f"The agent loaded {list(loaded)} but the deployment requires "
                f"{list(derivation.required)}; {list(missing)} did not load. "
                "A2rchi's MCP loader keeps the tools that DID load when a "
                "server fails and discards the per-server failure map "
                "(tools/mcp.py:26-41), so an agent that lost its OKG server "
                "would otherwise answer this question from its remaining "
                "non-graph tools and read as healthy."
            ),
        )


# ---------------------------------------------------------------------------
# The message adapter — fail-closed on any role A2rchi would demote
# ---------------------------------------------------------------------------

#: ``history_utils.py:9`` — the roles A2rchi maps EXPLICITLY to HumanMessage.
A2RCHI_HUMAN_SPEAKERS: frozenset[str] = frozenset({"user", "human"})
#: ``history_utils.py:11`` — the roles A2rchi maps EXPLICITLY to AIMessage.
A2RCHI_AI_SPEAKERS: frozenset[str] = frozenset(
    {"agent", "ai", "assistant", "archi"}
)


def a2rchi_infer_speaker_kind(speaker: str) -> str:
    """A replica of A2rchi's ``infer_speaker``, INCLUDING its default.

    Returns ``"human"``, ``"ai"``, or ``"human_by_default"``. The third value
    is the whole point: it is what A2rchi does to ``system`` — demote it to a
    user turn with only a ``logger.warning`` — and it exists here so the test
    suite can DEMONSTRATE the trap rather than assert the absence of it. A
    "system is not demoted" test that never shows the demotion is real proves
    nothing.
    """
    lowered = speaker.lower()
    if lowered in A2RCHI_HUMAN_SPEAKERS:
        return "human"
    if lowered in A2RCHI_AI_SPEAKERS:
        return "ai"
    return "human_by_default"


class UnmappedRole(Exception):
    """An OpenAI role with no honest A2rchi speaker. Raised, never demoted."""

    def __init__(self, role: str, *, index: int) -> None:
        super().__init__(
            f"message {index} carries role {role!r}, which has no A2rchi "
            f"speaker. A2rchi's infer_speaker "
            f"(utils/history_utils.py:7-14) recognises only "
            f"{sorted(A2RCHI_HUMAN_SPEAKERS | A2RCHI_AI_SPEAKERS)} and "
            f"DEFAULTS anything else to HumanMessage with a log warning, so "
            f"passing this role through would silently turn it into a user "
            f"turn. Refused instead."
        )
        self.role = role
        self.index = index


@dataclass(frozen=True)
class AdaptedTurn:
    """An Open WebUI request body, in the shape A2rchi's agent takes.

    ``history`` is the ``(speaker, content)`` pair list
    ``base_react.py:1161`` builds messages from. ``system`` is carried
    SEPARATELY and never appears in ``history`` — that is the whole fix.
    """

    history: tuple[tuple[str, str], ...]
    prompt: str
    system: tuple[str, ...] = field(default=())

    def to_dict(self) -> dict[str, Any]:
        return {
            "history": [list(pair) for pair in self.history],
            "prompt": self.prompt,
            "system": list(self.system),
        }


def adapt_messages(messages: Sequence[Mapping[str, Any]]) -> AdaptedTurn:
    """Map an OpenAI-style message list onto A2rchi's history, fail-closed.

    ``system`` is routed OUT of the history into :attr:`AdaptedTurn.system`.
    It is not raised on, because Open WebUI prepends the preset's system
    prompt to every turn and raising would refuse every request; it is not
    passed through either, because A2rchi would turn it into a user turn.

    Every other unrecognised role RAISES :class:`UnmappedRole`. ``tool``,
    ``function`` and ``developer`` are the roles this catches today; the rule
    is a whitelist, so a role added by a future vendor version raises too
    rather than arriving as a human turn.

    The last ``user`` message is the prompt and is not repeated in the
    history.
    """
    system: list[str] = []
    pairs: list[tuple[str, str]] = []
    for index, message in enumerate(messages):
        role = str(message.get("role", ""))
        content = message.get("content")
        content = "" if content is None else str(content)
        lowered = role.lower()
        if lowered == "system":
            system.append(content)
            continue
        if lowered in A2RCHI_HUMAN_SPEAKERS or lowered in A2RCHI_AI_SPEAKERS:
            pairs.append((lowered, content))
            continue
        raise UnmappedRole(role, index=index)

    prompt = ""
    for offset in range(len(pairs) - 1, -1, -1):
        speaker, content = pairs[offset]
        if speaker in A2RCHI_HUMAN_SPEAKERS:
            prompt = content
            pairs.pop(offset)
            break
    return AdaptedTurn(
        history=tuple(pairs), prompt=prompt, system=tuple(system),
    )


# ---------------------------------------------------------------------------
# The sync/async boundary
# ---------------------------------------------------------------------------

async def run_off_event_loop(call: Callable[[], T]) -> T:
    """Run a blocking agent call on a worker thread.

    ``functions.py:154-157`` calls a synchronous ``pipe()`` directly, so
    anything blocking inside it stalls the instance's event loop for the whole
    turn — and A2rchi's ``invoke``/``stream`` are synchronous
    (``base_react.py:256``, ``:306``) and bridge to MCP through a background
    loop thread (``base_react.py:1088-1092``). ``asyncio.to_thread`` is the
    boundary; nothing else in this module may call the agent.
    """
    return await asyncio.to_thread(call)


# ---------------------------------------------------------------------------
# The two visible failure turns
# ---------------------------------------------------------------------------

def refusal_turn(refusal: BridgeRefusal) -> str:
    """The operator-visible refusal. Never an empty answer, never an apology.

    Reports the derived set and every step to it, names what did and did not
    load, and states that the refusal is sticky.
    """
    header = (
        "**OKG bridge refused this turn.** No answer was produced, and none "
        "will be produced from the agent's remaining non-graph tools."
    )
    body = [
        header,
        "",
        f"Reason: `{refusal.reason}`.",
        refusal.detail,
        "",
        refusal.derivation.explain(),
        f"  loaded inventory: {list(refusal.loaded)}",
    ]
    if refusal.missing:
        body.append(f"  MISSING: {list(refusal.missing)}")
    body.extend(["", STICKY_REFUSAL_STATEMENT])
    return "\n".join(body)


def agent_failure_turn(
    exc: BaseException, *, deployment: str, derivation: ToolSetDerivation | None = None,
) -> str:
    """An exception inside the agent, surfaced as a visible error turn.

    Names the failure. Not an empty string, and deliberately not a generic
    apology: an "I'm sorry, I couldn't find that" reads like a real reply and
    would be scored as one.
    """
    lines = [
        f"**OKG bridge error while running the agent for deployment "
        f"`{deployment}`.** This turn produced no answer.",
        "",
        f"Failure: `{type(exc).__name__}: {exc}`",
    ]
    if derivation is not None:
        lines.extend(["", derivation.explain()])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The composition the okg-deployments Pipe calls
# ---------------------------------------------------------------------------

async def answer_turn(
    *,
    deployment: str,
    messages: Sequence[Mapping[str, Any]],
    derivation: ToolSetDerivation,
    load_tool_inventory: Callable[[], Sequence[str]],
    invoke_agent: Callable[[AdaptedTurn], str],
) -> str:
    """One chat turn: adapt, gate, delegate — and never rewrite the answer.

    Both injected callables are BLOCKING and both are run off the event loop.
    ``invoke_agent``'s return value is passed back verbatim; this function
    reformats nothing on the success path, which is what "the Pipe MUST NOT
    rewrite the agent's output" means operationally.

    Failures are turns, not silence: an unmapped role, a refusal, and an
    exception inside the agent each render a visible error string.
    """
    try:
        adapted = adapt_messages(messages)
    except UnmappedRole as exc:
        return agent_failure_turn(exc, deployment=deployment, derivation=derivation)

    try:
        loaded = tuple(await run_off_event_loop(load_tool_inventory))
    except Exception as exc:  # noqa: BLE001 — a loader failure is a visible turn
        return agent_failure_turn(exc, deployment=deployment, derivation=derivation)

    try:
        preflight(derivation, loaded)
    except BridgeRefusal as refusal:
        return refusal_turn(refusal)

    try:
        return await run_off_event_loop(lambda: invoke_agent(adapted))
    except Exception as exc:  # noqa: BLE001 — an agent failure is a visible turn
        return agent_failure_turn(exc, deployment=deployment, derivation=derivation)
