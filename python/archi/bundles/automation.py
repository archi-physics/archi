"""Checks for automation packs (ADR 0002 sections 7.2 and 7.3, PACT r3-ops-runtime).

`validate_pack` calls `automation_problems` for a pack of kind `automation`
once the JSON schema has passed. The schema fixes the shape; this module
checks what it cannot see: that the playbook exists, that every action class
the automation names is defined once, that each mediator resolves, that every
JSON schema is a valid JSON schema, that the context gate is a real
`GateSpec`, that the budget is positive and that no level is above L2.

The manifest shape:

    automation:
      trigger: {type: ticket_queue, system: redmine, queue: cms-comp-ops}
      context_gate: {sources: [...], max_staleness: 24h, invariants: [...]}
      playbook: triage.md              # under the pack's `playbooks` directory
      action_classes: [draft_reply]    # defined in the `action_classes` file
      proposal_schema: {...}           # JSON Schema; requires the ops_proposal fields
      budget: {cost_per_run: 0.5, cost_per_day: 10, runs_per_hour: 20}
      default_level: L1
      max_level: L2

A `ticket_queue` trigger names one system or a list of them, and a queue; `queue: "*"` means every ticket
the instance's ticket sources ingest (an instance narrows it with `queues:` in ops.yaml).

`narrow_to_selected: true` is for an automation that can read several optional packs (ticket-triage reads
Redmine, JIRA or ServiceNow, whichever the instance has). The composer then keeps only the gate sources,
and the trigger systems, whose pack the instance selected, and refuses an instance that selected none of
them (`archi.bundles.compose._automation`). Without it, every gate source must be installed, or the gate
fails on every run.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from archi.bundles.manifest import PACK_FILE, Problem, _schema_problems, load_unique_yaml
from archi.ops import steps
from archi.ops.context_gate import GateSpec, GateSpecError
from archi.ops.levels import LEVELS, level_problem, rank

#: The ops_proposal fields an automation's proposal carries
#: (bundles/packs/ops-record/schemas/ops_record.yaml). The runtime fills the
#: rest: incident_id, mode, autonomy_level, generation_read, and so on. It also
#: serialises `payload` to JSON text, which is how the record stores it.
PROPOSAL_FIELDS = ("summary", "evidence_ids", "action_class", "payload", "confidence", "rollback")

ID = re.compile(r"^[a-z][a-z0-9_]*$")
TOOL_ENTRY = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$")
CRON_FIELD = re.compile(r"^[0-9A-Za-z*/,\-?]+$")
SYSTEMS = ("redmine", "jira", "servicenow")
#: The shipped packs that publish each ticket system's tickets (JIRA: the jira pack, or the tickets pack's
#: generic JIRA reader). `narrow_to_selected` keeps a trigger system when one of its packs is selected.
SYSTEM_PACKS = {"redmine": ("tickets",), "jira": ("jira", "tickets"), "servicenow": ("servicenow",)}
TRIGGER_KEYS = {
    "schedule": ("cron",),
    "ticket_queue": ("system", "queue"),
    "signal": ("source", "condition"),
}


def _cron_values(spec: str, low: int, high: int) -> Optional[set[int]]:
    """The values a cron minute or hour field matches, or None if it is not plain numeric.

    Handles `*`, `*/n`, `a`, `a-b`, `a-b/n` and comma lists of those.
    """
    values: set[int] = set()
    for part in spec.split(","):
        m = re.fullmatch(r"(\*|\d+(?:-\d+)?)(?:/(\d+))?", part)
        if not m:
            return None
        rng, step = m.group(1), int(m.group(2) or 1)
        if step < 1:
            return None
        if rng == "*":
            a, b = low, high
        elif "-" in rng:
            a, b = (int(x) for x in rng.split("-"))
        else:
            a = int(rng)
            b = high if m.group(2) else a
        if not (low <= a <= b <= high):
            return None
        values.update(range(a, b + 1, step))
    return values


def cron_runs_per_hour(cron: str) -> int:
    """The most times `cron` can fire within one hour.

    Reads the minute and hour fields only: the busiest allowed hour fires once
    per allowed minute. Day-of-month, month and weekday fields can only fire
    less often, so they are treated as unrestricted, which keeps this an upper
    bound. Not handled: `@daily` style macros, names (JAN, MON) in the minute
    or hour field, and seconds or year fields. Those raise ValueError.
    """
    fields = cron.split()
    if len(fields) != 5:
        raise ValueError("not a five-field cron expression")
    minutes = _cron_values(fields[0], 0, 59)
    hours = _cron_values(fields[1], 0, 23)
    if minutes is None or hours is None:
        raise ValueError("the minute and hour fields must be numbers, ranges, lists or steps")
    if not minutes or not hours:
        raise ValueError("the minute and hour fields match nothing")
    return len(minutes)


def _schema_check(schema: Any, where: str) -> list[str]:
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import SchemaError

    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        return [f"{where}: not a valid JSON Schema: {exc.message}"]
    if not isinstance(schema, Mapping) or schema.get("type") != "object":
        return [f"{where}: must be an object schema (`type: object`)"]
    return []


def _trigger_messages(trigger: Mapping[str, Any], budget: Mapping[str, Any]) -> list[str]:
    kind = trigger["type"]
    out = []
    allowed = TRIGGER_KEYS[kind]
    for key in allowed:
        if key not in trigger:
            out.append(f"automation.trigger: a {kind} trigger needs `{key}`")
    for key in sorted(set(trigger) - {"type"} - set(allowed)):
        out.append(f"automation.trigger: `{key}` does not belong to a {kind} trigger (allowed: {', '.join(allowed)})")
    if kind == "schedule" and "cron" in trigger:
        fields = trigger["cron"].split()
        if len(fields) != 5 or not all(CRON_FIELD.match(f) for f in fields):
            out.append(f"automation.trigger.cron: {trigger['cron']!r} is not a five-field cron expression")
        else:
            try:
                fires = cron_runs_per_hour(trigger["cron"])
            except ValueError as exc:
                out.append(f"automation.trigger.cron: {trigger['cron']!r}: {exc}")
            else:
                limit = budget.get("runs_per_hour")
                if isinstance(limit, int) and not isinstance(limit, bool) and limit > 0 and fires > limit:
                    out.append(
                        f"automation.trigger.cron: {trigger['cron']!r} can fire {fires} times an hour, "
                        f"over budget.runs_per_hour {limit}"
                    )
    if kind == "ticket_queue" and "system" in trigger:
        systems = trigger_systems(trigger)
        for system in systems:
            if system not in SYSTEMS:
                out.append(f"automation.trigger.system: {system!r} is not one of {SYSTEMS}")
        if len(set(systems)) != len(systems):
            out.append("automation.trigger.system: a system is listed more than once")
    return out


def trigger_systems(trigger: Mapping[str, Any]) -> list[str]:
    """The ticket systems a `ticket_queue` trigger names: one (`system: jira`) or a list."""
    value = trigger.get("system")
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def _required_tools(
    data: Mapping[str, Any], packs: Optional[Mapping[str, Any]]
) -> tuple[set[str], list[str], set[str]]:
    """Live tools of this pack and of the packs it requires, the required packs not resolved,
    and the tools among them declared `idempotent_tools`.

    Required packs come from `packs`, the catalog (any root). Without a
    catalog none resolve; nothing is guessed from neighbouring directories.
    """
    tools = set(data.get("live_tools", ()))
    idempotent = set(data.get("idempotent_tools", ()))
    unresolved = []
    for dep in data.get("requires", {}).get("packs", ()):
        if packs is not None and dep in packs:
            tools |= set(packs[dep].live_tools)
            idempotent |= set(packs[dep].idempotent_tools)
        else:
            unresolved.append(dep)
    return tools, unresolved, idempotent


def _mediator_message(
    where: str, cls: Mapping[str, Any], tools: set[str], unresolved: list[str], has_catalog: bool, check_requires: bool,
    idempotent: Optional[set[str]] = None,
    own_signals: bool = False,
) -> Optional[str]:
    mediator = cls["mediator"]
    if TOOL_ENTRY.match(mediator):
        if not check_requires or mediator in tools:
            if (check_requires and cls["max_level"] == "L2" and idempotent is not None
                    and mediator not in idempotent):
                return (
                    f"{where}: live tool {mediator!r} can run at L2 but its pack does not declare it in "
                    "`idempotent_tools:`. The runtime passes every mediator an idempotency key and never "
                    "calls a mediator twice for one key, yet a tool that may have run once before a crash "
                    "must make a repeat harmless; declare it only if it is"
                )
            return None
        if unresolved and not has_catalog:
            return (
                f"{where}: mediator {mediator!r} is not a live tool of this pack; "
                f"required pack not resolvable without a catalog ({', '.join(unresolved)})"
            )
        hint = f" (required pack(s) not found: {', '.join(unresolved)})" if unresolved else ""
        return f"{where}: mediator {mediator!r} is not a live tool of this pack or of a pack it requires{hint}"
    step = steps.get_step(mediator)
    if step is not None and step.impl and not own_signals:
        return (
            f"{where}: workflow step {mediator!r} runs at L1 with no escalation and is allowed only in a pack that "
            "declares `automation.evidence: own_signals`"
        )
    if step is None:
        return (
            f"{where}: mediator {mediator!r} is neither a live tool entry (archi.tools.x:fn) "
            f"nor a registered workflow step ({', '.join(steps.step_names())}); a step that "
            "changes an external system must be a live tool of a tool pack"
        )
    if cls["max_level"] in LEVELS and rank(cls["max_level"]) > rank(step.max_level):
        return (
            f"{where}: max_level {cls['max_level']} is above {step.max_level}, the highest level of "
            f"workflow step {mediator!r}, which has no external effect; an action that runs after "
            "approval must be a live tool of a tool pack (ADR 0002 section 7.2)"
        )
    return None


def _action_class_messages(
    directory: Path,
    data: Mapping[str, Any],
    names: list[str],
    packs: Optional[Mapping[str, Any]],
    check_requires: bool,
) -> list[str]:
    """Messages for the `action_classes` file and the classes the automation names."""
    rel = data.get("action_classes")
    if rel is None:
        return ["action_classes: an automation pack must declare an `action_classes:` file"]
    path = directory / rel
    try:
        body = load_unique_yaml(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return []  # validate_pack already reported the missing file or the parse error
    classes = body.get("action_classes") if isinstance(body, Mapping) else None
    if not isinstance(classes, Mapping):
        return []  # validate_pack already reports the shape

    tools, unresolved, idempotent = _required_tools(data, packs)
    out = []
    for cid, cls in classes.items():
        where = f"{rel}: action class {cid!r}"
        if not isinstance(cid, str) or not ID.match(cid):
            out.append(f"{rel}: action class id {cid!r} must match {ID.pattern}")
        problems = _schema_problems("action_class", cls, path)
        if problems:
            out.extend(f"{where}: {p.message}" for p in problems)
            continue
        out.extend(f"{where}: {m}" for m in _schema_check(cls["payload_schema"], "payload_schema"))
        msg = level_problem(cls["max_level"], f"{where}: max_level")
        if msg:
            out.append(msg)
        msg = _mediator_message(where, cls, tools, unresolved, packs is not None, check_requires, idempotent,
                                own_signals=data["automation"].get("evidence") == "own_signals")
        if msg:
            out.append(msg)
    for name in names:
        if name not in classes:
            out.append(f"automation.action_classes: {name!r} is not defined in {rel}")
    return out


def _proposal_messages(proposal: Mapping[str, Any]) -> list[str]:
    where = "automation.proposal_schema"
    out = _schema_check(proposal, where)
    if out:
        return out
    props = proposal.get("properties", {})
    for field in sorted(set(props) - set(PROPOSAL_FIELDS)):
        out.append(f"{where}: property {field!r} is not an ops_proposal field ({', '.join(PROPOSAL_FIELDS)})")
    missing = [f for f in PROPOSAL_FIELDS if f not in proposal.get("required", ())]
    if missing:
        out.append(f"{where}: `required` must list {', '.join(missing)}")

    def prop(name: str) -> Mapping[str, Any]:
        value = props.get(name)
        return value if isinstance(value, Mapping) else {}

    for name in ("summary", "action_class", "rollback"):
        if prop(name).get("type") != "string":
            out.append(f"{where}: property {name!r} must have `type: string`")
    if prop("payload").get("type") != "object":
        out.append(f"{where}: property 'payload' must have `type: object`")
    ev = prop("evidence_ids")
    items = ev.get("items")
    if ev.get("type") != "array" or not isinstance(items, Mapping) or items.get("type") != "string":
        out.append(f"{where}: property 'evidence_ids' must be an array of strings (`type: array`, `items: {{type: string}}`)")
    conf = prop("confidence")
    lo, hi = conf.get("minimum"), conf.get("maximum")
    numbers = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (lo, hi))
    if conf.get("type") != "number" or not numbers or lo < 0 or hi > 1 or lo > hi:
        out.append(f"{where}: property 'confidence' must be a number within [0, 1] (`type: number`, `minimum` >= 0, `maximum` <= 1)")
    return out


def automation_problems(
    directory: Path,
    data: Mapping[str, Any],
    *,
    packs: Optional[Mapping[str, Any]] = None,
    check_requires: bool = True,
) -> list[Problem]:
    """Problems with the automation a (schema-valid) pack manifest declares.

    `packs` is the catalog used to resolve live-tool mediators of required
    packs. With `check_requires=False` that one lookup is skipped.
    """
    manifest = directory / PACK_FILE
    if data["kind"] != "automation":
        if "automation" in data:
            return [Problem(manifest, "automation: only a pack of kind `automation` may declare an automation")]
        return []
    if "automation" not in data:
        return [Problem(manifest, "an automation pack must declare an `automation:` block")]

    auto = data["automation"]
    messages: list[str] = []
    own_signals = auto.get("evidence") == "own_signals"
    if own_signals and level_problem(auto["default_level"], "x") is None and rank(auto["default_level"]) > rank("L1"):
        messages.append("automation.evidence: own_signals is for a monitor that starts at L0 or L1; default_level is "
                        f"{auto['default_level']}")

    messages.extend(_trigger_messages(auto["trigger"], auto["budget"]))
    gate_sources = auto["context_gate"].get("sources") if isinstance(auto["context_gate"], Mapping) else None
    if auto.get("narrow_to_selected") and not gate_sources:
        messages.append("automation.narrow_to_selected: the context gate names no source to narrow")

    try:
        gate = GateSpec.from_mapping(auto["context_gate"])
    except GateSpecError as exc:
        messages.append(f"automation.context_gate: {exc}")
    else:
        measured = gate.sources or gate.invariants or gate.min_context_eval_score is not None
        if not measured and not (gate.published and own_signals):
            messages.append(
                "automation.context_gate: declares no check; name at least one source, invariant or "
                "min_context_eval_score (ADR 0002 D12). `published: true` alone is allowed only with "
                "`automation.evidence: own_signals`"
            )

    playbooks = data.get("playbooks")
    if playbooks is None:
        messages.append("playbooks: an automation pack must declare a `playbooks:` directory")
    elif not (directory / playbooks / auto["playbook"]).is_file():
        messages.append(f"automation.playbook: {auto['playbook']!r} is not a file under {playbooks!r}")

    names = auto["action_classes"]
    for name in sorted({n for n in names if names.count(n) > 1}):
        messages.append(f"automation.action_classes: {name!r} is listed more than once")
    messages.extend(_action_class_messages(directory, data, list(dict.fromkeys(names)), packs, check_requires))
    messages.extend(_proposal_messages(auto["proposal_schema"]))

    budget = auto["budget"]
    for key in ("cost_per_run", "cost_per_day", "runs_per_hour"):
        value = budget[key]
        if isinstance(value, bool) or value <= 0:
            messages.append(f"automation.budget.{key}: must be greater than zero, got {value!r}")
    if all(not isinstance(budget[k], bool) and budget[k] > 0 for k in ("cost_per_run", "cost_per_day")):
        if budget["cost_per_run"] > budget["cost_per_day"]:
            messages.append("automation.budget: cost_per_run exceeds cost_per_day")

    levels_ok = True
    for key in ("default_level", "max_level"):
        msg = level_problem(auto[key], f"automation.{key}")
        if msg:
            messages.append(msg)
            levels_ok = False
    if levels_ok and rank(auto["default_level"]) > rank(auto["max_level"]):
        messages.append(
            f"automation: default_level {auto['default_level']} is above max_level {auto['max_level']}"
        )

    return [Problem(manifest, m) for m in messages]
