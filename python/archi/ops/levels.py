"""Autonomy levels and the rule that resolves them (ADR 0002 section 7.3).

Three places can set a level for one automation:

- the pack: `automation.default_level` (where it starts when nothing else says)
  and `automation.max_level` (the highest the pack allows);
- the bundle: `automations.<name>.max_level` in bundle.yaml, an upper bound;
- the instance: `automations.<name>.level` in archi.yaml, the level to run at.

The effective level is the lowest of the instance level (the pack default when
the instance sets none), the bundle bound and the pack maximum. 3.0 stops at
L2 (D13): L3, or any value that is not L0, L1 or L2, is refused wherever it
appears, so a bad level never reaches the runtime. The run-start refusal is
`effective_level`, which the runtime calls.
"""
from __future__ import annotations

from typing import Any, Mapping, NamedTuple, Optional

#: Autonomy levels the 3.0 runtime accepts, lowest first.
LEVELS = ("L0", "L1", "L2")
D13 = "ADR 0002 D13: 3.0 stops at L2, L3 is reserved and refused"


class LevelError(ValueError):
    """A level above L2, or not a level at all."""


def level_problem(level: Any, where: str) -> Optional[str]:
    """Why `level` is refused, or None when it is L0, L1 or L2."""
    if isinstance(level, str) and level in LEVELS:
        return None
    if level == "L3":
        return f"{where}: L3 is refused ({D13})"
    return f"{where}: unknown level {level!r} is refused; use one of {LEVELS} ({D13})"


def rank(level: str) -> int:
    return LEVELS.index(level)


class Effective(NamedTuple):
    level: str
    #: which input set the result: instance, default, bundle or pack
    limited_by: str


def resolve_level(
    *,
    instance_level: Optional[str] = None,
    default_level: str,
    bundle_max: Optional[str] = None,
    pack_max: str,
) -> Effective:
    """The effective level and what limited it. Raises LevelError on L3 or an unknown level."""
    named = [
        ("instance", instance_level),
        ("default", default_level),
        ("bundle", bundle_max),
        ("pack", pack_max),
    ]
    for source, level in named:
        if level is None and source in ("instance", "bundle"):
            continue
        message = level_problem(level, f"{source} level")
        if message:
            raise LevelError(message)
    wanted = ("instance", instance_level) if instance_level is not None else ("default", default_level)
    best = wanted
    for source, bound in (("bundle", bundle_max), ("pack", pack_max)):
        if bound is not None and rank(bound) < rank(best[1]):
            best = (source, bound)
    return Effective(best[1], best[0])


def effective_level(
    *,
    instance_level: Optional[str] = None,
    default_level: str,
    bundle_max: Optional[str] = None,
    pack_max: str,
) -> str:
    """min(instance level, bundle bound, pack maximum); the pack default stands in
    for an instance that sets no level. Raises LevelError on L3 or an unknown level."""
    return resolve_level(
        instance_level=instance_level,
        default_level=default_level,
        bundle_max=bundle_max,
        pack_max=pack_max,
    ).level


def min_level(*levels: Optional[str]) -> Optional[str]:
    """The lowest of the given levels, ignoring None; None when none are given."""
    given = [lv for lv in levels if lv is not None]
    return min(given, key=rank) if given else None


def settings_problems(settings: Any, key: str, where: str) -> list[str]:
    """Problems with an `automations:` mapping, `{<name>: {<key>: <level>}}`.

    `key` is `level` for archi.yaml and `max_level` for bundle.yaml.
    """
    if settings is None:
        return []
    if not isinstance(settings, Mapping):
        return [f"{where}: must be a mapping of automation name to {{{key}: L0|L1|L2}}"]
    out: list[str] = []
    for name, body in settings.items():
        if not isinstance(body, Mapping) or set(body) != {key}:
            out.append(f"{where}.{name}: must be exactly {{{key}: L0|L1|L2}}")
            continue
        message = level_problem(body[key], f"{where}.{name}.{key}")
        if message:
            out.append(message)
    return out


def unknown_automations(settings: Mapping[str, Any], locked: Mapping[str, Any]) -> list[str]:
    """Names in the instance's `automations:` that no selected automation pack declares.

    `locked` is `composed.automations` of archi.lock. A misspelled name would
    otherwise be ignored and the automation would run at its default level.
    """
    return sorted(n for n in settings if n not in locked)


def unknown_automation_message(unknown: list[str], locked: Mapping[str, Any]) -> str:
    known = ", ".join(sorted(locked)) or "none: no selected pack is an automation pack"
    return (
        f"archi.yaml automations names {', '.join(map(repr, unknown))}, which no selected "
        f"automation pack declares (selected: {known})"
    )


def instance_levels(instance_dir: Any) -> dict[str, Effective]:
    """The effective level of every automation of the instance, for the runtime.

    Reads archi.yaml and archi.lock. Raises InstanceError for an unknown
    automation name or a bad level in archi.yaml, and LevelError for a level
    the lock carries above L2.
    """
    from archi.bundles.lock import read_lock
    from archi.stack.instance import Instance, InstanceError

    instance = Instance.load(instance_dir)
    locked = read_lock(instance.dir).get("composed", {}).get("automations") or {}
    unknown = unknown_automations(instance.automations, locked)
    if unknown:
        raise InstanceError(unknown_automation_message(unknown, locked))
    return {
        name: resolve_level(
            instance_level=instance.automations.get(name, {}).get("level"),
            default_level=auto["default_level"],
            bundle_max=auto.get("bundle_max_level"),
            pack_max=auto["pack_max_level"],
        )
        for name, auto in locked.items()
    }
