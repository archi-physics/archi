"""Pack resolution (ADR 0002 section 3.3).

    packs = companions(closure(requires, ∪ default(extends*(A, B, ...))) ∪ with) − without

`resolve` computes that set in a fixed order, the *resolution order*, which
later decides the order of prompt fragments:

1. Bundles: each `--bundle` in the order given, preceded by everything it
   extends (parents first). A bundle reached twice counts once.
2. Their default packs, in bundle order then listed order, each preceded by
   the packs it requires (depth first). This is the requires closure.
3. `--with` packs, in the order given, appended if not already selected.
4. Companions (amended 2026-10-31): for each selected pack, in order, the
   packs the bundles name as its companions, each preceded by the packs it
   requires, unless the companion or its trigger is given to `--without`.
   A companion added this way can trigger its own companions.
5. `--without` packs removed. Removing a companion the bundles declare is
   the opt-out, so it is not refused as "not selected", even before its
   trigger is selected.

The formula closes `requires` over the defaults only. A `--with` pack whose
requirements are not selected, or a `--without` that removes a pack another
selected pack requires, is refused with the names involved rather than
rendering a profile that cannot load.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional

from packaging.specifiers import SpecifierSet

from archi.bundles.discovery import Catalog
from archi.bundles.manifest import Bundle, Pack, extends_closure
from archi.bundles.versions import OkgRangeError, intersect

#: Chat keys taken from one bundle (ADR 0002 section 3.3, "chat scalars").
CHAT_SCALARS = ("site_name", "default_model", "description")


class CompositionError(Exception):
    """The requested selection cannot be composed. `code` names the rule."""

    def __init__(self, code: str, message: str, **data: Any):
        self.code = code
        self.data = data
        super().__init__(message)


@dataclass(frozen=True)
class Resolution:
    """What `resolve` selected, in resolution order."""

    given: tuple[str, ...]
    bundles: tuple[Bundle, ...]
    packs: tuple[Pack, ...]
    with_packs: tuple[str, ...]
    without_packs: tuple[str, ...]
    okg_range: SpecifierSet
    chat: Mapping[str, Any]
    #: Question defaults the bundles set, later bundles overriding earlier ones.
    answers: Mapping[str, Any] = field(default_factory=dict)
    #: Conflicts resolved by rule rather than refused, e.g. two bundles
    #: setting different chat site names. Reported, never silent.
    notes: tuple[str, ...] = field(default=())

    @property
    def pack_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.packs)


def _bundle_order(catalog: Catalog, given: Iterable[str]) -> list[str]:
    order: list[str] = []
    for name in given:
        if name not in catalog.bundles:
            raise CompositionError(
                "unknown_bundle",
                f"unknown bundle {name!r}; known: {sorted(catalog.bundles) or 'none'}",
                bundle=name,
            )
        try:
            chain = extends_closure(name, catalog.bundles)
        except ValueError as exc:
            raise CompositionError("extends_cycle", f"bundle extends cycle: {exc}") from exc
        for item in chain:
            if item not in order:
                order.append(item)
    return order


def _require(catalog: Catalog, name: str, why: str) -> Pack:
    pack = catalog.packs.get(name)
    if pack is None:
        raise CompositionError(
            "unknown_pack", f"unknown pack {name!r} ({why}); known: {sorted(catalog.packs)}",
            pack=name,
        )
    return pack


def _closure(catalog: Catalog, roots: Iterable[str], why: str = "a bundle default") -> list[str]:
    out: list[str] = []

    def visit(name: str, stack: tuple[str, ...], why: str) -> None:
        if name in stack:
            raise CompositionError(
                "requires_cycle", "pack requires cycle: " + " -> ".join(stack + (name,))
            )
        if name in out:
            return
        pack = _require(catalog, name, why)
        for dep in pack.requires_packs:
            visit(dep, stack + (name,), f"required by {name}")
        out.append(name)

    for root in roots:
        visit(root, (), why)
    return out


def _companions(catalog: Catalog, order: Iterable[str]) -> dict[str, list[str]]:
    """Trigger pack -> companions, merged over the bundles in resolution order."""
    out: dict[str, list[str]] = {}
    for name in order:
        for trigger, companions in catalog.bundles[name].companions.items():
            merged = out.setdefault(trigger, [])
            merged.extend(c for c in companions if c not in merged)
    return out


def _chat(bundles_given_last_first: list[list[Bundle]]) -> tuple[dict[str, Any], list[str]]:
    """Scalars from the last --bundle given; suggestions as a union.

    For each scalar, the bundles are searched from the last given to the
    first, and within one given bundle from itself to its furthest parent.
    The first value found wins; any other distinct value is reported.
    """
    chat: dict[str, Any] = {}
    notes: list[str] = []
    searched = [b for chain in bundles_given_last_first for b in chain]
    for key in CHAT_SCALARS:
        found = [(b.name, b.chat[key]) for b in searched if key in b.chat]
        if not found:
            continue
        winner_bundle, winner = found[0]
        chat[key] = winner
        for name, value in found[1:]:
            if value != winner:
                notes.append(
                    f"chat.{key}: {winner_bundle!r} sets {winner!r}; "
                    f"{name!r} sets {value!r}, which is not used"
                )
    suggestions: list[str] = []
    for bundle in reversed(searched):
        for text in bundle.chat.get("suggestions", ()):
            if text not in suggestions:
                suggestions.append(text)
    if suggestions:
        chat["suggestions"] = suggestions
    return chat, notes


def resolve(
    catalog: Catalog,
    bundles: Iterable[str],
    with_packs: Iterable[str] = (),
    without_packs: Iterable[str] = (),
) -> Resolution:
    given = tuple(bundles)
    if not given:
        raise CompositionError("no_bundle", "name at least one bundle")
    with_packs = tuple(dict.fromkeys(with_packs))
    without_packs = tuple(dict.fromkeys(without_packs))

    order = _bundle_order(catalog, given)
    defaults: list[str] = []
    for name in order:
        for pack in catalog.bundles[name].default_packs:
            if pack not in defaults:
                defaults.append(pack)
    selected = _closure(catalog, defaults)

    for name in with_packs:
        _require(catalog, name, "--with")
        if name not in selected:
            selected.append(name)

    companions = _companions(catalog, order)
    # Any companion the bundles declare may be given to --without, selected
    # or not: the opt-out can be recorded before its trigger is added
    # (`archi add <trigger>` keeps the instance's --without list).
    offered: set[str] = {c for cs in companions.values() for c in cs}
    companion_notes: list[str] = []
    i = 0
    while i < len(selected):  # grows while it is walked: companions can trigger
        trigger = selected[i]
        i += 1
        if trigger in without_packs:
            continue
        for companion in companions.get(trigger, ()):
            if companion in without_packs or companion in selected:
                continue
            # An unknown companion stops here as unknown_pack, naming its
            # trigger; bundle validation reports it earlier.
            why = f"companion of {trigger!r} in the bundle"
            for name in _closure(catalog, [companion], why):
                if name not in selected:
                    selected.append(name)
            companion_notes.append(f"{companion} comes with {trigger} (opt out with --without {companion})")

    for name in without_packs:
        if name in offered and name not in selected:
            triggers = sorted(t for t, cs in companions.items() if name in cs)
            if not any(t in selected for t in triggers):
                companion_notes.append(
                    f"--without {name}: recorded; it would come with {', '.join(triggers)}, "
                    "which is not selected"
                )
            continue  # opting out of a companion
        if name not in selected:
            raise CompositionError(
                "without_not_selected",
                f"--without {name}: pack {name!r} is not selected",
                pack=name,
            )
        selected.remove(name)

    for name in selected:
        missing = [dep for dep in catalog.packs[name].requires_packs if dep not in selected]
        if missing:
            removed = [m for m in missing if m in without_packs]
            hint = (
                f"do not remove {removed}" if removed
                else "add " + " ".join(f"--with {m}" for m in missing)
            )
            raise CompositionError(
                "requires_unmet",
                f"pack {name!r} requires {missing}, which is not selected; {hint}",
                pack=name,
                missing=missing,
            )

    bundle_objs = [catalog.bundles[n] for n in order]
    try:
        okg_range = intersect(b.okg for b in bundle_objs)
    except OkgRangeError as exc:
        raise CompositionError(
            "okg_range_empty", str(exc), ranges={b.name: b.okg for b in bundle_objs}
        ) from exc

    chains = [
        list(reversed([catalog.bundles[n] for n in extends_closure(g, catalog.bundles)]))
        for g in reversed(given)
    ]
    chat, notes = _chat(chains)

    return Resolution(
        given=given,
        bundles=tuple(bundle_objs),
        packs=tuple(catalog.packs[n] for n in selected),
        with_packs=with_packs,
        without_packs=without_packs,
        okg_range=okg_range,
        chat=chat,
        answers={k: v for b in bundle_objs for k, v in b.answers.items()},
        notes=tuple(notes) + tuple(companion_notes),
    )
