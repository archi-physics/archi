"""okg version ranges: parse, intersect, and tell whether the result is empty.

A bundle names the AKMON releases it supports as PEP 440 specifiers
(`">=0.2.0,<0.3"`). Composing bundles intersects their ranges (ADR 0002
section 3.3), and an empty intersection stops the render. `packaging` can
combine specifier sets but cannot say whether one admits any version, so
`is_empty` reduces the set to one interval and checks that.
"""
from __future__ import annotations

from typing import Iterable, Optional

from packaging.specifiers import InvalidSpecifier, Specifier, SpecifierSet
from packaging.version import Version

SUPPORTED_OPERATORS = frozenset({">=", ">", "<=", "<", "==", "!=", "~="})


class OkgRangeError(ValueError):
    """An okg range is malformed, or a set of ranges admits no version."""


def parse_range(text: str) -> SpecifierSet:
    try:
        spec = SpecifierSet(text)
    except InvalidSpecifier as exc:
        raise OkgRangeError(f"{text!r} is not a PEP 440 version range") from exc
    if not len(spec):
        raise OkgRangeError(f"{text!r} names no bound")
    for item in spec:
        if item.operator not in SUPPORTED_OPERATORS:
            raise OkgRangeError(f"{text!r}: operator {item.operator!r} is not supported")
    return spec


def _bump(release: tuple[int, ...]) -> Version:
    return Version(".".join(str(x) for x in release[:-1] + (release[-1] + 1,)))


def _interval(spec: SpecifierSet):
    """(lower, lower_inclusive, upper, upper_inclusive, excluded points)."""
    lower: Optional[Version] = None
    lower_inc = True
    upper: Optional[Version] = None
    upper_inc = True
    excluded: set[Version] = set()

    def raise_lower(v: Version, inclusive: bool) -> None:
        nonlocal lower, lower_inc
        if lower is None or v > lower or (v == lower and not inclusive):
            lower, lower_inc = v, inclusive

    def cut_upper(v: Version, inclusive: bool) -> None:
        nonlocal upper, upper_inc
        if upper is None or v < upper or (v == upper and not inclusive):
            upper, upper_inc = v, inclusive

    for item in spec:
        op, text = item.operator, item.version
        if op == "==" and text.endswith(".*"):
            base = Version(text[:-2])
            raise_lower(base, True)
            cut_upper(_bump(base.release), False)
            continue
        v = Version(text)
        if op == ">=":
            raise_lower(v, True)
        elif op == ">":
            raise_lower(v, False)
        elif op == "<=":
            cut_upper(v, True)
        elif op == "<":
            cut_upper(v, False)
        elif op == "==":
            raise_lower(v, True)
            cut_upper(v, True)
        elif op == "~=":
            raise_lower(v, True)
            cut_upper(_bump(v.release[:-1]), False)
        elif op == "!=":
            excluded.add(v)
    return lower, lower_inc, upper, upper_inc, excluded


def is_empty(spec: SpecifierSet) -> bool:
    lower, lower_inc, upper, upper_inc, excluded = _interval(spec)
    if lower is None or upper is None:
        return False
    if lower > upper:
        return True
    if lower == upper:
        return not (lower_inc and upper_inc) or lower in excluded
    return False


def intersect(ranges: Iterable[str]) -> SpecifierSet:
    """The intersection of `ranges`; raises OkgRangeError when it is empty."""
    ranges = list(ranges)
    combined = SpecifierSet()
    for text in ranges:
        combined &= parse_range(text)
    if is_empty(combined):
        raise OkgRangeError(
            "the okg ranges " + ", ".join(repr(r) for r in ranges) + " have no version in common"
        )
    return combined


def canonical(spec: SpecifierSet) -> str:
    """A stable spelling: specifiers sorted, comma-joined."""
    return ",".join(sorted({str(Specifier(str(s))) for s in spec}))
