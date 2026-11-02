"""Durations as packs write them: a positive integer and one unit, `36h`, `7d`.

Units are `s m h d w`. The pack schema (`pack.schema.json`, `freshness`) pins
the same pattern. ISO-8601 (`PT36H`) is not accepted: one spelling keeps pack
manifests, `archi doctor --json` and context gates comparable by eye.
"""
from __future__ import annotations

import re

UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
DURATION = re.compile(r"^([1-9][0-9]*)([smhdw])$")


def parse_duration(text: str) -> int:
    """Seconds in `text`; ValueError when it is not `<n><s|m|h|d|w>`."""
    match = DURATION.fullmatch(text.strip()) if isinstance(text, str) else None
    if match is None:
        raise ValueError(
            f"{text!r} is not a duration: write a positive integer and one unit "
            "of s, m, h, d or w (for example 36h or 7d)"
        )
    return int(match.group(1)) * UNIT_SECONDS[match.group(2)]


def format_age(seconds: float) -> str:
    """A short human reading of an age: `45s`, `5m`, `31h 20m`, `3d 4h`."""
    total = int(max(seconds, 0))
    if total < 60:
        return f"{total}s"
    if total < 3600:
        return f"{total // 60}m"
    if total < 2 * 86400:
        hours, rest = divmod(total, 3600)
        minutes = rest // 60
        return f"{hours}h {minutes}m" if minutes else f"{hours}h"
    days, rest = divmod(total, 86400)
    hours = rest // 3600
    return f"{days}d {hours}h" if hours else f"{days}d"
