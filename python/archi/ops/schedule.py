"""Cron times for `schedule` triggers (ADR 0002 section 7.1).

An automation whose trigger is `{type: schedule, cron: "7 * * * *"}` has no ticket queue or alert
behind it: the scanner makes one signal per cron time. `latest_fire(cron, now)` is the most recent
time at or before `now` at which the cron fires, in UTC; the scanner turns it into the signal id
`tick-<time>`, so a scan that runs many times in the same hour adds the same signal and the table
keeps one.

The grammar is the one `archi.bundles.automation` accepts for the minute and hour fields (numbers,
ranges, lists, `*` and steps); day-of-month, month and day-of-week take the same forms. Names (`MON`,
`JAN`) and `@daily` macros are not supported. As in cron, when both day-of-month and day-of-week are
restricted, either one matching is enough.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

#: how far back `latest_fire` looks; a cron that fires less often than yearly is refused
LOOKBACK_DAYS = 400

_RANGES = ((0, 59), (0, 23), (1, 31), (1, 12), (0, 7))


class CronError(ValueError):
    """The cron expression cannot be read."""


def _field(text: str, low: int, high: int, label: str) -> frozenset[int]:
    values: set[int] = set()
    for part in text.split(","):
        step = 1
        if "/" in part:
            part, _, step_text = part.partition("/")
            if not step_text.isdigit() or int(step_text) < 1:
                raise CronError(f"{label}: bad step {step_text!r}")
            step = int(step_text)
        if part == "*":
            start, end = low, high
        elif "-" in part:
            a, _, b = part.partition("-")
            if not (a.isdigit() and b.isdigit()):
                raise CronError(f"{label}: bad range {part!r}")
            start, end = int(a), int(b)
        elif part.isdigit():
            start = int(part)
            end = high if step > 1 else start
        else:
            raise CronError(f"{label}: cannot read {part!r}")
        if not (low <= start <= end <= high):
            raise CronError(f"{label}: {part!r} is outside {low}-{high}")
        values.update(range(start, end + 1, step))
    return frozenset(values)


def parse_cron(cron: str) -> tuple[frozenset[int], ...]:
    """(minutes, hours, days of month, months, days of week 0-6 with 0 = Sunday)."""
    parts = str(cron).split()
    if len(parts) != 5:
        raise CronError(f"cron {cron!r} needs five fields")
    fields = [_field(p, lo, hi, name) for p, (lo, hi), name in
              zip(parts, _RANGES, ("minute", "hour", "day of month", "month", "day of week"))]
    fields[4] = frozenset(d % 7 for d in fields[4])  # 7 is Sunday too
    return tuple(fields)


def latest_fire(cron: str, now: datetime) -> Optional[datetime]:
    """The latest cron time at or before `now` (aware, UTC), or None within LOOKBACK_DAYS."""
    minutes, hours, days, months, weekdays = parse_cron(cron)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    now = now.astimezone(timezone.utc).replace(second=0, microsecond=0)
    day_open = str(cron).split()[2] == "*"
    weekday_open = str(cron).split()[4] == "*"
    for back in range(LOOKBACK_DAYS):
        day = (now - timedelta(days=back)).date()
        if day.month not in months:
            continue
        by_day, by_weekday = day.day in days, (day.isoweekday() % 7) in weekdays
        if day_open or weekday_open:
            matches = by_day and by_weekday
        else:
            matches = by_day or by_weekday
        if not matches:
            continue
        for hour in sorted(hours, reverse=True):
            for minute in sorted(minutes, reverse=True):
                moment = datetime(day.year, day.month, day.day, hour, minute, tzinfo=timezone.utc)
                if moment <= now:
                    return moment
    return None


def tick_id(moment: datetime) -> str:
    """The signal id of a cron time: no `~`, so it is a legal run id part."""
    return "tick-" + moment.astimezone(timezone.utc).strftime("%Y%m%dT%H%MZ")
