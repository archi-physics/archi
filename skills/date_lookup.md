# CMS Date Lookup

Use this skill for questions asking when an operational change happened.

## Search Strategy

- Search the literal entity name first.
- Search aliases and historical names, including old RSE names, endpoint
  hostnames, site names, and service labels.
- Search operational verbs: enabled, disabled, moved, migrated, commissioned,
  restored, re-enabled, cleanup, deactivated, removed, and switched.
- Search ticket families and parent epics when an isolated ticket gives only a
  partial event.

## Date Discipline

- Return a date only when it is directly supported by ticket metadata,
  documented prose, or graph evidence.
- Distinguish ticket creation date, update date, command execution date, and
  operator-stated effective date.
- If only a latest-known-by date exists, say "by <date>" instead of claiming an
  exact cutover.
- If evidence is absent, return a low-confidence gap with the searched entity
  aliases.
