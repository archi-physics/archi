# CMS Site Profile Arbitration

Use this skill for questions about a CMS site, RSE, storage endpoint, facility,
or monitoring profile.

## Source Families

Keep these evidence families separate:

- Site catalog facts: site tier, country, facility, status, and contained
  endpoints.
- RSE/storage facts: storage endpoint subtype, RSE name, hosted datasets, and
  Rucio overlays.
- Monitoring facts: Condor, SAM, transfer, and other monitoring snapshots.
- Ticket/document facts: incidents, policy discussions, migrations, and
  historical notes.

## Conflict Handling

- If source families disagree, cite both sides and mark the answer as a source
  conflict.
- Do not silently treat monitoring absence as storage absence.
- Do not treat a historical ticket as current status unless current catalog or
  monitoring evidence supports it.
- Name the missing source family when a profile is incomplete.
