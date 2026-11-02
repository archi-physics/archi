# Rucio Command Extraction

Use this skill when a CMS question asks for a Rucio CLI command or operational
syntax.

## Command Rules

- Return canonical option-based syntax first when possible.
- Then include positional syntax or CMS examples found in OKG evidence.
- Preserve placeholders clearly: `<DID>`, `<SCOPE:NAME>`, `<RSE_EXPRESSION>`,
  `<SECONDS>`, `<ACCOUNT>`, and `<COMMENT>`.
- State which options depend on CMS policy: approval, lifetime, activity,
  account, grouping, and comment.

## Evidence

- Cite documentation page or chunk IDs containing the syntax.
- Cite ticket examples only as examples, not as the canonical command contract,
  unless no documentation page exists.
- Do not invent flags; leave unknown flags out or mark them as policy-dependent.
