# CMS Evidence Compiler

After retrieval, compile evidence before writing prose.

Classify the question as one primary operator knowledge type:
`procedure`, `command`, `config_parameter`, `operational_term`,
`operational_process`, `timeline`, `policy_rule`, `incident`, `definition`, or
`out_of_scope`.

Build an online fact object from retrieved evidence:

- `fact_type`: the selected operator knowledge type.
- `query_terms`: exact terms from the question.
- `expanded_terms`: aliases, source-family terms, or timestamp fields actually
  used during retrieval.
- `filled_fields`: source-backed fields needed for the answer type.
- `missing_fields`: required fields not found in the pinned generation.
- `assertion_level`: `documented`, `inferred`, `conflict`, or `missing`.

Answer from the compiled fact, not from raw top-hit order. Keep the answer
operator-useful: give commands, paths, current status, timeline scope, policy
arbitration, or root cause directly when the OKG supports it. Mark historical,
current, inferred, conflict, and missing facts explicitly.

## Operator Answer Contract

Every final answer should expose this contract, either as explicit JSON fields
when the schema allows them or as compact prose inside `short_answer`:

- `answerability`: one of `answered_from_pinned_okg`,
  `partial_from_pinned_okg`, `live_source_required`, `insufficient_evidence`,
  or `out_of_scope`.
- `supported_finding`: the strongest OKG-backed conclusion, with real evidence
  IDs.
- `operator_action_path`: the next concrete checks or commands an operator can
  run. Include where to run them when the evidence supports that.
- `decision_criteria`: how to interpret the possible outcomes. Use explicit
  `if X, then Y; if not, check Z` logic when diagnosing incidents, quotas,
  workflow failures, current status, or source conflicts.
- `source_authority`: the source family that owns the truth for the question,
  such as current docs, JIRA, Rucio, CRAB, ReqMgr, Unified, WMAgent, Condor,
  or monitoring snapshots.
- `remaining_gap`: the decisive field or live system not present in the pinned
  OKG generation.

If the graph has historical evidence but not live state, answer the historical
part and make the operator action path the completion path. Do not turn the
absence of live proof into a dead-end answer.

Before final synthesis, check whether the best evidence is too narrow:

- If the strongest source is a single chunk from a procedure, policy, or
  runbook, retrieve parent/source metadata and enough neighboring or ordered
  chunks to fill the answer shape.
- If a question asks for a current procedure or source authority, record which
  timestamp/source-family fields were inspected and whether newer conflicting
  evidence was found.
- If the graph has related evidence but not the decisive field, answer the
  supported part, name the missing decisive field, and provide the next live
  authority check.

Do not finish with retrieval labels such as "strongest OKG evidence" unless the
question asked for evidence discovery. Summarize the evidence and keep IDs as
citations.
