# CompOps-42 agent arms on the comp-ops instance: before and after file text

Evidence for `acc.r3-context-quality.raw-arm` (req.r3-context-quality.context-eval). The question
set, the gold evidence ids, the answer key and the agents' answers are private to the instance and
are not in this repository; this file holds only the method and the numbers.

## Result

On the comp-ops instance, the agent with the graph's MCP tools (agent-graph) does at least as well as
the agent with bash over the raw files (agent-bash) once the graph holds the text of the git
repository files:

| | agent-bash | agent-graph |
|---|---|---|
| gold-evidence recall | 0.725 | 0.729 |
| answer correctness, LLM judge | 0.571 | 0.702 |
| answer correctness, blind reviewer | 0.657 | 0.783 |
| model calls per try | 14.3 | 10.0 |
| dollars per try | 0.0098 | 0.0059 |

Before that fix, agent-graph found far less evidence (0.259 against 0.747), because the instance held
each git file as a name with no text.

## Setup (both runs)

- Questions: CompOps-42 (the okg comp-ops benchmark; decision row of 2026-10-27), the 33 questions
  with file-backed gold evidence. Gold mapping: drafted with `python -m
  archi.evaluation.draft_gold_sources`, reviewed by an independent reviewer and verified, then frozen
  (gold v3: 76 gold ids over the 33 questions, 66 distinct records, 65 of them available to the
  agents: 43 git files and 22 tickets).
- `archi eval --context --agents agent-bash,agent-graph` at archi_v3 2426842c (okg 0.2.0): model
  gpt-6-luna, reasoning medium, Responses API; 3 tries per question; caps 30 turns, 200 tool calls,
  900 s per try; one spend ledger and a hard cap per run.
- agent-bash: bash in bubblewrap over the raw corpus the graph was built from (corpus digest
  `sha256:036684f98e5069a2fc7129c904618dafea93959691ceab4c36b7583a4c197136`, 1,910,219 record files),
  identical in both runs.
- agent-graph: the instance's MCP tools (search, inspect, expand, filter, map, aggregate).
- Evidence recall: a gold record counts only when a tool response the model saw shows its text (the
  task.raw-arm-scoring rule); per try, averaged; error tries left out; 95 scored tries per arm.
- Answer correctness: 0, 0.5 or 1 against CompOps-42 answer key v1 (the answer and its required facts;
  optional facts marked optional). LLM judge: gpt-6-luna, reasoning high, two passes, archi's judge
  prompt. Blind reviewer: three independent reviewers grading a shuffled packet with no arm and no tool
  calls. 99 tries per arm; error tries score 0.

## The fix between the runs (instance only)

- Before: each repository came in through git-code only, which ingests files as metadata nodes (path,
  size, commits) with no text. Reading a file node's text returned `no_text_chunks` on all 66 such reads
  in the first run.
- After: the code-text source (okg DocCorpusSource) on each of the 18 repositories, reading every text
  file: code as shipped, plus Markdown, YAML, shell, JSON, config and extensionless scripts routed to
  plain-text chunks. As shipped, code-text reads code extensions only and would have covered 19 of the
  44 git files in the gold.
- 303 files were left out after two secret scans (a credential-pattern scan, then okg's own leak
  patterns: literal values, base64 blobs or keys, every authorization header, credential URL or token
  value); none is a gold file.
- Published as a new generation, not blocked: 48,502 nodes and 92,839 edges added, none retracted.
  59 of the 66 failed reads now reach the file's text; the other 7 are a file excluded by the scan and
  6 old paths of moved or deleted files. Every gold document is readable and in the search index.

## Before and after

| | bash before | bash after | graph before | graph after |
|---|---|---|---|---|
| evidence recall, all gold | 0.747 | 0.725 | 0.259 | 0.729 |
| evidence recall, git-file gold | 0.652 | 0.669 | 0.007 | 0.684 |
| evidence recall, ticket gold | 0.847 | 0.725 | 0.803 | 0.713 |
| answers, judge | 0.576 | 0.571 | 0.540 | 0.702 |
| answers, blind reviewer | 0.646 | 0.657 | 0.530 | 0.783 |
| answers, judge, git-only questions | 0.562 | 0.542 | 0.450 | 0.692 |
| answers, judge, ticket-only questions | 0.719 | 0.688 | 0.729 | 0.719 |
| answers, judge, mixed questions | 0.400 | 0.500 | 0.600 | 0.717 |
| model calls (turns) per try | 14.3 | 14.3 | 12.8 | 10.0 |
| tool calls per try | 14.0 | 13.8 | 22.5 | 18.0 |
| dollars per try | 0.0107 | 0.0098 | 0.0076 | 0.0059 |
| tries cut off at 30 turns | 2 | 3 | 4 | 0 |
| tries ended by a model error | 4 | 4 | 0 | 0 |

Judge and blind reviewer agree exactly on 79% of answers after (82% before) and are never more than
half a point apart; the judge is the stricter. The judge's two passes agree on 96%.

## Why records are missed

Every try with a miss was replayed with no model against the same generation and corpus, and each
missed gold record classified.

Before: of 170 graph misses, 154 were git files the graph could not show (in 55 the agent had found the
right file node); the other 16 were tickets (never surfaced, listed but not opened, or too short for
the scorer to credit).

After:

| cause | agent-graph | agent-bash |
|---|---|---|
| never surfaced by any response | 53 | 51 |
| named in a response, never read | 11 | 18 |
| read, too little text shown to credit | 6 | 2 |
| try ended by a model error | 0 | 5 |
| total missed record-tries | 70 | 76 |

The arms now miss for the same reasons, mostly large source modules where the answer sits in one
function that neither agent's search reached.

## Caveats

- Ticket recall fell in both arms between the runs although nothing on the ticket side changed:
  run-to-run variance of one three-try run. On evidence the arms are at parity, not graph ahead.
- The after report warns that the corpus was built at the earlier generation; only file text was
  added to the graph, and the raw corpus already held those files.
- The fix is instance configuration. Whether code-text should read every text file by default is a
  separate product decision.

## Spend

Before: agent arms $1.81, judge $0.07. After: agent arms $1.72 (one complete run, two attempts
stopped early by a host event and a disk rule), judge $0.08.
