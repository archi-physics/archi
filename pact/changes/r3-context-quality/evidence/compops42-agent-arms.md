# CompOps-42 agent arms on the comp-ops instance: before and after file text

Evidence for `acc.r3-context-quality.raw-arm` and `acc.r3-context-quality.run`
(req.r3-context-quality.context-eval). The question set, the gold evidence ids, the answer key and
the agents' answers are private to the instance and are not in this repository; this file holds only
the method and the numbers.

## Result

On the comp-ops instance, once the graph holds the text of the git repository files, the agent with
the graph's MCP tools (agent-graph) finds as much gold evidence as the agent with bash over the raw
files (agent-bash), and its answers score higher with both graders:

| | agent-bash | agent-graph |
|---|---|---|
| gold-evidence recall | 0.725 | 0.729 |
| answer correctness, LLM judge | 0.571 | 0.702 |
| answer correctness, blind second grader | 0.657 | 0.783 |
| model calls per try | 14.3 | 10.0 |
| dollars per try | 0.0098 | 0.0059 |

This is one run of three tries per question, with no significance test. Before the fix, agent-graph
found far less evidence (0.259 against 0.747), because the instance held each git file as a name with
no text.

## Setup (both runs)

- Questions: CompOps-42 (the okg comp-ops benchmark; decision row of 2026-10-27). 33 of the 42 have
  file-backed gold evidence and are run; the other 9 have none and are not scored. Gold mapping:
  drafted with `python -m archi.evaluation.draft_gold_sources`, reviewed by an independent agent
  reviewer (Jason, 2026-10-29), checked by a fresh verifier, then frozen. Gold v3: 76 gold ids over the
  33 questions, 66 distinct records: 44 git files and 22 tickets. One git file is unavailable to the
  scorer (every recognisable line of it also appears in another corpus file), so 65 records count: 43
  git files and
  22 tickets.
- Facts only a live system can answer: 4 of the 33 questions carry 9 such facts in the answer key (live
  Rucio state). Neither arm can reach them, so they cap answer correctness for both.
- `archi eval --context --agents agent-bash,agent-graph` at archi_v3 2426842c (okg 0.2.0): model
  gpt-6-luna, reasoning medium, Responses API; one prompt for both arms but for the paragraph on where
  knowledge lives; 3 tries per question; caps per try: 30 turns, 200 tool calls, 900 s; 16,000 output
  tokens per model call; one spend ledger and a hard cap per run; judge off in the runs.
- agent-bash: bash in bubblewrap over the raw corpus the graph was built from (corpus digest
  `sha256:036684f98e5069a2fc7129c904618dafea93959691ceab4c36b7583a4c197136`, 1,910,219 record files),
  identical in both runs.
- agent-graph: the instance's MCP tools (search, inspect, expand, filter, map, aggregate).
- Evidence recall: a gold record counts only when a tool response the model saw shows its text (the
  task.raw-arm-scoring rule); per try, averaged over 95 tries per arm. Tries where an arm failed are left
  out of both arms (bash failed 4 tries in each run).
- Answer correctness, graded after each run: 0, 0.5 or 1 against CompOps-42 answer key v1 (the answer
  and its required facts; optional facts marked optional), all 99 tries per arm; a failed try scores 0.
  - LLM judge: a lane script around `archi.evaluation.raw_arm.Judge` (archi's judge prompt),
    gpt-6-luna (the same model as the agents), reasoning high, two passes averaged.
  - Blind second grader: three Claude subagents, each grading a shuffled third of the answers (66 each)
    with no arm and no tool calls shown; one grade per answer.

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
| answers, LLM judge | 0.576 | 0.571 | 0.540 | 0.702 |
| answers, blind second grader | 0.646 | 0.657 | 0.530 | 0.783 |
| answers, judge, git-only questions (60 tries) | 0.562 | 0.542 | 0.450 | 0.692 |
| answers, judge, ticket-only questions (24 tries) | 0.719 | 0.688 | 0.729 | 0.719 |
| answers, judge, mixed questions (15 tries) | 0.400 | 0.500 | 0.600 | 0.717 |
| model calls (turns) per try | 14.3 | 14.3 | 12.8 | 10.0 |
| tool calls per try | 14.0 | 13.8 | 22.5 | 18.0 |
| dollars per try | 0.0107 | 0.0098 | 0.0076 | 0.0059 |
| tries cut off at 30 turns | 2 | 3 | 4 | 0 |
| tries that hit the output-token cap before answering | 4 | 4 | 0 | 0 |

Per question (mean over its tries), bash better / tie / graph better: before, judge 13 / 13 / 7 and
second grader 19 / 6 / 8; after, judge 6 / 15 / 12 and second grader 9 / 11 / 13.

The judge and the second grader agree exactly on 79% of answers after (82% before) and are never more
than half a point apart. The second grader is the more lenient: after, it scored 39 answers higher than
the judge and 2 lower (before, 23 and 12), so part of its larger graph advantage is leniency. The
judge's two passes agree with each other on 96% of answers.

## Model-free search arms (diagnostic)

The instance's search tool called directly, no model; recall@10 of the gold over the same 33 questions.

| | before | after |
|---|---|---|
| graph search (server default) | 0.153 (MRR 0.155) | 0.143 (MRR 0.165) |
| lexical baseline | 0.143 (MRR 0.152) | 0.128 (MRR 0.160) |

Both runs report status pass. Plain recall@10 fell slightly after the fix while MRR rose; a likely
cause, not measured, is that the 48,502 new document and chunk nodes compete for the top ten. The
agents, which search, read and search again, gained.

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
| try hit the output-token cap | 0 | 5 |
| total missed record-tries | 70 | 76 |

The arms now miss for the same reasons, mostly large source modules where the answer sits in one
function that neither agent's search reached. The replay matched the recorded scores on every try but
4 graph tries in the after run, where search results differed on replay; their causes may be off.

## Caveats

- Ticket recall fell in both arms between the runs although nothing on the ticket side changed:
  run-to-run variance of one three-try run. On evidence the arms are at parity, not graph ahead.
- The LLM judge is the same model as the agents; the second grader is a different model family.
- The after report warns that the corpus was built at the earlier generation; only file text was
  added to the graph, and the raw corpus already held those files.
- The fix is instance configuration. Whether code-text should read every text file by default is a
  separate product decision.

## Spend

Before: agent arms $1.81, judge $0.07. After: agent arms $1.72 over four attempts (the complete run
$1.56; one killed by a host event at try 11, $0.04; one stopped at try 24 by a disk rule, $0.13; one
stopped before its first model call, $0, with no ledger written, per the lane's run log), judge $0.08.
