# Evaluating an Archi deployment: a practical guide

This guide shows how to measure an Archi deployment the way the CMS CompOps team did in September 2026. It uses real
questions, a fixed judge, and several setups compared question by question. It is for any team running Archi v2, and
it complements the [Evaluation Guide](evaluation.md), which documents `archi eval qa` itself.

## At a glance

For every setup you test, you get:

- **correct:** the share of the required facts its answers contain;
- **wrong fact:** the share of its answers that state something false;
- both split by where each fact comes from (docs, tickets, code, git history, live services) and by question type;
- a page to read every answer, verdict and tool call.

The work has two stages:

1. **Build the golden set (most of the work).** Real conversations → candidates (verbatim questions, in your
   traffic's mix) → review and approve (a domain expert, one item at a time) → golden set (gold answers and atoms,
   frozen before the run).
2. **Run every setup, then compare.** Dataset (`export_dataset.py`) → run each setup (`archi eval qa`: prepare, run,
   score) → compare (`report.py`: ranges, pairs, by source) → read the lost answers (`review_page.py`) → fix a setup,
   then re-run.

Stage 1 happens once per version of the set. Stage 2 repeats for every setup and every fix.

## Get the tools

One branch has everything: [`preview/eval`](https://github.com/archi-physics/archi/tree/preview/eval) = Archi `main` +
our 8 open fixes + the eval kit. Use it until the fixes are merged; all 922 unit tests pass on it.

```bash
git clone https://github.com/archi-physics/archi.git
cd archi
git checkout preview/eval
pip install -e .
archi eval qa --help
```

The evaluation command itself is Antonio Battaglia's `archi eval qa`, already in `main`. Its guide,
[Evaluation Guide](evaluation.md), documents the dataset, config, prompt and judge-profile formats and every option.
This guide adds what we learned using it.

| Piece | What it gives you | Where |
| --- | --- | --- |
| gpt-5.5+ / gpt-6 through the Responses API | These models can call tools at all | [#655](https://github.com/archi-physics/archi/pull/655) |
| Context-window fallback | New model names no longer skip history trimming | [#656](https://github.com/archi-physics/archi/pull/656) |
| MCP fixes | Servers without `env` load; `allowed_tools` hides tools of a server | [#657](https://github.com/archi-physics/archi/pull/657) |
| Judge fixes | Reasoning-model judges work; "not found" is never a wrong fact; the judge sees the gold answer | [#658](https://github.com/archi-physics/archi/pull/658) |
| Run options | A no-tools setup; a per-answer time limit; token usage per attempt | [#659](https://github.com/archi-physics/archi/pull/659) |
| File-search cap | One huge match cannot overflow the model | [#660](https://github.com/archi-physics/archi/pull/660) |
| CERN AI Gateway provider | CERN-hosted models, with their own key | [#661](https://github.com/archi-physics/archi/pull/661) |
| Faster catalogue grep | File search stays fast on large catalogues | [#662](https://github.com/archi-physics/archi/pull/662) |
| Eval kit | Dataset export, report, review page, answer import | [`scripts/eval`](https://github.com/archi-physics/archi/tree/preview/eval/scripts/eval) |

The eval kit needs only Python 3.9+ and PyYAML; it reads run folders and never imports Archi:

- **`golden_item_template.yaml`:** one question per file.
- **`export_dataset.py`:** item files → the dataset (`benchmark.json`) plus `labels.json`.
- **`report.py`:** scored runs → a Markdown report with ranges, paired comparisons and splits by source and label.
- **`review_page.py`:** scored runs → one HTML page to read every answer, verdict and tool call.
- **`import_answers.py`:** answers from any other agent → a run folder the same judge scores.

## Build the golden set

The golden set is most of the work, and it decides what your numbers mean. Take questions from real traffic, split
each gold answer into small facts, and freeze the set before the final run.

### Questions

- **Take them verbatim from your own traffic.** Rated conversations are the best source, and answers your users
  disliked are the best seeds. Never write a question backwards from where you know the answer is.
- **Match your traffic's mix of use cases.** First classify a sample of real conversations: how-to, troubleshooting,
  live status, history, multi-turn. Then fill the set in the same proportions. Our first set had 24% "when did X
  change" questions against 3% in the traffic.
- **Keep a separate tuning set** of about 10 questions for prompt work. Never tune a prompt on golden questions.

### Gold answers and atoms

- **Write the gold answer from the sources.** Check live facts yourself on the day you write them.
- **Split the gold answer into atoms: the smallest single facts.**
    - A rule and its exception are two atoms.
    - No "and", "except" or brackets inside one atom.
    - Mark each atom required or optional.
- **Implicit atoms are allowed.** Some facts are not literally asked for, but leaving them out leaves the user worse
  off, such as the command after "how do I". Record why.
- **Give each atom its `answer_source`:** `ticket`, `doc`, `source_code`, `git_history`, `repo_config`,
  `live_<service>` or `monitoring`. This lets you split scores by source later, which is where you learn what to fix.
- **Record where you verified each atom:** a file and line at a commit, a ticket id, a URL, or a live command with its
  output and date.
- **Test every atom: could a model say this unaided?** Run a no-tools setup (see
  [Choose what to compare](#choose-what-to-compare)). Atoms it gets right measure the model, not your system. Mark
  them `parametric` and report scores with and without them.
- **Don't shape the set by what your tools can reach today.** Keep an atom that no tool can reach and mark it. It
  shows the gap.
- **End the gold answer with known wrong answers** ("These answers are wrong: …"). The judge sees the gold answer and
  uses it to recognise false claims.

### Live questions

- **Refresh their gold answers on the run day.**
- **Prefer answers that change rarely,** such as "the largest tape site". Avoid answers that change every hour.
- **Keep the answers that were true once, with what changed.** They help the judge and the reviewers tell outdated
  from wrong.

### Review and freeze

- **The pipeline:** traffic dump → candidates → review one by one → approve → export → freeze. A small web page that
  shows one candidate at a time, with approve and reject buttons, makes the review fast.
- **A domain expert approves every item.** Reviewing takes longer than drafting.
- **Freeze before the final run.** Any change after that is a new version of the set.

### How many questions

- With 42 questions and 1 attempt, each setup's score is uncertain by about ±8–9 points.
- To see a 5-point difference between two setups you need about 110 questions, or fewer questions with 3 attempts
  each.
- Start with about 40 to find problems; grow towards 100 before you rank close setups.

The item template is in [Reference](#reference), at the end.

## Choose what to compare

Each configuration you test is a "setup": one agent config plus one prompt. Change one thing per setup, so a
difference in score has one cause.

- **A reference setup:** your production configuration on your production data. Everything else is compared with it.
- **One setup per change you want to measure:** model, knowledge store, prompt, tools, step limit, or agent. Mark any
  setup that changes several things at once; its differences cannot be attributed.
- **A no-tools setup:** the same model with no tools at all. It shows which facts the model knows unaided (see
  [Build the golden set](#build-the-golden-set)).
- **Equal content when you compare stores.** Our vector store and our knowledge graph first scored differently mainly
  because they had ingested different things, not because of how they search. Before the run, sample facts from each
  source and check that both stores hold them.
- **Same context window when you compare models.** A model with a 64k window overflowed on tool-heavy answers; that
  measured the window, not the model's reasoning.
- **Point every service at the test instance.** A config copied from production can silently read production's files
  or databases. Check the tool traces of the first run.

## Run it

Every setup answers the same dataset and is scored by the same judge; only the agent config and prompt differ.

**1. Export the dataset.** Only items with `status: approved` are exported; the manifest lists warnings, such as
required atoms without a source.

```bash
python scripts/eval/export_dataset.py --set golden/ --out datasets/v1
# -> datasets/v1/benchmark.json, labels.json, export_manifest.json
```

**2. Pin one judge.** Use a strong model, and keep the same profile for every setup and every rerun. On a sample,
check it against a judge from another vendor; ours agreed on 95.7% of verdicts.

```yaml
# evaluator.yaml
version: 1
qa:
  atoms_extractor: {provider: openai, model: gpt-6-astra, timeout: 300}
  evaluator: {provider: openai, model: gpt-6-astra, timeout: 300}
```

**3. Write one config and one prompt per setup.** The config is your deployment YAML, copied from production and
pointed at the test instance. The prompt is the agent spec: a name, the tool list and the system prompt. The
[Evaluation Guide](evaluation.md) has both formats. Useful settings:

- `services.chat_app.recursion_limit`: the step budget. Give at least 50 (see
  [Mistakes that cost us time](#mistakes-that-cost-us-time)).
- `services.chat_app.answer_time_limit_seconds`: stops a hung answer; we used 900.
- `tools: []` in the agent spec: the no-tools setup.
- `allowed_tools: […]` under an MCP server: only those tools of that server reach the agent.

**4. Run each setup.**

```bash
archi eval qa --dataset datasets/v1/benchmark.json --evaluator-profile evaluator.yaml \
  --agent-config setups/prod.yaml --agent-spec setups/prod.md \
  --output-dir runs/prod --attempts 3 --run-workers 2 --score-workers 4
```

- **Attempts:** 1 while you develop the set and the setups; 3 for the final run, because answers vary from run to run.
- **Load:** 2 workers per setup; at most 3 setups using the same live service at once, started a few minutes apart.
- **Phases:** `prepare`, `run` and `score` also exist as separate commands. Use them to re-score without re-running,
  or to import answers from another agent.
- **Live questions:** refresh their gold answers, re-export, and run all setups on the same day, ideally at the same
  time.
- **Failures:** re-run start-up failures (tools that did not connect) in every setup alike.

**5. Other agents.** `archi eval qa run` drives only Archi's own agent. For anything else (another framework, a plain
tool loop, a coding assistant), prepare a run folder, ask your agent the questions yourself, and import its answers.
The same judge then scores them. The kit's `import_answers.py` documents the answer format.

```bash
archi eval qa prepare --dataset datasets/v1/benchmark.json --evaluator-profile evaluator.yaml --output-dir runs/other
# your runner -> answers.jsonl (one row per question and attempt)
python scripts/eval/import_answers.py runs/other answers.jsonl other_config.yaml other_prompt.md \
  --agent-name my-agent --provider openai --model gpt-5.5
archi eval qa score runs/other --evaluator-profile evaluator.yaml
```

**Cost.** Tokens are recorded per attempt, and the report shows them. For 42 questions × 1 attempt, the answers cost
us about $0.50 with a cheap model and $60–80 with a top model. The judge added about $2 per run. Tool-heavy answers
dominate: several hundred thousand input tokens per answer is normal.

## Read the results

Compare setups question by question, with ranges, and then read the answers they lost. The totals alone mislead.

```bash
python scripts/eval/report.py --labels datasets/v1/labels.json \
  --pair prod:new-model --no-tools no-tools \
  prod=runs/prod new-model=runs/new-model no-tools=runs/no-tools > report.md

python scripts/eval/review_page.py --dataset datasets/v1/benchmark.json --labels datasets/v1/labels.json \
  --out review.html prod=runs/prod new-model=runs/new-model
```

- **Correct and wrong fact go together.** "Correct" is the share of required facts found; "wrong fact" is the share
  of answers stating something false. A cheaper model that finds almost as many facts but invents more is not almost
  as good.
- **Use the paired comparison, not the difference of totals.** `--pair A:B` gives B − A per question with a 95%
  range. If the range includes 0, you have not shown a difference.
- **Split by answer source and by label.** That is where you learn what to fix: a setup that loses only on git
  history questions needs a different fix than one that loses on tickets.
- **Separate what the model knows unaided.** With `--no-tools`, the report shows scores without the questions the
  bare model already answers.
- **Read the lost answers in the review page.** It shows each question's atoms with every setup's verdict, the
  answer, and each tool call with its input and output. Most of our findings came from there:
    - a tool the agent misunderstood;
    - a store that lacked a field the other one had;
    - a wrap-up after the step limit that dropped everything found;
    - a tool server upgrade that returned empty results.
- **Label failed attempts apart from wrong answers.** The report labels them as context overflow, oversized tool
  output, time limit, step limit, API error, or other.
- **Keep a change log** of every deviation and fix, with its date, while you go.

Run folders, reports and review pages contain your questions, answers and tool outputs. Share them only where that
data may go, and keep them out of public repositories.

## Mistakes that cost us time

Each of these cost us a rerun or a wrong conclusion. Check them before your final run.

- [ ] **Test configs point at the test instance.** A config copied from production read production's data manager,
  and the "test" setup silently used production files.
- [ ] **Every MCP server's tools load.** One broken server removes all MCP tools of the agent. Check the tool list at
  the start of each run.
- [ ] **Tool output is capped.** One live call returned 29 MB and broke the answer; one grep hit returned 2.6 M
  characters. The preview branch caps file search; cap your own tools too.
- [ ] **The agent has enough steps.** When Archi runs out of steps, its wrap-up answer gets none of the tool results,
  so everything found is lost. Give at least 50 steps until this is fixed.
- [ ] **File search scales to your catalogue.** It fetched metadata for every file before matching: 12 minutes per
  search on 165k files. Fixed in the preview branch.
- [ ] **The judge does not count "I could not find it" as a wrong fact.** Fixed in the preview branch's judge prompt.
- [ ] **Reasoning models get no temperature, and gpt-5.5+/gpt-6 models use the Responses API.** Both fixed in the
  preview branch.
- [ ] **At most 3 setups share a live service at once, with staggered starts.** Under more load our Rucio calls hung
  until the tool timeout.
- [ ] **Start-up failures are re-run in every setup alike.** Tools that failed to connect are infrastructure problems;
  report them apart from answer failures such as timeouts and context overflows.
- [ ] **Models are compared at similar context windows.**
- [ ] **Every deviation goes into a change log as you go.** You will need it to explain the numbers weeks later.

## Reference

### Golden item template

One file per question, in one folder. The same file is in the kit as
[`scripts/eval/golden_item_template.yaml`](https://github.com/archi-physics/archi/blob/preview/eval/scripts/eval/golden_item_template.yaml).

```yaml
# Golden item template: one file per question. Comments say what each field is for; delete them in real items.
# export_dataset.py reads every .yaml file of a directory (files starting with "_" are skipped).
id: myset-q001                      # stable id, never reused
status: approved                    # draft | approved; only approved items are exported
provenance: user_traffic            # where the question came from: user_traffic | ticket | expert | ...
provenance_ref: conversation 1234, 2026-08-07, turn 1   # enough to find it again; no user names
seed_reaction: dislike              # the user's rating of the original answer, if any: like | dislike | null
original_question: how do i delete a rule that has a child rule   # verbatim, typos included
question: How do I delete a rule that has a child rule?          # the question as asked, fixed only for spelling
gold: |
  The full reference answer, written from the sources and checked. The judge sees it as the reference.
  End with the known wrong answers, so the judge can recognise them:
  These answers are wrong: "child rules are the per-dataset rules of a container rule".
atoms:                              # the smallest facts the answer must (required) or may (optional) contain
- id: delete_blocked
  text: a rule with a child rule cannot be deleted directly
  required: true
  answer_source: source_code        # doc | ticket | source_code | git_history | repo_config | live_<service> | monitoring
                                    # alternatives: "doc or source_code"; both needed: "ticket+live_<service>"
  asked_by: How do I delete a rule that has a child rule?   # the part of the question that asks for it
  source: 'where it was verified: file:line at a commit, a ticket id, a URL, or "LIVE <date>: <command> -> <output>"'
- id: detach_command
  text: the child rule must first be detached from its parent rule
  required: true
  answer_source: source_code
  asked_by: How do I delete a rule that has a child rule?
  implicit: true                    # not literally asked ...
  implicit_because: without it the user cannot act on the answer   # ... but leaving it out leaves them worse off
  source: '...'
- id: owner_needed
  text: only the rule's owner or an admin can change it
  required: false                   # nice to have: reported, but not part of the score
  answer_source: doc
  reachable: false                  # no tool of the tested setups can reach this source today; keep the atom anyway
  source: '...'
contradiction_traps:                # known wrong answers, for reviewers (also put them at the end of `gold`)
- child rules are the per-dataset rules of a container rule
past_answers:                       # answers that were true once, with what changed (exported after the gold answer)
- text: 'Use the old command X.'
  changed: 'Removed in version 2; replaced by Y.'
  source: 'where the change is documented'
tier: golden                        # golden | tuning (tuning items are for prompt work, never in the golden set)
time_sensitive: false               # true if the answer depends on live state: refresh the gold on the run day
labels:                             # describe the question, never a setup's ability; used to split the scores
  task_type: procedure              # e.g. procedure | troubleshooting | status | history | explanation
  domain: data_management           # your own domains
  multi_record: 'no'                # does the answer need several records (tickets, files) combined?
  implicit_ask: 'yes'               # does it have implicit atoms?
```

### Files

| File | Written by | Holds |
| --- | --- | --- |
| `benchmark.json` | `export_dataset.py` | The dataset for `--dataset`: id, question, gold answer (with outdated answers), atoms with their required flag |
| `labels.json` | `export_dataset.py` | Per question: its labels, computed `multi_source` and `changed_over_time`; per atom: answer source, implicit, reachable |
| `answers.jsonl` | `archi eval qa run` or `import_answers.py` | Per attempt: status, answer or error, duration, every tool call (name, input, output), token usage |
| `evaluation_results.jsonl` | `archi eval qa score` | Per attempt: each atom's verdict (entailed, not mentioned, contradicted) with the judge's reason |
| `manifest.json`, `report.md` | `archi eval qa` | The run's phases, hashes of every input and output, and a one-run summary |
| `*.resolved.yaml`, `*.resolved.md` | `archi eval qa` | The exact config, prompt and judge profile used. Check for secrets before sharing |
