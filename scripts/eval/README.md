# Eval kit for `archi eval qa`

Small, dependency-light scripts around Archi's evaluation command (`archi eval qa`, see
`docs/docs/evaluation.md`). They cover the parts the command leaves to you: writing the golden set,
comparing several setups, reviewing answers, and scoring agents other than Archi's own.

| script | what it does |
|---|---|
| `golden_item_template.yaml` | One question per file: gold answer, atoms (required facts) with their source, labels. |
| `export_dataset.py` | Golden item files → `benchmark.json` (the `--dataset` of `archi eval qa`) + `labels.json`. |
| `report.py` | Several scored runs → Markdown: correct and wrong-fact rates with 95% ranges, paired comparisons, scores by answer source and by label, failed attempts, tokens. |
| `review_page.py` | Several scored runs → one HTML page: per question, every setup's answer, verdicts and tool calls. |
| `import_answers.py` | Answers from any other agent → a run folder that `archi eval qa score` judges like the others. |

They need Python ≥ 3.9 and PyYAML only, and do not import Archi.

## Typical flow

```bash
# 1. golden set -> dataset
python scripts/eval/export_dataset.py --set golden/ --out datasets/v1

# 2. one run per setup (same dataset and judge profile; one config + prompt per setup)
archi eval qa --dataset datasets/v1/benchmark.json --evaluator-profile evaluator.yaml \
  --agent-config setups/prod.yaml --agent-spec setups/prod.md --output-dir runs/prod --attempts 3 --run-workers 2
archi eval qa --dataset datasets/v1/benchmark.json --evaluator-profile evaluator.yaml \
  --agent-config setups/new-model.yaml --agent-spec setups/new-model.md --output-dir runs/new-model --attempts 3 --run-workers 2

# 3. compare and review
python scripts/eval/report.py --labels datasets/v1/labels.json --pair prod:new-model \
  prod=runs/prod new-model=runs/new-model > report.md
python scripts/eval/review_page.py --dataset datasets/v1/benchmark.json --labels datasets/v1/labels.json \
  --out review.html prod=runs/prod new-model=runs/new-model
```

Run folders, reports and review pages contain your questions, answers and tool outputs. Keep them out of
public repositories.
