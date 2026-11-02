# Sealed Benchmark Mode

Use this skill when answering CMS benchmark questions or replaying a curated
question set.

## Contract

- Accept only sanitized fields: question ID, category, question text, and
  explicitly allowed non-answer metadata.
- Do not read, request, quote, summarize, or infer from `reference_answer`,
  `reference_source`, baseline answer files, external comparator reports, or
  prior benchmark comparisons.
- Answer from the CMS OKG graph and cited evidence IDs only.
- Return confidence and gaps. A gap is acceptable; baseline leakage is not.

## Output Shape

For every answer, include:

- `question_id`
- short answer
- evidence IDs: node IDs, chunk IDs, documentation page IDs, view names, or
  bounded SQL evidence
- confidence: high, medium, or low
- gaps or ambiguity

## Evaluation Boundary

The answerer does not compare against baseline answers. A separate evaluator may
compare sealed answers privately and report only redacted outcomes such as
aligned, partial, miss, conflict, deferred, and failure stage.
