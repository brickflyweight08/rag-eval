# 11. Report and compare details

Status: accepted (2026-11-07)

## Context

R6 fixes the main columns and the plain-language comparison line but leaves the details.

## Decision

- Main table: Hit@k, Recall@k, MRR, Groundedness, Citation precision, Correctness (only
  when some run has it), Refusal accuracy, p50/p95 latency. Precision@k,
  hallucinated-citation rate and false-refusal rate go in a "Diagnostics" table.
- Cells read `mean [low, high] (n=…)`. Missing data is `-`.
- Gold IDs for the failure log come from the questions file named in `config.json`
  (or `--questions`); if it cannot be found, the log says so instead of failing.
- Failure log order: lowest recall, then lowest groundedness (a missing value counts as
  1.0, i.e. "not a failure"), then question ID.
- `compare` pairs only questions where the metric exists in both runs, reports B − A in
  percentage points, says "identical on every question" for a zero-width zero
  difference, "not clear at this sample size" when the CI includes zero, and otherwise
  "B is clearly higher/lower", adding "(lower is better for this metric)" where that
  applies. It never uses the word "winner". Runs with different `k` may be compared
  and are labelled with both values; different `questions_sha256` is refused.
- Globs like `runs/*` are expanded by rag-eval itself because PowerShell does not, and
  non-run folders are skipped.
- Console output is plain ASCII where rag-eval controls it, and stdout uses
  `errors="replace"`, so a Windows code page never crashes the CLI.

## Consequences

- Latency is not compared statistically (ADR 0004).
