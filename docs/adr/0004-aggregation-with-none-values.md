# 4. Aggregating metrics that can be None (refusals, latency, errors)

Status: accepted (2026-11-07)

## Context

Many per-question fields are `None`: retrieval metrics for unanswerable questions,
groundedness with zero claims, `refused` for retrieval-only runs, latency when a system
fails. The spec asks for `mean [95% CI]` per metric and p50/p95 latency but does not say
how `None` is aggregated.

## Decision

- **`None` means "not applicable or not measured"** and is dropped before averaging.
  Each metric reports its own `n` (the number of questions it was computed on); the
  report prints `n` in every cell.
- **Errors are `None`, not 0.** A question whose `retrieve()` raised has no retrieval
  metrics; the run header shows the error count and the failure log lists each error.
  Counting errors as zeros would mix reliability with quality.
- **Refusal metrics** are means of per-question indicators: refusal accuracy = mean of
  `refused` over unanswerable questions with `refused` not `None`; false-refusal rate =
  the same over answerable questions. This makes them bootstrappable and pairable.
- **Latency** p50/p95 use linear-interpolation percentiles over non-`None` values and
  are shown **without a CI** (they are descriptive, not the quality claim being tested,
  and `compare` does not test them). Source: the system's own `latency_ms` if given,
  else wall-clock time of `answer()`, or of `retrieve()` in retrieval-only runs.
- Bootstrap: 2,000 resamples, seed 0, percentile intervals; a single value gives a
  zero-width interval.
- Cohen's kappa is `nan` (printed "undefined") when both raters used one label only.

## Consequences

- Two runs can have different `n` for the same metric (for example if one had more
  errors); `compare` only pairs questions scored in both runs.
