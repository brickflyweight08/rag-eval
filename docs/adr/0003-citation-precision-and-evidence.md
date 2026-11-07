# 3. Citation precision denominator and evidence IDs

Status: accepted (2026-11-07)

## Context

The spec defines citation precision as "share of valid citations whose chunk the
verifier used as evidence for at least one claim". It does not say what happens with
zero valid citations, repeated citations, unsupported claims that still name a chunk,
or an `evidence_id` that is not in the context.

## Decision

- Citations are de-duplicated (first occurrence kept) before scoring.
- **Valid citations** = distinct cited IDs present in `output.context`. **Hallucinated
  citations** = distinct cited IDs not in `output.context`.
- Citation precision = |valid citations used as evidence| / |valid citations|.
- It is **`None`, not 0, when there are zero valid citations** (nothing cited, or only
  hallucinated IDs cited) **or zero claims** (including refusals). With no
  denominator, or nothing to support, there is nothing to measure; hallucinated
  citations are reported separately, so citing only made-up IDs is still visible.
- Only **supported** claims contribute evidence. An `evidence_id` returned for an
  unsupported claim is discarded, and an `evidence_id` that is not one of the context
  chunk IDs is set to `None` (brackets around it are stripped first).
- Run-level "hallucinated-citation rate" = share of answered questions with at least
  one hallucinated citation.

## Consequences

- A system that never cites has `n = 0` for citation precision rather than a misleading
  0.0 or 1.0; the report shows `n` next to every number.
- Citation precision depends on the judge's choice of a single best `evidence_id` per
  claim, so a valid citation of a second supporting chunk can count as unused. This is
  a known underestimate.
