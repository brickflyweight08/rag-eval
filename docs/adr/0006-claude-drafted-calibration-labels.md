# 6. The demo's calibration labels are a Claude-drafted placeholder

Status: accepted (2026-11-07)

## Context

R8 asks for `examples/toy_handbook/human_labels.jsonl`, about 10 hand-labelled claims
for the calibration demo. No human labeller was available when the repo was built, and
the design rules say rag-eval never generates gold labels with an LLM.

## Decision

- Claude (the coding assistant that wrote the repo) drafted 12 labels (7 supported,
  5 not supported, including deliberately tricky ones such as "matches contributions
  up to 7%"). Each line carries `"labelled_by": "claude-draft (needs human review)"`,
  and the README and `examples/toy_handbook/README.md` say the same.
- The labels are never described as human labels. The file keeps the name the spec
  gives it so the documented commands work.
- The same applies to the toy benchmark: its gold IDs and reference answers were written
  by Claude together with the fictional corpus, so they are correct by construction, but
  they too deserve a human read-through.
- This does not change the rule for rag-eval itself: the library never produces gold
  labels, reference answers or calibration labels at run time.

## Consequences

- A calibration result on these labels measures agreement with Claude's judgement, not
  with a person's. A human should review (and correct) every line, then delete the
  `labelled_by` note or change it to their name.
