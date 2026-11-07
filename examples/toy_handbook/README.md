# Toy handbook demo

An original, **fictional** employee handbook for *Thimblewood Instruments*, a made-up
maker of barometers in the made-up town of Pellmoor. Because the company does not exist,
an LLM cannot answer from memory, which keeps the groundedness test meaningful.

| File | What it is |
|---|---|
| `corpus.jsonl` | 30 short passages (leave, equipment, travel, security, benefits) |
| `questions.jsonl` | 15 questions: 5 lookup, 4 multi-passage, 3 paraphrase (near-miss wording), 3 unanswerable |
| `systems.py` | `make_bm25` (rank-bm25) and `make_dense` (sentence-transformers) demo systems |
| `human_labels.jsonl` | 12 labelled claims for `rag-eval calibrate` (see the warning below) |
| `results/` | Report and comparison from the retrieval-only demo runs |

> **Warning: `human_labels.jsonl` is a Claude-labelled draft that needs human review.**
> The labels were written by Claude (the AI assistant that built this repo), not by a
> person. Each line says so in `"labelled_by"`. Review every label before using the
> calibration result. See [ADR 0006](../../docs/adr/0006-claude-drafted-calibration-labels.md).
>
> The questions' gold IDs and reference answers were also written by Claude, together
> with the corpus, so they match it by construction; a human read-through is still wise.

## Running

From the repository root:

```bash
rag-eval run --system examples.toy_handbook.systems:make_bm25 \
    --questions examples/toy_handbook/questions.jsonl --k 3 --out runs/bm25_k3 --retrieval-only
```

Drop `--retrieval-only` once an LLM is configured. The demo generator reads
`RAG_EVAL_DEMO_BASE_URL`, `RAG_EVAL_DEMO_API_KEY` and `RAG_EVAL_DEMO_MODEL`, and falls
back to the `RAG_EVAL_JUDGE_*` settings. Without a key, each answer is recorded as an
error (retrieval metrics are still scored).
