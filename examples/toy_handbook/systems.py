"""Two demo RAG systems over the fictional Thimblewood Instruments handbook.

- ``BM25System``: keyword retrieval with ``rank-bm25``.
- ``DenseSystem``: ``sentence-transformers`` embeddings with cosine similarity.

Both share the same small answer step: an LLM (any OpenAI-compatible endpoint) answers
from the retrieved chunks only and cites them as ``[chunk_id]``.

Use them through rag-eval's plugin loader::

    rag-eval run --system examples.toy_handbook.systems:make_bm25 \\
        --questions examples/toy_handbook/questions.jsonl --k 3 --out runs/bm25_k3

The generator is configured with RAG_EVAL_DEMO_BASE_URL / _API_KEY / _MODEL, falling back
to the judge settings (RAG_EVAL_JUDGE_*). This is example code, not part of rag-eval.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rag_eval import RetrievedChunk, SystemOutput

CORPUS_PATH = Path(__file__).with_name("corpus.jsonl")
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CITATION = re.compile(r"\[([A-Za-z0-9_.:/-]+)\]")
_STOPWORDS_TEXT = (
    "a an and are at be by can do does for from how i if in is it my of on or the to "
    "what when which who with you your"
)
STOPWORDS = frozenset(_STOPWORDS_TEXT.split())

ANSWER_PROMPT = """You answer questions about the Thimblewood Instruments employee handbook.
Use ONLY the passages below. Cite every passage you use by its ID in square brackets,
for example [leave-01]. If the passages do not contain the answer, reply exactly:
"I don't have enough information to answer that."

PASSAGES:
{passages}

QUESTION: {question}
ANSWER:"""

Generator = Callable[[str], str]


def load_corpus(path: str | Path = CORPUS_PATH) -> list[dict[str, str]]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


def _env(name: str) -> str | None:
    return os.environ.get(f"RAG_EVAL_DEMO_{name}") or os.environ.get(f"RAG_EVAL_JUDGE_{name}")


def openai_generator(
    model: str | None = None, base_url: str | None = None, api_key: str | None = None
) -> Generator:
    """An OpenAI-compatible chat call at temperature 0 (created lazily, on first use)."""
    state: dict[str, Any] = {}

    def generate(prompt: str) -> str:
        key = api_key or _env("API_KEY")
        if not key:
            raise RuntimeError(
                "the demo generator has no API key: set RAG_EVAL_DEMO_API_KEY or "
                "RAG_EVAL_JUDGE_API_KEY, or run rag-eval with --retrieval-only"
            )
        if "client" not in state:
            import openai

            url = base_url or _env("BASE_URL") or "http://localhost:11434/v1"
            state["client"] = openai.OpenAI(base_url=url, api_key=key)
        response = state["client"].chat.completions.create(
            model=model or _env("MODEL") or "llama3.1:8b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content or ""

    return generate


class _HandbookSystem:
    name = "handbook"

    def __init__(
        self,
        corpus_path: str | Path = CORPUS_PATH,
        generator: Generator | None = None,
        llm_model: str | None = None,
        llm_base_url: str | None = None,
        llm_api_key: str | None = None,
    ) -> None:
        self.docs = load_corpus(corpus_path)
        self.generate = generator or openai_generator(llm_model, llm_base_url, llm_api_key)

    def _scores(self, question: str) -> list[float]:
        raise NotImplementedError

    def retrieve(self, question: str, k: int) -> list[RetrievedChunk]:
        scores = self._scores(question)
        ranked = sorted(range(len(self.docs)), key=lambda i: (-scores[i], i))[:k]
        return [
            RetrievedChunk(
                chunk_id=self.docs[i]["chunk_id"],
                text=self.docs[i]["text"],
                score=float(scores[i]),
                metadata={"title": self.docs[i]["title"]},
            )
            for i in ranked
        ]

    def answer(self, question: str, k: int) -> SystemOutput:
        context = self.retrieve(question, k)
        passages = "\n\n".join(f"[{c.chunk_id}] {c.text}" for c in context)
        text = self.generate(ANSWER_PROMPT.format(passages=passages, question=question)).strip()
        citations = list(dict.fromkeys(CITATION.findall(text)))
        return SystemOutput(answer=text, citations=citations, context=context)


class BM25System(_HandbookSystem):
    name = "bm25"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        from rank_bm25 import BM25Okapi

        self.bm25 = BM25Okapi([tokenize(d["title"] + " " + d["text"]) for d in self.docs])

    def _scores(self, question: str) -> list[float]:
        return list(self.bm25.get_scores(tokenize(question)))


class DenseSystem(_HandbookSystem):
    name = "dense"

    def __init__(self, model_name: str = DENSE_MODEL, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")
        texts = [d["title"] + ". " + d["text"] for d in self.docs]
        self.embeddings = self.model.encode(texts, normalize_embeddings=True)

    def _scores(self, question: str) -> list[float]:
        query = self.model.encode([question], normalize_embeddings=True)[0]
        return list(self.embeddings @ query)  # cosine similarity (unit vectors)


def make_bm25(**kwargs: Any) -> BM25System:
    return BM25System(**kwargs)


def make_dense(**kwargs: Any) -> DenseSystem:
    return DenseSystem(**kwargs)
