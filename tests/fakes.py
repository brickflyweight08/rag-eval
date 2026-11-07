"""Test doubles: FakeRAGSystem (fixed outputs) and FakeJudgeClient (scripted JSON)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from rag_eval.schema import RetrievedChunk, SystemOutput

CHUNKS = {
    "c1": "Staff get 25 days of paid leave per year.",
    "c2": "Unused leave can be carried over up to 5 days.",
    "c3": "Laptops are replaced every three years.",
    "c4": "Travel must be booked through the internal portal.",
}


class FakeRAGSystem:
    """Returns fixed chunks and answers, looked up by question text.

    - ``retrievals``: question -> ranked chunk ids returned by ``retrieve``.
    - ``answers``: question -> (answer text, cited ids, context ids).
    - ``fail_on``: questions for which ``answer`` raises RuntimeError.
    - ``retrieval_only``: ``answer`` raises NotImplementedError.
    """

    def __init__(
        self,
        name: str = "fake",
        retrievals: dict[str, list[str]] | None = None,
        answers: dict[str, tuple[str, list[str], list[str]]] | None = None,
        fail_on: set[str] | None = None,
        retrieval_only: bool = False,
        chunks: dict[str, str] | None = None,
    ) -> None:
        self.name = name
        self.retrievals = retrievals or {}
        self.answers = answers or {}
        self.fail_on = fail_on or set()
        self.retrieval_only = retrieval_only
        self.chunks = chunks or CHUNKS
        self.answer_calls = 0

    def _chunk(self, chunk_id: str) -> RetrievedChunk:
        return RetrievedChunk(chunk_id=chunk_id, text=self.chunks.get(chunk_id, "?"))

    def retrieve(self, question: str, k: int) -> list[RetrievedChunk]:
        return [self._chunk(cid) for cid in self.retrievals.get(question, [])][:k]

    def answer(self, question: str, k: int) -> SystemOutput:
        self.answer_calls += 1
        if self.retrieval_only:
            raise NotImplementedError("retrieval-only system")
        if question in self.fail_on:
            raise RuntimeError("generator exploded")
        text, cited, context_ids = self.answers[question]
        return SystemOutput(
            answer=text, citations=cited, context=[self._chunk(c) for c in context_ids]
        )


Response = dict | list | Exception | Callable[[dict[str, Any]], Any]


class FakeJudgeClient:
    """Scripted judge. ``responses`` maps prompt name to one of:

    - a dict (returned every time),
    - a list (one item popped per call),
    - a callable taking the template variables,
    - an Exception (raised).
    """

    def __init__(self, responses: dict[str, Response], model: str = "fake-judge") -> None:
        self.responses = responses
        self.model = model
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def complete_json(
        self,
        prompt_name: str,
        variables: dict[str, Any],
        schema: type[BaseModel] | None = None,
    ) -> dict:
        self.calls.append((prompt_name, dict(variables)))
        response = self.responses[prompt_name]
        if isinstance(response, list):
            response = response.pop(0)
        if callable(response) and not isinstance(response, Exception):
            response = response(variables)
        if isinstance(response, Exception):
            raise response
        if schema is not None:
            schema.model_validate(response)
        return response

    def count(self, prompt_name: str) -> int:
        return sum(1 for name, _ in self.calls if name == prompt_name)
