"""The plugin protocol any RAG system implements to be evaluated live."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rag_eval.schema import RetrievedChunk, SystemOutput


@runtime_checkable
class RAGSystem(Protocol):
    """A system under test.

    A system may be retrieval-only: if ``answer`` raises ``NotImplementedError`` the
    runner skips answer metrics for the whole run.
    """

    name: str

    def retrieve(self, question: str, k: int) -> list[RetrievedChunk]: ...

    def answer(self, question: str, k: int) -> SystemOutput: ...
