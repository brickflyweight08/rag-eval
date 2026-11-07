"""rag-eval: a small, general-purpose toolkit for evaluating RAG systems.

Public API (keep stable): EvalQuestion, RetrievedChunk, SystemOutput, Prediction,
RAGSystem, run_live, score_offline, load_questions. Everything else is internal.
"""

from rag_eval.loading import load_questions
from rag_eval.protocols import RAGSystem
from rag_eval.runner import run_live, score_offline
from rag_eval.schema import EvalQuestion, Prediction, RetrievedChunk, SystemOutput
from rag_eval.version import __version__

__all__ = [
    "EvalQuestion",
    "Prediction",
    "RAGSystem",
    "RetrievedChunk",
    "SystemOutput",
    "__version__",
    "load_questions",
    "run_live",
    "score_offline",
]
