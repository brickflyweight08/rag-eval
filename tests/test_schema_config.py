import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

import rag_eval
from rag_eval.config import load_settings
from rag_eval.schema import EvalQuestion, QuestionResult, SystemOutput

ROOT = Path(__file__).resolve().parents[1]


def test_public_exports():
    expected = {
        "EvalQuestion",
        "RetrievedChunk",
        "SystemOutput",
        "Prediction",
        "RAGSystem",
        "load_questions",
        "run_live",
        "score_offline",
    }
    assert expected <= set(rag_eval.__all__)
    for name in expected:
        assert hasattr(rag_eval, name)


def test_version_matches_pyproject():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["version"] == rag_eval.__version__


def test_question_defaults():
    q = EvalQuestion(id="q1", question="What?")
    assert q.qtype == "default" and q.answerable and q.gold_ids == [] and q.metadata == {}


def test_system_output_requires_context():
    with pytest.raises(ValidationError):
        SystemOutput(answer="hi")


def test_correctness_literal():
    base = dict(
        question_id="q",
        qtype="t",
        answerable=True,
        retrieved_ids=[],
        hit=None,
        recall=None,
        precision=None,
        reciprocal_rank=None,
        answer=None,
    )
    assert QuestionResult(**base, correctness="partial").correctness == "partial"
    with pytest.raises(ValidationError):
        QuestionResult(**base, correctness="maybe")


def test_settings_precedence(monkeypatch):
    monkeypatch.setenv("RAG_EVAL_JUDGE_MODEL", "env-model")
    monkeypatch.delenv("RAG_EVAL_JUDGE_API_KEY", raising=False)
    monkeypatch.delenv("RAG_EVAL_JUDGE_BASE_URL", raising=False)
    settings = load_settings()
    assert settings.model == "env-model"
    assert settings.api_key is None
    assert settings.base_url == "http://localhost:11434/v1"
    assert load_settings(model="flag-model").model == "flag-model"
