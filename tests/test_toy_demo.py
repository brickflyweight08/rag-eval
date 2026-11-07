"""The toy handbook demo: data integrity and the BM25 system (dense is opt-in: slow)."""

import json
import os
import re
from pathlib import Path

import pytest

from fakes import FakeJudgeClient
from rag_eval import load_questions, run_live
from rag_eval.calibrate import load_labels

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "toy_handbook"
SPEC_BM25 = "examples.toy_handbook.systems:make_bm25"


def corpus():
    lines = (DEMO / "corpus.jsonl").read_text(encoding="utf-8").splitlines()
    return {d["chunk_id"]: d for d in map(json.loads, lines)}


def test_src_has_no_domain_coupling():
    forbidden = re.compile(r"arango|mysql|restaurant", re.IGNORECASE)
    for path in (ROOT / "src").rglob("*"):
        if path.is_file() and path.suffix in {".py", ".txt", ".md", ".toml"}:
            assert not forbidden.search(path.read_text(encoding="utf-8")), path


def test_corpus_and_questions_are_consistent():
    docs = corpus()
    assert 25 <= len(docs) <= 35
    questions = load_questions(DEMO / "questions.jsonl")
    assert len(questions) == 15
    assert sum(not q.answerable for q in questions) == 3
    for q in questions:
        assert set(q.gold_ids) <= set(docs), q.id


def test_labels_are_marked_as_claude_drafts_and_match_corpus():
    docs = corpus()
    labels = load_labels(DEMO / "human_labels.jsonl")
    assert len(labels) >= 10
    for raw in (DEMO / "human_labels.jsonl").read_text(encoding="utf-8").splitlines():
        assert "claude-draft" in json.loads(raw)["labelled_by"]
    for label in labels:
        for chunk in label.context:
            assert chunk.text == docs[chunk.chunk_id]["text"]


@pytest.fixture
def in_repo_root(monkeypatch):
    monkeypatch.chdir(ROOT)


def test_bm25_retrieves_the_obvious_chunk(in_repo_root):
    from examples.toy_handbook.systems import make_bm25

    system = make_bm25(generator=lambda prompt: "unused")
    top = system.retrieve("How long must a password for a company account be?", 3)
    assert top[0].chunk_id == "security-01"
    assert len(top) == 3 and top[0].score >= top[1].score


def test_bm25_answer_parses_citations(in_repo_root):
    from examples.toy_handbook.systems import make_bm25

    system = make_bm25(generator=lambda p: "27 days [leave-01], see also [leave-01] [nope-9].")
    output = system.answer("How many days of annual leave?", 2)
    assert output.citations == ["leave-01", "nope-9"]
    assert len(output.context) == 2


def test_bm25_without_key_records_clear_errors(in_repo_root, monkeypatch):
    for var in ("RAG_EVAL_DEMO_API_KEY", "RAG_EVAL_JUDGE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    artifacts = run_live(SPEC_BM25, DEMO / "questions.jsonl", k=3, judge=FakeJudgeClient({}))
    assert all("no API key" in r.error for r in artifacts.results)
    assert artifacts.summary["overall"]["hit"]["n"] == 12  # retrieval still scored


def test_bm25_end_to_end_with_fake_generator_and_judge(in_repo_root, tmp_path):
    judge = FakeJudgeClient(
        {
            "extract_claims": {"claims": ["A claim."]},
            "verify_claim": {"supported": True, "evidence_id": None},
        }
    )
    artifacts = run_live(
        SPEC_BM25,
        DEMO / "questions.jsonl",
        k=3,
        system_kwargs={"generator": lambda p: "I don't know." if "dog" in p else "Yes [leave-01]."},
        judge=judge,
        out_dir=tmp_path / "run",
    )
    assert "<function" in artifacts.config.system_kwargs["generator"]
    refused = {r.question_id: r.refused for r in artifacts.results}
    assert refused["q13"] is True and refused["q01"] is False
    assert artifacts.summary["overall"]["groundedness"]["mean"] == 1.0


@pytest.mark.skipif(
    not os.environ.get("RAG_EVAL_TEST_DENSE"),
    reason="loads a sentence-transformers model (slow); set RAG_EVAL_TEST_DENSE=1",
)
def test_dense_retrieves_a_paraphrase(in_repo_root):
    from examples.toy_handbook.systems import make_dense

    system = make_dense(generator=lambda p: "")
    top = system.retrieve("Is there any reward for pedalling to the office?", 3)
    assert "benefits-04" in [c.chunk_id for c in top]
