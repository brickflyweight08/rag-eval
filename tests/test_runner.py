import json
import logging
import re

import pytest
from typer.testing import CliRunner

from fakes import FakeJudgeClient, FakeRAGSystem
from rag_eval import run_live, score_offline
from rag_eval.aggregate import summarize
from rag_eval.cli import app
from rag_eval.llm import JudgeCache, JudgeClient, JudgeConfigError
from rag_eval.schema import EvalQuestion, QuestionResult

QUESTIONS = [
    EvalQuestion(id="q1", question="How much leave?", qtype="lookup", gold_ids=["c1"]),
    EvalQuestion(id="q2", question="Carry over?", qtype="lookup", gold_ids=["c2", "c1"]),
    EvalQuestion(id="q3", question="Laptops?", qtype="equipment", gold_ids=["c3"]),
    EvalQuestion(id="q4", question="Pets at work?", qtype="unanswerable", answerable=False),
]

RETRIEVALS = {
    "How much leave?": ["c1", "c2"],
    "Carry over?": ["c2", "c4"],
    "Laptops?": ["c4", "c1"],
    "Pets at work?": ["c4"],
}
ANSWERS = {
    "How much leave?": ("25 days [c1].", ["c1"], ["c1", "c2"]),
    "Carry over?": ("Up to 5 days [c2] [c9].", ["c2", "c9"], ["c2", "c4"]),
    "Pets at work?": ("I don't know.", [], ["c4"]),
}
CLAIMS = {
    "25 days [c1].": ["Staff get 25 days."],
    "Up to 5 days [c2] [c9].": ["5 days carry over.", "Carry-over expires in March."],
}
VERDICTS = {
    "Staff get 25 days.": {"supported": True, "evidence_id": "c1"},
    "5 days carry over.": {"supported": True, "evidence_id": "c2"},
    "Carry-over expires in March.": {"supported": False, "evidence_id": None},
}


def fake_system(**kw):
    return FakeRAGSystem(retrievals=RETRIEVALS, answers=ANSWERS, fail_on={"Laptops?"}, **kw)


def fake_judge():
    return FakeJudgeClient(
        {
            "extract_claims": lambda v: {"claims": CLAIMS[v["answer"]]},
            "verify_claim": lambda v: VERDICTS[v["claim"]],
        }
    )


def by_id(artifacts):
    return {r.question_id: r for r in artifacts.results}


def test_live_run_scores_and_survives_a_failing_question(tmp_path):
    out = tmp_path / "run"
    artifacts = run_live(fake_system(), QUESTIONS, k=2, out_dir=out, judge=fake_judge())
    r = by_id(artifacts)
    assert (r["q1"].hit, r["q1"].recall, r["q1"].reciprocal_rank) == (True, 1.0, 1.0)
    assert r["q1"].groundedness == 1.0 and r["q1"].citation_precision == 1.0
    assert r["q2"].recall == 0.5
    assert r["q2"].groundedness == 0.5
    assert r["q2"].hallucinated_citations == ["c9"]
    assert r["q2"].citation_precision == 1.0
    # q3: retrieval scored, answer() raised -> error recorded, run continued
    assert r["q3"].hit is False and r["q3"].answer is None
    assert r["q3"].error == "answer: RuntimeError: generator exploded"
    # q4: unanswerable + refused
    assert r["q4"].hit is None and r["q4"].refused is True and r["q4"].groundedness is None
    assert [x.question_id for x in artifacts.results] == ["q1", "q2", "q3", "q4"]

    assert {p.name for p in out.iterdir()} == {"config.json", "results.jsonl", "summary.json"}
    config = json.loads((out / "config.json").read_text())
    assert config["run_id"] == "run" and config["k"] == 2
    assert config["judge_model"] == "fake-judge"
    assert set(config["prompt_versions"]) == {"extract_claims", "verify_claim"}
    assert config["questions_path"] == "<in-memory>" and len(config["questions_sha256"]) == 64
    summary = json.loads((out / "summary.json").read_text())
    assert summary["n_errors"] == 1
    assert summary["overall"]["hit"]["mean"] == pytest.approx(2 / 3)
    assert summary["overall"]["refusal_accuracy"]["mean"] == 1.0
    assert summary["per_qtype"]["lookup"]["metrics"]["recall"]["mean"] == 0.75
    lines = (out / "results.jsonl").read_text().splitlines()
    assert [QuestionResult.model_validate_json(x).question_id for x in lines][0] == "q1"


def test_progress_is_logged_per_question(caplog):
    with caplog.at_level(logging.INFO, logger="rag_eval"):
        run_live(fake_system(), QUESTIONS, k=2, judge=fake_judge())
    assert "[1/4] q1" in caplog.text and "[4/4] q4" in caplog.text
    assert "ERROR answer: RuntimeError" in caplog.text


def test_retrieval_only_flag_never_calls_answer_or_judge():
    system, judge = fake_system(), fake_judge()
    artifacts = run_live(system, QUESTIONS, k=2, judge=judge, retrieval_only=True)
    assert system.answer_calls == 0 and judge.calls == []
    assert artifacts.config.judge_model is None and artifacts.config.prompt_versions == {}
    assert all(r.answer is None and r.refused is None for r in artifacts.results)
    assert all(r.latency_ms is not None for r in artifacts.results)


def test_not_implemented_answer_makes_run_retrieval_only():
    system = fake_system(retrieval_only=True)
    artifacts = run_live(system, QUESTIONS, k=2, judge=fake_judge())
    assert system.answer_calls == 1  # stopped asking after the first NotImplementedError
    assert all(r.error is None for r in artifacts.results)
    assert artifacts.summary["overall"]["groundedness"]["n"] == 0


def test_retrieve_exception_is_recorded():
    class Broken(FakeRAGSystem):
        def retrieve(self, question, k):
            raise ConnectionError("db down")

    artifacts = run_live(Broken(), QUESTIONS[:1], k=2, retrieval_only=True)
    (r,) = artifacts.results
    assert r.error == "retrieve: ConnectionError: db down" and r.hit is None


def test_judge_config_error_aborts_the_run():
    judge = FakeJudgeClient({"extract_claims": JudgeConfigError("no key")})
    with pytest.raises(JudgeConfigError):
        run_live(fake_system(), QUESTIONS, k=2, judge=judge)


def test_load_system_from_string_and_questions_file(tmp_path):
    qpath = tmp_path / "q.jsonl"
    qpath.write_text("".join(q.model_dump_json() + "\n" for q in QUESTIONS), encoding="utf-8")
    artifacts = run_live(
        "fake_plugins:make_good",
        qpath,
        k=2,
        system_kwargs={"name": "plug", "retrievals": RETRIEVALS},
        retrieval_only=True,
    )
    assert artifacts.config.system == "fake_plugins:make_good"
    assert artifacts.config.system_kwargs["name"] == "plug"
    assert artifacts.config.questions_path == str(qpath)
    assert artifacts.config.run_id == "plug-k2"


def _deterministic_completion(calls):
    def complete(prompt):
        calls.append(prompt)
        if "atomic factual claims" in prompt:
            return json.dumps({"claims": ["Claim one.", "Claim two."]})
        return json.dumps({"supported": "Claim one." in prompt, "evidence_id": "c2"})

    return complete


def _blank_latency(path):
    """results.jsonl bytes with latency values blanked (timing is never reproducible)."""
    return re.sub(rb'"latency_ms":[^,}]*', b'"latency_ms":null', path.read_bytes())


def test_second_run_is_identical_and_makes_zero_judge_calls(tmp_path):
    calls = []
    cache_file = tmp_path / "cache.sqlite"
    judges = []
    for name in ("a", "b"):
        judge = JudgeClient(
            "m", cache=JudgeCache(cache_file), completion_fn=_deterministic_completion(calls)
        )
        run_live(fake_system(), QUESTIONS, k=2, out_dir=tmp_path / name, judge=judge)
        judges.append(judge)
    assert judges[0].n_network_calls > 0
    assert judges[1].n_network_calls == 0
    a = _blank_latency(tmp_path / "a" / "results.jsonl")
    assert a == _blank_latency(tmp_path / "b" / "results.jsonl")
    assert b"Claim one." in a


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_score_offline(tmp_path, caplog):
    chunk = lambda cid: {"chunk_id": cid, "text": "t"}  # noqa: E731
    preds = [
        {"question_id": "q1", "retrieved": [chunk("c1"), chunk("c1"), chunk("c2")]},
        {
            "question_id": "q2",
            "retrieved": [chunk("c2")],
            "output": {
                "answer": "Up to 5 days [c2] [c9].",
                "citations": ["c2", "c9"],
                "context": [chunk("c2")],
                "latency_ms": 12.5,
            },
        },
        {"question_id": "zzz", "retrieved": []},
    ]
    ppath = write_jsonl(tmp_path / "p.jsonl", preds)
    with caplog.at_level(logging.WARNING):
        artifacts = score_offline(ppath, QUESTIONS[:3], k=2, judge=fake_judge())
    r = by_id(artifacts)
    assert r["q1"].retrieved_ids == ["c1", "c2"] and r["q1"].answer is None
    assert r["q2"].groundedness == 0.5 and r["q2"].latency_ms == 12.5
    assert r["q3"].error == "no prediction for this question" and r["q3"].hit is None
    assert artifacts.config.system == f"offline:{ppath}"
    assert "unknown question IDs: ['zzz']" in caplog.text


def test_summarize_hand_computed():
    def res(qid, qtype, hit, recall, latency, answerable=True, refused=None):
        return QuestionResult(
            question_id=qid,
            qtype=qtype,
            answerable=answerable,
            retrieved_ids=[],
            hit=hit,
            recall=recall,
            precision=None,
            reciprocal_rank=None,
            answer=None,
            refused=refused,
            latency_ms=latency,
        )

    results = [
        res("a", "x", True, 1.0, 10.0),
        res("b", "x", False, 0.0, 20.0, refused=True),
        res("c", "y", True, 0.5, None),
        res("d", "y", None, None, 40.0, answerable=False, refused=False),
    ]
    s = summarize(results, k=3, n_resamples=200)
    assert s["n_questions"] == 4 and s["n_unanswerable"] == 1
    assert s["overall"]["hit"]["mean"] == pytest.approx(2 / 3) and s["overall"]["hit"]["n"] == 3
    assert s["overall"]["recall"]["mean"] == pytest.approx(0.5)
    assert s["overall"]["false_refusal_rate"] == {"mean": 1.0, "low": 1.0, "high": 1.0, "n": 1}
    assert s["overall"]["refusal_accuracy"]["mean"] == 0.0
    assert s["overall"]["groundedness"] == {"mean": None, "low": None, "high": None, "n": 0}
    assert s["latency_ms"] == {"p50": 20.0, "p95": pytest.approx(38.0), "n": 3}
    assert s["per_qtype"]["y"]["n_questions"] == 2
    assert s["per_qtype"]["y"]["metrics"]["hit"]["n"] == 1


cli = CliRunner()


def test_cli_run_retrieval_only(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    qpath = tmp_path / "q.jsonl"
    qpath.write_text("".join(q.model_dump_json() + "\n" for q in QUESTIONS), encoding="utf-8")
    kwargs = json.dumps({"retrievals": RETRIEVALS})
    args = ["run", "--system", "fake_plugins:make_good", "--system-kwargs", kwargs]
    args += [
        "--questions",
        str(qpath),
        "--k",
        "2",
        "--out",
        str(tmp_path / "r"),
        "--retrieval-only",
    ]
    result = cli.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "Hit@2" in result.output
    assert (tmp_path / "r" / "summary.json").is_file()
    assert not (tmp_path / ".rag_eval_cache.sqlite").exists()


def test_cli_run_bad_kwargs_and_bad_system(tmp_path):
    qpath = tmp_path / "q.jsonl"
    qpath.write_text(QUESTIONS[0].model_dump_json() + "\n", encoding="utf-8")
    base = ["run", "--questions", str(qpath), "--out", str(tmp_path / "r")]
    bad_json = cli.invoke(
        app, [*base, "--system", "fake_plugins:make_good", "--system-kwargs", "{"]
    )
    assert bad_json.exit_code == 1 and "not valid JSON" in bad_json.output
    bad_sys = cli.invoke(app, [*base, "--system", "fake_plugins:nope"])
    assert bad_sys.exit_code == 1 and "has no attribute 'nope'" in bad_sys.output


def test_cli_score_without_outputs_needs_no_judge(tmp_path):
    qpath = tmp_path / "q.jsonl"
    qpath.write_text(QUESTIONS[0].model_dump_json() + "\n", encoding="utf-8")
    ppath = write_jsonl(
        tmp_path / "p.jsonl", [{"question_id": "q1", "retrieved": [{"chunk_id": "c1", "text": ""}]}]
    )
    args = ["score", "--predictions", str(ppath), "--questions", str(qpath)]
    result = cli.invoke(app, [*args, "--out", str(tmp_path / "o")])
    assert result.exit_code == 0, result.output
    assert "MRR" in result.output
