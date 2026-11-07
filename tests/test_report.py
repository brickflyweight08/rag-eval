import re

import pytest
from typer.testing import CliRunner

from fakes import FakeRAGSystem
from rag_eval import run_live
from rag_eval.aggregate import METRICS_BY_KEY
from rag_eval.cli import app
from rag_eval.report import (
    CompareError,
    compare_runs,
    describe_difference,
    expand_run_paths,
    fmt_stat,
    load_run,
    md_table,
    render_compare,
    render_report,
    worst_questions,
    write_report,
)
from test_runner import ANSWERS, QUESTIONS, RETRIEVALS, fake_judge

BETTER_RETRIEVALS = {**RETRIEVALS, "Carry over?": ["c2", "c1"], "Laptops?": ["c3", "c1"]}


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    qpath = tmp_path / "questions.jsonl"
    qpath.write_text("".join(q.model_dump_json() + "\n" for q in QUESTIONS), encoding="utf-8")
    for name, retrievals in (("base", RETRIEVALS), ("better", BETTER_RETRIEVALS)):
        system = FakeRAGSystem(retrievals=retrievals, answers=ANSWERS, fail_on={"Laptops?"})
        run_live(
            system, "questions.jsonl", k=2, out_dir=tmp_path / "runs" / name, judge=fake_judge()
        )
    return tmp_path


def _check_tables(markdown):
    """Every row of a markdown table must have as many cells as its header."""
    width = None
    for line in markdown.splitlines():
        if not line.startswith("|"):
            width = None
            continue
        cells = len(re.findall(r"(?<!\\)\|", line)) - 1
        width = width or cells
        assert cells == width, line


def test_fmt_stat():
    assert (
        fmt_stat({"mean": 0.8, "low": 0.6, "high": 1.0, "n": 12}) == "0.800 [0.600, 1.000] (n=12)"
    )
    assert fmt_stat({"mean": None, "low": None, "high": None, "n": 0}) == "–"


def test_md_table_escapes_pipes():
    table = md_table(["a", "b"], [["x|y", 1]])
    assert "x\\|y" in table
    _check_tables(table)


def test_report_contents(runs):
    loaded = [load_run(p) for p in expand_run_paths([str(runs / "runs" / "*")])]
    assert [r.name for r in loaded] == ["base", "better"]
    out = write_report(loaded, runs / "REPORT.md")
    text = out.read_text(encoding="utf-8")
    _check_tables(text)
    assert "# rag-eval report" in text
    assert "fake-judge" in text and loaded[0].config.questions_sha256[:12] in text
    assert "| Run | Hit@2 | Recall@2 | MRR | Groundedness |" in text
    assert "Correctness" not in text.split("## Diagnostics")[0]  # not run -> no column
    assert "### Hit@2 by qtype" in text
    assert "## Failure log" in text and "Missed gold" in text
    assert "answer: RuntimeError: generator exploded" in text
    assert (runs / "REPORT_hit_by_run.png").is_file()
    assert (runs / "REPORT_groundedness_by_qtype.png").is_file()
    assert "![Hit@2 by run](REPORT_hit_by_run.png)" in text


def test_retrieval_only_report_has_no_groundedness_chart(tmp_path):
    system = FakeRAGSystem(retrievals=RETRIEVALS)
    run_live(system, QUESTIONS, k=2, out_dir=tmp_path / "r", retrieval_only=True)
    out = write_report([load_run(tmp_path / "r")], tmp_path / "R.md")
    text = out.read_text(encoding="utf-8")
    assert "retrieval-only" in text
    assert not (tmp_path / "R_groundedness_by_qtype.png").exists()
    assert "(questions file not found)" in text  # in-memory questions: no gold IDs


def test_worst_questions_order(runs):
    run = load_run(runs / "runs" / "base")
    worst = worst_questions(run.results)
    assert [r.question_id for r in worst] == ["q3", "q2", "q1"]  # recall 0, 0.5, 1


def test_describe_difference_wording():
    spec = METRICS_BY_KEY["hit"]
    clear = {"diff": 0.125, "low": 0.025, "high": 0.225, "n": 40}
    line = describe_difference("Hit@5", clear, spec)
    assert line.startswith("Hit@5 +12.5 pts [95% CI +2.5, +22.5] (n=40)")
    assert "B is clearly higher" in line
    unclear = {"diff": 0.05, "low": -0.05, "high": 0.15, "n": 40}
    assert "not clear at this sample size" in describe_difference("Hit@5", unclear, spec)
    lower = {"diff": -0.2, "low": -0.3, "high": -0.1, "n": 10}
    text = describe_difference("FRR", lower, METRICS_BY_KEY["false_refusal_rate"])
    assert "clearly lower" in text and "lower is better" in text


def test_compare_runs(runs):
    a, b = load_run(runs / "runs" / "base"), load_run(runs / "runs" / "better")
    result = compare_runs(a, b)
    recall = result["overall"]["recall"]
    # recall per question: base (1, .5, 0) -> better (1, 1, 1)
    assert recall["diff"] == pytest.approx(0.5) and recall["n"] == 3
    text = render_compare(a, b, result)
    assert "Recall@2 +50.0 pts" in text
    assert "## qtype `lookup`" in text


def test_compare_refuses_different_question_files(runs):
    a, b = load_run(runs / "runs" / "base"), load_run(runs / "runs" / "better")
    b.config.questions_sha256 = "0" * 64
    with pytest.raises(CompareError, match="different question files"):
        compare_runs(a, b)


def test_cli_report_and_compare(runs):
    cli = CliRunner()
    result = cli.invoke(app, ["report", "runs/*", "--out", "out/REPORT.md"])
    assert result.exit_code == 0, result.output
    assert (runs / "out" / "REPORT.md").is_file()
    result = cli.invoke(app, ["compare", "runs/base", "runs/better", "--out", "C.md"])
    assert result.exit_code == 0, result.output
    assert "Recall@2 +50.0 pts" in result.output and (runs / "C.md").is_file()
    result = cli.invoke(app, ["compare", "runs/base", "nowhere"])
    assert result.exit_code == 1 and "not a run folder" in result.output


def test_cli_compare_mismatch_errors_clearly(runs):
    config = runs / "runs" / "better" / "config.json"
    text = config.read_text(encoding="utf-8")
    sha = load_run(runs / "runs" / "better").config.questions_sha256
    config.write_text(text.replace(sha, "f" * 64), encoding="utf-8")
    result = CliRunner().invoke(app, ["compare", "runs/base", "runs/better"])
    assert result.exit_code == 1
    assert "different question files" in result.output


def test_render_report_requires_runs():
    with pytest.raises(ValueError):
        render_report([])
