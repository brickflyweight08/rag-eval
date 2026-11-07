import json

from typer.testing import CliRunner

from rag_eval.cli import app

runner = CliRunner()


def write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_validate_prints_summary(tmp_path):
    rows = [
        {"id": f"q{i}", "question": "x?", "qtype": "lookup", "gold_ids": ["c1"]} for i in range(3)
    ] + [{"id": "u1", "question": "y?", "qtype": "unanswerable", "answerable": False}]
    result = runner.invoke(app, ["validate", str(write(tmp_path / "q.jsonl", rows))])
    assert result.exit_code == 0, result.output
    assert "4 valid question(s)" in result.output
    assert "answerable: 3   unanswerable: 1" in result.output
    assert "lookup" in result.output
    assert "Warning: qtype 'unanswerable' has only 1" in result.output
    assert "OK" in result.output


def test_validate_reports_problems(tmp_path):
    rows = [{"id": "q1", "question": "x?"}, {"id": "q1", "question": "x?", "gold_ids": ["a"]}]
    result = runner.invoke(app, ["validate", str(write(tmp_path / "q.jsonl", rows))])
    assert result.exit_code == 1
    assert "line 1: answerable question 'q1' has no gold_ids" in result.output
    assert "line 2: duplicate id 'q1'" in result.output


def test_validate_missing_file(tmp_path):
    result = runner.invoke(app, ["validate", str(tmp_path / "nope.jsonl")])
    assert result.exit_code == 1
    assert "file not found" in result.output
