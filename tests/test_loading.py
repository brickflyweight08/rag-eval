import json
import logging

import pytest

from rag_eval.loading import (
    DatasetError,
    SystemLoadError,
    load_predictions,
    load_questions,
    load_system,
    validate_questions_file,
)


def write_jsonl(path, rows):
    lines = [r if isinstance(r, str) else json.dumps(r) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def q(id_, **kw):
    row = {"id": id_, "question": f"question {id_}?", "qtype": "lookup", "gold_ids": ["c1"]}
    row.update(kw)
    return row


def test_load_valid_questions(tmp_path):
    path = write_jsonl(tmp_path / "q.jsonl", [q("q1"), q("q2"), q("q3")])
    questions = load_questions(path)
    assert [x.id for x in questions] == ["q1", "q2", "q3"]
    assert questions[0].metadata == {}


def test_reports_all_errors_with_line_numbers(tmp_path):
    rows = [
        q("q1"),
        "{not json",
        {"id": "q3"},  # missing question
        q("q1"),  # duplicate id
        q("q5", gold_ids=[]),  # answerable without gold
        q("q6", answerable=False),  # unanswerable with gold
        q("q7", question="   "),  # empty text
        "",  # blank lines are ignored
    ]
    path = write_jsonl(tmp_path / "q.jsonl", rows)
    with pytest.raises(DatasetError) as info:
        load_questions(path)
    problems = info.value.problems
    assert any(p.startswith("line 2: invalid JSON") for p in problems)
    assert any(p.startswith("line 3: question") for p in problems)
    assert "line 4: duplicate id 'q1' (first seen on line 1)" in problems
    assert "line 5: answerable question 'q5' has no gold_ids" in problems
    assert "line 6: unanswerable question 'q6' must have no gold_ids" in problems
    assert "line 7: question text is empty (id 'q7')" in problems
    assert len(problems) == 6
    assert "6 problem(s)" in str(info.value)


def test_small_qtype_is_a_warning_not_an_error(tmp_path, caplog):
    rows = [q("q1"), q("q2"), q("q3"), q("q4", qtype="rare")]
    path = write_jsonl(tmp_path / "q.jsonl", rows)
    report = validate_questions_file(path)
    assert report.ok
    assert report.warnings == [
        "qtype 'rare' has only 1 question(s); per-type numbers will be very noisy"
    ]
    with caplog.at_level(logging.WARNING):
        assert len(load_questions(path)) == 4
    assert "rare" in caplog.text


def test_load_predictions_rejects_duplicates(tmp_path):
    pred = {"question_id": "q1", "retrieved": [{"chunk_id": "c1", "text": "t"}]}
    path = write_jsonl(tmp_path / "p.jsonl", [pred, pred])
    with pytest.raises(DatasetError, match="line 2: duplicate prediction for 'q1'"):
        load_predictions(path)


def test_load_predictions_ok(tmp_path):
    pred = {
        "question_id": "q1",
        "retrieved": [{"chunk_id": "c1", "text": "t", "score": 0.5}],
        "output": {
            "answer": "a",
            "citations": ["c1"],
            "context": [{"chunk_id": "c1", "text": "t"}],
        },
    }
    path = write_jsonl(tmp_path / "p.jsonl", [pred])
    (p,) = load_predictions(path)
    assert p.output is not None and p.output.citations == ["c1"]


def test_load_system_ok_passes_kwargs():
    system = load_system("fake_plugins:make_good", {"name": "custom"})
    assert system.name == "custom"


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("no_colon_here", "expected 'module.path:factory'"),
        ("does_not_exist_mod:make", "could not import module 'does_not_exist_mod'"),
        ("fake_plugins:nope", "has no attribute 'nope'"),
        ("fake_plugins:NOT_A_FACTORY", "is not callable"),
        ("fake_plugins:make_raises", "raised ValueError('bad configuration')"),
        ("fake_plugins:make_missing_answer", "missing method `answer(question, k)`"),
    ],
)
def test_load_system_helpful_errors(spec, message):
    with pytest.raises(SystemLoadError, match=None) as info:
        load_system(spec)
    assert message in str(info.value)
