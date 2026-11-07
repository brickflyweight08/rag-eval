import json

import pytest
from typer.testing import CliRunner

from fakes import FakeJudgeClient
from rag_eval.calibrate import calibrate, load_labels, render_calibration
from rag_eval.cli import app
from rag_eval.llm import JudgeError
from rag_eval.loading import DatasetError

HUMAN = [True, True, False, False, True, False, True, True, False, True]
JUDGE = [True, False, False, False, True, False, True, True, True, True]


def labels_file(tmp_path, human=HUMAN):
    rows = [
        {
            "question_id": f"q{i}",
            "claim": f"claim {i}",
            "context": [{"chunk_id": "c1", "text": "some text"}],
            "human_supported": h,
            "labelled_by": "test",
        }
        for i, h in enumerate(human)
    ]
    path = tmp_path / "labels.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def scripted_judge(verdicts=JUDGE):
    table = {f"claim {i}": v for i, v in enumerate(verdicts)}
    return FakeJudgeClient(
        {
            "verify_claim": lambda v: {
                "supported": table[v["claim"]],
                "evidence_id": "c1" if table[v["claim"]] else None,
            }
        }
    )


def test_kappa_matches_hand_computed_value(tmp_path):
    result = calibrate(load_labels(labels_file(tmp_path)), scripted_judge())
    # p_o = 0.8; p_e = 0.6*0.6 + 0.4*0.4 = 0.52; kappa = (0.8-0.52)/(1-0.52)
    assert result.n == 10
    assert result.agreement == pytest.approx(0.8)
    assert result.kappa == pytest.approx(0.28 / 0.48)
    assert [d["question_id"] for d in result.disagreements] == ["q1", "q8"]
    assert result.disagreements[0]["human_supported"] is True
    assert not result.trustworthy  # kappa 0.583 < 0.6
    text = render_calibration(result)
    assert "Agreement: 80.0%" in text and "Cohen's kappa: 0.583" in text
    assert "should not be trusted yet" in text
    assert "[q8] human=False judge=True" in text


def test_perfect_agreement_is_trustworthy(tmp_path):
    result = calibrate(load_labels(labels_file(tmp_path)), scripted_judge(HUMAN))
    assert result.agreement == 1.0 and result.kappa == pytest.approx(1.0)
    assert result.trustworthy and "agrees well" in render_calibration(result)


def test_judge_errors_are_reported_and_excluded(tmp_path):
    judge = scripted_judge()
    inner = judge.responses["verify_claim"]
    judge.responses["verify_claim"] = lambda v: (
        JudgeError("bad JSON") if v["claim"] == "claim 0" else inner(v)
    )

    result = calibrate(load_labels(labels_file(tmp_path)), judge)
    assert result.n == 9 and len(result.errors) == 1
    assert "judge error" in render_calibration(result)


def test_bad_labels_file(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"question_id": "q1", "claim": "x"}\n', encoding="utf-8")
    with pytest.raises(DatasetError, match="line 1: context"):
        load_labels(path)


def test_cli_calibrate_without_key_fails_clearly(tmp_path, monkeypatch):
    monkeypatch.delenv("RAG_EVAL_JUDGE_API_KEY", raising=False)
    path = labels_file(tmp_path)
    args = ["calibrate", "--labels", str(path), "--no-cache"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 1
    assert "no judge API key configured" in result.output
