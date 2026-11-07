import pytest

from fakes import FakeJudgeClient
from rag_eval.llm import JudgeConfigError, JudgeError
from rag_eval.metrics.answer import AnswerOptions, score_answer
from rag_eval.metrics.citations import (
    citation_precision,
    hallucinated_citations,
    valid_citations,
)
from rag_eval.metrics.correctness import judge_correctness
from rag_eval.metrics.groundedness import (
    format_context,
    groundedness_score,
    score_groundedness,
    verify_claim,
)
from rag_eval.metrics.refusal import (
    false_refusal_rate,
    is_refusal,
    judge_refusal,
    refusal_accuracy,
)
from rag_eval.schema import EvalQuestion, RetrievedChunk, SystemOutput

CONTEXT = [
    RetrievedChunk(chunk_id="c1", text="Staff get 25 days of paid leave."),
    RetrievedChunk(chunk_id="c2", text="Up to 5 unused days carry over."),
    RetrievedChunk(chunk_id="c3", text="Laptops are replaced every three years."),
]

VERDICTS = {
    "Staff get 25 days of leave.": {"supported": True, "evidence_id": "c1"},
    "Five days carry over.": {"supported": True, "evidence_id": "[c2]"},  # brackets stripped
    "Leave doubles after ten years.": {"supported": False, "evidence_id": "c3"},  # ignored
}


def make_judge(claims=None, verdicts=None, **extra):
    responses = {
        "extract_claims": {"claims": list(claims if claims is not None else VERDICTS)},
        "verify_claim": lambda v: (verdicts or VERDICTS)[v["claim"]],
    }
    responses.update(extra)
    return FakeJudgeClient(responses)


def test_format_context():
    assert format_context(CONTEXT[:2]) == (
        "[c1] Staff get 25 days of paid leave.\n\n[c2] Up to 5 unused days carry over."
    )


def test_groundedness_hand_computed():
    judge = make_judge()
    result = score_groundedness(judge, "Leave?", "some answer", CONTEXT)
    assert result.groundedness == pytest.approx(2 / 3)
    assert result.claims == [
        {"claim": "Staff get 25 days of leave.", "supported": True, "evidence_id": "c1"},
        {"claim": "Five days carry over.", "supported": True, "evidence_id": "c2"},
        {"claim": "Leave doubles after ten years.", "supported": False, "evidence_id": None},
    ]
    assert judge.count("extract_claims") == 1 and judge.count("verify_claim") == 3
    # The verifier sees only the context, each chunk prefixed with its ID.
    assert judge.calls[1][1]["context"].startswith("[c1] Staff get")


def test_zero_claims_gives_none():
    assert groundedness_score([]) is None
    result = score_groundedness(make_judge(claims=[]), "q", "Hello!", CONTEXT)
    assert result.groundedness is None and result.claims == []


def test_evidence_id_outside_context_is_dropped():
    judge = make_judge(verdicts={"x": {"supported": True, "evidence_id": "c99"}})
    assert verify_claim(judge, "x", CONTEXT) == {
        "claim": "x",
        "supported": True,
        "evidence_id": None,
    }


def test_empty_context_means_unsupported_without_judge_calls():
    judge = make_judge()
    assert verify_claim(judge, "anything", [])["supported"] is False
    assert judge.calls == []


def test_citations_hand_computed():
    claims = [
        {"claim": "a", "supported": True, "evidence_id": "c1"},
        {"claim": "b", "supported": True, "evidence_id": "c2"},
        {"claim": "c", "supported": False, "evidence_id": None},
    ]
    cited = ["c1", "c3", "c9", "c1"]
    assert hallucinated_citations(cited, CONTEXT) == ["c9"]
    assert valid_citations(cited, CONTEXT) == ["c1", "c3"]
    # valid = {c1, c3}; used as evidence = {c1, c2} -> 1 / 2
    assert citation_precision(cited, CONTEXT, claims) == 0.5


def test_citation_precision_none_cases():
    claims = [{"claim": "a", "supported": True, "evidence_id": "c1"}]
    assert citation_precision([], CONTEXT, claims) is None  # nothing cited
    assert citation_precision(["c9"], CONTEXT, claims) is None  # only hallucinated
    assert citation_precision(["c1"], CONTEXT, []) is None  # no claims to support


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("I don't know.", True),
        ("Sorry, I don’t have enough   information to answer that.", True),
        ("That detail is NOT IN THE CONTEXT provided.", True),
        ("Staff get 25 days of leave [c1].", False),
    ],
)
def test_is_refusal_default_phrases(answer, expected):
    assert is_refusal(answer) is expected


def test_is_refusal_custom_phrases():
    assert is_refusal("No idea, mate.", ["no idea"]) is True
    assert is_refusal("I don't know.", ["no idea"]) is False


def test_judge_refusal():
    judge = FakeJudgeClient({"refusal": {"refused": True}})
    assert judge_refusal(judge, "q", "Hmm, can't say.") is True


def test_refusal_rates_hand_computed():
    refused = [True, False, None, True, False]
    answerable = [False, False, False, True, True]
    assert refusal_accuracy(refused, answerable) == 0.5  # 1 of 2 scored unanswerable
    assert false_refusal_rate(refused, answerable) == 0.5  # 1 of 2 answerable
    assert refusal_accuracy([None], [False]) is None


def test_judge_correctness():
    judge = FakeJudgeClient({"correctness": {"verdict": "partial", "reason": "misses x"}})
    assert judge_correctness(judge, "q", "a", "ref") == ("partial", "misses x")


QUESTION = EvalQuestion(id="q1", question="Leave?", gold_ids=["c1"], reference_answer="25 days")
OUTPUT = SystemOutput(
    answer="25 days [c1] [c3] [c9]", citations=["c1", "c3", "c9"], context=CONTEXT
)


def test_score_answer_end_to_end():
    judge = make_judge(correctness={"verdict": "correct", "reason": "ok"})
    scores = score_answer(judge, QUESTION, OUTPUT, AnswerOptions(correctness=True))
    assert scores.refused is False
    assert scores.groundedness == pytest.approx(2 / 3)
    assert scores.hallucinated_citations == ["c9"]
    assert scores.citation_precision == 0.5
    assert scores.correctness == "correct"
    assert scores.error is None


def test_refusal_gives_none_groundedness_not_zero():
    judge = make_judge()
    output = SystemOutput(answer="I don't have enough information.", context=CONTEXT)
    scores = score_answer(judge, QUESTION, output, AnswerOptions())
    assert scores.refused is True
    assert scores.groundedness is None
    assert scores.citation_precision is None
    assert judge.calls == []  # no claims are extracted from a refusal


def test_correctness_skipped_without_reference():
    judge = make_judge(correctness={"verdict": "correct", "reason": ""})
    question = QUESTION.model_copy(update={"reference_answer": None})
    scores = score_answer(judge, question, OUTPUT, AnswerOptions(correctness=True))
    assert scores.correctness is None and judge.count("correctness") == 0


def test_refusal_judge_option_uses_llm():
    judge = make_judge(refusal={"refused": True})
    scores = score_answer(judge, QUESTION, OUTPUT, AnswerOptions(refusal_judge=True))
    assert scores.refused is True and scores.groundedness is None


def test_judge_error_is_recorded_not_raised():
    judge = make_judge()
    judge.responses["extract_claims"] = JudgeError("bad JSON twice")
    scores = score_answer(judge, QUESTION, OUTPUT, AnswerOptions())
    assert scores.error == "judge: bad JSON twice"
    assert scores.groundedness is None
    assert scores.hallucinated_citations == ["c9"]  # non-judge metrics are kept


def test_judge_config_error_propagates():
    judge = make_judge()
    judge.responses["extract_claims"] = JudgeConfigError("no key")
    with pytest.raises(JudgeConfigError):
        score_answer(judge, QUESTION, OUTPUT, AnswerOptions())
