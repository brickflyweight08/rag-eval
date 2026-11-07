"""Judge calibration: how often does the claim verifier agree with human labels?

Run the same verification step used for groundedness on human-labelled claims, then
report percentage agreement, Cohen's kappa and every disagreement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from rag_eval.llm import Judge, JudgeConfigError, JudgeError
from rag_eval.loading import DatasetError, read_jsonl
from rag_eval.metrics.groundedness import verify_claim
from rag_eval.schema import RetrievedChunk
from rag_eval.stats import cohen_kappa

AGREEMENT_THRESHOLD = 0.80
KAPPA_THRESHOLD = 0.60


class CalibrationLabel(BaseModel):
    """One line of a labels file. Extra fields (e.g. who labelled it) are ignored."""

    question_id: str
    claim: str
    context: list[RetrievedChunk]
    human_supported: bool


@dataclass
class CalibrationResult:
    judge_model: str
    n: int  # labels the judge scored (judge errors excluded)
    agreement: float | None
    kappa: float | None  # nan when undefined (all labels identical)
    disagreements: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)

    @property
    def trustworthy(self) -> bool:
        return (
            self.agreement is not None
            and self.kappa is not None
            and not math.isnan(self.kappa)
            and self.agreement >= AGREEMENT_THRESHOLD
            and self.kappa >= KAPPA_THRESHOLD
        )


def load_labels(path: str | Path) -> list[CalibrationLabel]:
    numbered, problems = read_jsonl(path, CalibrationLabel)
    if problems:
        raise DatasetError(path, problems)
    if not numbered:
        raise DatasetError(path, ["the labels file is empty"])
    return [label for _, label in numbered]


def calibrate(labels: list[CalibrationLabel], judge: Judge) -> CalibrationResult:
    human: list[bool] = []
    machine: list[bool] = []
    disagreements: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for label in labels:
        try:
            verdict = verify_claim(judge, label.claim, label.context)
        except JudgeConfigError:
            raise
        except JudgeError as exc:
            errors.append(
                {"question_id": label.question_id, "claim": label.claim, "error": str(exc)}
            )
            continue
        human.append(label.human_supported)
        machine.append(verdict["supported"])
        if verdict["supported"] != label.human_supported:
            disagreements.append(
                {
                    "question_id": label.question_id,
                    "claim": label.claim,
                    "human_supported": label.human_supported,
                    "judge_supported": verdict["supported"],
                    "judge_evidence_id": verdict["evidence_id"],
                }
            )
    n = len(human)
    agreement = sum(h == m for h, m in zip(human, machine, strict=True)) / n if n else None
    kappa = cohen_kappa(human, machine) if n else None
    return CalibrationResult(judge.model, n, agreement, kappa, disagreements, errors)


def _fmt_kappa(kappa: float | None) -> str:
    if kappa is None:
        return "n/a"
    if math.isnan(kappa):
        return "undefined (every label was the same)"
    return f"{kappa:.3f}"


def render_calibration(result: CalibrationResult) -> str:
    agreement = "n/a" if result.agreement is None else f"{result.agreement:.1%}"
    lines = [
        f"Judge model: {result.judge_model}",
        f"Labels scored: {result.n}"
        + (f" ({len(result.errors)} judge error(s))" if result.errors else ""),
        f"Agreement: {agreement}",
        f"Cohen's kappa: {_fmt_kappa(result.kappa)}",
        "",
    ]
    if result.trustworthy:
        lines.append(
            f"Agreement >= {AGREEMENT_THRESHOLD:.0%} and kappa >= {KAPPA_THRESHOLD}: the judge "
            "agrees well with the labels on this sample. Keep checking as prompts change."
        )
    else:
        lines.append(
            f"WARNING: agreement below about {AGREEMENT_THRESHOLD:.0%} or kappa below about "
            f"{KAPPA_THRESHOLD} means groundedness numbers should not be trusted yet. Improve "
            "the verification prompt or use a stronger judge model, then calibrate again."
        )
    if result.disagreements:
        lines += ["", f"Disagreements ({len(result.disagreements)}):"]
        for d in result.disagreements:
            lines.append(
                f"- [{d['question_id']}] human={d['human_supported']} "
                f"judge={d['judge_supported']} (evidence {d['judge_evidence_id']}): {d['claim']}"
            )
    if result.errors:
        lines += ["", "Judge errors:"]
        lines += [f"- [{e['question_id']}] {e['error']}" for e in result.errors]
    return "\n".join(lines) + "\n"
