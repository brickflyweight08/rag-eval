"""Loading inputs: questions and predictions (JSONL) and plugin systems ("module:factory")."""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from rag_eval.protocols import RAGSystem
from rag_eval.schema import EvalQuestion, Prediction

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

MIN_QUESTIONS_PER_QTYPE = 3


class DatasetError(ValueError):
    """Raised when a JSONL file has problems. ``problems`` lists every one of them."""

    def __init__(self, path: str | Path, problems: list[str]) -> None:
        self.path = str(path)
        self.problems = problems
        lines = "\n".join(f"  - {p}" for p in problems)
        super().__init__(f"{len(problems)} problem(s) in {self.path}:\n{lines}")


class SystemLoadError(RuntimeError):
    """Raised when a "module:factory" plugin cannot be loaded."""


@dataclass
class QuestionsReport:
    """Everything ``rag-eval validate`` prints about a questions file."""

    questions: list[EvalQuestion] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def file_sha256(path: str | Path) -> str:
    """SHA-256 of the raw file bytes (used to detect benchmark changes)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _format_validation_error(line_no: int, exc: ValidationError) -> list[str]:
    problems = []
    for err in exc.errors():
        where = ".".join(str(part) for part in err["loc"]) or "<line>"
        problems.append(f"line {line_no}: {where}: {err['msg']}")
    return problems


def read_jsonl(path: str | Path, model: type[T]) -> tuple[list[tuple[int, T]], list[str]]:
    """Parse and validate every line. Returns (items with line numbers, all problems)."""
    items: list[tuple[int, T]] = []
    problems: list[str] = []
    text = Path(path).read_text(encoding="utf-8")
    for line_no, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            problems.append(f"line {line_no}: invalid JSON: {exc.msg}")
            continue
        try:
            items.append((line_no, model.model_validate(data)))
        except ValidationError as exc:
            problems.extend(_format_validation_error(line_no, exc))
    return items, problems


def check_questions(numbered: list[tuple[int, EvalQuestion]]) -> tuple[list[str], list[str]]:
    """Dataset-level checks. Returns (errors, warnings)."""
    errors: list[str] = []
    first_seen: dict[str, int] = {}
    for line_no, q in numbered:
        if q.id in first_seen:
            errors.append(
                f"line {line_no}: duplicate id {q.id!r} (first seen on line {first_seen[q.id]})"
            )
        else:
            first_seen[q.id] = line_no
        if not q.question.strip():
            errors.append(f"line {line_no}: question text is empty (id {q.id!r})")
        if q.answerable and not q.gold_ids:
            errors.append(f"line {line_no}: answerable question {q.id!r} has no gold_ids")
        if not q.answerable and q.gold_ids:
            errors.append(f"line {line_no}: unanswerable question {q.id!r} must have no gold_ids")
    counts = Counter(q.qtype for _, q in numbered)
    warnings = [
        f"qtype {qtype!r} has only {n} question(s); per-type numbers will be very noisy"
        for qtype, n in sorted(counts.items())
        if n < MIN_QUESTIONS_PER_QTYPE
    ]
    return errors, warnings


def validate_questions_file(path: str | Path) -> QuestionsReport:
    """Read, validate and check a questions file without raising."""
    numbered, problems = read_jsonl(path, EvalQuestion)
    errors, warnings = check_questions(numbered)
    return QuestionsReport(
        questions=[q for _, q in numbered], errors=problems + errors, warnings=warnings
    )


def load_questions(path: str | Path) -> list[EvalQuestion]:
    """Load a questions JSONL file. Raises ``DatasetError`` listing every problem."""
    report = validate_questions_file(path)
    if not report.ok:
        raise DatasetError(path, report.errors)
    for warning in report.warnings:
        logger.warning("%s: %s", path, warning)
    return report.questions


def load_predictions(path: str | Path) -> list[Prediction]:
    """Load an offline predictions JSONL file (one ``Prediction`` per line)."""
    numbered, problems = read_jsonl(path, Prediction)
    first_seen: dict[str, int] = {}
    for line_no, pred in numbered:
        if pred.question_id in first_seen:
            problems.append(
                f"line {line_no}: duplicate prediction for {pred.question_id!r} "
                f"(first seen on line {first_seen[pred.question_id]})"
            )
        else:
            first_seen[pred.question_id] = line_no
    if problems:
        raise DatasetError(path, problems)
    return [pred for _, pred in numbered]


def _import_module(module_name: str) -> Any:
    # Like `python -m`, make modules in the current directory importable
    # (e.g. `examples.toy_handbook.systems` when run from the repo root). See ADR 0007.
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name and module_name.startswith(exc.name):
            raise SystemLoadError(
                f"could not import module {module_name!r}: {exc}. "
                "Is the package installed, or are you running from the right directory?"
            ) from exc
        raise SystemLoadError(
            f"module {module_name!r} failed to import one of its dependencies: {exc}"
        ) from exc
    except Exception as exc:  # any error raised while importing the user's module
        raise SystemLoadError(f"error while importing module {module_name!r}: {exc!r}") from exc


def _protocol_problems(system: object) -> list[str]:
    problems = []
    if not hasattr(system, "name"):
        problems.append("missing attribute `name`")
    for method in ("retrieve", "answer"):
        if not callable(getattr(system, method, None)):
            problems.append(f"missing method `{method}(question, k)`")
    return problems


def load_system(spec: str, kwargs: dict[str, Any] | None = None) -> RAGSystem:
    """Import ``"module.path:factory"``, call ``factory(**kwargs)`` and check the result."""
    module_name, sep, attr = spec.partition(":")
    if not sep or not module_name or not attr:
        raise SystemLoadError(
            f"invalid system spec {spec!r}: expected 'module.path:factory', "
            "e.g. 'my_pkg.my_module:make_system'"
        )
    module = _import_module(module_name)
    if not hasattr(module, attr):
        public = sorted(n for n in dir(module) if not n.startswith("_"))
        raise SystemLoadError(
            f"module {module_name!r} has no attribute {attr!r}. Available names: {public}"
        )
    factory = getattr(module, attr)
    if not callable(factory):
        raise SystemLoadError(f"{spec!r} is not callable; it must be a factory function or class")
    try:
        system = factory(**(kwargs or {}))
    except Exception as exc:
        raise SystemLoadError(f"factory {spec!r} raised {exc!r}") from exc
    problems = _protocol_problems(system)
    if problems:
        raise SystemLoadError(
            f"object returned by {spec!r} ({type(system).__name__}) does not satisfy the "
            f"RAGSystem protocol: {'; '.join(problems)}."
        )
    return system
