"""Command-line interface: ``rag-eval validate | run | score | report | compare | calibrate``."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import typer

from rag_eval.loading import validate_questions_file

app = typer.Typer(
    help="Evaluate retrieval-augmented generation (RAG) systems.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def main() -> None:
    """Evaluate retrieval-augmented generation (RAG) systems."""


def _fail(message: str) -> None:
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=1)


@app.command()
def validate(questions: Path = typer.Argument(..., help="Questions JSONL file.")) -> None:
    """Check a questions file and print a summary (count per qtype, answerable vs not)."""
    if not questions.is_file():
        _fail(f"file not found: {questions}")
    report = validate_questions_file(questions)
    qs = report.questions
    n_answerable = sum(q.answerable for q in qs)
    typer.echo(f"{questions}: {len(qs)} valid question(s)")
    typer.echo(f"  answerable: {n_answerable}   unanswerable: {len(qs) - n_answerable}")
    typer.echo("  per qtype:")
    for qtype, n in sorted(Counter(q.qtype for q in qs).items()):
        typer.echo(f"    {qtype:<24} {n}")
    for warning in report.warnings:
        typer.echo(f"Warning: {warning}")
    if report.errors:
        typer.echo(f"{len(report.errors)} problem(s):", err=True)
        for problem in report.errors:
            typer.echo(f"  - {problem}", err=True)
        raise typer.Exit(code=1)
    typer.echo("OK")


if __name__ == "__main__":  # pragma: no cover
    app()
