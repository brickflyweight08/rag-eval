"""Command-line interface: ``rag-eval validate | run | score | report | compare | calibrate``."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import typer

from rag_eval.aggregate import METRICS
from rag_eval.config import load_settings
from rag_eval.llm import JudgeClient, JudgeConfigError
from rag_eval.loading import DatasetError, SystemLoadError, validate_questions_file
from rag_eval.runner import RunArtifacts, run_live, score_offline

app = typer.Typer(
    help="Evaluate retrieval-augmented generation (RAG) systems.",
    no_args_is_help=True,
    add_completion=False,
)

# Options shared by every command that may call the judge.
JUDGE_BASE_URL = typer.Option(None, help="Judge endpoint [env RAG_EVAL_JUDGE_BASE_URL].")
JUDGE_API_KEY = typer.Option(None, help="Judge API key [env RAG_EVAL_JUDGE_API_KEY].")
JUDGE_MODEL = typer.Option(None, help="Judge model [env RAG_EVAL_JUDGE_MODEL].")
CACHE_PATH = typer.Option(None, help="sqlite cache file [env RAG_EVAL_CACHE_PATH].")
NO_CACHE = typer.Option(False, "--no-cache", help="Bypass the judge cache.")
NO_JSON_MODE = typer.Option(
    False, "--no-json-mode", help="Do not send response_format=json_object (old endpoints)."
)
K = typer.Option(5, "--k", "-k", min=1, help="Number of retrieved chunks to score.")
CORRECTNESS = typer.Option(False, "--correctness", help="Judge answers vs reference_answer.")
RETRIEVAL_ONLY = typer.Option(False, "--retrieval-only", help="Skip answers and the judge.")
REFUSAL_JUDGE = typer.Option(
    False, "--refusal-judge", help="Detect refusals with the judge instead of phrases."
)
REFUSAL_PHRASE = typer.Option(
    None, "--refusal-phrase", help="Refusal phrase (repeatable); replaces the defaults."
)


@app.callback()
def main(
    verbose: bool = typer.Option(True, "--verbose/--quiet", help="Per-question progress."),
) -> None:
    """Evaluate retrieval-augmented generation (RAG) systems."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING, format="%(message)s", force=True
    )
    for noisy in ("httpx", "openai", "sentence_transformers", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _fail(message: str) -> None:
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=1)


def _make_judge(
    base_url: str | None,
    api_key: str | None,
    model: str | None,
    cache_path: str | None,
    no_cache: bool,
    no_json_mode: bool,
) -> JudgeClient:
    settings = load_settings(base_url, api_key, model, cache_path)
    return JudgeClient.from_settings(settings, use_cache=not no_cache, json_mode=not no_json_mode)


def _parse_kwargs(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        _fail(f"--system-kwargs is not valid JSON: {exc.msg} (got {raw!r})")
    if not isinstance(value, dict):
        _fail('--system-kwargs must be a JSON object, e.g. \'{"strategy": "hybrid"}\'')
    return value


def _fmt(stat: dict[str, Any]) -> str:
    if stat["mean"] is None:
        return "n/a"
    return f"{stat['mean']:.3f} [{stat['low']:.3f}, {stat['high']:.3f}] (n={stat['n']})"


def _print_summary(artifacts: RunArtifacts) -> None:
    summary = artifacts.summary
    typer.echo(
        f"\nRun {artifacts.config.run_id}: {summary['n_questions']} questions, "
        f"{summary['n_errors']} error(s), judge={artifacts.config.judge_model or 'none'}"
    )
    for spec in METRICS:
        stat = summary["overall"][spec.key]
        if stat["n"]:
            typer.echo(f"  {spec.title(summary['k']):<28} {_fmt(stat)}")
    latency = summary["latency_ms"]
    if latency["n"]:
        typer.echo(f"  {'Latency p50 / p95 (ms)':<28} {latency['p50']:.0f} / {latency['p95']:.0f}")


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


@app.command()
def run(
    system: str = typer.Option(..., help="Plugin factory, e.g. my_pkg.module:make_system."),
    questions: Path = typer.Option(..., help="Questions JSONL file."),
    out: Path = typer.Option(..., help="Output folder for this run."),
    system_kwargs: str = typer.Option("{}", help="JSON object passed to the factory."),
    k: int = K,
    correctness: bool = CORRECTNESS,
    retrieval_only: bool = RETRIEVAL_ONLY,
    refusal_judge: bool = REFUSAL_JUDGE,
    refusal_phrase: list[str] | None = REFUSAL_PHRASE,
    run_id: str | None = typer.Option(None, help="Run name (default: the --out folder name)."),
    judge_base_url: str | None = JUDGE_BASE_URL,
    judge_api_key: str | None = JUDGE_API_KEY,
    judge_model: str | None = JUDGE_MODEL,
    cache_path: str | None = CACHE_PATH,
    no_cache: bool = NO_CACHE,
    no_json_mode: bool = NO_JSON_MODE,
) -> None:
    """Evaluate a live system through the RAGSystem plugin protocol."""
    judge = _make_judge(
        judge_base_url, judge_api_key, judge_model, cache_path, no_cache, no_json_mode
    )
    try:
        artifacts = run_live(
            system,
            questions,
            k=k,
            out_dir=out,
            system_kwargs=_parse_kwargs(system_kwargs),
            judge=judge,
            retrieval_only=retrieval_only,
            correctness=correctness,
            refusal_judge=refusal_judge,
            refusal_phrases=refusal_phrase or None,
            run_id=run_id,
        )
    except (SystemLoadError, DatasetError, JudgeConfigError, FileNotFoundError) as exc:
        _fail(str(exc))
    _print_summary(artifacts)
    typer.echo(f"Wrote {out}/config.json, results.jsonl, summary.json")


@app.command()
def score(
    predictions: Path = typer.Option(..., help="Predictions JSONL file (one Prediction/line)."),
    questions: Path = typer.Option(..., help="Questions JSONL file."),
    out: Path = typer.Option(..., help="Output folder for this run."),
    k: int = K,
    correctness: bool = CORRECTNESS,
    retrieval_only: bool = RETRIEVAL_ONLY,
    refusal_judge: bool = REFUSAL_JUDGE,
    refusal_phrase: list[str] | None = REFUSAL_PHRASE,
    run_id: str | None = typer.Option(None, help="Run name (default: the --out folder name)."),
    judge_base_url: str | None = JUDGE_BASE_URL,
    judge_api_key: str | None = JUDGE_API_KEY,
    judge_model: str | None = JUDGE_MODEL,
    cache_path: str | None = CACHE_PATH,
    no_cache: bool = NO_CACHE,
    no_json_mode: bool = NO_JSON_MODE,
) -> None:
    """Score an offline predictions file produced by any external system."""
    judge = _make_judge(
        judge_base_url, judge_api_key, judge_model, cache_path, no_cache, no_json_mode
    )
    try:
        artifacts = score_offline(
            predictions,
            questions,
            k=k,
            out_dir=out,
            judge=judge,
            retrieval_only=retrieval_only,
            correctness=correctness,
            refusal_judge=refusal_judge,
            refusal_phrases=refusal_phrase or None,
            run_id=run_id,
        )
    except (DatasetError, JudgeConfigError, FileNotFoundError) as exc:
        _fail(str(exc))
    _print_summary(artifacts)
    typer.echo(f"Wrote {out}/config.json, results.jsonl, summary.json")


if __name__ == "__main__":  # pragma: no cover
    app()
