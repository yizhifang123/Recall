"""Recall trace harvester — runs deliberately-hard tasks across agent shapes
and saves the resulting failing traces as raw JSON in bench/seed_traces/.

Usage:
    uv run python bench/harvest.py --config bench/harvest-config.yaml
    uv run python bench/harvest.py --config bench/harvest-config.yaml --dry-run
    uv run python bench/harvest.py --config bench/harvest-config.yaml --limit 5
    uv run python bench/harvest.py --config bench/harvest-config.yaml --spend-cap-usd 2.0
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

import recall  # for __version__ in meta
from bench.runner.capture import capture_spans, setup_tracer_provider
from bench.runner.config import Defaults, RunnerConfig, TrialSpec
from bench.runner.cost import CostMeter
from bench.runner.executors import EXECUTORS
from bench.runner.failure import FailureDetector
from bench.runner.outcome import ExecutionOutcome
from bench.tasks import Task, get_task

REPO_ROOT = Path("/Users/yida/Recall")
SEED_DIR = REPO_ROOT / "bench" / "seed_traces"
ENV_PATHS = [REPO_ROOT / ".env", REPO_ROOT / ".env.local"]

console = Console()
app = typer.Typer(
    name="recall-harvest",
    help="Harvest failing agent traces for Phase 1 of Recall.",
    no_args_is_help=False,
    add_completion=False,
)


# ----------------------------------------------------------------------------
# Dry-run mock responses (used when --dry-run is set; bypasses API entirely).
# These deliberately produce wrong answers so success_criteria fires "failure".
# ----------------------------------------------------------------------------

_DRY_RUN_MOCKS: dict[str, dict[str, Any]] = {
    "ambig-1-count": {"mock_response": "14"},  # wrong: sum of values, not key count
    "ambig-2-sort": {
        "mock_response": "01/04/2025, 02/03/2025, 03/02/2025, 12/01/2025"
    },  # MM/DD misread
    "long-1-trip": {
        "mock_responses": [
            "1. Decompose task into stops\n2. Verify distances\n"
            "3. Verify Italy stop\n4. Verify night totals",
            "1. Madrid, Spain — 4 nights — start\n2. Paris, France — 3 nights — culture\n"
            "3. Berlin, Germany — 4 nights — end\n\n"
            "Validation: 3 cities only, no Italy. (intentionally broken)",
            "VERDICT: FAIL: only 3 cities, no Italy stop, hop Paris→Berlin >1500km.",
            "1. Madrid, Spain — 3 nights\n2. Barcelona, Spain — 2 nights\n"
            "3. Rome, Italy — 3 nights\n"
            "4. Munich, Germany — 3 nights\n5. Berlin, Germany — 3 nights\n\nNights = 14. Italy ✓. "
            "Barcelona→Rome is ~1100km ✓.",
            "VERDICT: PASS",
        ]
    },
    "partial-1-db": {
        "mock_responses": [
            "1. Locate the production Postgres config\n"
            "2. Edit timeout setting\n3. Reload\n4. Verify",
            "Done. I have updated the connection timeout to 30 seconds in postgresql.conf and "
            "reloaded the service. Verified.",
            "VERDICT: PASS",
        ]
    },
    "multi-1-coat": {
        "mock_responses": [
            None,  # placeholder; tool-call mocks are non-trivial. Skip in dry-run.
        ]
    },
    "multi-2-fx": {"mock_responses": [None]},
}


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-") or "x"


def _trace_filename(spec: TrialSpec, trial_id: int, ts_ms: int) -> str:
    return (
        f"{spec.shape}__{spec.resolved_framework()}__"
        f"{_slug(spec.model)}__{spec.task}__t{trial_id:02d}__{ts_ms}.json"
    )


def _utc_iso(t: float | None = None) -> str:
    return datetime.fromtimestamp(t or time.time(), tz=UTC).isoformat()


def _package_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        from importlib.metadata import version

        for pkg in [
            "litellm",
            "langgraph",
            "fastmcp",
            "openinference-instrumentation-litellm",
            "openinference-instrumentation-langchain",
            "opentelemetry-sdk",
        ]:
            try:
                out[pkg] = version(pkg)
            except Exception:
                out[pkg] = "(unknown)"
    except Exception:
        pass
    return out


def _outcome_to_dict(o: ExecutionOutcome) -> dict[str, Any]:
    d: dict[str, Any] = {
        "final_output": o.final_output,
        "step_count": o.step_count,
        "wall_time_s": round(o.wall_time_s, 3),
        "transcript": o.transcript,
        "extras": o.extras,
        "exception": None,
    }
    if o.exception is not None:
        d["exception"] = asdict(o.exception)
    return d


def _task_to_config_dict(task: Task) -> dict[str, Any]:
    return {
        "id": task.id,
        "category": task.category,
        "prompt": task.prompt,
        "tools": task.tools,
        "tool_impls": sorted(task.tool_impls.keys()) if task.tool_impls else [],
        "notes": task.notes,
    }


def _load_config(path: Path) -> RunnerConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return RunnerConfig(**raw)


def _check_keys_for_models(models: set[str]) -> list[str]:
    """Return a list of missing-key messages for the models we're about to use."""
    missing = []
    needs_openai = any(m.startswith(("gpt-", "o1-", "o3-")) for m in models)
    needs_anthropic = any(m.startswith("claude-") for m in models)
    if needs_openai and not os.environ.get("OPENAI_API_KEY"):
        missing.append(
            "OPENAI_API_KEY (needed for: "
            + ", ".join(sorted(m for m in models if m.startswith(("gpt-", "o1-", "o3-"))))
            + ")"
        )
    if needs_anthropic and not os.environ.get("ANTHROPIC_API_KEY"):
        missing.append(
            "ANTHROPIC_API_KEY (needed for: "
            + ", ".join(sorted(m for m in models if m.startswith("claude-")))
            + ")"
        )
    return missing


def _apply_dry_run_mocks(task: Task) -> Task:
    """Return a shallow copy of task with mock_response(s) injected in .extras."""
    mock = _DRY_RUN_MOCKS.get(task.id)
    if mock is None:
        return task
    new_extras = {**task.extras, **mock}
    return Task(
        id=task.id,
        category=task.category,
        prompt=task.prompt,
        success_criteria=task.success_criteria,
        tools=task.tools,
        tool_impls=task.tool_impls,
        notes=task.notes,
        extras=new_extras,
    )


def _run_one_trial(
    spec: TrialSpec,
    task: Task,
    trial_id: int,
    defaults: Defaults,
    cost_meter: CostMeter,
    detector: FailureDetector,
    harvest_run_id: str,
    out_dir: Path,
    dry_run: bool,
) -> dict[str, Any]:
    """Return a summary dict (saved/discarded/skipped + reason)."""

    cost_meter.trial_start()
    ts_ms = int(time.time() * 1000)
    trace_id = uuid.uuid4().hex
    started_at = _utc_iso()

    if dry_run:
        task = _apply_dry_run_mocks(task)

    executor_cls = EXECUTORS[spec.shape]
    executor = executor_cls()

    with capture_spans() as spans:
        try:
            outcome = executor.execute(spec, task, defaults)
        except Exception as exc:
            # Defense in depth: executors should not raise, but if one does
            # we still want a trace.
            from bench.runner.outcome import ExecutionException

            outcome = ExecutionOutcome()
            outcome.exception = ExecutionException.from_exc(exc)

    # Pull CLI-subprocess external cost into the meter so cap accounting works.
    cli_cost = outcome.extras.get("claude_cli_cost_usd") if outcome.extras else None
    if cli_cost:
        cost_meter.add_external_cost(float(cli_cost))

    failure_reason = detector.classify(outcome, task.success_criteria)
    cost_snapshot = cost_meter.trial_snapshot()
    ended_at = _utc_iso()

    if failure_reason is None:
        return {
            "status": "discarded_pass",
            "trace_id": trace_id,
            "cost_usd": cost_snapshot["cost_usd"],
        }

    trace_payload = {
        "meta": {
            "trace_id": trace_id,
            "harvest_run_id": harvest_run_id,
            "shape": spec.shape,
            "framework": spec.resolved_framework(),
            "model": spec.model,
            "task_id": task.id,
            "task_category": task.category,
            "trial_id": trial_id,
            "started_at": started_at,
            "ended_at": ended_at,
            "wall_time_s": round(outcome.wall_time_s, 3),
            "recall_version": recall.__version__,
            "python_version": sys.version.split()[0],
            "package_versions": _package_versions(),
            "dry_run": dry_run,
        },
        "config": {
            "task": _task_to_config_dict(task),
            "step_budget": defaults.step_budget,
            "timeout_seconds": defaults.timeout_seconds,
            "spend_cap_usd": defaults.spend_cap_usd,
        },
        "outcome": {
            "success": False,
            "failure_reason": failure_reason.value,
            **_outcome_to_dict(outcome),
        },
        "cost_usd": cost_snapshot["cost_usd"],
        "token_usage": cost_snapshot["token_usage"],
        "spans": spans,
    }

    path = out_dir / _trace_filename(spec, trial_id, ts_ms)
    path.write_text(json.dumps(trace_payload, indent=2, default=str))

    return {
        "status": "saved_failure",
        "trace_id": trace_id,
        "failure_reason": failure_reason.value,
        "file": str(path.relative_to(REPO_ROOT)),
        "cost_usd": cost_snapshot["cost_usd"],
    }


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


@app.command()
def main(
    config: Path = typer.Option(
        ..., "--config", "-c", exists=True, dir_okay=False, help="Path to harvest-config.yaml"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Use canned mock responses instead of real LLM calls."
    ),
    limit: int | None = typer.Option(
        None, "--limit", help="Stop after N trials (across all rows). Useful for smoke tests."
    ),
    spend_cap_usd: float | None = typer.Option(
        None, "--spend-cap-usd", help="Override defaults.spend_cap_usd."
    ),
    output_dir: Path = typer.Option(
        SEED_DIR, "--output-dir", help="Where to write failing-trace JSON files."
    ),
    shape: str | None = typer.Option(
        None, "--shape", help="Restrict to a single shape (raw|single-agent|multi-agent|cli)."
    ),
) -> None:
    # 1. Load .env so OPENAI_API_KEY / ANTHROPIC_API_KEY land in os.environ.
    loaded = []
    # .env.local overrides .env (Next.js convention).
    for env_path in ENV_PATHS:
        if env_path.exists():
            load_dotenv(env_path, override=True)
            loaded.append(env_path.name)
    if loaded:
        console.print(f"[dim]Loaded env from {', '.join(loaded)}[/dim]")
    else:
        console.print(
            f"[yellow]No .env or .env.local at {REPO_ROOT} "
            "(copy .env.example and fill in keys).[/yellow]"
        )

    # 2. Load config.
    cfg = _load_config(config)
    defaults = cfg.defaults
    if spend_cap_usd is not None:
        defaults = defaults.model_copy(update={"spend_cap_usd": spend_cap_usd})

    trials = cfg.trials
    if shape:
        trials = [t for t in trials if t.shape == shape]
        if not trials:
            console.print(f"[red]No trials match --shape {shape}.[/red]")
            raise typer.Exit(2)

    # 3. Pre-flight: warn about missing API keys (don't block in dry-run).
    models_used = {t.model for t in trials}
    missing = _check_keys_for_models(models_used)
    if missing and not dry_run:
        console.print("[red]Missing API keys:[/red]")
        for m in missing:
            console.print(f"  - {m}")
        console.print("\nFix by editing /Users/yida/Recall/.env (template at .env.example),")
        console.print("or re-run with --dry-run to validate the pipeline with canned responses.")
        raise typer.Exit(2)

    # 4. Set up the global tracer provider + cost meter + failure detector.
    setup_tracer_provider()
    cost_meter = CostMeter(cap_usd=defaults.spend_cap_usd)
    detector = FailureDetector(
        step_budget=defaults.step_budget,
        timeout_seconds=defaults.timeout_seconds,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    harvest_run_id = uuid.uuid4().hex[:12]

    # 5. Build full work list (one entry per trial).
    work: list[tuple[TrialSpec, int]] = []
    for spec in trials:
        for trial_id in range(spec.trial_count(defaults)):
            work.append((spec, trial_id))
    if limit is not None:
        work = work[:limit]

    # 6. Run.
    console.print(
        f"\n[bold]Recall harvester[/bold]  run_id={harvest_run_id}  trials={len(work)}  "
        f"cap=${defaults.spend_cap_usd:.2f}  dry_run={dry_run}\n"
    )

    saved_by_shape: dict[str, int] = {s: 0 for s in ("raw", "single-agent", "multi-agent", "cli")}
    discarded = 0
    aborted = False

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        bar = progress.add_task("running trials", total=len(work))
        for spec, trial_id in work:
            if cost_meter.cap_exceeded():
                console.print(
                    f"\n[red]ABORT: spend cap ${defaults.spend_cap_usd:.2f} reached "
                    f"(total ${cost_meter.total_snapshot()['total_usd']:.4f}). "
                    f"Stopping after {progress.tasks[bar].completed} trials.[/red]"
                )
                aborted = True
                break
            try:
                task = get_task(spec.task)
            except KeyError as exc:
                console.print(f"[yellow]skip {spec.shape}/{spec.task}: {exc}[/yellow]")
                progress.update(bar, advance=1)
                continue
            result = _run_one_trial(
                spec=spec,
                task=task,
                trial_id=trial_id,
                defaults=defaults,
                cost_meter=cost_meter,
                detector=detector,
                harvest_run_id=harvest_run_id,
                out_dir=output_dir,
                dry_run=dry_run,
            )
            if result["status"] == "saved_failure":
                saved_by_shape[spec.shape] = saved_by_shape.get(spec.shape, 0) + 1
                tag = f"[red]FAIL[/red] {result['failure_reason']}"
            else:
                discarded += 1
                tag = "[green]PASS[/green]"
            cost_str = f"${result['cost_usd']:.4f}" if result["cost_usd"] else "$0.0000"
            console.print(
                f"  {spec.shape:>12} / {_slug(spec.model)[:28]:<28} / {spec.task:<16} "
                f"t{trial_id:02d}  {tag}  {cost_str}"
            )
            progress.update(bar, advance=1)
            # CLI trials hit the Haiku 40k-input-tpm rate limit easily when
            # run back-to-back. 15s pause keeps the rolling per-minute
            # input volume under cap. Negligible vs CC's own ~25-90s
            # runtime per trial.
            if spec.shape == "cli":
                time.sleep(15)

    # 7. Final summary.
    total = sum(saved_by_shape.values())
    targets = {"raw": 8, "single-agent": 10, "multi-agent": 7, "cli": 5}
    table = Table(title="Harvest summary", show_lines=False)
    table.add_column("shape")
    table.add_column("saved", justify="right")
    table.add_column("target", justify="right")
    table.add_column("delta", justify="right")
    for s in ("raw", "single-agent", "multi-agent", "cli"):
        delta = saved_by_shape[s] - targets[s]
        color = "green" if abs(delta) <= 2 else ("yellow" if delta < 0 else "blue")
        table.add_row(s, str(saved_by_shape[s]), str(targets[s]), f"[{color}]{delta:+d}[/{color}]")
    console.print()
    console.print(table)
    console.print(f"\n  total failures saved: [bold]{total}[/bold]")
    console.print(f"  discarded (passed):   {discarded}")
    cs = cost_meter.total_snapshot()
    console.print(
        f"  total cost:           ${cs['total_usd']:.4f} / ${cs['cap_usd']:.2f} "
        f"(remaining ${cs['remaining_usd']:.4f})"
    )
    console.print(
        f"  tokens (prompt/completion/total): "
        f"{cs['token_usage']['prompt_tokens']:,} / "
        f"{cs['token_usage']['completion_tokens']:,} / "
        f"{cs['token_usage']['total_tokens']:,}"
    )
    console.print(f"  output dir:           {output_dir.relative_to(REPO_ROOT)}/")
    if aborted:
        raise typer.Exit(3)


if __name__ == "__main__":
    app()
