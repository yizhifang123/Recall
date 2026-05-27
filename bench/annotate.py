"""Annotation scaffolder for harvested Phase 1 traces.

Two commands:

  scaffold  — build/update bench/seed_traces/annotations.jsonl from saved
              trace JSONs. Preserves any user_summary / user_hypothesis /
              user_tags you've already written; only adds new entries for
              traces that lack one. Use --overwrite to wipe user fields.

  validate  — read annotations.jsonl and flag empty or vague entries.
              Vagueness heuristics: phrases like "agent failed", "model
              error", "didn't work" — meaningful for a one-line gloss but
              not useful for taxonomy derivation. Aim for specific
              mechanisms ("planner dropped the end-city constraint
              between steps 2 and 3").

Usage:
  uv run python bench/annotate.py scaffold
  uv run python bench/annotate.py validate
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

REPO_ROOT = Path("/Users/yida/Recall")
SEED_DIR = REPO_ROOT / "bench" / "seed_traces"
ANNOTATIONS_FILE = SEED_DIR / "annotations.jsonl"

console = Console()
app = typer.Typer(
    name="recall-annotate",
    help="Scaffold and validate Phase 1 trace annotations.",
    no_args_is_help=True,
    add_completion=False,
)


_VAGUE_PATTERNS = (
    "agent failed",
    "got it wrong",
    "model error",
    "didn't work",
    "did not work",
    "wrong answer",
    "incorrect",
    "bad output",
    "didn't follow",
    "did not follow",
    "messed up",
    "failed to ",
    "couldn't ",
    "broken",
    "no good",
)


def _outcome_summary(d: dict[str, Any]) -> str:
    """Auto-generated one-line gloss for context — NEVER edit this in the JSONL."""
    outcome = d["outcome"]
    if outcome["exception"]:
        return f"EXCEPTION {outcome['exception']['type']}: {outcome['exception']['message'][:120]}"
    final = (outcome["final_output"] or "").replace("\n", " ")[:160]
    return f"step_count={outcome['step_count']}, final={final!r}"


def _load_existing() -> dict[str, dict[str, Any]]:
    if not ANNOTATIONS_FILE.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for raw in ANNOTATIONS_FILE.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            out[d["trace_id"]] = d
        except (json.JSONDecodeError, KeyError):
            continue
    return out


@app.command()
def scaffold(
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Wipe existing user_summary / user_hypothesis / user_tags. Dangerous.",
    ),
) -> None:
    """Build/update annotations.jsonl from saved trace JSONs."""
    if not SEED_DIR.exists():
        console.print(f"[red]Missing dir: {SEED_DIR}[/red]")
        raise typer.Exit(1)

    existing = {} if overwrite else _load_existing()
    trace_paths = sorted(p for p in SEED_DIR.glob("*.json") if p.name != "annotations.jsonl")
    if not trace_paths:
        console.print(
            f"[yellow]No traces in {SEED_DIR.relative_to(REPO_ROOT)} — run harvest first.[/yellow]"
        )
        raise typer.Exit(1)

    new_count = 0
    preserved_count = 0
    lines: list[str] = []

    for tpath in trace_paths:
        with tpath.open() as f:
            d = json.load(f)
        trace_id = d["meta"]["trace_id"]
        prev = existing.get(trace_id, {})

        entry = {
            "trace_id": trace_id,
            "trace_file": tpath.name,
            "shape": d["meta"]["shape"],
            "framework": d["meta"]["framework"],
            "task_id": d["meta"]["task_id"],
            "task_category": d["meta"]["task_category"],
            "model": d["meta"]["model"],
            "failure_reason": d["outcome"]["failure_reason"],
            "step_count": d["outcome"]["step_count"],
            "wall_time_s": d["meta"]["wall_time_s"],
            "auto_outcome_summary": _outcome_summary(d),
            # User-fill fields. Preserve unless --overwrite.
            "user_summary": prev.get("user_summary", ""),
            "user_hypothesis": prev.get("user_hypothesis", ""),
            "user_tags": prev.get("user_tags", []),
        }
        lines.append(json.dumps(entry))
        if trace_id in existing:
            preserved_count += 1
        else:
            new_count += 1

    ANNOTATIONS_FILE.write_text("\n".join(lines) + "\n")
    console.print(
        f"[green]Wrote {len(lines)} annotations[/green] to "
        f"{ANNOTATIONS_FILE.relative_to(REPO_ROOT)}"
    )
    console.print(f"  {new_count} new stubs")
    console.print(f"  {preserved_count} updated (user fields preserved)")
    console.print()
    console.print(
        f"[bold]Next:[/bold] open {ANNOTATIONS_FILE.relative_to(REPO_ROOT)} "
        "in your editor and fill in:"
    )
    console.print("  user_summary    — one sentence: what specifically went wrong (be concrete)")
    console.print("  user_hypothesis — your tentative failure-mode hypothesis, free-form")
    console.print("  user_tags       — optional list of free-form labels")
    console.print()
    console.print("Then: [cyan]uv run python bench/annotate.py validate[/cyan]")


@app.command()
def validate() -> None:
    """Check annotations.jsonl for empty or vague user_summary / user_hypothesis."""
    if not ANNOTATIONS_FILE.exists():
        console.print(
            f"[red]No {ANNOTATIONS_FILE.relative_to(REPO_ROOT)} — "
            "run [cyan]scaffold[/cyan] first.[/red]"
        )
        raise typer.Exit(1)

    issues_by_trace: list[tuple[str, str, str, list[str]]] = []
    annotated = 0
    total = 0

    for raw in ANNOTATIONS_FILE.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        d = json.loads(line)
        total += 1
        tid = d["trace_id"][:8]
        shape = d["shape"]
        task = d["task_id"]

        issues: list[str] = []
        us = (d.get("user_summary") or "").strip()
        uh = (d.get("user_hypothesis") or "").strip()

        if not us:
            issues.append("user_summary EMPTY")
        else:
            annotated += 1
            if len(us) < 25:
                issues.append(f"user_summary too short ({len(us)} chars): {us!r}")
            else:
                # Vagueness only flagged if summary is short-ish AND matches a vague phrase
                lower_us = us.lower()
                for vp in _VAGUE_PATTERNS:
                    if vp in lower_us and len(us) < 80:
                        issues.append(
                            f"user_summary may be vague — contains '{vp}' but is brief "
                            f"({len(us)} chars). What specifically?"
                        )
                        break

        if not uh:
            issues.append("user_hypothesis EMPTY")
        elif len(uh) < 10:
            issues.append(f"user_hypothesis too short ({len(uh)} chars): {uh!r}")

        if issues:
            issues_by_trace.append((tid, shape, task, issues))

    console.print()
    console.print(f"[bold]Annotation progress:[/bold] {annotated}/{total} traces have user_summary")
    console.print(f"[bold]Issues:[/bold] {len(issues_by_trace)} traces need attention")
    console.print()

    if not issues_by_trace:
        console.print(
            "[green]All annotations look good.[/green] Ready for the frequency-table step."
        )
        return

    for tid, shape, task, issues in issues_by_trace:
        console.print(f"  [yellow]{tid}[/yellow]  {shape:<13} / {task}")
        for i in issues:
            console.print(f"      - {i}")


if __name__ == "__main__":
    app()
