"""Recall CLI entry point."""

import typer

from recall import __version__

app = typer.Typer(
    name="recall",
    help="Recall — flight recorder for LLM agents.",
    no_args_is_help=True,
)


@app.callback()
def _root() -> None:
    """Root callback — forces Typer into multi-command dispatch mode."""


@app.command()
def version() -> None:
    """Print the installed Recall version."""
    typer.echo(__version__)
