"""Shape executors registry."""

from bench.runner.executors.base import ShapeExecutor
from bench.runner.executors.claude_cli import ClaudeCliExecutor
from bench.runner.executors.langgraph_agent import LangGraphExecutor
from bench.runner.executors.raw import RawExecutor
from bench.runner.executors.single_agent import SingleAgentExecutor

EXECUTORS: dict[str, type[ShapeExecutor]] = {
    "raw": RawExecutor,
    "single-agent": SingleAgentExecutor,
    "multi-agent": LangGraphExecutor,
    "cli": ClaudeCliExecutor,
}

__all__ = [
    "EXECUTORS",
    "ShapeExecutor",
    "RawExecutor",
    "SingleAgentExecutor",
    "LangGraphExecutor",
    "ClaudeCliExecutor",
]
