"""Execution outcome dataclasses shared across all shape executors."""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class FailureReason(StrEnum):
    EXCEPTION = "exception"
    STEP_BUDGET_EXCEEDED = "step_budget_exceeded"
    TIMEOUT = "timeout"
    SUCCESS_CRITERIA_FALSE = "success_criteria_false"


@dataclass
class ExecutionException:
    type: str
    message: str
    traceback: str

    @classmethod
    def from_exc(cls, exc: BaseException) -> ExecutionException:
        return cls(
            type=type(exc).__name__,
            message=str(exc),
            traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        )


@dataclass
class ExecutionOutcome:
    """What a shape executor returns after running one trial.

    The executor populates fields it can. The FailureDetector then classifies.
    """

    final_output: str = ""
    step_count: int = 0
    wall_time_s: float = 0.0
    exception: ExecutionException | None = None
    transcript: list[dict[str, Any]] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)
