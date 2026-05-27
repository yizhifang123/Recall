"""Classify a completed trial as failure (and why) or pass (discard)."""

from __future__ import annotations

from collections.abc import Callable

from bench.runner.outcome import ExecutionOutcome, FailureReason


class FailureDetector:
    def __init__(self, step_budget: int, timeout_seconds: float) -> None:
        self.step_budget = step_budget
        self.timeout_seconds = timeout_seconds

    def classify(
        self,
        outcome: ExecutionOutcome,
        success_criteria: Callable[[ExecutionOutcome], bool] | None,
    ) -> FailureReason | None:
        """Return a FailureReason if the trial failed, else None.

        Order matters: exception > step budget > timeout > success_criteria.
        We classify on the FIRST trigger so traces have a single, clean
        failure label.
        """
        if outcome.exception is not None:
            return FailureReason.EXCEPTION

        if outcome.step_count > self.step_budget:
            return FailureReason.STEP_BUDGET_EXCEEDED

        if outcome.wall_time_s and outcome.wall_time_s > self.timeout_seconds:
            return FailureReason.TIMEOUT

        if success_criteria is not None:
            try:
                if not success_criteria(outcome):
                    return FailureReason.SUCCESS_CRITERIA_FALSE
            except Exception:
                # A criteria callback that itself errors counts as a failure
                # (the criteria couldn't confirm success, so we save the trace).
                return FailureReason.SUCCESS_CRITERIA_FALSE

        return None
