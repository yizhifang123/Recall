"""ShapeExecutor abstract base."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from bench.runner.outcome import ExecutionOutcome

if TYPE_CHECKING:
    from bench.runner.config import Defaults, TrialSpec


class ShapeExecutor(ABC):
    """Run one trial of one shape and return an ExecutionOutcome.

    Implementations MUST:
      - never raise; catch all exceptions and stash them in
        outcome.exception via ExecutionException.from_exc(exc)
      - populate step_count incrementally so failure detection has signal
        even when the executor crashes mid-trial
      - populate transcript[] with human-readable per-step entries
      - never write to disk; the runner handles serialization
    """

    name: str = "shape"

    @abstractmethod
    def execute(
        self,
        spec: TrialSpec,
        task: Any,
        defaults: Defaults,
    ) -> ExecutionOutcome:
        raise NotImplementedError
