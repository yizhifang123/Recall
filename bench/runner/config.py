"""Pydantic config models for harvest-config.yaml."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ShapeName = Literal["raw", "single-agent", "multi-agent", "cli"]
FrameworkName = Literal["litellm", "langgraph", "claude-code"]

_DEFAULT_FRAMEWORK: dict[ShapeName, FrameworkName] = {
    "raw": "litellm",
    "single-agent": "litellm",
    "multi-agent": "langgraph",
    "cli": "claude-code",
}


class Defaults(BaseModel):
    step_budget: int = 25
    timeout_seconds: float = 120.0
    n_trials: int = 2
    spend_cap_usd: float = 5.0


class TrialSpec(BaseModel):
    shape: ShapeName
    model: str
    task: str
    framework: FrameworkName | None = None
    n_trials: int | None = None

    def resolved_framework(self) -> FrameworkName:
        return self.framework or _DEFAULT_FRAMEWORK[self.shape]

    def trial_count(self, defaults: Defaults) -> int:
        return self.n_trials if self.n_trials is not None else defaults.n_trials


class RunnerConfig(BaseModel):
    defaults: Defaults = Field(default_factory=Defaults)
    trials: list[TrialSpec]
