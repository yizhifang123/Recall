"""Cost meter — subscribes to litellm.success_callback and tracks USD + tokens.

Per-trial cost is computed as a delta: snapshot total at trial start, subtract
at trial end. The hard cap is checked between trials by the runner.
"""

from __future__ import annotations

import threading
from typing import Any

import litellm


class CostMeter:
    def __init__(self, cap_usd: float) -> None:
        self.cap_usd = cap_usd
        self.total_usd = 0.0
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self._trial_start_total = 0.0
        self._trial_start_tokens = dict(self.token_usage)
        self._lock = threading.Lock()
        self._install_callback()

    def _install_callback(self) -> None:
        existing = list(litellm.success_callback or [])
        if self._record not in existing:
            existing.append(self._record)
            litellm.success_callback = existing

    def _record(
        self,
        kwargs: dict[str, Any],
        completion_response: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        try:
            cost = kwargs.get("response_cost") or 0.0
            usage = getattr(completion_response, "usage", None)
            with self._lock:
                self.total_usd += float(cost or 0)
                if usage is not None:
                    for key, attr in [
                        ("prompt_tokens", "prompt_tokens"),
                        ("completion_tokens", "completion_tokens"),
                        ("total_tokens", "total_tokens"),
                    ]:
                        val = getattr(usage, attr, 0) or 0
                        self.token_usage[key] += int(val)
        except Exception:
            # Cost meter must never break the harvest.
            pass

    def trial_start(self) -> None:
        with self._lock:
            self._trial_start_total = self.total_usd
            self._trial_start_tokens = dict(self.token_usage)

    def add_external_cost(
        self, usd: float, prompt_tokens: int = 0, completion_tokens: int = 0
    ) -> None:
        """For costs not flowing through litellm (e.g. Claude Code subprocess)."""
        with self._lock:
            self.total_usd += usd
            self.token_usage["prompt_tokens"] += prompt_tokens
            self.token_usage["completion_tokens"] += completion_tokens
            self.token_usage["total_tokens"] += prompt_tokens + completion_tokens

    def trial_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "cost_usd": round(self.total_usd - self._trial_start_total, 6),
                "token_usage": {
                    k: self.token_usage[k] - self._trial_start_tokens[k] for k in self.token_usage
                },
            }

    def cap_exceeded(self) -> bool:
        with self._lock:
            return self.total_usd >= self.cap_usd

    def total_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_usd": round(self.total_usd, 6),
                "cap_usd": self.cap_usd,
                "token_usage": dict(self.token_usage),
                "remaining_usd": round(max(0.0, self.cap_usd - self.total_usd), 6),
            }
