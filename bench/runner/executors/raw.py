"""Raw shape: bare LLM call, no tools, no framework."""

from __future__ import annotations

import time
from typing import Any

import litellm

from bench.runner.executors.base import ShapeExecutor
from bench.runner.outcome import ExecutionException, ExecutionOutcome


class RawExecutor(ShapeExecutor):
    name = "raw"

    def execute(self, spec, task, defaults) -> ExecutionOutcome:
        start = time.monotonic()
        outcome = ExecutionOutcome()
        user_msg = {"role": "user", "content": task.prompt}
        outcome.transcript.append(user_msg)

        try:
            kwargs: dict[str, Any] = dict(
                model=spec.model,
                messages=[user_msg],
                temperature=0.7,
                max_tokens=1024,
            )
            mock = task.extras.get("mock_response") if hasattr(task, "extras") else None
            if mock is not None:
                kwargs["mock_response"] = mock

            response = litellm.completion(**kwargs)
            outcome.step_count = 1
            content = response.choices[0].message.content or ""
            outcome.final_output = content
            outcome.transcript.append({"role": "assistant", "content": content})
        except Exception as exc:
            outcome.exception = ExecutionException.from_exc(exc)

        outcome.wall_time_s = time.monotonic() - start
        return outcome
