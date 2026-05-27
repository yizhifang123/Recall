"""Single-agent + tools shape: litellm.completion with manual tool loop."""

from __future__ import annotations

import json
import time
from typing import Any

import litellm

from bench.runner.executors.base import ShapeExecutor
from bench.runner.outcome import ExecutionException, ExecutionOutcome


class SingleAgentExecutor(ShapeExecutor):
    name = "single-agent"

    def execute(self, spec, task, defaults) -> ExecutionOutcome:
        start = time.monotonic()
        outcome = ExecutionOutcome()

        messages: list[dict[str, Any]] = [{"role": "user", "content": task.prompt}]
        outcome.transcript.append(messages[0])
        tools = task.tools or None
        tool_impls = task.tool_impls or {}
        mocks = (task.extras.get("mock_responses") if hasattr(task, "extras") else None) or []
        mock_idx = 0

        try:
            for step in range(defaults.step_budget):
                outcome.step_count = step + 1
                kwargs: dict[str, Any] = dict(
                    model=spec.model,
                    messages=messages,
                    tools=tools,
                    temperature=0.7,
                    max_tokens=1024,
                )
                if mock_idx < len(mocks):
                    kwargs["mock_response"] = mocks[mock_idx]
                    mock_idx += 1

                response = litellm.completion(**kwargs)
                msg = response.choices[0].message
                content = msg.content or ""
                tool_calls = list(getattr(msg, "tool_calls", None) or [])

                assistant_entry: dict[str, Any] = {"role": "assistant", "content": content}
                if tool_calls:
                    assistant_entry["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ]
                messages.append(assistant_entry)
                outcome.transcript.append(assistant_entry)

                if not tool_calls:
                    outcome.final_output = content
                    return _finish(outcome, start)

                # Dispatch every tool call requested.
                for tc in tool_calls:
                    name = tc.function.name
                    raw_args = tc.function.arguments or "{}"
                    try:
                        args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        args = {"_raw": raw_args, "_error": "invalid json"}

                    impl = tool_impls.get(name)
                    if impl is None:
                        result: Any = {"error": f"unknown tool: {name}"}
                    else:
                        try:
                            result = impl(**args) if isinstance(args, dict) else impl(args)
                        except Exception as exc:
                            result = {
                                "error": f"{type(exc).__name__}: {exc}",
                            }

                    tool_entry = {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": name,
                        "content": json.dumps(result, default=str),
                    }
                    messages.append(tool_entry)
                    outcome.transcript.append(tool_entry)

            # Loop exited via step_budget rather than `return` — overrun.
            outcome.step_count = defaults.step_budget + 1  # signal overrun
            outcome.final_output = messages[-1].get("content") or ""
        except Exception as exc:
            outcome.exception = ExecutionException.from_exc(exc)

        return _finish(outcome, start)


def _finish(outcome: ExecutionOutcome, start: float) -> ExecutionOutcome:
    outcome.wall_time_s = time.monotonic() - start
    return outcome
