"""Multi-agent shape: LangGraph StateGraph (planner -> executor -> verifier) per D-001-A."""

from __future__ import annotations

import time
from typing import Any, TypedDict

import litellm
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from bench.runner.executors.base import ShapeExecutor
from bench.runner.outcome import ExecutionException, ExecutionOutcome


class _State(TypedDict, total=False):
    task: str
    plan: str
    execution: str
    verification: str
    iterations: int
    final: str


class LangGraphExecutor(ShapeExecutor):
    name = "multi-agent"

    def execute(self, spec, task, defaults) -> ExecutionOutcome:
        start = time.monotonic()
        outcome = ExecutionOutcome()
        max_iterations = max(2, defaults.step_budget // 6)
        mocks = (task.extras.get("mock_responses") if hasattr(task, "extras") else None) or []
        mock_idx = [0]  # closure-mutable

        def llm_call(messages: list[dict[str, Any]]) -> str:
            outcome.step_count += 1
            kwargs: dict[str, Any] = dict(
                model=spec.model,
                messages=messages,
                temperature=0.6,
                max_tokens=1024,
            )
            if mock_idx[0] < len(mocks):
                kwargs["mock_response"] = mocks[mock_idx[0]]
                mock_idx[0] += 1
            r = litellm.completion(**kwargs)
            return r.choices[0].message.content or ""

        def planner(state: _State) -> _State:
            plan = llm_call(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are the planner agent. Decompose the user's task into "
                            "3-6 concrete numbered steps. Identify constraints. Output "
                            "the plan only."
                        ),
                    },
                    {"role": "user", "content": state["task"]},
                ]
            )
            outcome.transcript.append({"role": "planner", "content": plan})
            return {"plan": plan, "iterations": state.get("iterations", 0)}

        def executor(state: _State) -> _State:
            exe = llm_call(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are the executor agent. Carry out the plan and produce a "
                            "final answer to the task. Be explicit about how each constraint "
                            "is satisfied. If a verifier previously rejected your work, "
                            "address its feedback."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Task: {state['task']}\n\n"
                            f"Plan:\n{state.get('plan', '(no plan)')}\n\n"
                            f"Previous verifier feedback: {state.get('verification', '(none)')}"
                        ),
                    },
                ]
            )
            outcome.transcript.append({"role": "executor", "content": exe})
            return {"execution": exe}

        def verifier(state: _State) -> _State:
            v = llm_call(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are the verifier agent. Decide if the executor's answer "
                            "actually satisfies every constraint in the task. Reply on a "
                            "single line: 'VERDICT: PASS' or 'VERDICT: FAIL: <one-line reason>'."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Task: {state['task']}\n\n"
                            f"Plan:\n{state.get('plan', '(no plan)')}\n\n"
                            f"Execution:\n{state.get('execution', '(none)')}"
                        ),
                    },
                ]
            )
            outcome.transcript.append({"role": "verifier", "content": v})
            iters = state.get("iterations", 0) + 1
            verdict_pass = "VERDICT: PASS" in v.upper()
            patch: _State = {"verification": v, "iterations": iters}
            if verdict_pass or iters >= max_iterations:
                patch["final"] = state.get("execution", "")
            return patch

        def route(state: _State) -> str:
            if state.get("final") is not None:
                return END
            if state.get("iterations", 0) >= max_iterations:
                return END
            return "executor"

        try:
            g: StateGraph = StateGraph(_State)
            g.add_node("planner", planner)
            g.add_node("executor", executor)
            g.add_node("verifier", verifier)
            g.add_edge(START, "planner")
            g.add_edge("planner", "executor")
            g.add_edge("executor", "verifier")
            g.add_conditional_edges("verifier", route, {"executor": "executor", END: END})

            graph = g.compile(checkpointer=MemorySaver())
            final_state = graph.invoke(
                {"task": task.prompt, "iterations": 0},
                config={
                    "configurable": {"thread_id": f"harvest-{int(start * 1000)}"},
                    "recursion_limit": max_iterations * 4 + 6,
                },
            )
            outcome.final_output = final_state.get("final") or final_state.get("execution") or ""
            outcome.extras["langgraph_final_state"] = {
                k: v
                for k, v in final_state.items()
                if isinstance(v, (str, int, float, bool, type(None)))
            }
            outcome.extras["langgraph_iterations"] = final_state.get("iterations", 0)
        except Exception as exc:
            if outcome.exception is None:
                outcome.exception = ExecutionException.from_exc(exc)

        outcome.wall_time_s = time.monotonic() - start
        return outcome
