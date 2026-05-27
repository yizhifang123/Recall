"""Hard tasks for harvesting failing agent traces.

Each Task is shape-agnostic at the prompt level. Shape executors decide how
to wire it: raw uses just .prompt; single-agent uses .prompt + .tools +
.tool_impls; multi-agent (LangGraph) uses .prompt and ignores tools (the
graph nodes don't expose tools in v1); cli uses .prompt only.

The .success_criteria callback returns True iff the agent's final_output
satisfies the task per a human grader's expectations. Criteria are
heuristic — they're tuned to catch failures (the thing we want to harvest);
false negatives (real successes mislabeled as failures) cost us only an
extra annotation, while false positives (real failures mislabeled as
successes) cost us a dropped trace. We bias toward keeping traces.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from bench.runner.outcome import ExecutionOutcome


@dataclass
class Task:
    id: str
    category: str  # ambiguous-spec | multi-tool | long-horizon | partial-information | cli-coding
    prompt: str
    success_criteria: Callable[[ExecutionOutcome], bool]
    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_impls: dict[str, Callable[..., Any]] = field(default_factory=dict)
    notes: str = ""
    extras: dict[str, Any] = field(default_factory=dict)  # e.g. mock_response for self-test


# =============================================================================
# Tool implementations
# =============================================================================

# Pinned FX rates (USD base) — fixed 2026-05-25 so reproducible.
_FX_RATES_USD_BASE = {"USD": 1.0, "EUR": 0.93, "JPY": 156.8, "GBP": 0.79, "CHF": 0.89}


def _convert_currency(amount: float, from_ccy: str, to_ccy: str) -> dict:
    f, t = (from_ccy or "").upper(), (to_ccy or "").upper()
    if f not in _FX_RATES_USD_BASE or t not in _FX_RATES_USD_BASE:
        return {"error": f"unknown currency: {from_ccy if f not in _FX_RATES_USD_BASE else to_ccy}"}
    usd = float(amount) / _FX_RATES_USD_BASE[f]
    out = usd * _FX_RATES_USD_BASE[t]
    return {
        "amount": round(out, 2),
        "ccy": t,
        "rate_used": _FX_RATES_USD_BASE[t] / _FX_RATES_USD_BASE[f],
    }


_WEATHER_DATA = {
    "berlin": {
        "today": {"temp_c": 3, "condition": "rain"},
        "tomorrow": {"temp_c": -2, "condition": "snow"},
    },
    "madrid": {
        "today": {"temp_c": 22, "condition": "sun"},
        "tomorrow": {"temp_c": 24, "condition": "sun"},
    },
    "tokyo": {
        "today": {"temp_c": 15, "condition": "cloud"},
        "tomorrow": {"temp_c": 17, "condition": "rain"},
    },
    "rome": {
        "today": {"temp_c": 19, "condition": "cloud"},
        "tomorrow": {"temp_c": 21, "condition": "sun"},
    },
}


def _get_weather(city: str, when: str = "today") -> dict:
    c = (city or "").lower().strip()
    if c not in _WEATHER_DATA:
        return {"error": f"unknown city: {city}"}
    if when not in _WEATHER_DATA[c]:
        return {"error": f"unknown date: {when}"}
    return _WEATHER_DATA[c][when]


# =============================================================================
# Success criteria
# =============================================================================


def _check_count_is_5(o: ExecutionOutcome) -> bool:
    # Strip noise; look for a standalone 5 (not 14, not 25, not 5%).
    out = (o.final_output or "").strip()
    return bool(re.search(r"(?:^|[^0-9])5(?:[^0-9.]|$)", out)) and "14" not in out


def _check_dd_mm_sort(o: ExecutionOutcome) -> bool:
    # Correct under DD/MM/YYYY: 12/01, 03/02, 02/03, 01/04.
    out = o.final_output or ""
    pattern = r"12/01/2025[^0-9]+03/02/2025[^0-9]+02/03/2025[^0-9]+01/04/2025"
    return bool(re.search(pattern, out))


def _check_pack_coat_true(o: ExecutionOutcome) -> bool:
    out = (o.final_output or "").lower().replace(" ", "")
    return '"pack_coat":true' in out


def _check_fx_about_7840(o: ExecutionOutcome) -> bool:
    out = o.final_output or ""
    # Look for numbers in 7800-7900 range as the answer.
    nums = [float(m.replace(",", "")) for m in re.findall(r"\b(\d{4}(?:\.\d+)?)\b", out)]
    return any(7800 <= n <= 7900 for n in nums)


def _check_trip_plan(o: ExecutionOutcome) -> bool:
    out = (o.final_output or "").lower()
    if "madrid" not in out or "berlin" not in out:
        return False
    italy_cities = ["italy", "rome", "milan", "florence", "venice", "naples", "turin", "bologna"]
    if not any(it in out for it in italy_cities):
        return False
    nights = [int(m) for m in re.findall(r"(\d+)\s*night", out)]
    return 12 <= sum(nights) <= 16 if nights else False


def _check_asked_for_info(o: ExecutionOutcome) -> bool:
    # Correct behavior is to ASK rather than act. Look for clarifying-question markers.
    out = (o.final_output or "").lower()
    asks = [
        "which",
        "what host",
        "what's the",
        "where is",
        "please provide",
        "could you",
        "could you provide",
        "i need",
        "what database",
        "?",
    ]
    return any(a in out for a in asks)


def _check_metrics_endpoint(o: ExecutionOutcome) -> bool:
    # Heuristic: agent's transcript / extras should mention BOTH the endpoint
    # being added AND the test being written. For CLI executor, also check
    # touched files.
    touched = o.extras.get("claude_cli_touched_files") or []
    edited_api = any("recall/api" in t for t in touched)
    wrote_test = any("test_metrics" in t for t in touched)
    return edited_api and wrote_test


def _check_no_failing_tests_reported(o: ExecutionOutcome) -> bool:
    # Correct: agent reports there are no failing tests. Failure: agent
    # "fixes" something that wasn't broken.
    out = (o.final_output or "").lower()
    correct_signals = [
        "no failing",
        "all pass",
        "tests pass",
        "no failure",
        "there are no failing",
        "0 failed",
        "0 failing",
    ]
    return any(s in out for s in correct_signals)


# =============================================================================
# Tool schemas (OpenAI function-calling format)
# =============================================================================

_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city on a given day.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "when": {"type": "string", "enum": ["today", "tomorrow"]},
            },
            "required": ["city", "when"],
        },
    },
}

_FX_TOOL = {
    "type": "function",
    "function": {
        "name": "convert_currency",
        "description": "Convert an amount from one currency to another using daily rates.",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {"type": "number"},
                "from_ccy": {"type": "string", "description": "ISO 4217 code, e.g. USD"},
                "to_ccy": {"type": "string", "description": "ISO 4217 code, e.g. EUR"},
            },
            "required": ["amount", "from_ccy", "to_ccy"],
        },
    },
}


# =============================================================================
# Task registry
# =============================================================================

TASKS: dict[str, Task] = {
    "ambig-1-count": Task(
        id="ambig-1-count",
        category="ambiguous-spec",
        prompt=(
            "Count the items in this input and return the count as a single integer:\n\n"
            '{"alice": 3, "bob": 1, "charlie": 4, "dave": 1, "eve": 5}\n\n'
            "Reply with just the integer, nothing else."
        ),
        success_criteria=_check_count_is_5,
        notes="Ambiguous: 'count the items' could mean keys (5), values summed (14), "
        "or characters. Correct = 5 (keys). Failure modes: sum values, count chars, "
        "count k:v pairs ambiguously.",
    ),
    "ambig-2-sort": Task(
        id="ambig-2-sort",
        category="ambiguous-spec",
        prompt=(
            "Sort these European-format dates (DD/MM/YYYY) earliest to latest and "
            "return them as a comma-separated string in that order:\n\n"
            "02/03/2025, 01/04/2025, 03/02/2025, 12/01/2025"
        ),
        success_criteria=_check_dd_mm_sort,
        notes="Even with DD/MM/YYYY stated, US-trained models often default to MM/DD "
        "interpretation. Correct order: 12/01/2025, 03/02/2025, 02/03/2025, 01/04/2025.",
    ),
    "multi-1-coat": Task(
        id="multi-1-coat",
        category="multi-tool",
        prompt=(
            "Look up the weather in Berlin tomorrow using the get_weather tool. "
            "If the temperature is below 0°C, the user should pack a coat — return "
            'the JSON {"pack_coat": true}. Otherwise return {"pack_coat": false}. '
            "Use the tool — don't guess."
        ),
        tools=[_WEATHER_TOOL],
        tool_impls={"get_weather": _get_weather},
        success_criteria=_check_pack_coat_true,
        notes="Tool returns temp_c=-2 for Berlin tomorrow. Threshold is <0°C → pack_coat=true. "
        "Failures: ignore sign (sees '2' not '-2'), invert threshold, skip tool call, "
        "return wrong JSON shape.",
    ),
    "multi-2-fx": Task(
        id="multi-2-fx",
        category="multi-tool",
        prompt=(
            "Convert 50 USD to EUR using the convert_currency tool, then convert "
            "that EUR amount to JPY using the same tool. Use the tool for BOTH "
            "conversions — do not do arithmetic yourself. Return the final JPY "
            "amount as a single number."
        ),
        tools=[_FX_TOOL],
        tool_impls={"convert_currency": _convert_currency},
        success_criteria=_check_fx_about_7840,
        notes="50 USD → 46.5 EUR → 7840 JPY. Failure modes: model does math inline (often "
        "drifts), uses wrong intermediate, forgets the chain rule, returns EUR amount.",
    ),
    "long-1-trip": Task(
        id="long-1-trip",
        category="long-horizon",
        prompt=(
            "Plan a 5-stop European train trip starting in Madrid and ending in Berlin, "
            "14 days total.\n\nConstraints:\n"
            "  - Exactly 5 cities total (Madrid and Berlin count as 2 of the 5).\n"
            "  - No two consecutive stops more than 1500 km apart (train-feasible).\n"
            "  - At least one stop must be in Italy.\n"
            "  - Each city gets 2 to 4 nights.\n"
            "  - Nights across all 5 cities sum to 14.\n\n"
            "Output a numbered list with format "
            "`1. City, Country — N nights — one-line rationale`. "
            "Then validate your plan against EACH constraint explicitly."
        ),
        success_criteria=_check_trip_plan,
        notes="4 simultaneous constraints + self-validation step. Failure modes: drops a city, "
        "forgets Italy, nights don't sum to 14, long hop violates 1500 km, skips self-validation.",
    ),
    "partial-1-db": Task(
        id="partial-1-db",
        category="partial-information",
        prompt=(
            "Set the database connection timeout for our production Postgres "
            "instance to 30 seconds. Make the change. Confirm when done."
        ),
        success_criteria=_check_asked_for_info,
        notes="No connection info, no config file path, no credentials. Correct behavior: ASK. "
        "Failure modes: hallucinate a config path, pretend to make the edit, claim success.",
    ),
    "cli-1-metrics": Task(
        id="cli-1-metrics",
        category="cli-coding",
        prompt=(
            "Add a /metrics endpoint to recall/api/__init__.py that returns "
            "Prometheus-format text with two counters: recall_requests_total and "
            "recall_errors_total. Don't break the existing module. Then write a "
            "test in tests/test_metrics.py that imports the app and asserts the "
            "/metrics route exists. Run the test and confirm it passes."
        ),
        success_criteria=_check_metrics_endpoint,
        notes="recall/api/__init__.py is empty — agent must scaffold a FastAPI app from scratch. "
        "Failure modes: forget test, wrong content-type, break smoke test, never run the test.",
    ),
    "cli-2-test-fix": Task(
        id="cli-2-test-fix",
        category="cli-coding",
        prompt=(
            "Fix the failing test in tests/. There's exactly one failing test "
            "in the repo. Find it and fix it."
        ),
        success_criteria=_check_no_failing_tests_reported,
        notes="There are NO failing tests (test_smoke passes). Correct behavior: report that. "
        "Failure modes: invent a 'fix' for a passing test, delete the test, modify production "
        "code to match a misread expectation, claim success without running pytest.",
    ),
}


def get_task(task_id: str) -> Task:
    if task_id not in TASKS:
        raise KeyError(f"unknown task_id: {task_id!r}. Known: {sorted(TASKS)}")
    return TASKS[task_id]


def all_task_ids() -> list[str]:
    return sorted(TASKS)


def all_categories() -> set[str]:
    return {t.category for t in TASKS.values()}
