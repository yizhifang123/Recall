"""CLI coding shape: spawn Claude Code headless and capture its stream-json output.

For v1, isolation is provided by running each trial inside a fresh temp
directory containing a snapshot of recall/, tests/, pyproject.toml, etc.
The agent's filesystem mutations are observable as a diff against the
snapshot and never affect the real repo.

Full OTel-export-from-CC walkthrough (writing spans to a local file) is
documented in docs/cli-otel.md; the executor uses stream-json output as the
canonical capture surface and synthesizes pseudo-spans from the events.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from bench.runner.executors.base import ShapeExecutor
from bench.runner.outcome import ExecutionException, ExecutionOutcome

# Files copied into each trial's isolated workdir.
_REPO_SNAPSHOT_FILES = [
    "recall",
    "tests",
    "pyproject.toml",
    "ruff.toml",
    "Makefile",
    ".python-version",
    "uv.lock",
    "README.md",
]

# CC tools the harvested agent is allowed to use. We intentionally scope to
# read + workdir-local edits + a narrow Bash allowlist for pytest. Without
# this, --permission-mode=bypassPermissions would give a confused/adversarial
# agent unrestricted Bash, which is dangerous even though the cwd is a
# disposable temp dir (Bash can still read $HOME, hit the network, etc.).
_ALLOWED_TOOLS = [
    "Read",
    "Write",
    "Edit",
    "Grep",
    "Glob",
    "LS",
    "Bash(uv run pytest:*)",
    "Bash(pytest:*)",
]

# Env vars the child CC subprocess is allowed to inherit. We do NOT pass
# OPENAI_API_KEY, AWS_*, or any other parent secrets — only what CC itself
# needs to function (PATH for binaries, HOME for config, ANTHROPIC_API_KEY
# for model auth, plus a few locale/term vars for clean text output).
_SAFE_ENV_PASSTHROUGH = (
    "PATH",
    "HOME",
    "USER",
    "TERM",
    "LANG",
    "LC_ALL",
    "TMPDIR",
    "SHELL",
)


def _build_child_env() -> dict[str, str]:
    """Return an explicit env dict for the CC subprocess.

    Pass only safe locale/path vars + ANTHROPIC_API_KEY. Never inherit the
    full parent env (no OPENAI_API_KEY, AWS creds, etc. — CC has no business
    seeing them and a misled agent could exfiltrate).
    """
    env = {k: os.environ[k] for k in _SAFE_ENV_PASSTHROUGH if k in os.environ}
    env["CLAUDE_CODE_DISABLE_AUTOUPDATER"] = "1"
    if "ANTHROPIC_API_KEY" in os.environ:
        env["ANTHROPIC_API_KEY"] = os.environ["ANTHROPIC_API_KEY"]
    return env


class ClaudeCliExecutor(ShapeExecutor):
    name = "cli"
    repo_root = "/Users/yida/Recall"

    def execute(self, spec, task, defaults) -> ExecutionOutcome:
        start = time.monotonic()
        outcome = ExecutionOutcome()

        claude_bin = shutil.which("claude") or os.path.expanduser("~/.local/bin/claude")
        if not Path(claude_bin).exists():
            outcome.exception = ExecutionException(
                type="FileNotFoundError",
                message=f"claude CLI not found at {claude_bin}",
                traceback="",
            )
            outcome.wall_time_s = time.monotonic() - start
            return outcome

        # Isolated workdir per trial. Symlinks aren't safe (agent writes would
        # mutate the real repo); deep-copy the snapshot.
        workdir = Path(f"/tmp/recall-cli-{int(start * 1000)}")
        workdir.mkdir(parents=True, exist_ok=True)
        try:
            for item in _REPO_SNAPSHOT_FILES:
                src = Path(self.repo_root) / item
                dst = workdir / item
                if not src.exists():
                    continue
                if src.is_dir():
                    shutil.copytree(
                        src, dst, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache")
                    )
                else:
                    shutil.copy2(src, dst)

            cmd = [
                claude_bin,
                "-p",
                task.prompt,
                "--output-format",
                "stream-json",
                "--verbose",
                "--model",
                spec.model,
                "--allowedTools",
                ",".join(_ALLOWED_TOOLS),
            ]
            outcome.extras["claude_cli_cmd"] = cmd
            outcome.extras["claude_cli_workdir"] = str(workdir)
            outcome.extras["claude_cli_allowed_tools"] = list(_ALLOWED_TOOLS)

            events: list[dict] = []
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=str(workdir),
                    text=True,
                    env=_build_child_env(),
                )
            except FileNotFoundError as exc:
                outcome.exception = ExecutionException.from_exc(exc)
                outcome.wall_time_s = time.monotonic() - start
                return outcome

            try:
                assert proc.stdout is not None
                for raw in proc.stdout:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        evt = json.loads(line)
                    except json.JSONDecodeError:
                        evt = {"type": "raw_line", "text": line}
                    events.append(evt)
                    outcome.transcript.append(evt)
                    et = evt.get("type")
                    if et == "assistant":
                        outcome.step_count += 1
                    elif et == "tool_use":
                        outcome.step_count += 1

                stderr = (proc.stderr.read() if proc.stderr else "") or ""
                proc.wait(timeout=defaults.timeout_seconds)
            except subprocess.TimeoutExpired:
                proc.kill()
                outcome.exception = ExecutionException(
                    type="TimeoutExpired",
                    message=f"claude subprocess exceeded {defaults.timeout_seconds}s",
                    traceback="",
                )
                stderr = ""

            outcome.extras["claude_cli_stderr_tail"] = stderr[-2000:]
            outcome.extras["claude_cli_returncode"] = getattr(proc, "returncode", None)
            outcome.extras["claude_cli_event_count"] = len(events)

            # Final output: prefer the explicit `result` event; fall back to last assistant text.
            result_events = [e for e in events if e.get("type") == "result"]
            if result_events:
                outcome.final_output = result_events[-1].get("result") or ""
                # Per-trial cost from the result event if present
                cost_usd = result_events[-1].get("total_cost_usd") or 0.0
                if cost_usd:
                    outcome.extras["claude_cli_cost_usd"] = float(cost_usd)
            else:
                assistants = [e for e in events if e.get("type") == "assistant"]
                if assistants:
                    msg = assistants[-1].get("message", {}) or {}
                    blocks = msg.get("content", []) or []
                    text = " ".join(
                        (b.get("text") or "")
                        for b in blocks
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                    outcome.final_output = text

            # Workdir diff: which files did the agent touch?
            touched = []
            for item in _REPO_SNAPSHOT_FILES:
                src = Path(self.repo_root) / item
                dst = workdir / item
                if src.is_file() and dst.exists():
                    if src.read_bytes() != dst.read_bytes():
                        touched.append(item)
                elif src.is_dir() and dst.exists():
                    for sub in dst.rglob("*"):
                        if sub.is_file():
                            rel = sub.relative_to(workdir)
                            orig = Path(self.repo_root) / rel
                            if not orig.exists() or orig.read_bytes() != sub.read_bytes():
                                touched.append(str(rel))
            outcome.extras["claude_cli_touched_files"] = touched
        except Exception as exc:
            outcome.exception = ExecutionException.from_exc(exc)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

        outcome.wall_time_s = time.monotonic() - start
        return outcome
