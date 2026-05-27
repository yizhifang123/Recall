# Security model

Recall is a local-first developer tool. There is no hosted backend, no
shared infrastructure, no user accounts. The threat model below is
scoped accordingly — a solo developer running the harvester on their
own laptop against API endpoints they own keys for.

## Key handling

- API keys live in `/Users/<you>/Recall/.env` or `.env.local`
  (both gitignored). The harvester loads them at startup via
  `python-dotenv`.
- Keys are never logged, never serialized into trace JSON, never sent
  to any endpoint other than the configured provider (OpenAI /
  Anthropic via litellm).
- The Claude Code subprocess executor (`bench/runner/executors/claude_cli.py`)
  builds an explicit child env: only `PATH`, `HOME`, `USER`, `TERM`,
  `LANG`, `LC_ALL`, `TMPDIR`, `SHELL`, `CLAUDE_CODE_DISABLE_AUTOUPDATER`,
  and `ANTHROPIC_API_KEY` cross the process boundary. `OPENAI_API_KEY`
  and any other parent-process secrets are **deliberately not inherited**.

## Claude Code subprocess executor — the load-bearing decision

When the CLI shape runs, the harvester invokes Claude Code headless
(`claude -p`) in a fresh `/tmp/recall-cli-<timestamp>/` deep-copy of the
repo snapshot. CC runs with `--permission-mode bypassPermissions`, which
means it can use any tool — including arbitrary Bash — without
prompting.

This is deliberate. We tried `--allowedTools` to restrict the surface,
but live testing showed CC's `--allowedTools` in headless `-p` mode does
NOT enforce a restrict-to allowlist — it's an additional-allow list, not
a constraint. Without a custom `--permission-prompt-tool` script (~30
LOC of bash + JSON parsing), CC's headless mode is effectively
permission-free.

For Phase 1 trace harvesting on a developer laptop, the actual security
model is:

1. **Workdir isolation.** Each trial runs in a fresh, disposable temp
   dir. Mutations never reach the real repo. The dir is `shutil.rmtree`'d
   on trial exit. Touched files are recorded in `extras.claude_cli_touched_files`
   for trace analysis.
2. **Env scrub.** OPENAI / AWS / other unrelated secrets do not cross
   into the CC subprocess.
3. **Bounded duration.** `defaults.timeout_seconds` kills the subprocess
   if it runs over budget.

### Residual risks (accepted)

- CC can run arbitrary Bash inside the workdir. That Bash can read
  `$HOME`, make outbound network calls, write to `TMPDIR`, exhaust
  disk, or read any file the parent user can read. The temp-dir cwd
  bounds *mutations* but not *reads*.
- CC has access to `ANTHROPIC_API_KEY` (it needs it to function); a
  misled CC could in principle log it. Anthropic's own model alignment
  is the mitigation; we treat this risk as low for benign coding tasks.
- The Anthropic rate limit on the trial's model becomes a denial-of-service
  surface if you run many CLI trials concurrently. The harvester is
  single-threaded so this is naturally bounded.

### When to harden further

If you reuse `ClaudeCliExecutor` outside Phase 1 trace harvesting —
in CI, against untrusted prompts, on a shared box, or for any production
use — replace `--permission-mode bypassPermissions` with a real
`--permission-prompt-tool` script that classifies tool requests and
denies anything outside an explicit allowlist. The CC docs document
the JSON protocol the script must implement.

## Reporting

This is a personal-project codebase with no public deployment. Report
issues by opening a GitHub issue at
[`github.com/yizhifang123/Recall`](https://github.com/yizhifang123/Recall/issues).
