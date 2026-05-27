# Decision Log

Lightweight ADR-style log for Recall. One entry per material decision.
Each entry: ID, date, status, context, decision, consequences.

---

## D-000 — Project named Recall

- **Date:** 2026-05-25
- **Status:** Accepted

### Context

Bootstrapping a 13-week project: a flight recorder and black-box analyzer for
LLM agents. The toolkit needs a short, memorable name that's lockable across
GitHub, PyPI, a primary domain, and Twitter.

Shortlist (in preference order): Recall, Bisect, Postmortem, Rewind, with
recallai and bisectllm as fallbacks.

### Availability findings (2026-05-25)

| Name       | PyPI                       | GitHub          | .com            | .dev   |
|------------|----------------------------|-----------------|-----------------|--------|
| recall     | taken (abandoned, 2014)    | taken (user)    | taken (1999)    | taken  |
| bisect     | available (but stdlib name)| taken (user)    | taken (2025)    | taken  |
| postmortem | taken (active, 2021)       | taken (user)    | taken (2000)    | taken  |
| rewind     | taken (abandoned, 2012)    | taken (user)    | taken (1996)    | taken  |
| recallai   | taken (Recall.ai SDK)      | taken (org, 76) | taken (2017)    | taken  |
| bisectllm  | available                  | available       | available       | avail. |

Twitter handle availability not reliably verifiable from automation; deferred.

### Decision

- Project name: **Recall**.
- PyPI distribution name: **`recall-trace`** (the bare `recall` name on PyPI
  is squatted by an abandoned 2014 RPC framework; namespacing the
  distribution preserves the brand without blocking on PEP 541 abandonment,
  which can take months).
- Import name in code stays as `import recall`.
- GitHub: `github.com/yizhifang123/Recall` (personal namespace; org-level
  `github.com/recall` was unavailable but not required for a single-author
  project at this stage).
- Domain: deferred. Neither `recall.com` nor `recall.dev` is available.
  Revisit at launch — likely candidates: `recall.tools`, `recall.run`,
  `recallhq.dev`, or accept project-page hosting on GitHub.
- Twitter handle: to be claimed manually post-bootstrap.

### Consequences

- Users install with `pip install recall-trace`, import as `import recall`.
  Mirrors prior art like `python-dateutil` → `import dateutil`.
- We accept the risk that `recall-trace` looks slightly less clean than
  bare `recall` would, in exchange for keeping the brand intact.
- If PyPI `recall` later becomes claimable (PEP 541), we can migrate the
  distribution name without breaking import compatibility.
- Future docs and the README must consistently surface `recall-trace` as
  the installable name; never let the bare `recall` PyPI name leak into
  install instructions.

---

## D-001-A — Multi-agent trace harvesting uses LangGraph

- **Date:** 2026-05-25
- **Status:** Accepted

### Context

Phase 1 trace harvesting needs a multi-agent representative shape (~7 of the
30 target seed traces). The candidate frameworks are AutoGen 0.4 (popular,
well-documented role-drift and repeated-work failures), CrewAI (role-based
hierarchical agents, known for over-delegation), and LangGraph (`StateGraph`
+ cycles + checkpointers, known for state corruption and conditional-edge
mispicks).

### Decision

Use **LangGraph** for the multi-agent shape in v1. Concretely: a
planner-executor-verifier graph built with `StateGraph`, at least three
nodes, conditional edges between them, and a checkpointer (`MemorySaver`
is sufficient for harvesting).

### Rationale

- Aligns Phase 1 trace harvesting with the primary multi-agent framework
  integration planned for Phase 2.4, so the failure modes we collect now
  match the framework we'll instrument deeply later.
- Avoids scope creep — supporting multiple multi-agent frameworks in v1
  doubles the executor surface and the trace-normalization work for
  marginal coverage gain.

### Consequences

- AutoGen and CrewAI multi-agent harvesting are out of scope for v1 and
  tracked on `ROADMAP.md`.
- LangGraph traces use `openinference-instrumentation-langchain` for OTel
  capture (LangGraph builds on LangChain primitives).
- Multi-agent trace coverage reflects LangGraph's failure modes
  specifically. Failures induced by AutoGen's role-based chat ceremony or
  CrewAI's hierarchical delegation pattern are NOT represented in v1 seed
  traces; the taxonomy derived from this corpus must explicitly note that
  limitation.
