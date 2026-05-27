# Roadmap

Items deliberately deferred out of the v1 scope. Each entry: what, why
deferred, when to reconsider.

## Multi-agent framework coverage

- **AutoGen 0.4** — popular for chat-based multi-agent, well-documented
  role-drift and repeated-work failures. Deferred because v1 multi-agent
  trace harvesting uses LangGraph only (see [D-001-A](docs/decisions.md)).
  Supporting multiple multi-agent frameworks in v1 doubles the executor
  and trace-normalization surface for marginal coverage gain. Reconsider
  after Phase 2 ingestion is stable and the unified taxonomy has been
  validated against the LangGraph corpus.
- **CrewAI** — role-based hierarchical agents, known for role drift and
  over-delegation failures. Same deferral logic as AutoGen.

## PyPI dist name

- Attempt **PEP 541 abandonment claim** on the bare `recall` PyPI name
  (currently squatted by an abandoned 2014 RPC framework — no upload
  since 2014, no contact). Reconsider at v0.2 launch. If successful,
  add `recall` as an alias dist that re-exports `recall-trace`'s
  contents.

## Domain

- Neither `recall.com` nor `recall.dev` was available at name-lock time
  (2026-05-25). Candidates to evaluate: `recall.tools`, `recall.run`,
  `recallhq.dev`. Defer until docs site goes live.
