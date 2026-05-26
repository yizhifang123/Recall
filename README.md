# Recall

> Flight recorder and black-box analyzer for LLM agents.

Recall records agent runs, replays them deterministically, and classifies failures against a unified taxonomy drawn from published research (MAST, AgentRx, AgentAtlas, Butterfly Effects). Agent-shape agnostic: single prompts, single-agent + tools, multi-agent swarms, CLI coding agents. Local-first, no hosted backend.

## Install

```bash
# Not yet on PyPI. Install from source:
uv pip install git+https://github.com/yizhifang123/Recall.git

# Once published:
# pip install recall-trace
```

## Quickstart

Coming in Phase 1.

## Status

Pre-alpha. See [`docs/decisions.md`](docs/decisions.md) for the design log and [`docs/observed-failures.md`](docs/observed-failures.md) for the running failure catalog.

## License

MIT — see [LICENSE](LICENSE).
