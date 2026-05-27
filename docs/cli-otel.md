# Capturing OpenTelemetry from Claude Code

The v1 `ClaudeCliExecutor` (`bench/runner/executors/claude_cli.py`) uses
`claude -p ... --output-format=stream-json` as its capture surface. That is
correct, complete, and easy. This doc covers the alternative — pulling
**OTel spans** directly from Claude Code — for the day you want richer trace
data than stream-json gives you (e.g. fine-grained sub-tool spans, parent /
child relationships across nested tool calls, or alignment with the same
openinference span schema that `litellm` and `langchain` already emit).

---

## What Claude Code emits

When telemetry is enabled, Claude Code uses the OpenTelemetry SDK to emit
spans for: each user turn, each model call, each tool invocation, and any
sub-agents it spawns. Span attributes follow a Claude-Code-internal naming
convention; they overlap with but are not identical to openinference
semantic conventions, so any merged Phase 2 ingestion will need a small
normalization layer.

## Enabling export

Three env vars do all the work:

```bash
export CLAUDE_CODE_ENABLE_TELEMETRY=1
export OTEL_TRACES_EXPORTER=<exporter>          # otlp | console
export OTEL_EXPORTER_OTLP_ENDPOINT=<url>        # if exporter=otlp
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf  # or http/json, grpc
```

Optional but useful:

```bash
export OTEL_SERVICE_NAME=recall-cli-harvest
export OTEL_RESOURCE_ATTRIBUTES="harvest.trial_id=42,harvest.task=cli-1-metrics"
export OTEL_LOG_LEVEL=debug                     # see exporter activity
```

You can pass these as `subprocess.Popen(env=...)` from the harvester
instead of setting them in the parent shell — the `claude` subprocess will
inherit them.

---

## Pattern 1 — In-process aiohttp OTLP receiver (richest)

Run a tiny HTTP server inside the harvester process listening on
`localhost:4318` (the OTLP/HTTP default). Each CC subprocess POSTs spans
to it; the harvester buffers them, decodes the protobuf, and merges into
the trace JSON.

**Skeleton** (~80 LOC):

```python
import asyncio
from aiohttp import web
from opentelemetry.proto.collector.trace.v1 import trace_service_pb2

class OtlpReceiver:
    def __init__(self, port: int = 4318):
        self.port = port
        self.spans: list[dict] = []
        self._runner: web.AppRunner | None = None

    async def _handle_traces(self, request: web.Request) -> web.Response:
        body = await request.read()
        req = trace_service_pb2.ExportTraceServiceRequest()
        req.ParseFromString(body)
        for rs in req.resource_spans:
            for ss in rs.scope_spans:
                for span in ss.spans:
                    self.spans.append(_span_pb_to_dict(span, rs.resource))
        return web.Response(text='{"partial_success":{}}',
                            content_type="application/json")

    async def start(self) -> None:
        app = web.Application()
        app.router.add_post("/v1/traces", self._handle_traces)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", self.port)
        await site.start()

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()

# Usage from ClaudeCliExecutor:
#   receiver = OtlpReceiver(4318)
#   await receiver.start()
#   subprocess.run([...], env={..., "OTEL_EXPORTER_OTLP_ENDPOINT": "http://127.0.0.1:4318"})
#   await receiver.stop()
#   outcome.extras["cc_otel_spans"] = receiver.spans
```

**Pros:** richest data, schema-aligned with what Phase 2 expects.
**Cons:** ~80 LOC of receiver code; need `opentelemetry-proto` to decode
protobuf (already a transitive dep via `opentelemetry-exporter-otlp-proto-http`);
port conflicts if multiple harvesters run on the same machine — randomize
the port per trial if you parallelize.

`_span_pb_to_dict` is left as an exercise; the OTel protobuf schema is at
`opentelemetry/proto/trace/v1/trace.proto`. The minimum viable mapping is
`{span_id, parent_span_id, name, start_time_unix_nano, end_time_unix_nano,
attributes, status}` — ~30 LOC including the attribute-type switch.

---

## Pattern 2 — Console exporter + stderr scrape (simplest)

Set `OTEL_TRACES_EXPORTER=console`. Claude Code will print each span as a
JSON-ish blob to stderr at end-of-batch. Capture stderr from the subprocess
and parse.

```python
proc = subprocess.run(
    [claude_bin, "-p", task.prompt, "--output-format", "stream-json"],
    env={
        **_build_child_env(),
        "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
        "OTEL_TRACES_EXPORTER": "console",
    },
    capture_output=True, text=True,
)
spans = []
for line in proc.stderr.splitlines():
    if line.startswith("{") and '"trace_id"' in line:
        try:
            spans.append(json.loads(line))
        except json.JSONDecodeError:
            continue
```

**Pros:** zero extra deps, ~10 LOC.
**Cons:** the console format is human-readable, not stable. CC may emit
multi-line span representations that span boundaries you can't cleanly
detect by `\n`. Acceptable for exploration; brittle for production.

---

## Pattern 3 — External OTLP collector (production-grade)

Run a real `otel-collector` binary as a sidecar, point CC at it via
`OTEL_EXPORTER_OTLP_ENDPOINT`, and have the collector write spans to a
local file via the `file` exporter:

```yaml
# otel-collector-config.yaml
receivers:
  otlp:
    protocols:
      http:
        endpoint: 127.0.0.1:4318
exporters:
  file:
    path: /tmp/cc-spans.jsonl
service:
  pipelines:
    traces:
      receivers: [otlp]
      exporters: [file]
```

```bash
otelcol --config otel-collector-config.yaml &
```

**Pros:** rock-solid, schema-stable, scales to many concurrent harvesters.
**Cons:** extra binary to install + supervise; overkill for solo-laptop
harvesting.

---

## Trade-off vs stream-json (what v1 actually uses)

| Property | stream-json (v1) | OTel-from-CC |
|---|---|---|
| Setup complexity | zero | low → high depending on pattern |
| Schema stability | high (CC commits to event types) | medium (CC internal attrs) |
| Span hierarchy | flat event list | true parent/child tree |
| Tool I/O detail | full inputs + outputs | full inputs + outputs |
| Per-step timing | yes | yes, sub-ms precise |
| Cost / token data | yes (`usage` events + `result.total_cost_usd`) | yes (attribute) |
| Cross-shape schema match | no (CC-specific) | partial (CC ≠ openinference) |
| Extra deps | none | `opentelemetry-proto` (Pattern 1+3) |

For Phase 1 (build a failure corpus → classify by hand → derive taxonomy),
stream-json is enough. The annotation phase doesn't need sub-ms timing or
true span trees; it needs "what did the agent do, where did it go wrong."

Switch to OTel-from-CC when:
- Phase 2's SQLite ingestion lands and you want CLI traces in the same
  span table as litellm/langchain traces.
- You add latency-budget analysis that needs sub-ms span timing.
- You want true parent/child links between sub-agents and their tools
  (e.g. CC spawning Task agents).

---

## Notes on the security model

If you adopt Pattern 1, the in-process receiver is reachable on
`127.0.0.1:<port>`. That's fine on a single-user laptop but a real
multi-user box would let any other local user POST spans to your
receiver. Mitigations: bind to a Unix socket instead of TCP, require an
`Authorization` header populated from a per-trial random token also
injected into CC's env, or use `OTEL_EXPORTER_OTLP_HEADERS`.

For Pattern 3, the same applies to the collector's listen address.
