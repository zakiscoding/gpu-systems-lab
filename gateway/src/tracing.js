// Minimal OpenTelemetry tracing for the Worker: W3C trace context in, spans out as OTLP/JSON.
//
//   client --traceparent?--> Worker [SERVER span] --traceparent--> vLLM [vLLM's own spans]
//                                   \-- [CLIENT span] upstream call (vllm or workers-ai)
//
// No SDK: the Workers runtime has no Node OpenTelemetry SDK, and two span kinds plus one
// export call don't need one. Spans are exported after the response is sent (ctx.waitUntil)
// through the OTEL binding, a Workers VPC Service reaching the iMac's collector over the tunnel.
// Without the binding, tracing is off and nothing is buffered.
//
// Spans carry routes, statuses, the client's key *name* and the backend. Never bodies or keys.
//
// Clock: Workers freeze Date.now() between I/O, so a span's duration is the time spent waiting
// on I/O (the upstream fetch), not CPU time. That is the part worth measuring here.

const KIND = { SERVER: 2, CLIENT: 3 };
const STATUS = { UNSET: 0, OK: 1, ERROR: 2 };
const TRACEPARENT = /^00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$/;

function randomHex(bytes) {
  const buf = crypto.getRandomValues(new Uint8Array(bytes));
  return [...buf].map((b) => b.toString(16).padStart(2, "0")).join("");
}

// Returns { traceId, spanId, flags } for a valid version-00 header, else null.
// All-zero IDs are invalid per the spec and start a new trace instead.
export function parseTraceparent(header) {
  const match = TRACEPARENT.exec((header ?? "").trim().toLowerCase());
  if (!match || /^0+$/.test(match[1]) || /^0+$/.test(match[2])) return null;
  return { traceId: match[1], spanId: match[2], flags: match[3] };
}

const nowNanos = () => BigInt(Date.now()) * 1000000n;

class Span {
  constructor(tracer, name, kind, parentSpanId, attributes) {
    this.tracer = tracer;
    this.name = name;
    this.kind = kind;
    this.traceId = tracer.traceId;
    this.spanId = randomHex(8);
    this.parentSpanId = parentSpanId;
    this.attributes = { ...attributes };
    this.start = nowNanos();
    this.end = null;
    this.status = STATUS.UNSET;
  }

  set(attributes) {
    for (const [key, value] of Object.entries(attributes)) {
      if (value !== null && value !== undefined) this.attributes[key] = value;
    }
    return this;
  }

  // W3C header for a request made *from* this span.
  traceparent() {
    return `00-${this.traceId}-${this.spanId}-${this.tracer.flags}`;
  }

  finish(error = false) {
    if (this.end === null) {
      this.end = nowNanos();
      if (error) this.status = STATUS.ERROR;
      this.tracer.finished.push(this);
    }
  }
}

export class Tracer {
  // `incoming` is the request's traceparent header; a valid one is continued.
  constructor(incoming, resource = { "service.name": "vllm-gateway" }) {
    const parent = parseTraceparent(incoming);
    this.traceId = parent?.traceId ?? randomHex(16);
    this.remoteParent = parent?.spanId ?? null;
    this.flags = parent?.flags ?? "01";
    this.resource = resource;
    this.finished = [];
  }

  server(name, attributes) {
    return new Span(this, name, KIND.SERVER, this.remoteParent, attributes);
  }

  client(parent, name, attributes) {
    return new Span(this, name, KIND.CLIENT, parent.spanId, attributes);
  }

  // OTLP/JSON (ExportTraceServiceRequest). IDs are hex strings and timestamps decimal strings,
  // as the OTLP/HTTP JSON encoding specifies.
  toOtlp() {
    const value = (v) => (typeof v === "number"
      ? (Number.isInteger(v) ? { intValue: String(v) } : { doubleValue: v })
      : typeof v === "boolean" ? { boolValue: v } : { stringValue: String(v) });
    const attrs = (obj) => Object.entries(obj).map(([key, v]) => ({ key, value: value(v) }));
    return {
      resourceSpans: [{
        resource: { attributes: attrs(this.resource) },
        scopeSpans: [{
          scope: { name: "vllm-gateway", version: "1" },
          spans: this.finished.map((s) => ({
            traceId: s.traceId,
            spanId: s.spanId,
            ...(s.parentSpanId ? { parentSpanId: s.parentSpanId } : {}),
            name: s.name,
            kind: s.kind,
            startTimeUnixNano: String(s.start),
            endTimeUnixNano: String(s.end),
            attributes: attrs(s.attributes),
            status: { code: s.status },
          })),
        }],
      }],
    };
  }

  // Sends finished spans to the collector. Never throws: tracing must not break requests.
  async export(binding) {
    if (!binding || this.finished.length === 0) return false;
    try {
      const res = await binding.fetch("http://otel-collector/v1/traces", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(this.toOtlp()),
      });
      await res.body?.cancel();
      return res.ok;
    } catch {
      return false;
    }
  }
}
