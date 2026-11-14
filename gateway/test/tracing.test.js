import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { test } from "node:test";

import { handle } from "../src/index.js";
import { parseTraceparent, Tracer } from "../src/tracing.js";

const KEY = "client-secret-key";
const PROMPT = "a private prompt that must never be exported";
const BASE = "https://api.example.com";
const INCOMING = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
const hash = (key) => createHash("sha256").update(key).digest("hex");
const silent = () => {};

// env with stub VLLM, AI and OTEL bindings. `exports` collects the OTLP bodies the Worker sends.
function makeEnv({ reply = () => Response.json({ ok: true }), otel = true, otelReply, ai } = {}) {
  const calls = [];
  const exports = [];
  return {
    calls,
    exports,
    VLLM_API_KEY: "upstream-secret",
    GATEWAY_KEYS: `alice=${hash(KEY)}`,
    FALLBACK_MODEL: "@cf/test/model",
    CLIENT_LIMITER: {
      idFromName: (name) => name,
      get: () => ({ fetch: async () => Response.json({ success: true }) }),
    },
    VLLM: {
      async fetch(url, init) {
        calls.push({ url, init });
        return reply();
      },
    },
    AI: ai ?? { run: async () => ({ choices: [{ message: { content: "fallback" } }] }) },
    ...(otel ? {
      OTEL: {
        async fetch(url, init) {
          exports.push({ url, body: JSON.parse(init.body) });
          return otelReply ? otelReply() : new Response(null, { status: 200 });
        },
      },
    } : {}),
  };
}

function chat(headers = {}) {
  return new Request(`${BASE}/v1/chat/completions`, {
    method: "POST",
    headers: { authorization: `Bearer ${KEY}`, "content-type": "application/json", ...headers },
    body: JSON.stringify({ model: "m", messages: [{ role: "user", content: PROMPT }] }),
  });
}

// Runs a request with a ctx whose waitUntil promises are awaited, like the runtime does.
async function run(request, env) {
  const pending = [];
  const res = await handle(request, env, silent, { waitUntil: (p) => pending.push(p) });
  await Promise.all(pending);
  return { res, pending };
}

const spansOf = (env) => env.exports.flatMap((e) => e.body.resourceSpans[0].scopeSpans[0].spans);
const attr = (span, key) => {
  const a = span.attributes.find((x) => x.key === key);
  return a && (a.value.stringValue ?? Number(a.value.intValue));
};

test("parseTraceparent accepts valid headers and rejects bad or all-zero ones", () => {
  assert.deepEqual(parseTraceparent(INCOMING), {
    traceId: "4bf92f3577b34da6a3ce929d0e0e4736", spanId: "00f067aa0ba902b7", flags: "01",
  });
  for (const bad of [null, "", "garbage", "01-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
    "00-00000000000000000000000000000000-00f067aa0ba902b7-01",
    "00-4bf92f3577b34da6a3ce929d0e0e4736-0000000000000000-01"]) {
    assert.equal(parseTraceparent(bad), null, String(bad));
  }
});

test("a chat request exports a server span and a child client span in one trace", async () => {
  const env = makeEnv();
  const { res, pending } = await run(chat(), env);
  assert.equal(res.status, 200);
  assert.equal(pending.length, 1);
  assert.equal(env.exports.length, 1);
  assert.equal(env.exports[0].url, "http://otel-collector/v1/traces");

  const spans = spansOf(env);
  const server = spans.find((s) => s.kind === 2);
  const client = spans.find((s) => s.kind === 3);
  assert.equal(spans.length, 2);
  assert.equal(server.name, "POST /v1/chat/completions");
  assert.equal(attr(server, "http.route"), "/v1/chat/completions");
  assert.equal(attr(server, "http.response.status_code"), 200);
  assert.equal(attr(server, "gateway.backend"), "gpu");
  assert.equal(attr(server, "gateway.client"), "alice");
  assert.equal(server.parentSpanId, undefined);
  assert.equal(client.traceId, server.traceId);
  assert.equal(client.parentSpanId, server.spanId);
  assert.equal(attr(client, "url.path"), "/v1/chat/completions");
  assert.ok(BigInt(server.endTimeUnixNano) >= BigInt(server.startTimeUnixNano));
  assert.equal(res.headers.get("x-trace-id"), server.traceId);
});

test("vLLM receives a traceparent pointing at the client span", async () => {
  const env = makeEnv();
  await run(chat(), env);
  const client = spansOf(env).find((s) => s.kind === 3);
  assert.equal(env.calls[0].init.headers.traceparent, `00-${client.traceId}-${client.spanId}-01`);
  assert.equal(env.calls[0].init.headers.authorization, "Bearer upstream-secret");
});

test("an incoming traceparent is continued", async () => {
  const env = makeEnv();
  await run(chat({ traceparent: INCOMING }), env);
  const server = spansOf(env).find((s) => s.kind === 2);
  assert.equal(server.traceId, "4bf92f3577b34da6a3ce929d0e0e4736");
  assert.equal(server.parentSpanId, "00f067aa0ba902b7");
  assert.ok(env.calls[0].init.headers.traceparent.startsWith("00-4bf92f3577b34da6a3ce929d0e0e4736-"));
});

test("an invalid incoming traceparent starts a new trace", async () => {
  const env = makeEnv();
  await run(chat({ traceparent: "00-zz-yy-01" }), env);
  const server = spansOf(env).find((s) => s.kind === 2);
  assert.match(server.traceId, /^[0-9a-f]{32}$/);
  assert.equal(server.parentSpanId, undefined);
});

test("GPU unreachable: the vLLM span is an error and the fallback has its own span", async () => {
  const env = makeEnv({ reply: () => { throw new Error("tunnel down"); } });
  const { res } = await run(chat(), env);
  assert.equal(res.status, 200);
  const spans = spansOf(env);
  assert.equal(spans.length, 3);
  const server = spans.find((s) => s.kind === 2);
  const vllm = spans.find((s) => s.name === "POST vllm");
  const ai = spans.find((s) => s.name === "workers-ai run");
  assert.equal(attr(server, "gateway.backend"), "workers-ai");
  assert.equal(vllm.status.code, 2);
  assert.equal(ai.status.code, 0);
  assert.equal(ai.parentSpanId, server.spanId);
});

test("upstream 5xx marks the client span as an error", async () => {
  const env = makeEnv({ reply: () => new Response("bad", { status: 500 }) });
  await run(chat(), env);
  const vllm = spansOf(env).find((s) => s.name === "POST vllm");
  assert.equal(attr(vllm, "http.response.status_code"), 500);
  assert.equal(vllm.status.code, 2);
});

test("exported spans contain no prompt, client key or upstream key", async () => {
  const env = makeEnv();
  await run(chat(), env);
  const dump = JSON.stringify(env.exports);
  for (const secret of [PROMPT, KEY, hash(KEY), "upstream-secret"]) {
    assert.ok(!dump.includes(secret), secret);
  }
});

test("unknown routes and bad keys are traced with a bounded route label", async () => {
  const env = makeEnv();
  await run(new Request(`${BASE}/v1/embeddings`, { method: "POST" }), env);
  await run(new Request(`${BASE}/v1/models`), env);
  const [unknown, unauthorised] = spansOf(env);
  assert.equal(attr(unknown, "http.route"), "unknown");
  assert.equal(attr(unknown, "http.response.status_code"), 404);
  assert.equal(attr(unauthorised, "http.route"), "/v1/models");
  assert.equal(attr(unauthorised, "http.response.status_code"), 401);
});

test("/health is not traced", async () => {
  const env = makeEnv();
  const { res } = await run(new Request(`${BASE}/health`), env);
  assert.equal(res.status, 200);
  assert.equal(env.exports.length, 0);
  assert.equal(res.headers.get("x-trace-id"), null);
});

test("without the OTEL binding nothing is traced or propagated", async () => {
  const env = makeEnv({ otel: false });
  const { res, pending } = await run(chat(), env);
  assert.equal(res.status, 200);
  assert.equal(pending.length, 0);
  assert.equal(env.calls[0].init.headers.traceparent, undefined);
  assert.equal(res.headers.get("x-trace-id"), null);
});

test("a failing collector never affects the response", async () => {
  for (const otelReply of [() => { throw new Error("tunnel down"); }, () => new Response("no", { status: 503 })]) {
    const env = makeEnv({ otelReply });
    const { res } = await run(chat(), env);
    assert.equal(res.status, 200);
    assert.deepEqual(await res.json(), { ok: true });
  }
});

test("export reports success and failure without throwing", async () => {
  const tracer = new Tracer(null);
  tracer.server("s", {}).finish();
  assert.equal(await tracer.export(null), false);
  assert.equal(await tracer.export({ fetch: async () => new Response(null, { status: 200 }) }), true);
  assert.equal(await tracer.export({ fetch: async () => { throw new Error("x"); } }), false);
});
