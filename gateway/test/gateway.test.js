import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { test } from "node:test";

import { handle } from "../src/index.js";

const KEY = "client-secret-key";
const BASE = "https://api.example.com";
const hash = (key) => createHash("sha256").update(key).digest("hex");

// env with a stub VLLM binding that records calls. `reply` builds each response.
function makeEnv({ reply = () => Response.json({ ok: true }), allowed = true, ...vars } = {}) {
  const calls = [];
  const limits = [];
  return {
    calls,
    limits,
    VLLM_API_KEY: "upstream-secret",
    GATEWAY_KEYS: `other=${hash("other-key")},alice=${hash(KEY)}`,
    MAX_BODY_BYTES: "1024",
    MAX_TOKENS: "100",
    RATE_LIMITER: {
      async limit(options) {
        limits.push(options);
        return { success: allowed };
      },
    },
    VLLM: {
      async fetch(url, init) {
        calls.push({ url, init });
        return reply();
      },
    },
    ...vars,
  };
}

function post(path, body, headers = {}) {
  return new Request(`${BASE}${path}`, {
    method: "POST",
    headers: { authorization: `Bearer ${KEY}`, "content-type": "application/json", ...headers },
    body: typeof body === "string" ? body : JSON.stringify(body),
  });
}

const chat = (body = { model: "m", messages: [] }, headers) => post("/v1/chat/completions", body, headers);
const silent = () => {};
const sent = (env) => JSON.parse(env.calls[0].init.body);

test("health is public and does not touch upstream", async () => {
  const env = makeEnv();
  const res = await handle(new Request(`${BASE}/health`), env, silent);
  assert.equal(res.status, 200);
  assert.deepEqual(await res.json(), { status: "ok" });
  assert.equal(env.calls.length, 0);
});

test("unlisted paths and methods return 404 without reaching upstream", async () => {
  const env = makeEnv();
  const requests = [
    post("/tokenize", {}),
    post("/v1/embeddings", {}),
    new Request(`${BASE}/metrics`),
    new Request(`${BASE}/v1/chat/completions`),
    new Request(`${BASE}/v1/models`, { method: "POST" }),
  ];
  for (const request of requests) {
    assert.equal((await handle(request, env, silent)).status, 404, request.url);
  }
  assert.equal(env.calls.length, 0);
});

test("missing, wrong and malformed keys return 401", async () => {
  const env = makeEnv();
  const headers = [{}, { authorization: "Bearer wrong" }, { authorization: KEY }, { authorization: "Basic abc" }];
  for (const h of headers) {
    const res = await handle(new Request(`${BASE}/v1/models`, { headers: h }), env, silent);
    assert.equal(res.status, 401);
    assert.equal(res.headers.get("www-authenticate"), "Bearer");
  }
  assert.equal(env.calls.length, 0);
});

test("empty or unset GATEWAY_KEYS rejects everyone", async () => {
  for (const GATEWAY_KEYS of ["", undefined, ",", "broken", "=abc"]) {
    const env = makeEnv({ GATEWAY_KEYS });
    assert.equal((await handle(chat(), env, silent)).status, 401);
  }
});

test("a revoked key stops working", async () => {
  const env = makeEnv({ GATEWAY_KEYS: `other=${hash("other-key")}` });
  assert.equal((await handle(chat(), env, silent)).status, 401);
});

test("rate limit is keyed by client name and returns 429", async () => {
  const env = makeEnv({ allowed: false });
  const res = await handle(chat(), env, silent);
  assert.equal(res.status, 429);
  assert.equal(res.headers.get("retry-after"), "60");
  assert.deepEqual(env.limits, [{ key: "alice" }]);
  assert.equal(env.calls.length, 0);
});

test("a failing rate limiter fails closed with a JSON 503", async () => {
  const env = makeEnv();
  env.RATE_LIMITER.limit = async () => { throw new Error("internal error"); };
  const res = await handle(chat(), env, silent);
  assert.equal(res.status, 503);
  assert.equal((await res.json()).error.type, "rate_limiter_unavailable");
  assert.equal(env.calls.length, 0);
});

test("bodies over the cap return 413, with or without Content-Length", async () => {
  const large = JSON.stringify({ prompt: "x".repeat(2048) });
  const env = makeEnv();
  assert.equal((await handle(post("/v1/completions", large), env, silent)).status, 413);

  const chunked = new Request(`${BASE}/v1/completions`, {
    method: "POST",
    duplex: "half",
    headers: { authorization: `Bearer ${KEY}` },
    body: new ReadableStream({
      start(controller) {
        controller.enqueue(new TextEncoder().encode(large));
        controller.close();
      },
    }),
  });
  assert.equal((await handle(chunked, env, silent)).status, 413);
  assert.equal(env.calls.length, 0);
});

test("non-JSON and non-object bodies return 400", async () => {
  const env = makeEnv();
  for (const body of ["not json", "null", "[1]", "3"]) {
    assert.equal((await handle(chat(body), env, silent)).status, 400, body);
  }
  assert.equal(env.calls.length, 0);
});

test("token limits are clamped, and small ones are left alone", async () => {
  const cases = [
    [{ max_tokens: 5000 }, { max_tokens: 100 }],
    [{ max_tokens: 50 }, { max_tokens: 50 }],
    [{ max_completion_tokens: 5000 }, { max_completion_tokens: 100 }],
    [{}, { max_tokens: 100 }],
  ];
  for (const [input, expected] of cases) {
    const env = makeEnv();
    await handle(chat(input), env, silent);
    assert.deepEqual(sent(env), expected);
  }
});

test("completions keeps vLLM's default when no limit is set", async () => {
  const env = makeEnv();
  await handle(post("/v1/completions", { prompt: "hi" }), env, silent);
  assert.deepEqual(sent(env), { prompt: "hi" });
});

test("client credentials and other headers are not forwarded", async () => {
  const env = makeEnv();
  await handle(chat(undefined, { cookie: "a=b", "cf-connecting-ip": "1.2.3.4", accept: "text/event-stream" }), env, silent);
  assert.equal(env.calls[0].url, "http://vllm/v1/chat/completions");
  assert.deepEqual(env.calls[0].init.headers, {
    authorization: "Bearer upstream-secret",
    accept: "text/event-stream",
    "content-type": "application/json",
  });
});

test("the query string is not forwarded", async () => {
  const env = makeEnv();
  await handle(new Request(`${BASE}/v1/models?x=1`, { headers: { authorization: `Bearer ${KEY}` } }), env, silent);
  assert.equal(env.calls[0].url, "http://vllm/v1/models");
  assert.equal(env.calls[0].init.method, "GET");
});

test("streaming responses pass through incrementally", async () => {
  const encoder = new TextEncoder();
  let controller;
  const stream = new ReadableStream({ start: (c) => { controller = c; } });
  const env = makeEnv({
    reply: () => new Response(stream, {
      headers: { "content-type": "text/event-stream", "set-cookie": "leak=1", server: "uvicorn" },
    }),
  });
  const res = await handle(chat({ stream: true }), env, silent);
  assert.equal(res.status, 200);
  assert.equal(res.headers.get("content-type"), "text/event-stream");
  assert.equal(res.headers.get("set-cookie"), null);
  assert.equal(res.headers.get("server"), null);

  // The first chunk must arrive while upstream is still open.
  const reader = res.body.getReader();
  controller.enqueue(encoder.encode("data: one\n\n"));
  assert.equal(new TextDecoder().decode((await reader.read()).value), "data: one\n\n");
  controller.enqueue(encoder.encode("data: two\n\n"));
  controller.close();
  assert.equal(new TextDecoder().decode((await reader.read()).value), "data: two\n\n");
  assert.equal((await reader.read()).done, true);
});

test("offline upstream returns 503 when the binding throws", async () => {
  const env = makeEnv({ reply: () => { throw new Error("connection failed"); } });
  const res = await handle(chat(), env, silent);
  assert.equal(res.status, 503);
  assert.equal(res.headers.get("retry-after"), "60");
  assert.equal((await res.json()).error.type, "upstream_unavailable");
});

test("tunnel 5xx statuses become 503 and hide the upstream body", async () => {
  for (const status of [502, 503, 504, 524, 530]) {
    const env = makeEnv({ reply: () => new Response("<html>tunnel error</html>", { status }) });
    const res = await handle(chat(), env, silent);
    assert.equal(res.status, 503);
    assert.equal((await res.json()).error.type, "upstream_unavailable");
  }
});

test("upstream 401 is a gateway fault, not the client's", async () => {
  const env = makeEnv({ reply: () => new Response("unauthorized", { status: 401 }) });
  const res = await handle(chat(), env, silent);
  assert.equal(res.status, 502);
  assert.equal((await res.json()).error.type, "gateway_misconfigured");
});

test("vLLM validation errors reach the client unchanged", async () => {
  const env = makeEnv({ reply: () => Response.json({ error: "bad model" }, { status: 400 }) });
  const res = await handle(chat(), env, silent);
  assert.equal(res.status, 400);
  assert.deepEqual(await res.json(), { error: "bad model" });
});

test("logs hold metadata only, never prompts or keys", async () => {
  const lines = [];
  await handle(chat({ messages: [{ role: "user", content: "PRIVATE-PROMPT" }] }), makeEnv(), (l) => lines.push(l));
  assert.equal(lines.length, 1);
  assert.ok(!lines[0].includes("PRIVATE-PROMPT"));
  assert.ok(!lines[0].includes(KEY));
  assert.ok(!lines[0].includes("upstream-secret"));
  const entry = JSON.parse(lines[0]);
  assert.equal(entry.client, "alice");
  assert.equal(entry.path, "/v1/chat/completions");
  assert.equal(entry.status, 200);
});
