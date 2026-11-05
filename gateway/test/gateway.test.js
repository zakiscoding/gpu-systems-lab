import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { test } from "node:test";

import { ClientRateLimiter, handle } from "../src/index.js";

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
    CLIENT_LIMITER: {
      idFromName: (name) => name,
      get: (id) => ({
        async fetch(url) {
          limits.push({ id, url });
          return Response.json(allowed ? { success: true } : { success: false, retryAfter: 17 });
        },
      }),
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
  assert.equal(res.headers.get("retry-after"), "17");
  assert.deepEqual(env.limits, [{ id: "alice", url: "http://limiter/check?limit=60" }]);
  assert.equal(env.calls.length, 0);
});

// The real limiter on fake storage, called with a controllable clock.
function limiterAt(clock) {
  const data = new Map();
  const limiter = new ClientRateLimiter({
    storage: { get: async (k) => data.get(k), put: async (k, v) => data.set(k, v) },
  });
  return async (limit = 3) => {
    const realNow = Date.now;
    Date.now = () => clock.now;
    try {
      return await (await limiter.fetch(new Request(`http://limiter/check?limit=${limit}`))).json();
    } finally {
      Date.now = realNow;
    }
  };
}

test("limiter allows exactly the limit, then refuses with an accurate Retry-After", async () => {
  const clock = { now: 1_000_000 };
  const check = limiterAt(clock);
  for (let i = 0; i < 3; i++) assert.equal((await check()).success, true);
  clock.now += 10_000;
  assert.deepEqual(await check(), { success: false, retryAfter: 50 });
});

test("limiter window slides: capacity returns only as each request ages out", async () => {
  const clock = { now: 1_000_000 };
  const check = limiterAt(clock);
  await check();
  clock.now += 30_000;
  await check();
  await check();
  assert.equal((await check()).success, false);
  clock.now += 30_001; // only the first request has aged out
  assert.equal((await check()).success, true);
  assert.equal((await check()).success, false);
});

test("refused requests don't count against the client", async () => {
  const clock = { now: 1_000_000 };
  const check = limiterAt(clock);
  for (let i = 0; i < 3; i++) await check();
  for (let i = 0; i < 10; i++) await check();
  clock.now += 60_001;
  assert.equal((await check()).success, true);
});

test("a failing rate limiter fails closed with a JSON 503", async () => {
  const env = makeEnv();
  env.CLIENT_LIMITER.get = () => ({ fetch: async () => { throw new Error("internal error"); } });
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

// env whose GPU is offline, with a stub Workers AI binding that records its input.
function offlineEnv(run = async () => ({ object: "chat.completion", model: "fallback" })) {
  const inputs = [];
  return {
    inputs,
    env: makeEnv({
      reply: () => { throw new Error("connection failed"); },
      FALLBACK_MODEL: "@cf/test/model",
      AI: {
        async run(model, input) {
          inputs.push({ model, input });
          return run(input);
        },
      },
    }),
  };
}

test("GPU responses are labelled, and online requests never touch Workers AI", async () => {
  const { env, inputs } = offlineEnv();
  env.VLLM.fetch = async () => Response.json({ ok: true });
  const res = await handle(chat(), env, silent);
  assert.equal(res.headers.get("x-gateway-backend"), "gpu");
  assert.equal(inputs.length, 0);
});

test("offline chat falls back to Workers AI and says so", async () => {
  const { env, inputs } = offlineEnv();
  const res = await handle(chat({ model: "m", messages: [{ role: "user", content: "hi" }], max_tokens: 5000, temperature: 0.5, seed: 7 }), env, silent);
  assert.equal(res.status, 200);
  assert.equal(res.headers.get("x-gateway-backend"), "workers-ai");
  assert.equal((await res.json()).model, "fallback");
  assert.deepEqual(inputs, [{
    model: "@cf/test/model",
    input: { messages: [{ role: "user", content: "hi" }], max_tokens: 100, temperature: 0.5 },
  }]);
});

test("tunnel 5xx also falls back", async () => {
  const { env, inputs } = offlineEnv();
  env.VLLM.fetch = async () => new Response("down", { status: 530 });
  assert.equal((await handle(chat(), env, silent)).status, 200);
  assert.equal(inputs[0].input.max_tokens, 100);
});

test("max_completion_tokens carries over as the fallback limit", async () => {
  const { env, inputs } = offlineEnv();
  await handle(chat({ messages: [], max_completion_tokens: 50 }), env, silent);
  assert.equal(inputs[0].input.max_tokens, 50);
});

test("fallback streams drop the non-OpenAI event, even split across chunks", async () => {
  const encoder = new TextEncoder();
  const parts = ['data: {"choices":[{"delta":{"content":"hi"}}]}\n\nda', 'ta: {"respon', 'se":"","usage":{}}\n\ndata: [DONE]\n\n'];
  const { env, inputs } = offlineEnv(async () => new ReadableStream({
    start(controller) {
      for (const part of parts) controller.enqueue(encoder.encode(part));
      controller.close();
    },
  }));
  const res = await handle(chat({ messages: [], stream: true }), env, silent);
  assert.equal(inputs[0].input.stream, true);
  assert.equal(res.headers.get("content-type"), "text/event-stream");
  const text = await res.text();
  assert.ok(text.includes('"content":"hi"'));
  assert.ok(text.includes("data: [DONE]"));
  assert.ok(!text.includes('"response"'));
});

test("requests needing unforwarded features get the 503, not a wrong answer", async () => {
  for (const extra of [{ tools: [{}] }, { functions: [{}] }, { response_format: { type: "json_object" } }]) {
    const { env, inputs } = offlineEnv();
    assert.equal((await handle(chat({ messages: [], ...extra }), env, silent)).status, 503);
    assert.equal(inputs.length, 0);
  }
});

test("only chat falls back", async () => {
  const { env, inputs } = offlineEnv();
  assert.equal((await handle(post("/v1/completions", { prompt: "hi" }), env, silent)).status, 503);
  const models = new Request(`${BASE}/v1/models`, { headers: { authorization: `Bearer ${KEY}` } });
  assert.equal((await handle(models, env, silent)).status, 503);
  assert.equal(inputs.length, 0);
});

test("no fallback when upstream rejects the gateway's key", async () => {
  const { env, inputs } = offlineEnv();
  env.VLLM.fetch = async () => new Response("no", { status: 401 });
  assert.equal((await handle(chat(), env, silent)).status, 502);
  assert.equal(inputs.length, 0);
});

test("a failing fallback returns the 503 and logs only the error code", async () => {
  const lines = [];
  const { env } = offlineEnv(async () => { throw new Error("3036: quota, input was PRIVATE-PROMPT"); });
  const res = await handle(chat(), env, (l) => lines.push(l));
  assert.equal(res.status, 503);
  assert.equal((await res.json()).error.type, "upstream_unavailable");
  assert.ok(!lines[0].includes("PRIVATE-PROMPT"));
  assert.equal(JSON.parse(lines[0]).fallbackError, "3036");
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
