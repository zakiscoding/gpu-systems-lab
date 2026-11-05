// Public OpenAI-compatible gateway for the vLLM server on zak-gpu-node.
//
//   client --gateway key--> this Worker --VPC Service binding--> Tunnel --> vllm:8000
//                                       \--offline, chat only--> Workers AI
//
// Clients never see VLLM_API_KEY, and only the routes below reach vLLM.

const DEFAULT_MAX_BODY_BYTES = 262144;
const DEFAULT_MAX_TOKENS = 1024;
const DEFAULT_RATE_LIMIT = 60;
const RATE_LIMIT_PERIOD_SECONDS = 60;
const OFFLINE_RETRY_SECONDS = 60;

const CHAT_PATH = "/v1/chat/completions";
const UPSTREAM_ROUTES = new Set([
  `POST ${CHAT_PATH}`,
  "POST /v1/completions",
  "GET /v1/models",
]);
const FORWARDED_RESPONSE_HEADERS = ["content-type", "cache-control", "x-request-id"];

// OpenAI-style error body, so SDK clients surface the message.
function errorResponse(status, type, message, headers = {}) {
  return new Response(JSON.stringify({ error: { message, type } }), {
    status,
    headers: { "content-type": "application/json", ...headers },
  });
}

async function sha256Hex(text) {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

function equalConstantTime(a, b) {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

// GATEWAY_KEYS is "name=sha256hex,name=sha256hex". Returns the client name or null.
// Checks every entry, without stopping at a match.
async function authenticate(request, gatewayKeys) {
  const match = /^Bearer\s+(\S+)$/i.exec(request.headers.get("authorization") ?? "");
  if (!match) return null;
  const hash = await sha256Hex(match[1]);
  let client = null;
  for (const entry of (gatewayKeys ?? "").split(",")) {
    const [name, stored] = entry.trim().split("=");
    if (name && stored && equalConstantTime(hash, stored.toLowerCase())) client = name;
  }
  return client;
}

// Returns the body bytes, or null past the cap. Content-Length can be absent
// (chunked uploads), so the running total is what enforces the cap.
async function readCapped(request, maxBytes) {
  if (Number(request.headers.get("content-length") ?? 0) > maxBytes) return null;
  if (!request.body) return new Uint8Array();
  const reader = request.body.getReader();
  const chunks = [];
  let total = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    total += value.length;
    if (total > maxBytes) {
      await reader.cancel();
      return null;
    }
    chunks.push(value);
  }
  const body = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    body.set(chunk, offset);
    offset += chunk.length;
  }
  return body;
}

// Chat has no server-side default limit, so an absent limit gets the cap there.
// Completions defaults to 16 tokens, so it only needs clamping.
function clampTokens(payload, path, maxTokens) {
  for (const field of ["max_tokens", "max_completion_tokens"]) {
    if (typeof payload[field] === "number" && payload[field] > maxTokens) {
      payload[field] = maxTokens;
    }
  }
  if (path === CHAT_PATH && payload.max_tokens == null && payload.max_completion_tokens == null) {
    payload.max_tokens = maxTokens;
  }
}

// Builds the upstream request, or returns an error Response for a bad body.
async function buildUpstreamInit(request, path, env) {
  const headers = { authorization: `Bearer ${env.VLLM_API_KEY}` };
  const accept = request.headers.get("accept");
  if (accept) headers.accept = accept;
  if (request.method === "GET") return { method: "GET", headers };

  const maxBytes = Number(env.MAX_BODY_BYTES) || DEFAULT_MAX_BODY_BYTES;
  const bytes = await readCapped(request, maxBytes);
  if (bytes === null) {
    return errorResponse(413, "request_too_large", `Request body exceeds ${maxBytes} bytes.`);
  }
  let payload;
  try {
    payload = JSON.parse(new TextDecoder().decode(bytes));
  } catch {
    return errorResponse(400, "invalid_request_error", "Request body must be JSON.");
  }
  if (payload === null || typeof payload !== "object" || Array.isArray(payload)) {
    return errorResponse(400, "invalid_request_error", "Request body must be a JSON object.");
  }
  clampTokens(payload, path, Number(env.MAX_TOKENS) || DEFAULT_MAX_TOKENS);
  headers["content-type"] = "application/json";
  return { method: "POST", headers, body: JSON.stringify(payload) };
}

function upstreamUnavailable() {
  return errorResponse(
    503,
    "upstream_unavailable",
    "The GPU node is offline. Try again later.",
    { "retry-after": String(OFFLINE_RETRY_SECONDS) },
  );
}

// Workers AI ends its streams with a {"response":...} event that has no
// `choices`, which breaks OpenAI clients. Everything else is already OpenAI format.
function dropNonOpenAIEvents(stream) {
  const decoder = new TextDecoder();
  const encoder = new TextEncoder();
  const keep = (line) => !line.startsWith('data: {"response"');
  let partial = "";
  return stream.pipeThrough(new TransformStream({
    transform(chunk, controller) {
      const lines = (partial + decoder.decode(chunk, { stream: true })).split("\n");
      partial = lines.pop();
      const kept = lines.filter(keep);
      if (kept.length) controller.enqueue(encoder.encode(kept.join("\n") + "\n"));
    },
    flush(controller) {
      if (partial && keep(partial)) controller.enqueue(encoder.encode(partial));
    },
  }));
}

// Answers a chat request from Workers AI while the GPU is offline. Requests that
// depend on features this path doesn't forward get the 503 instead of a wrong answer.
async function fallback(path, init, env, context) {
  if (path !== CHAT_PATH || !env.FALLBACK_MODEL || !env.AI) return upstreamUnavailable();
  const payload = JSON.parse(init.body);
  if (payload.tools || payload.functions || payload.response_format) return upstreamUnavailable();
  const input = {
    messages: payload.messages,
    max_tokens: payload.max_tokens ?? payload.max_completion_tokens,
  };
  for (const field of ["temperature", "top_p", "stream"]) {
    if (payload[field] != null) input[field] = payload[field];
  }
  let result;
  try {
    result = await env.AI.run(env.FALLBACK_MODEL, input);
  } catch (error) {
    // The message can quote the input, so log only its numeric code (3036 is the daily quota).
    context.fallbackError = /^\d+/.exec(error?.message ?? "")?.[0] ?? "unknown";
    return upstreamUnavailable();
  }
  context.backend = "workers-ai";
  const headers = { "x-gateway-backend": "workers-ai" };
  if (input.stream) {
    return new Response(dropNonOpenAIEvents(result), {
      headers: { ...headers, "content-type": "text/event-stream" },
    });
  }
  return Response.json(result, { headers });
}

async function forward(path, init, env, context) {
  let upstream;
  try {
    // The binding routes the request; this host only sets the Host header.
    upstream = await env.VLLM.fetch(`http://vllm${path}`, init);
  } catch {
    return fallback(path, init, env, context);
  }
  if (upstream.status === 401) {
    await upstream.body?.cancel();
    return errorResponse(502, "gateway_misconfigured", "The gateway could not authenticate upstream.");
  }
  // 502 and up come from the tunnel when the node is down. A 500 is vLLM's own.
  if (upstream.status >= 502) {
    await upstream.body?.cancel();
    return fallback(path, init, env, context);
  }
  context.backend = "gpu";
  const headers = new Headers({ "x-gateway-backend": "gpu" });
  for (const name of FORWARDED_RESPONSE_HEADERS) {
    const value = upstream.headers.get(name);
    if (value) headers.set(name, value);
  }
  // Passing the stream through keeps SSE incremental and propagates client disconnects.
  return new Response(upstream.body, { status: upstream.status, headers });
}

async function route(request, env, url, context) {
  if (request.method === "GET" && url.pathname === "/health") {
    return Response.json({ status: "ok" });
  }
  if (!UPSTREAM_ROUTES.has(`${request.method} ${url.pathname}`)) {
    return errorResponse(404, "not_found", "Unknown route.");
  }
  context.client = await authenticate(request, env.GATEWAY_KEYS);
  if (!context.client) {
    return errorResponse(401, "invalid_api_key", "Missing or invalid API key.", {
      "www-authenticate": "Bearer",
    });
  }
  let verdict;
  try {
    const limiter = env.CLIENT_LIMITER.get(env.CLIENT_LIMITER.idFromName(context.client));
    const limit = Number(env.RATE_LIMIT_PER_MINUTE) || DEFAULT_RATE_LIMIT;
    verdict = await (await limiter.fetch(`http://limiter/check?limit=${limit}`)).json();
  } catch {
    // Fail closed: the limiter is what protects the GPU from a runaway client.
    return errorResponse(503, "rate_limiter_unavailable", "Rate limiter unavailable. Try again later.", {
      "retry-after": String(RATE_LIMIT_PERIOD_SECONDS),
    });
  }
  if (!verdict.success) {
    return errorResponse(429, "rate_limit_exceeded", "Rate limit exceeded.", {
      "retry-after": String(verdict.retryAfter),
    });
  }
  const init = await buildUpstreamInit(request, url.pathname, env);
  if (init instanceof Response) return init;
  return forward(url.pathname, init, env, context);
}

// One instance per client. A Durable Object handles its requests one at a time, so
// concurrent bursts can't slip past the count the way they do with per-location limits.
// Storing the timestamps makes this a true sliding window: never more than `limit`
// requests in any 60 s. It's a plain class with a fetch handler so tests can run it in Node.
export class ClientRateLimiter {
  constructor(ctx) {
    this.storage = ctx.storage;
  }

  async fetch(request) {
    const limit = Number(new URL(request.url).searchParams.get("limit"));
    const now = Date.now();
    const periodMs = RATE_LIMIT_PERIOD_SECONDS * 1000;
    const recent = ((await this.storage.get("requests")) ?? []).filter((t) => t > now - periodMs);
    if (recent.length >= limit) {
      const retryAfter = Math.max(1, Math.ceil((recent[recent.length - limit] + periodMs - now) / 1000));
      return Response.json({ success: false, retryAfter });
    }
    recent.push(now);
    await this.storage.put("requests", recent);
    return Response.json({ success: true });
  }
}

export async function handle(request, env, log = console.log) {
  const start = Date.now();
  const url = new URL(request.url);
  const context = { client: null, backend: null, fallbackError: null };
  const response = await route(request, env, url, context);
  // Metadata only: prompts, completions and keys must never reach the logs.
  log(JSON.stringify({
    client: context.client,
    backend: context.backend,
    fallbackError: context.fallbackError,
    method: request.method,
    path: url.pathname,
    status: response.status,
    ms: Date.now() - start,
  }));
  return response;
}

export default {
  fetch(request, env) {
    return handle(request, env);
  },
};
