// Public OpenAI-compatible gateway for the vLLM server on zak-gpu-node.
//
//   client --gateway key--> this Worker --VPC Service binding--> Tunnel --> vllm:8000
//
// Clients never see VLLM_API_KEY, and only the routes below reach vLLM.

const DEFAULT_MAX_BODY_BYTES = 262144;
const DEFAULT_MAX_TOKENS = 1024;
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

async function forward(path, init, env) {
  let upstream;
  try {
    // The binding routes the request; this host only sets the Host header.
    upstream = await env.VLLM.fetch(`http://vllm${path}`, init);
  } catch {
    return upstreamUnavailable();
  }
  if (upstream.status === 401) {
    await upstream.body?.cancel();
    return errorResponse(502, "gateway_misconfigured", "The gateway could not authenticate upstream.");
  }
  // 502 and up come from the tunnel when the node is down. A 500 is vLLM's own.
  if (upstream.status >= 502) {
    await upstream.body?.cancel();
    return upstreamUnavailable();
  }
  const headers = new Headers();
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
  let success;
  try {
    ({ success } = await env.RATE_LIMITER.limit({ key: context.client }));
  } catch {
    // Fail closed: the limiter is what protects the GPU from a runaway client.
    return errorResponse(503, "rate_limiter_unavailable", "Rate limiter unavailable. Try again later.", {
      "retry-after": String(RATE_LIMIT_PERIOD_SECONDS),
    });
  }
  if (!success) {
    return errorResponse(429, "rate_limit_exceeded", "Rate limit exceeded.", {
      "retry-after": String(RATE_LIMIT_PERIOD_SECONDS),
    });
  }
  const init = await buildUpstreamInit(request, url.pathname, env);
  if (init instanceof Response) return init;
  return forward(url.pathname, init, env);
}

export async function handle(request, env, log = console.log) {
  const start = Date.now();
  const url = new URL(request.url);
  const context = { client: null };
  const response = await route(request, env, url, context);
  // Metadata only: prompts, completions and keys must never reach the logs.
  log(JSON.stringify({
    client: context.client,
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
