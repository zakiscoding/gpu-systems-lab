import assert from "node:assert/strict";
import { test } from "node:test";

import { dashboards, handle } from "../src/index.js";

const BASE = "https://grafana-proxy.example.workers.dev";
const TOKEN = "eace2e730ce6450191fc1681b69d57d1";
const OTHER = "0123456789abcdef0123456789abcdef";

function makeEnv(reply = () => new Response("grafana", { status: 200 })) {
  const calls = [];
  return {
    calls,
    PUBLIC_DASHBOARDS: JSON.stringify([{ title: "GopherGrid <test>", description: "Scheduler", token: TOKEN }]),
    GRAFANA: {
      async fetch(url, init) {
        calls.push({ url, init });
        return reply();
      },
    },
  };
}

const request = (path, init = {}) => new Request(`${BASE}${path}`, init);
const query = (token = TOKEN, init = {}) => request(`/api/public/dashboards/${token}/panels/3/query`, {
  method: "POST", body: '{"intervalMs":1000}', headers: { "content-type": "application/json", "content-length": "19" }, ...init,
});

test("the landing page lists the dashboards and escapes titles", async () => {
  const env = makeEnv();
  const res = await handle(request("/"), env);
  assert.equal(res.status, 200);
  const html = await res.text();
  assert.match(html, new RegExp(`href="/public-dashboards/${TOKEN}"`));
  assert.match(html, /GopherGrid &#60;test&#62;/);
  assert.equal(env.calls.length, 0);
});

test("forwards a listed dashboard page, its API, annotations and panel queries", async () => {
  for (const req of [
    request(`/public-dashboards/${TOKEN}`),
    request(`/api/public/dashboards/${TOKEN}`),
    request(`/api/public/dashboards/${TOKEN}/annotations?from=1&to=2`),
    query(),
  ]) {
    const env = makeEnv();
    const res = await handle(req, env);
    assert.equal(res.status, 200, req.url);
    assert.equal(env.calls.length, 1);
  }
});

test("forwards Grafana's static files", async () => {
  const env = makeEnv();
  assert.equal((await handle(request("/public/build/app.js"), env)).status, 200);
  assert.equal(env.calls[0].url, `${BASE.replace("https", "http")}/public/build/app.js`);
});

test("unlisted dashboard tokens get a 404 and are never forwarded", async () => {
  for (const req of [request(`/public-dashboards/${OTHER}`), request(`/api/public/dashboards/${OTHER}`), query(OTHER)]) {
    const env = makeEnv();
    assert.equal((await handle(req, env)).status, 404, req.url);
    assert.equal(env.calls.length, 0);
  }
});

test("the rest of Grafana is unreachable", async () => {
  for (const path of [
    "/login", "/d/gophergrid", "/explore", "/api/dashboards/uid/gophergrid", "/api/datasources",
    "/api/ds/query", "/api/datasources/proxy/uid/prometheus/api/v1/query", "/api/users", "/api/admin/settings",
    "/api/public/dashboards", "/api/dashboards/public-dashboards", "/metrics", "/api/health", "/avatar/x",
    `/api/public/dashboards/${TOKEN}/panels/3/query/../../../../datasources`,
  ]) {
    const env = makeEnv();
    const res = await handle(request(path), env);
    assert.equal(res.status, 404, path);
    assert.equal(env.calls.length, 0, path);
  }
});

test("only reads are allowed, except panel queries, which must be POST", async () => {
  const env = makeEnv();
  assert.equal((await handle(request(`/public-dashboards/${TOKEN}`, { method: "POST", body: "x" }), env)).status, 405);
  assert.equal((await handle(request("/public/build/app.js", { method: "DELETE" }), env)).status, 405);
  assert.equal((await handle(request(`/api/public/dashboards/${TOKEN}/panels/3/query`), env)).status, 405);
  assert.equal(env.calls.length, 0);
});

test("panel query bodies over 64 KiB, or without a length, are refused", async () => {
  const env = makeEnv();
  const big = query(TOKEN, { headers: { "content-type": "application/json", "content-length": String(64 * 1024 + 1) } });
  assert.equal((await handle(big, env)).status, 413);
  const chunked = new Request(`${BASE}/api/public/dashboards/${TOKEN}/panels/3/query`, {
    method: "POST", body: new ReadableStream({ start(c) { c.enqueue(new Uint8Array(4)); c.close(); } }), duplex: "half",
  });
  assert.equal((await handle(chunked, env)).status, 413);
  assert.equal(env.calls.length, 0);
});

test("cookies and credentials never reach Grafana, and Set-Cookie never comes back", async () => {
  const env = makeEnv(() => new Response("ok", { headers: { "set-cookie": "grafana_session=abc", "content-type": "text/html" } }));
  const res = await handle(request(`/public-dashboards/${TOKEN}`, {
    headers: { cookie: "grafana_session=stolen", authorization: "Bearer x", "x-grafana-org-id": "1", "cf-connecting-ip": "1.2.3.4", accept: "text/html" },
  }), env);
  const sent = env.calls[0].init.headers;
  assert.equal(sent.get("cookie"), null);
  assert.equal(sent.get("authorization"), null);
  assert.equal(sent.get("x-grafana-org-id"), null);
  assert.equal(sent.get("cf-connecting-ip"), null);
  assert.equal(sent.get("accept"), "text/html");
  assert.equal(res.headers.get("set-cookie"), null);
  assert.equal(res.headers.get("x-robots-tag"), "noindex");
});

test("the login probe is answered locally with 401", async () => {
  const env = makeEnv();
  assert.equal((await handle(request("/apis/dashboard.grafana.app/"), env)).status, 401);
  assert.equal(env.calls.length, 0);
});

test("an offline iMac or tunnel becomes a 503", async () => {
  const thrown = makeEnv(() => { throw new Error("no route"); });
  assert.equal((await handle(request(`/public-dashboards/${TOKEN}`), thrown)).status, 503);
  const bad = makeEnv(() => new Response("bad gateway", { status: 502 }));
  assert.equal((await handle(request(`/public-dashboards/${TOKEN}`), bad)).status, 503);
});

test("a malformed or missing dashboard list exposes nothing", async () => {
  for (const value of [undefined, "", "not json", '{"token":"x"}', '[{"title":"x","token":"short"}]']) {
    assert.deepEqual(dashboards({ PUBLIC_DASHBOARDS: value }), []);
  }
  const env = { ...makeEnv(), PUBLIC_DASHBOARDS: "not json" };
  assert.equal((await handle(request(`/public-dashboards/${TOKEN}`), env)).status, 404);
});
