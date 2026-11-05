import assert from "node:assert/strict";
import { test } from "node:test";
import { createLocalJWKSet, exportJWK, generateKeyPair, SignJWT } from "jose";

import { handle, verifyAccessToken } from "../src/index.js";

const TEAM = "https://team.cloudflareaccess.com";
const AUD = "app-audience";
const BASE = "https://grafana-proxy.example.workers.dev";

const { publicKey, privateKey } = await generateKeyPair("RS256");
const other = await generateKeyPair("RS256");
const jwk = { ...(await exportJWK(publicKey)), kid: "k1", alg: "RS256" };
const keys = createLocalJWKSet({ keys: [jwk] });

const sign = (claims = {}, key = privateKey, expires = "1h") => new SignJWT(claims)
  .setProtectedHeader({ alg: "RS256", kid: "k1" })
  .setIssuer(claims.iss ?? TEAM)
  .setAudience(claims.aud ?? AUD)
  .setIssuedAt()
  .setExpirationTime(expires)
  .sign(key);

function makeEnv(reply = () => new Response("grafana", { status: 200 })) {
  const calls = [];
  return {
    calls,
    ACCESS_TEAM_DOMAIN: TEAM,
    ACCESS_AUD: AUD,
    GRAFANA: {
      async fetch(url, init) {
        calls.push({ url, init });
        return reply();
      },
    },
  };
}

const allow = async () => {};
const request = (path = "/", headers = {}, init = {}) => new Request(`${BASE}${path}`, { headers, ...init });
const withToken = (headers = {}) => ({ "cf-access-jwt-assertion": "token", ...headers });

test("refuses everything until Access is configured", async () => {
  for (const missing of ["ACCESS_TEAM_DOMAIN", "ACCESS_AUD"]) {
    const env = makeEnv();
    delete env[missing];
    const res = await handle(request("/", withToken()), env, allow);
    assert.equal(res.status, 503);
    assert.equal(env.calls.length, 0);
  }
});

test("refuses a request with no token", async () => {
  const env = makeEnv();
  assert.equal((await handle(request(), env, allow)).status, 403);
  assert.equal(env.calls.length, 0);
});

test("refuses when verification fails, and never reaches Grafana", async () => {
  const env = makeEnv();
  const res = await handle(request("/", withToken()), env, async () => { throw new Error("bad"); });
  assert.equal(res.status, 403);
  assert.equal(env.calls.length, 0);
});

test("a valid signed token passes", async () => {
  await verifyAccessToken(await sign(), { ACCESS_TEAM_DOMAIN: TEAM, ACCESS_AUD: AUD }, keys);
});

test("rejects wrong signer, audience, issuer and expired tokens", async () => {
  const env = { ACCESS_TEAM_DOMAIN: TEAM, ACCESS_AUD: AUD };
  const bad = [
    await sign({}, other.privateKey),
    await sign({ aud: "another-app" }),
    await sign({ iss: "https://evil.cloudflareaccess.com" }),
    await sign({}, privateKey, "-1h"),
    "not-a-token",
  ];
  for (const token of bad) await assert.rejects(verifyAccessToken(token, env, keys));
});

test("forwards the path and query, keeping the public host", async () => {
  const env = makeEnv();
  const res = await handle(request("/d/gpu-node?from=now-1h", withToken()), env, allow);
  assert.equal(res.status, 200);
  assert.equal(await res.text(), "grafana");
  assert.equal(env.calls[0].url, "http://grafana-proxy.example.workers.dev/d/gpu-node?from=now-1h");
});

test("Access credentials stop at the proxy; Grafana's own cookie goes through", async () => {
  const env = makeEnv();
  await handle(request("/", withToken({
    cookie: "CF_Authorization=secret; grafana_session=abc",
    "cf-access-authenticated-user-email": "someone@example.com",
  })), env, allow);
  const { headers } = env.calls[0].init;
  assert.equal(headers.get("cf-access-jwt-assertion"), null);
  assert.equal(headers.get("cf-access-authenticated-user-email"), null);
  assert.equal(headers.get("cookie"), "grafana_session=abc");
  assert.equal(headers.get("x-forwarded-proto"), "https");
});

test("drops the cookie header when only the Access cookie was sent", async () => {
  const env = makeEnv();
  await handle(request("/", withToken({ cookie: "CF_Authorization=secret" })), env, allow);
  assert.equal(env.calls[0].init.headers.get("cookie"), null);
});

test("POST bodies pass through, and redirects aren't followed", async () => {
  const env = makeEnv();
  await handle(request("/login", withToken(), { method: "POST", body: "{}", duplex: "half" }), env, allow);
  assert.equal(env.calls[0].init.method, "POST");
  assert.ok(env.calls[0].init.body);
  assert.equal(env.calls[0].init.redirect, "manual");
});

test("Grafana redirects and client errors reach the browser unchanged", async () => {
  for (const status of [302, 404, 500]) {
    const env = makeEnv(() => new Response(null, { status }));
    assert.equal((await handle(request("/", withToken()), env, allow)).status, status);
  }
});

test("an unreachable PC gives a clear 503", async () => {
  const thrown = makeEnv(() => { throw new Error("connection failed"); });
  assert.equal((await handle(request("/", withToken()), thrown, allow)).status, 503);
  const tunnel = makeEnv(() => new Response("down", { status: 530 }));
  assert.equal((await handle(request("/", withToken()), tunnel, allow)).status, 503);
});
