// Serves Grafana on workers.dev, behind Cloudflare Access.
//
//   browser --Access login--> this Worker --VPC Service binding--> Tunnel --> grafana:3000
//
// Grafana allows anonymous viewers, so this must never forward a request without a
// valid Access token. With Access unconfigured it refuses everything, which makes it
// safe to deploy before Access is turned on.
import { createRemoteJWKSet, jwtVerify } from "jose";

let remoteKeys;
let remoteKeysTeam;

export async function verifyAccessToken(token, env, keys) {
  if (!keys) {
    if (remoteKeysTeam !== env.ACCESS_TEAM_DOMAIN) {
      remoteKeys = createRemoteJWKSet(new URL(`${env.ACCESS_TEAM_DOMAIN}/cdn-cgi/access/certs`));
      remoteKeysTeam = env.ACCESS_TEAM_DOMAIN;
    }
    keys = remoteKeys;
  }
  await jwtVerify(token, keys, { issuer: env.ACCESS_TEAM_DOMAIN, audience: env.ACCESS_AUD });
}

function refuse(status, message) {
  return new Response(message, { status, headers: { "content-type": "text/plain; charset=utf-8" } });
}

// Grafana has no use for Access's credentials, so they stop here.
function forwardedHeaders(request) {
  const headers = new Headers(request.headers);
  headers.delete("cf-access-jwt-assertion");
  headers.delete("cf-access-authenticated-user-email");
  const cookies = (headers.get("cookie") ?? "").split(";").map((c) => c.trim())
    .filter((c) => c && !c.startsWith("CF_Authorization="));
  if (cookies.length) headers.set("cookie", cookies.join("; "));
  else headers.delete("cookie");
  headers.set("x-forwarded-proto", "https");
  return headers;
}

export async function handle(request, env, verify = verifyAccessToken) {
  if (!env.ACCESS_TEAM_DOMAIN || !env.ACCESS_AUD) {
    return refuse(503, "Cloudflare Access isn't configured for this Worker yet.");
  }
  const token = request.headers.get("cf-access-jwt-assertion");
  if (!token) return refuse(403, "A Cloudflare Access login is required.");
  try {
    await verify(token, env);
  } catch {
    return refuse(403, "The Cloudflare Access token isn't valid.");
  }

  const url = new URL(request.url);
  let upstream;
  try {
    // The binding routes the request. Keeping the public host in the URL sets the Host
    // header, which Grafana compares with Origin before accepting a login.
    upstream = await env.GRAFANA.fetch(`http://${url.host}${url.pathname}${url.search}`, {
      method: request.method,
      headers: forwardedHeaders(request),
      body: request.body,
      redirect: "manual",
    });
  } catch {
    return refuse(503, "Grafana is offline. The GPU PC may be off or rebooting.");
  }
  // 502 and up come from the tunnel when the PC is down.
  if (upstream.status >= 502) {
    await upstream.body?.cancel();
    return refuse(503, "Grafana is offline. The GPU PC may be off or rebooting.");
  }
  return upstream;
}

export default {
  fetch(request, env) {
    return handle(request, env);
  },
};
