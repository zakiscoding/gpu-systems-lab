// Serves the homelab's public, read-only Grafana dashboards on workers.dev.
//
//   browser --> this Worker --VPC Service binding--> Tunnel "mos-imac" --> grafana:3000
//
// Grafana's own "public dashboard" links do the read-only part: a link shows one dashboard
// and runs only that dashboard's saved queries. This Worker makes sure nothing else in
// Grafana is reachable. It forwards only:
//   - the pages and API calls of the dashboards listed in PUBLIC_DASHBOARDS,
//   - Grafana's static files (/public/...).
// Logins, the admin UI, the HTTP API, Explore and datasource proxies all get a 404, and no
// cookies or credentials pass in either direction.

const TOKEN = /^[0-9a-f]{32}$/;
const MAX_QUERY_BODY = 64 * 1024;

// Request headers Grafana needs to serve a public dashboard. Everything else (cookies,
// Authorization, client IPs) stays here.
const FORWARD_HEADERS = ["accept", "accept-language", "content-type", "user-agent", "if-none-match", "if-modified-since"];

export function dashboards(env) {
  let list;
  try {
    list = JSON.parse(env.PUBLIC_DASHBOARDS || "[]");
  } catch {
    return [];
  }
  return Array.isArray(list) ? list.filter((d) => d && TOKEN.test(d.token) && typeof d.title === "string") : [];
}

function text(status, message) {
  return new Response(message, { status, headers: { "content-type": "text/plain; charset=utf-8" } });
}

const escape = (s) => s.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);

function landing(list) {
  const items = list.map((d) => `<li><a href="/public-dashboards/${d.token}">${escape(d.title)}</a>` +
    (d.description ? `<span>${escape(d.description)}</span>` : "") + "</li>").join("\n");
  const html = `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Homelab dashboards</title>
<style>
:root{color-scheme:light dark;--bg:#fff;--fg:#1f2328;--muted:#59636e;--link:#0969da;--line:#d1d9e0}
@media (prefers-color-scheme:dark){:root{--bg:#111217;--fg:#e6edf3;--muted:#9198a1;--link:#58a6ff;--line:#30363d}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}
main{max-width:640px;margin:0 auto;padding:40px 16px}
h1{font-size:1.5rem;margin:0 0 4px}p{color:var(--muted);margin:0 0 24px}
ul{list-style:none;padding:0;margin:0}li{border-top:1px solid var(--line);padding:14px 0}
a{color:var(--link);font-weight:600;text-decoration:none}a:hover{text-decoration:underline}
span{display:block;color:var(--muted);font-size:.9rem}
</style></head><body><main>
<h1>Homelab dashboards</h1>
<p>Live, read-only metrics from a home cluster: an iMac, a laptop and a GPU PC, scraped by one Prometheus.</p>
<ul>
${items}
</ul>
</main></body></html>`;
  return new Response(html, { headers: { "content-type": "text/html; charset=utf-8", "cache-control": "max-age=300" } });
}

// Returns null when the request is allowed, otherwise the response to send instead.
export function check(request, allowed) {
  const { pathname } = new URL(request.url);
  const read = request.method === "GET" || request.method === "HEAD";

  if (pathname.startsWith("/public/")) return read ? null : text(405, "Method not allowed");

  let m = pathname.match(/^\/public-dashboards\/([^/]+)$/);
  if (m) {
    if (!allowed.has(m[1])) return text(404, "Not found");
    return read ? null : text(405, "Method not allowed");
  }

  m = pathname.match(/^\/api\/public\/dashboards\/([^/]+)(\/annotations|\/panels\/\d{1,6}\/query)?$/);
  if (m) {
    if (!allowed.has(m[1])) return text(404, "Not found");
    if (m[2]?.endsWith("/query")) {
      if (request.method !== "POST") return text(405, "Method not allowed");
      const length = Number(request.headers.get("content-length") ?? NaN);
      if (!(length <= MAX_QUERY_BODY)) return text(413, "Query body too large");
      return null;
    }
    return read ? null : text(405, "Method not allowed");
  }

  return text(404, "Not found");
}

export async function handle(request, env) {
  const list = dashboards(env);
  const url = new URL(request.url);
  if (url.pathname === "/" && (request.method === "GET" || request.method === "HEAD")) return landing(list);
  // The dashboard page probes this API to see whether a user is logged in. Answer it here
  // the way Grafana does for anonymous visitors, without forwarding anything.
  if (url.pathname === "/apis/dashboard.grafana.app/") {
    return new Response('{"message":"Unauthorized"}', { status: 401, headers: { "content-type": "application/json" } });
  }

  const refused = check(request, new Set(list.map((d) => d.token)));
  if (refused) return refused;

  const headers = new Headers();
  for (const name of FORWARD_HEADERS) {
    const value = request.headers.get(name);
    if (value) headers.set(name, value);
  }
  headers.set("x-forwarded-proto", "https");

  let upstream;
  try {
    upstream = await env.GRAFANA.fetch(`http://${url.host}${url.pathname}${url.search}`, {
      method: request.method,
      headers,
      body: request.method === "POST" ? request.body : undefined,
      redirect: "manual",
    });
  } catch {
    return text(503, "The dashboards are offline right now. The homelab iMac may be restarting.");
  }
  // 502 and up come from the tunnel when the iMac or Grafana is down.
  if (upstream.status >= 502) {
    await upstream.body?.cancel();
    return text(503, "The dashboards are offline right now. The homelab iMac may be restarting.");
  }
  const out = new Response(upstream.body, upstream);
  out.headers.delete("set-cookie");
  out.headers.set("x-robots-tag", "noindex");
  return out;
}

export default {
  fetch(request, env) {
    return handle(request, env);
  },
};
