// Builds the public results page from docs/RESULTS.md into dist/index.html.
//
// The repo is private and the doc describes the private network, so this drops
// the architecture map and runbooks, turns repo links into plain text, and
// refuses to build if anything that identifies the network survives.
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { marked } from "marked";

const SOURCE = new URL("../docs/RESULTS.md", import.meta.url);
const OUTPUT = new URL("./dist/", import.meta.url);
const DROPPED_SECTIONS = new Set(["Architecture", "Runbooks"]);
const FORBIDDEN = [
  /\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b/, // IP addresses
  /zak-gpu-node|mos-imac|hp-laptop|zakar|zak_pc|gpu-ts|zak-ts|imac-ts/i, // machine names
  /github\.com\/zakiscoding/i, // private repos
  /\]\(\.\.?\//, // leftover relative links
];

// Splits on "## " headings and drops the sections that describe the network.
export function publicMarkdown(source) {
  const [intro, ...sections] = source.replace(/\r\n/g, "\n").split(/^(?=## )/m);
  const kept = sections.filter((s) => !DROPPED_SECTIONS.has(s.split("\n")[0].slice(3).trim()));
  const title = intro.split("\n")[0];
  const summary = "Measurements from one RTX 5070 up to multi-GPU H100s on Modal and two-machine MPI, " +
    "with what they mean for choosing hardware.";
  const body = kept.join("")
    // "([modal/smoke.py](../modal/smoke.py))" after a heading or claim
    .replace(/ \((?:\[[^\]]+\]\([^)\s]+\)(?:, )?)+\)/g, "")
    // any other link becomes its text, since visitors can't open the repo
    .replace(/\[([^\]]+)\]\([^)\s]+\)/g, "$1");
  return `${title}\n\n${summary}\n\n${body}`;
}

export function assertPublishable(text) {
  for (const pattern of FORBIDDEN) {
    const match = pattern.exec(text);
    if (match) throw new Error(`refusing to publish, found ${pattern}: "${match[0]}"`);
  }
}

const PAGE = (content) => `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GPU Systems Lab results</title>
<meta name="description" content="GPU benchmarks from an RTX 5070 to multi-GPU H100s: kernels, NCCL, DDP, MPI, vLLM serving, FSDP and scheduling.">
<style>
:root { color-scheme: light dark; --bg: #fbfbf9; --fg: #1d1f21; --muted: #5d6368; --line: #dcdfe1; --stripe: #f1f2ef; --accent: #1a6b4f; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #15181a; --fg: #e6e8e9; --muted: #9ba2a7; --line: #2f3539; --stripe: #1d2124; --accent: #6fcfa8; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg); font: 16px/1.6 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 860px; margin: 0 auto; padding: 48px 24px 72px; }
h1 { font-size: 2rem; line-height: 1.2; margin: 0 0 12px; }
h1 + p { color: var(--muted); font-size: 1.1rem; margin: 0 0 32px; }
h2 { font-size: 1.45rem; margin: 48px 0 12px; padding-top: 24px; border-top: 1px solid var(--line); }
h3 { font-size: 1.1rem; margin: 32px 0 8px; }
code { font: 0.9em ui-monospace, Consolas, monospace; background: var(--stripe); padding: 0.1em 0.3em; border-radius: 3px; }
strong { font-weight: 650; }
li { margin: 6px 0; }
.table { overflow-x: auto; margin: 12px 0 20px; }
table { border-collapse: collapse; width: 100%; font-size: 0.93rem; font-variant-numeric: tabular-nums; }
th, td { padding: 7px 12px 7px 0; border-bottom: 1px solid var(--line); vertical-align: top; }
th { color: var(--muted); font-weight: 600; text-align: left; border-bottom-color: var(--muted); }
strong, h2, h3 { color: inherit; }
td strong { color: var(--accent); }
@media (max-width: 480px) { main { padding: 32px 16px 56px; } h1 { font-size: 1.6rem; } }
</style>
</head>
<body>
<main>
${content}
</main>
</body>
</html>
`;

const markdown = publicMarkdown(readFileSync(SOURCE, "utf8"));
assertPublishable(markdown);
const html = marked.parse(markdown)
  .replace(/<table>/g, '<div class="table"><table>')
  .replace(/<\/table>/g, "</table></div>");
mkdirSync(OUTPUT, { recursive: true });
writeFileSync(new URL("index.html", OUTPUT), PAGE(html));
console.log(`wrote dist/index.html, ${markdown.split("\n").length} lines of markdown`);
