// Builds the public results page from docs/RESULTS.md into dist/index.html.
//
// The doc describes the home network, so this drops the architecture map and
// runbooks, turns relative repo links into plain text, and refuses to build if
// anything that identifies the network survives.
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { marked } from "marked";

const SOURCE = new URL("../docs/RESULTS.md", import.meta.url);
const OUTPUT = new URL("./dist/", import.meta.url);
const DROPPED_SECTIONS = new Set(["Architecture", "Runbooks"]);
const FORBIDDEN = [
  /\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b/, // IP addresses
  /zak-gpu-node|mos-imac|hp-laptop|zakar|zak_pc|gpu-ts|zak-ts|imac-ts/i, // machine names
  /github\.com\/zakiscoding/i, // repo links, some of them to private repos
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
    // any other link becomes its text, since relative paths don't work on the site
    .replace(/\[([^\]]+)\]\([^)\s]+\)/g, "$1");
  return `${title}\n\n${summary}\n\n${body}`;
}

// Reads a measured table from the doc, so a chart can't drift from the numbers beside it.
function tableRows(markdown, firstHeaderCell) {
  const lines = markdown.split("\n");
  const start = lines.findIndex((l) => l.startsWith(`| ${firstHeaderCell}`));
  if (start < 0) throw new Error(`table not found: ${firstHeaderCell}`);
  const rows = [];
  for (let i = start + 2; lines[i]?.startsWith("|"); i++) {
    rows.push(lines[i].split("|").slice(1, -1).map((c) => c.trim()));
  }
  return rows;
}

function number(cell) {
  const value = Number(/[\d,.]+/.exec(cell)?.[0].replace(/,/g, ""));
  if (!Number.isFinite(value)) throw new Error(`no number in table cell: ${cell}`);
  return value;
}

// Puts a chart directly under its section heading. The table below stays as the text version.
function withChart(markdown, heading, title, note, spec) {
  const pattern = new RegExp(`^### ${heading}.*$`, "m");
  if (!pattern.test(markdown)) throw new Error(`heading not found: ${heading}`);
  const data = JSON.stringify(spec).replace(/&/g, "&amp;").replace(/'/g, "&#39;");
  const figure = `<figure class="chart" data-chart='${data}'><figcaption><strong>${title}</strong><span>${note}</span></figcaption>` +
    `<div class="legend"></div><div class="plot"></div><div class="tip" role="status" hidden></div></figure>`;
  return markdown.replace(pattern, (line) => `${line}\n\n${figure}\n`);
}

function addCharts(markdown) {
  const compute = tableRows(markdown, "8192² matmul").filter((row) => row[0].includes("TFLOPS"));
  const serving = tableRows(markdown, "Concurrent requests");
  const withCompute = withChart(markdown, "Single GPU", "Matmul speed by precision", "TFLOPS, higher is better", {
    type: "bars",
    unit: "TFLOPS",
    labels: compute.map((row) => row[0].replace(" TFLOPS", "")),
    series: [
      { name: "RTX 5070", values: compute.map((row) => number(row[1])) },
      { name: "H100 80GB", values: compute.map((row) => number(row[2])), labelled: true },
    ],
  });
  const precision = tableRows(markdown, "Concurrency");
  const latency = tableRows(markdown, "Precision");
  const withServing = withChart(withCompute, "Inference serving", "vLLM throughput as concurrency rises", "Output tokens/s", {
    type: "line",
    xName: "concurrent",
    labels: serving.map((row) => row[0]),
    series: [
      { name: "RTX 5070", values: serving.map((row) => number(row[1])) },
      { name: "H100", values: serving.map((row) => number(row[2])) },
    ],
  });
  const withLatency = withChart(withServing, "Inference optimization", "Latency at 64 concurrent requests",
    "End-to-end seconds, lower is better", {
      type: "bars",
      unit: "s",
      labels: latency.map((row) => row[0]),
      series: [
        { name: "p50", values: latency.map((row) => number(row[4].split("/")[0])) },
        { name: "p95", values: latency.map((row) => number(row[4].split("/")[1])), labelled: true },
      ],
    });
  const spec = tableRows(markdown, "Requests in flight");
  const withSpec = withChart(withLatency, "Speculative decoding", "Speculative decoding on a 1.5B model",
    "Output tokens/s by requests in flight, higher is better", {
      type: "line",
      xName: "in flight",
      labels: spec.map((row) => row[0]),
      series: [
        { name: "Off", values: spec.map((row) => number(row[1])) },
        { name: "N-gram", values: spec.map((row) => number(row[2])) },
        { name: "0.5B draft", values: spec.map((row) => number(row[3])) },
      ],
    });
  // Added last so it sits first under the heading.
  return withChart(withSpec, "Inference optimization", "FP16 vs AWQ vs FP8 on the RTX 5070",
    "Output tokens/s by concurrent requests", {
      type: "line",
      xName: "concurrent",
      labels: precision.map((row) => row[0]),
      series: [
        { name: "FP16", values: precision.map((row) => number(row[1])) },
        { name: "AWQ", values: precision.map((row) => number(row[2])) },
        { name: "FP8", values: precision.map((row) => number(row[3])) },
      ],
    });
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
:root { color-scheme: light dark; --bg: #fbfbf9; --fg: #1d1f21; --muted: #5d6368; --line: #dcdfe1; --stripe: #f1f2ef; --accent: #1a6b4f; --axis: #c3c2b7; --series-1: #2a78d6; --series-2: #eb6834; --series-3: #2e9e6a; --series-4: #8a5cc2; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #15181a; --fg: #e6e8e9; --muted: #9ba2a7; --line: #2f3539; --stripe: #1d2124; --accent: #6fcfa8; --axis: #454b4f; --series-1: #3987e5; --series-2: #d95926; --series-3: #45b884; --series-4: #a27ad6; }
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
.chart { display: none; position: relative; margin: 16px 0 8px; }
.js .chart { display: block; }
figcaption { display: flex; flex-wrap: wrap; gap: 0 10px; align-items: baseline; }
figcaption span { color: var(--muted); font-size: 0.88rem; }
.legend { display: flex; gap: 16px; margin: 4px 0 2px; color: var(--muted); font-size: 0.85rem; min-height: 1.4em; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.key-line { width: 16px; height: 2px; border-radius: 1px; }
.key-box { width: 10px; height: 10px; border-radius: 2px; }
.plot { height: 280px; }
.plot svg { display: block; }
.grid { stroke: var(--line); stroke-width: 1; }
.axis { stroke: var(--axis); stroke-width: 1; }
.crosshair { stroke: var(--muted); stroke-width: 1; pointer-events: none; }
.tick { fill: var(--muted); font-size: 12px; font-variant-numeric: tabular-nums; }
.label { fill: var(--fg); font-size: 12px; }
.value { fill: var(--fg); font-size: 12px; font-weight: 600; font-variant-numeric: tabular-nums; }
.series-line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.dot { stroke: var(--bg); stroke-width: 2; }
.bar.lifted { opacity: 0.75; }
.hit { fill: transparent; outline: none; }
.hit:focus-visible { stroke: var(--muted); stroke-width: 1; }
.tip { position: absolute; z-index: 1; pointer-events: none; background: var(--bg); border: 1px solid var(--axis); border-radius: 6px; padding: 6px 10px; font-size: 0.82rem; white-space: nowrap; box-shadow: 0 2px 8px rgb(0 0 0 / 0.15); }
.tip[hidden] { display: none; }
.tip-title { color: var(--muted); }
.tip-row { display: flex; align-items: center; gap: 6px; font-variant-numeric: tabular-nums; }
.tip-row i { width: 12px; height: 2px; }
.tip-row span { color: var(--muted); }
@media (max-width: 480px) { main { padding: 32px 16px 56px; } h1 { font-size: 1.6rem; } }
</style>
</head>
<body>
<main>
${content}
</main>
<script>${CLIENT}</script>
</body>
</html>
`;

const CLIENT = readFileSync(new URL("./charts.client.js", import.meta.url), "utf8");
const markdown = addCharts(publicMarkdown(readFileSync(SOURCE, "utf8")));
assertPublishable(markdown);
const html = marked.parse(markdown)
  .replace(/<table>/g, '<div class="table"><table>')
  .replace(/<\/table>/g, "</table></div>");
mkdirSync(OUTPUT, { recursive: true });
writeFileSync(new URL("index.html", OUTPUT), PAGE(html));
console.log(`wrote dist/index.html, ${markdown.split("\n").length} lines of markdown`);
