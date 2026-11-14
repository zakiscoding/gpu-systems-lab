"""Highest throughput that still meets a latency target, per engine, from bench_compare summaries.

An interactive-chat SLO: TTFT p95 <= --ttft-ms and TPOT p95 <= --tpot-ms. For each engine, the
answer is the best output tok/s among the measured concurrency levels that meet both (a step
function: no interpolation between levels). Also prints the plain peak for comparison.

    python serving/engines/goodput.py results/engines/*_summary_*.json
    python serving/engines/goodput.py --ttft-ms 500 --tpot-ms 20 results/engines/*_summary_*.json
"""

import argparse
import glob
import json


def best_under_slo(rows, ttft_ms, tpot_ms):
    """The row with the highest output tok/s whose p95 TTFT and TPOT meet the SLO, or None."""
    ok = [r for r in rows if r["ttft_p95_ms"] is not None and r["tpot_p95_ms"] is not None
          and r["ttft_p95_ms"] <= ttft_ms and r["tpot_p95_ms"] <= tpot_ms]
    return max(ok, key=lambda r: r["output_tok_s"]) if ok else None


def table(summaries, ttft_ms, tpot_ms):
    out = [f"| engine | peak tok/s (at conc) | **best tok/s with TTFT p95 <= {ttft_ms:g} ms and TPOT p95 <= {tpot_ms:g} ms** "
           "(at conc) | TTFT p95 / TPOT p95 there |",
           "|---|---:|---:|---|"]
    for s in summaries:
        peak = max(s["rows"], key=lambda r: r["output_tok_s"])
        b = best_under_slo(s["rows"], ttft_ms, tpot_ms)
        good = f"**{b['output_tok_s']:,.0f}** ({b['concurrency']})" if b else "none meets it"
        lat = f"{b['ttft_p95_ms']:.0f} / {b['tpot_p95_ms']:.1f} ms" if b else "-"
        out.append(f"| {s['label']} | {peak['output_tok_s']:,.0f} ({peak['concurrency']}) | {good} | {lat} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--ttft-ms", type=float, default=1000)
    ap.add_argument("--tpot-ms", type=float, default=30)
    a = ap.parse_args()
    files = sorted({f for pat in a.files for f in glob.glob(pat)})
    summaries = [json.load(open(f)) for f in files if "warmup" not in f]
    print(table(summaries, a.ttft_ms, a.tpot_ms))


if __name__ == "__main__":
    main()
