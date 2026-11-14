"""Roofline / speed-of-light numbers for the RTX 5070 from this repo's own measurements.

Every point is (FLOPs, bytes of DRAM traffic, measured time or rate) for a workload measured
elsewhere in the repo, with the source cited. From that:
    arithmetic intensity  AI = FLOPs / byte
    attainable            min(compute roof, AI x memory roof)
    bound                 memory-bound if AI < ridge point (compute roof / memory roof)
    % of roof             achieved / attainable at that AI

Roofs are *measured* ceilings on this card (empirical roofline), not datasheet numbers:
    memory  634 GB/s     cub::DeviceReduce::Sum, the fastest read stream measured (cuda/README.md);
                         the 672 GB/s GDDR7 spec is drawn as a reference only
    FP32    23.7 TFLOPS  cuBLAS FP32 GEMM 8192^2 (docs/RESULTS.md, Single GPU)
    BF16    67.7 TFLOPS  cuBLAS BF16 GEMM 8192^2 (same table)

    python tools/roofline.py            # prints the markdown table used in docs/RESULTS.md
    python tools/roofline.py --json     # machine-readable, for the site chart
"""

import argparse
import json

MEM_ROOF_GBS = 634.0
MEM_SPEC_GBS = 672.0
ROOFS_TFLOPS = {"fp32": 23.7, "bf16": 67.7}

# Qwen2.5-1.5B-Instruct: 1.54B parameters in bf16 (tied embedding/LM head read once per step)
QWEN15_PARAMS = 1.54e9
QWEN15_WEIGHT_BYTES = QWEN15_PARAMS * 2


def per_elem(name, flops_per_elem, bytes_per_elem, gbs, roof, source):
    """A streaming kernel measured as effective GB/s: FLOP rate = GB/s x AI."""
    ai = flops_per_elem / bytes_per_elem
    return {"name": name, "ai": ai, "gflops": gbs * ai, "gbs": gbs, "roof": roof, "source": source}


def gemm(name, n, tflops, roof, elem_bytes, source):
    """Square GEMM: 2N^3 FLOPs over the compulsory traffic (read A, B, write C once)."""
    ai = 2 * n ** 3 / (3 * n * n * elem_bytes)
    return {"name": name, "ai": ai, "gflops": tflops * 1e3, "gbs": tflops * 1e3 / ai, "roof": roof, "source": source}


def decode(name, conc, tok_s, source):
    """vLLM decode step, weights-only traffic model: each step reads all weights once
    (3.08 GB) and does 2 x params FLOPs per sequence in the batch. Ignores KV-cache reads
    (adds bytes) and prefill (adds FLOPs), so AI and bandwidth are estimates."""
    steps_s = tok_s / conc
    flops = 2 * QWEN15_PARAMS * conc
    return {"name": name, "ai": flops / QWEN15_WEIGHT_BYTES, "gflops": flops * steps_s / 1e9,
            "gbs": QWEN15_WEIGHT_BYTES * steps_s / 1e9, "roof": "bf16", "source": source, "estimate": True}


POINTS = [
    per_elem("vector_add (float4)", 1, 12, 590, "fp32", "cuda/README.md"),
    per_elem("reduction (warp shuffle)", 1, 4, 630, "fp32", "cuda/README.md"),
    per_elem("fused bias+GELU, CUDA", 10, 8, 491, "fp32", "kernels/triton/README.md"),
    per_elem("fused bias+GELU, Triton", 10, 8, 559, "fp32", "kernels/triton/README.md"),
    per_elem("RMSNorm custom op v2 (bf16)", 4, 4, 562, "fp32", "kernels/custom_op/README.md"),
    gemm("naive matmul FP32, N=4096", 4096, 1.962, "fp32", 4, "cuda/README.md"),
    gemm("cuBLAS SGEMM FP32, N=4096", 4096, 21.017, "fp32", 4, "cuda/README.md"),
    {"name": "GPT-136M train step, JAX (bf16)", "ai": None, "gflops": 48.75e3, "gbs": None, "roof": "bf16",
     "source": "jax-xla/README.md", "note": "model FLOPs (6 x params x tokens); compute-bound by construction"},
    decode("vLLM decode, Qwen2.5-1.5B, 1 request", 1, 172, "docs/RESULTS.md, Inference serving"),
    decode("vLLM decode, Qwen2.5-1.5B, 16 requests", 16, 1987, "docs/RESULTS.md, Inference serving"),
    decode("vLLM decode, Qwen2.5-1.5B, 64 requests", 64, 4430, "docs/RESULTS.md, Inference serving"),
]


def classify(p):
    """Add ridge point, attainable rate, bound and % of roof to a point (returns a new dict)."""
    peak = ROOFS_TFLOPS[p["roof"]] * 1e3
    ridge = peak / MEM_ROOF_GBS
    out = dict(p, ridge=ridge)
    if p["ai"] is None:                       # no traffic model: compare with the compute roof only
        out.update(attainable=peak, bound="compute", pct=100 * p["gflops"] / peak)
        return out
    attainable = min(peak, p["ai"] * MEM_ROOF_GBS)
    out.update(attainable=attainable, bound="memory" if p["ai"] < ridge else "compute",
               pct=100 * p["gflops"] / attainable)
    return out


def table(points):
    rows = ["| Workload | FLOP/byte | Achieved | Roof at that AI | Bound | % of roof |",
            "|---|---:|---:|---:|---|---:|"]
    for p in points:
        ai = "-" if p["ai"] is None else f"{p['ai']:.3g}"
        ach = f"{p['gflops'] / 1e3:.2f} TFLOPS" if p["gflops"] >= 1e3 else f"{p['gflops']:.0f} GFLOPS"
        if p["gbs"] is not None and p["bound"] == "memory":
            ach += f" ({p['gbs']:.0f} GB/s)"
        roof = f"{p['attainable'] / 1e3:.1f} TFLOPS" if p["attainable"] >= 1e3 else f"{p['attainable']:.0f} GFLOPS"
        name = p["name"] + (" *(est.)*" if p.get("estimate") else "")
        rows.append(f"| {name} | {ai} | {ach} | {roof} | {p['bound']} | {p['pct']:.0f}% |")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    pts = [classify(p) for p in POINTS]
    if args.json:
        print(json.dumps({"mem_roof_gbs": MEM_ROOF_GBS, "mem_spec_gbs": MEM_SPEC_GBS,
                          "roofs_tflops": ROOFS_TFLOPS, "points": pts}, indent=1))
    else:
        print(table(pts))


if __name__ == "__main__":
    main()
