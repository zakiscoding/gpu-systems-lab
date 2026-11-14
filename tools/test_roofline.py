"""python -m pytest tools/test_roofline.py"""

import roofline as r


def test_ridge_points():
    fp32 = r.classify(r.per_elem("x", 1, 4, 100, "fp32", "-"))
    assert round(fp32["ridge"], 1) == round(23.7e3 / 634, 1)    # ~37 FLOP/B
    bf16 = r.classify(r.per_elem("x", 1, 4, 100, "bf16", "-"))
    assert bf16["ridge"] > fp32["ridge"]


def test_streaming_kernel_is_memory_bound():
    p = r.classify(r.per_elem("reduction", 1, 4, 630, "fp32", "-"))
    assert p["bound"] == "memory"
    assert abs(p["gflops"] - 157.5) < 1e-9          # 630 GB/s x 0.25 FLOP/B
    assert abs(p["pct"] - 100 * 630 / 634) < 1e-9    # % of roof == % of memory roof


def test_gemm_is_compute_bound():
    p = r.classify(r.gemm("sgemm", 4096, 21.017, "fp32", 4, "-"))
    assert round(p["ai"]) == 683                     # N / 6 for fp32
    assert p["bound"] == "compute"
    assert abs(p["pct"] - 100 * 21.017 / 23.7) < 1e-6


def test_decode_intensity_scales_with_batch():
    one = r.classify(r.decode("c1", 1, 172, "-"))
    many = r.classify(r.decode("c64", 64, 4430, "-"))
    assert abs(one["ai"] - 1.0) < 1e-9 and abs(many["ai"] - 64.0) < 1e-9
    assert one["bound"] == "memory"
    assert 500 < one["gbs"] < 560                    # 172 steps/s x 3.08 GB
    assert many["gflops"] > 20 * one["gflops"]          # batching reuses each weight read 64x


def test_table_renders_every_point():
    out = r.table([r.classify(p) for p in r.POINTS])
    assert out.count("\n") == len(r.POINTS) + 1
    assert "*(est.)*" in out
