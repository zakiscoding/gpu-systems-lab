"""python -m pytest tools/test_benchstat.py"""

import math

import benchstat as b


def test_describe_matches_hand_computation():
    d = b.describe([10.0, 12.0, 11.0, 13.0, 9.0])
    assert d["mean"] == 11.0 and d["n"] == 5
    assert abs(d["sd"] - math.sqrt(2.5)) < 1e-4
    half = 2.776 * math.sqrt(2.5) / math.sqrt(5)            # t(0.975, df=4)
    assert abs(d["ci95_half"] - half) < 1e-3
    assert d["ci95"] == [round(11 - half, 4), round(11 + half, 4)]
    assert d["min"] == 9.0 and d["max"] == 13.0


def test_t_values():
    assert b.t95(1) == 12.706 and b.t95(9) == 2.262 and b.t95(100) == 1.96


def test_single_value_has_no_interval():
    d = b.describe([5.0])
    assert d["sd"] == 0.0 and math.isnan(d["ci95_half"])


def test_parse_cuda_table():
    out = """GPU: NVIDIA GeForce RTX 5070
kernel                               ms         GB/s    max_err   ok
---------------------------- ---------- ------------ ---------- ----
  add_bias alone                 1.0415        515.5          -    -
unfused (add_bias + gelu)        2.1005        511.2   1.19e-07  yes
fused bias_gelu                  1.0637        504.7   1.19e-07  yes
DRAM traffic: unfused 1024 MiB, fused 512 MiB; speedup 1.97x
PASS
"""
    rows = b.parse_rows(out)
    assert rows == {"add_bias alone": (1.0415, 515.5), "unfused (add_bias + gelu)": (2.1005, 511.2),
                    "fused bias_gelu": (1.0637, 504.7)}
