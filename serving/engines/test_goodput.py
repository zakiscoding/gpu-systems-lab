"""python -m pytest serving/engines/test_goodput.py"""

import goodput


def row(c, tok, ttft, tpot):
    return {"concurrency": c, "output_tok_s": tok, "ttft_p95_ms": ttft, "tpot_p95_ms": tpot}


ROWS = [row(1, 160, 35, 6), row(16, 1900, 300, 8), row(64, 4100, 900, 15), row(256, 4600, 2500, 50)]


def test_best_meets_both_limits():
    b = goodput.best_under_slo(ROWS, 1000, 30)
    assert b["concurrency"] == 64             # 256 is faster but breaks both limits


def test_tighter_slo_picks_lower_level():
    assert goodput.best_under_slo(ROWS, 500, 30)["concurrency"] == 16
    assert goodput.best_under_slo(ROWS, 10, 30) is None


def test_missing_percentiles_never_qualify():
    assert goodput.best_under_slo([row(1, 100, None, 5)], 1000, 30) is None


def test_table_marks_engines_without_a_qualifying_level():
    s = [{"label": "a", "rows": ROWS}, {"label": "b", "rows": [row(1, 10, 5000, 5)]}]
    t = goodput.table(s, 1000, 30)
    assert "**4,100** (64)" in t and "none meets it" in t
