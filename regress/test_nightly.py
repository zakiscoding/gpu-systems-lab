"""python -m pytest regress/test_nightly.py"""

import nightly


def run(f16, ok=True, label=""):
    return {"ok": ok, "label": label, "tflops": {"float16": f16, "bfloat16": f16 + 0.4}}


def test_no_verdict_until_enough_history():
    v = nightly.evaluate([run(63.0), run(63.5)], {"float16": 40.0})
    assert v["float16"]["baseline"] is None and v["float16"]["regression"] is False


def test_drop_over_5_percent_is_a_regression():
    hist = [run(x) for x in (63.0, 63.6, 63.2, 64.0, 63.4)]
    v = nightly.evaluate(hist, {"float16": 59.9, "bfloat16": 63.0})
    assert v["float16"]["baseline"] == 63.4                 # median of the 5
    assert v["float16"]["regression"] is True               # 59.9 / 63.4 = 0.945
    assert v["bfloat16"]["regression"] is False             # 63.0 / 63.8 = 0.987


def test_exactly_5_percent_is_not_flagged():
    hist = [run(100.0)] * 5
    assert nightly.evaluate(hist, {"float16": 95.0})["float16"]["regression"] is False


def test_baseline_uses_last_7_successful_unlabelled_runs():
    hist = [run(10.0)] * 5 + [run(60.0)] * 7 + [run(1.0, ok=False), run(1.0, label="contention-test")]
    v = nightly.evaluate(hist, {"float16": 60.0})
    assert v["float16"]["baseline"] == 60.0 and v["float16"]["n_baseline"] == 7


def test_prometheus_text():
    v = nightly.evaluate([run(63.0)] * 3, {"float16": 30.0})
    t = nightly.prometheus_text(v, True, 1_700_000_000, label="contention-test")
    assert 'gg_nightly_regression{precision="float16",label="contention-test"} 1' in t
    assert 'gg_nightly_baseline_tflops{precision="float16",label="contention-test"} 63.0' in t
    assert "gg_nightly_job_ok 1" in t and t.endswith("\n")


def test_tflops_by_precision():
    job = {"result": {"summary": {"matmul": [{"precision": "float16", "tflops": 63.6}, {"size": 1}]}}}
    assert nightly.tflops_by_precision(job) == {"float16": 63.6}
