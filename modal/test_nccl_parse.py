"""Unit tests for modal/nccl_parse.py (no GPU): python -m pytest modal/test_nccl_parse.py"""

import nccl_parse

SAMPLE = """\
# nccl-tests version 2.21.1 nccl-headers=22706 nccl-library=22706
#                                                              out-of-place                       in-place
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
           8             2     float     sum      -1    17.12    0.00    0.00      0    16.93    0.00    0.00      0
     1048576        262144     float     sum      -1    20.50   51.15   51.15      0    20.10   52.17   52.17      0
  8589934592    2147483648     float     sum      -1  24000.0  357.91  357.91      0  24100.0  356.43  356.43      0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 136.5
zak:1:1 [0] NCCL INFO Channel 00/0 : 0[0] -> 1[1] via P2P/CUMEM/read
zak:1:1 [0] NCCL INFO Channel 01/0 : 0[0] -> 1[1] via P2P/CUMEM/read
"""


def test_rows_and_peak():
    r = nccl_parse.parse(SAMPLE)
    assert [x["bytes"] for x in r["rows"]] == [8, 1048576, 8589934592]
    assert r["rows"][1]["busbw_gbs"] == 52.17  # best of out-of-place / in-place
    assert r["peak"] == {"bytes": 8589934592, "busbw_gbs": 357.91, "algbw_gbs": 357.91}
    assert r["wrong_total"] == 0
    assert r["avg_busbw_gbs"] == 136.5
    assert r["small_latency_us"] == 17.12
    assert r["nccl_version"] == "2.27.6"


def test_wrong_counted_and_na_columns():
    log = "        8   2  float  none  0  10.0  0.00  0.00  N/A  11.0  0.00  0.00  3\n"
    r = nccl_parse.parse(log)
    assert r["wrong_total"] == 3


def test_transports_dedup():
    assert nccl_parse.transports(SAMPLE) == ["P2P/CUMEM/read"]


def test_human():
    assert nccl_parse.human(8) == "8B"
    assert nccl_parse.human(64 * 2**20) == "64M"
    assert nccl_parse.human(8 * 2**30) == "8G"
