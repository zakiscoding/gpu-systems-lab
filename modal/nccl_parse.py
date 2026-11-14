"""Parse NVIDIA nccl-tests output (no Modal, no GPU needed).

A result row looks like (out-of-place then in-place columns):
    size  count  type  redop  root  time  algbw  busbw  #wrong  time  algbw  busbw  #wrong
For each size we keep the better of out-of-place / in-place busbw. busbw is
nccl-tests' "bus bandwidth": algbw scaled by the collective's traffic factor
(2(n-1)/n for all_reduce, (n-1)/n for all_gather / reduce_scatter / alltoall,
1 for broadcast / sendrecv), so it is comparable with the link speed.

Usage: python modal/nccl_parse.py <log> [<log> ...]
"""

import json
import re
import sys

ROW = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\w+)\s+(\S+)\s+(-?\d+)\s+(.+)$")


def _num(s: str):
    """float(s), or None for nccl-tests' 'N/A' columns."""
    try:
        return float(s)
    except ValueError:
        return None


def parse(text: str) -> dict:
    """Rows, per-size best busbw, the peak, the average-bus-bandwidth line,
    total wrong results and the NCCL version from one nccl-tests log."""
    rows = []
    for line in text.splitlines():
        m = ROW.match(line)
        if not m:
            continue
        size, count, dtype, redop, root, rest = m.groups()
        cols = rest.split()
        if len(cols) < 8:
            continue
        oop = [_num(c) for c in cols[0:4]]
        inp = [_num(c) for c in cols[4:8]]
        busbw = max(x for x in (oop[2], inp[2]) if x is not None)
        algbw = max(x for x in (oop[1], inp[1]) if x is not None)
        wrong = sum(int(x) for x in (oop[3], inp[3]) if x is not None)
        rows.append({"bytes": int(size), "dtype": dtype, "redop": redop,
                     "time_us": oop[0], "algbw_gbs": algbw, "busbw_gbs": busbw, "wrong": wrong})
    out = {"rows": rows, "wrong_total": sum(r["wrong"] for r in rows)}
    if rows:
        pk = max(rows, key=lambda r: r["busbw_gbs"])
        out["peak"] = {"bytes": pk["bytes"], "busbw_gbs": pk["busbw_gbs"], "algbw_gbs": pk["algbw_gbs"]}
        small = min(rows, key=lambda r: r["bytes"])
        out["small_latency_us"] = small["time_us"]
    m = re.search(r"Avg bus bandwidth\s*:\s*([\d.]+)", text)
    if m:
        out["avg_busbw_gbs"] = float(m.group(1))
    m = re.search(r"NCCL version (\d+\.\d+\.\d+)", text)
    if m:
        out["nccl_version"] = m.group(1)
    else:  # nccl-tests header: nccl-library=22706 -> 2.27.6
        m = re.search(r"nccl-library=(\d+)", text)
        if m:
            v = int(m.group(1))
            out["nccl_version"] = f"{v // 10000}.{v // 100 % 100}.{v % 100}"
    return out


def transports(text: str) -> list[str]:
    """Distinct 'via ...' transport phrases from an NCCL_DEBUG=INFO log, e.g.
    ['P2P/CUMEM', 'NVLS']; empty if the log has no INFO lines."""
    found = []
    for m in re.finditer(r"via (P2P/\S+|SHM/\S+|NET/\S+|NVLS|CollNet\S*)", text):
        t = m.group(1).rstrip(",")
        if t not in found:
            found.append(t)
    return found


def human(n) -> str:
    """Byte count as 8B / 64K / 1M / 8G; '-' for None."""
    if n is None:
        return "-"
    for unit, s in ((2**30, "G"), (2**20, "M"), (2**10, "K")):
        if n >= unit:
            return f"{n / unit:g}{s}"
    return f"{n}B"


if __name__ == "__main__":
    for p in sys.argv[1:]:
        r = parse(open(p).read())
        print(p, json.dumps({k: v for k, v in r.items() if k != "rows"}))
