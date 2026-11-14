"""Pure-Python parts of the serving benchmark (no GPU, no vLLM):
python -m unittest serving.test_bench_sweep"""

import json
import unittest
from types import SimpleNamespace
from unittest import mock

from serving import bench_sweep


class BenchSweepTest(unittest.TestCase):
    def test_parse_kv_reads_capacity_lines(self):
        log = ("INFO Model loading took 2.89 GiB memory\n"
               "INFO Available KV cache memory: 4.12 GiB\n"
               "INFO GPU KV cache size: 154,320 tokens\n"
               "INFO Maximum concurrency for 4,096 tokens per request: 37.68x\n")
        kv = bench_sweep.parse_kv(log)
        self.assertEqual(kv["kv_cache_tokens"], 154320)
        self.assertEqual(kv["kv_cache_memory_gib"], 4.12)
        self.assertEqual(kv["model_weights_gib"], 2.89)
        self.assertEqual(kv["max_concurrency_tokens_per_request"], 4096)
        self.assertEqual(len(kv["raw_lines"]), 4)

    def test_dataset_args(self):
        rnd = bench_sweep.dataset_args(SimpleNamespace(dataset="random", input_len=512), 256)
        self.assertEqual(rnd, ["--dataset-name", "random", "--random-input-len", "512", "--random-output-len", "256"])
        cus = bench_sweep.dataset_args(SimpleNamespace(dataset="custom", input_len=512), 128)
        self.assertIn("custom", cus)
        self.assertEqual(cus[-2:], ["--custom-output-len", "128"])
        self.assertTrue(bench_sweep.CUSTOM_PROMPTS.exists(), "run serving/make_prompts.py")

    def test_custom_prompts_are_valid_jsonl(self):
        rows = [json.loads(l) for l in bench_sweep.CUSTOM_PROMPTS.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 256)
        self.assertTrue(all(r["prompt"].strip() for r in rows))
        self.assertEqual(len({r["prompt"] for r in rows}), 256)

    def test_spec_counters_sums_labels_and_skips_comments(self):
        text = ("# HELP vllm:spec_decode_num_draft_tokens_total x\n"
                'vllm:spec_decode_num_draft_tokens_total{engine="0"} 100.0\n'
                'vllm:spec_decode_num_draft_tokens_total{engine="1"} 20.0\n'
                'vllm:spec_decode_num_accepted_tokens_total{engine="0"} 60.0\n')
        resp = mock.MagicMock()
        resp.__enter__.return_value.read.return_value = text.encode()
        with mock.patch("urllib.request.urlopen", return_value=resp):
            self.assertEqual(bench_sweep.spec_counters("http://x"), {"draft": 120.0, "accepted": 60.0})
        with mock.patch("urllib.request.urlopen", side_effect=OSError("down")):
            self.assertEqual(bench_sweep.spec_counters("http://x"), {})

    def test_markdown_has_p95_and_acceptance(self):
        row = {k: 1 for k in ("concurrency", "request_throughput", "output_tok_s", "ttft_p50_ms", "ttft_p95_ms",
                              "tpot_p50_ms", "tpot_p95_ms", "e2el_p50_ms", "e2el_p95_ms")}
        row["spec_acceptance_rate"] = 0.42
        md = bench_sweep.markdown({"label": "x", "model": "m", "gpu": {"gpu": "g"}, "vllm_version": "v",
                                   "input_len": 1, "output_len": 1, "kv_cache": {}, "rows": [row]})
        self.assertIn("TTFT p95", md)
        self.assertIn("| 0.42 |", md)


if __name__ == "__main__":
    unittest.main()
