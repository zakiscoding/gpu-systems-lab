"""Write serving/prompts/chat_prompts.jsonl: 256 natural-language prompts for
`bench_sweep.py --dataset custom`.

Deterministic (no randomness, no downloads), so every run and machine benchmarks
the same text. The mix leans on what chat traffic looks like: explanations,
code, rewriting and summaries. Some tasks repeat their input (rewrite, convert,
fix), which is where n-gram speculation can find matches; others don't.

    python serving/make_prompts.py
"""

import itertools
import json
from pathlib import Path

TOPICS = [
    "TCP congestion control", "garbage collection in Go", "Postgres MVCC", "the CAP theorem",
    "CUDA warps and occupancy", "KV caching in transformer inference", "Raft leader election",
    "B-trees versus LSM trees", "HTTP/2 multiplexing", "Linux cgroups", "consistent hashing",
    "gradient checkpointing", "the Rust borrow checker", "DNS resolution", "vector databases",
    "speculative decoding",
]

TASKS = [
    "Explain {t} to a new backend engineer in three short paragraphs.",
    "List five common mistakes people make with {t}, with one sentence on each.",
    "Write a Python function that demonstrates {t}, with comments.",
    "Compare {t} with the most common alternative. Give a short table, then a recommendation.",
    "Summarize {t} in exactly five bullet points.",
    "Write an interview question about {t} and a model answer.",
    "Rewrite this sentence to be clearer, then explain each change: \"{t} is a thing that is used for making systems work in a way that is better when there are many requests.\"",
    "Fix the grammar and keep the meaning: \"{t} are very importent because it make the system more faster and also it reduce the cost alot.\"",
    "Convert this note into a numbered checklist: we should read about {t}, then try it in a small project, then measure it, then write down what we found, then share it with the team.",
    "Write a short dialogue between a junior and a senior engineer about {t}.",
    "Give a step-by-step debugging plan for a production issue caused by misunderstanding {t}.",
    "Write the docstring and type hints for a module that implements {t}. Do not write the implementation.",
    "Translate this explanation into plain English for a non-technical manager: {t} trades memory or coordination for lower latency or higher throughput.",
    "Write a haiku about {t}, then explain the technical idea behind it.",
    "Draft a commit message for a change that adds {t} to an existing service.",
    "What are the first three things you would measure before optimizing a system that uses {t}? Explain why.",
]


def main():
    out = Path(__file__).resolve().parent / "prompts" / "chat_prompts.jsonl"
    out.parent.mkdir(exist_ok=True)
    rows = [{"prompt": task.format(t=topic)} for task, topic in itertools.product(TASKS, TOPICS)]
    out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(f"wrote {len(rows)} prompts to {out}")


if __name__ == "__main__":
    main()
