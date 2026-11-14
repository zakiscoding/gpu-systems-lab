"""Synthetic batches shared by jax_gpt.py and torch_gpt_bench.py, so both frameworks train on
exactly the same tokens. Same recipe as modal/fsdp_gpt.batch_for (a random 32-token motif
from the first --data-vocab ids, repeated), drawn from numpy's RNG seeded by (seed, step)."""

import numpy as np


def batch_for(step, args, n_rows=None):
    rows = n_rows or args.batch
    g = np.random.default_rng(args.seed * 1_000_003 + step * 1009)
    motif = g.integers(0, min(args.data_vocab, args.vocab), size=(rows, 32), dtype=np.int32)
    toks = np.tile(motif, (1, (args.seq + 1) // 32 + 1))[:, : args.seq + 1]
    return toks[:, :-1], toks[:, 1:]
