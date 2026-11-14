"""The JAX port computes the same function as the PyTorch model (CPU, fp32, no GPU needed).

Copies a randomly initialized modal/fsdp_gpt.GPT into the JAX parameter tree and checks the
logits, the loss and the gradients agree. Run: JAX_PLATFORMS=cpu python -m pytest jax-xla/test_parity.py
"""

import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "modal"))
sys.path.insert(0, HERE)
import jax_gpt  # noqa: E402
from fsdp_gpt import GPT, init_weights  # noqa: E402

V, T, D, L, H, B = 97, 16, 32, 2, 4, 3


def torch_to_jax(m):
    """torch GPT -> jax_gpt param tree (Linear weights transposed to (in, out))."""
    def t(x):
        return jnp.asarray(x.detach().numpy())
    return {"tok": t(m.tok.weight), "pos": t(m.pos.weight), "head": t(m.head.weight.T),
            "lnf_w": t(m.ln_f.weight), "lnf_b": t(m.ln_f.bias),
            "blocks": [{"ln1_w": t(b.ln1.weight), "ln1_b": t(b.ln1.bias), "qkv": t(b.qkv.weight.T),
                        "proj": t(b.proj.weight.T), "ln2_w": t(b.ln2.weight), "ln2_b": t(b.ln2.bias),
                        "fc1": t(b.fc1.weight.T), "fc2": t(b.fc2.weight.T)} for b in m.blocks]}


@pytest.fixture(scope="module")
def models():
    torch.manual_seed(0)
    m = GPT(V, T, D, L, H, ac=False)
    init_weights(m, L)
    return m, torch_to_jax(m)


def test_logits_match(models):
    m, params = models
    idx = np.random.default_rng(1).integers(0, V, size=(B, T)).astype(np.int64)
    ref = m(torch.from_numpy(idx)).detach().numpy()
    out = np.asarray(jax_gpt.forward(params, jnp.asarray(idx, jnp.int32), H, dtype=jnp.float32))
    assert out.shape == (B, T, V)
    np.testing.assert_allclose(out, ref, atol=2e-5, rtol=1e-4)


def test_loss_and_grads_match(models):
    m, params = models
    rng = np.random.default_rng(2)
    x = rng.integers(0, V, size=(B, T)).astype(np.int64)
    y = rng.integers(0, V, size=(B, T)).astype(np.int64)
    m.zero_grad()
    loss_t = torch.nn.functional.cross_entropy(m(torch.from_numpy(x)).view(-1, V), torch.from_numpy(y).view(-1))
    loss_t.backward()
    loss_j, g = jax.value_and_grad(jax_gpt.loss_fn)(params, jnp.asarray(x, jnp.int32), jnp.asarray(y, jnp.int32),
                                                     H, jnp.float32)
    assert abs(float(loss_j) - loss_t.item()) < 1e-5
    np.testing.assert_allclose(np.asarray(g["head"]), m.head.weight.grad.numpy().T, atol=1e-6, rtol=1e-4)
    np.testing.assert_allclose(np.asarray(g["blocks"][0]["qkv"]), m.blocks[0].qkv.weight.grad.numpy().T,
                               atol=1e-6, rtol=1e-3)
    np.testing.assert_allclose(np.asarray(g["tok"]), m.tok.weight.grad.numpy(), atol=1e-6, rtol=1e-3)


def test_data_is_deterministic():
    class A:
        batch, seq, vocab, data_vocab, seed = 2, 64, 4096, 512, 0
    x1, y1 = jax_gpt.batch_for(5, A)
    x2, _ = jax_gpt.batch_for(5, A)
    assert (x1 == x2).all() and x1.shape == (2, 64) and (x1[:, 1:] == y1[:, :-1]).all()
