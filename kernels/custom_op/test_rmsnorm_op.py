"""Tests for labops::rmsnorm (needs a CUDA GPU): python -m pytest kernels/custom_op/test_rmsnorm_op.py"""

import os
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def op():
    import rmsnorm_op
    return rmsnorm_op


@pytest.mark.parametrize("dtype,tol", [(torch.float32, 1e-5), (torch.float16, 2e-3), (torch.bfloat16, 2e-2)])
@pytest.mark.parametrize("shape", [(3, 7), (64, 1024), (8, 33, 4096), (2, 5000)])
def test_matches_reference(op, dtype, tol, shape):
    x = torch.randn(*shape, device="cuda", dtype=dtype)
    w = torch.randn(shape[-1], device="cuda", dtype=dtype)
    torch.testing.assert_close(op.rmsnorm(x, w), op.rmsnorm_reference(x, w), atol=tol, rtol=tol)
    torch.testing.assert_close(torch.ops.labops.rmsnorm_naive(x, w, 1e-6), op.rmsnorm_reference(x, w),
                               atol=tol, rtol=tol)


def test_unaligned_input_uses_scalar_path(op):
    """A view starting one bf16 element into a buffer is not 16-byte aligned, so the vector
    kernel must not run; the result must still be right."""
    flat = torch.randn(4 * 4096 + 1, device="cuda", dtype=torch.bfloat16)
    xm = flat[1:].view(4, 4096)
    w = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(op.rmsnorm(xm, w), op.rmsnorm_reference(xm, w), atol=2e-2, rtol=2e-2)


def test_opcheck(op):
    """PyTorch's own custom-op checker: schema, fake tensor, autograd registration, and
    the AOT-dispatch tests that torch.compile relies on."""
    x = torch.randn(16, 256, device="cuda", requires_grad=True)
    w = torch.randn(256, device="cuda", requires_grad=True)
    torch.library.opcheck(torch.ops.labops.rmsnorm.default, (x, w, 1e-6))


def test_gradients_match_reference(op):
    x = torch.randn(32, 512, device="cuda", dtype=torch.float32, requires_grad=True)
    w = torch.randn(512, device="cuda", dtype=torch.float32, requires_grad=True)
    gy = torch.randn(32, 512, device="cuda")
    op.rmsnorm(x, w).backward(gy)
    gx, gw = x.grad.clone(), w.grad.clone()
    x.grad = w.grad = None
    op.rmsnorm_reference(x, w).backward(gy)
    torch.testing.assert_close(gx, x.grad, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(gw, w.grad, atol=1e-3, rtol=1e-4)


def test_torch_compile_fullgraph(op):
    """The op traces under torch.compile(fullgraph=True): no graph break, same output."""
    lin = torch.nn.Linear(1024, 1024, device="cuda")
    w = torch.ones(1024, device="cuda")

    def f(x):
        return lin(op.rmsnorm(x, w)).relu()

    x = torch.randn(64, 1024, device="cuda")
    torch._dynamo.reset()
    out = torch.compile(f, fullgraph=True)(x)
    torch.testing.assert_close(out, f(x), atol=1e-5, rtol=1e-5)
