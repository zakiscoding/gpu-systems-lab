"""Load the labops::rmsnorm C++/CUDA operator and make it torch.compile-friendly.

    import rmsnorm_op
    y = torch.ops.labops.rmsnorm(x, weight, 1e-6)        # or rmsnorm_op.rmsnorm(x, weight)

What makes it work under torch.compile:
  * the op is defined in C++ with a schema (TORCH_LIBRARY), so Dynamo sees an opaque op with
    known semantics instead of an untraceable Python call;
  * register_fake gives it a meta/"fake tensor" implementation (output shape/dtype only), so
    Inductor can plan memory around it without running it;
  * register_autograd makes it differentiable: the backward is written with ordinary PyTorch ops,
    which Inductor is then free to fuse.
"""

import os

import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))

_ext = load(name="labops_rmsnorm", sources=[os.path.join(HERE, "rmsnorm.cu")],
            extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)


@torch.library.register_fake("labops::rmsnorm")
def _(x, weight, eps):
    torch._check(weight.shape == (x.shape[-1],), lambda: "weight must be [hidden]")
    return torch.empty_like(x)


@torch.library.register_fake("labops::rmsnorm_naive")
def _(x, weight, eps):
    return torch.empty_like(x)


def _setup_context(ctx, inputs, output):
    x, weight, eps = inputs
    ctx.save_for_backward(x, weight)
    ctx.eps = eps


def _backward(ctx, gy):
    # y = x * r * w with r = rsqrt(mean(x^2) + eps); in fp32 for accuracy.
    x, w = ctx.saved_tensors
    xf, wf, gf = x.float(), w.float(), gy.float()
    r = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + ctx.eps)
    gxhat = gf * wf
    gx = r * gxhat - xf * r.pow(3) * (gxhat * xf).mean(-1, keepdim=True)
    gw = (gf * xf * r).reshape(-1, x.shape[-1]).sum(0)
    return gx.to(x.dtype), gw.to(w.dtype), None


torch.library.register_autograd("labops::rmsnorm", _backward, setup_context=_setup_context)


def rmsnorm(x, weight, eps=1e-6):
    return torch.ops.labops.rmsnorm(x, weight, eps)


def rmsnorm_reference(x, weight, eps=1e-6):
    """Plain PyTorch RMSNorm (what torch.compile turns into an Inductor-generated Triton kernel)."""
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * weight.float()).to(x.dtype)
