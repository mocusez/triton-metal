"""Installed-wheel smoke test for the Triton Metal backend."""

from __future__ import annotations

import argparse

import torch
import triton
import triton.language as tl
from triton.backends import backends
from triton.backends.metal.driver import MetalDriver
from triton.compiler import ASTSource


@triton.jit
def _add_kernel(x, y, output, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    tl.store(output + offsets, tl.load(x + offsets) + tl.load(y + offsets))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-mps",
        action="store_true",
        help="fail instead of running compile-only validation when MPS is unavailable",
    )
    args = parser.parse_args()

    assert set(backends) == {"metal"}, f"unexpected backend payload: {sorted(backends)}"
    target = MetalDriver().get_current_target()
    source = ASTSource(
        fn=_add_kernel,
        signature={"x": "*fp32", "y": "*fp32", "output": "*fp32", "BLOCK_SIZE": "constexpr"},
        constexprs={"BLOCK_SIZE": 64},
    )
    compiled = triton.compile(source, target=target, options={"num_warps": 1})
    assert "metal" in compiled.asm

    mps_error = None
    try:
        mps_usable = torch.backends.mps.is_available() and torch.empty(1, device="mps").device.type == "mps"
    except RuntimeError as exc:
        mps_usable = False
        mps_error = exc

    if not mps_usable:
        if args.require_mps:
            raise RuntimeError("MPS is required for the installed-wheel numeric smoke test") from mps_error
        print(
            f"installed Metal wheel compiled a kernel for {target}; "
            "MPS numeric smoke skipped because this runner exposes no usable device"
        )
        return

    x = torch.arange(64, dtype=torch.float32, device="mps")
    y = torch.arange(64, dtype=torch.float32, device="mps")
    output = torch.empty_like(x)
    _add_kernel[(1,)](x, y, output, BLOCK_SIZE=64, num_warps=1)
    torch.mps.synchronize()
    torch.testing.assert_close(output.cpu(), (x + y).cpu(), rtol=0, atol=0)
    print(f"installed Metal wheel passed compile and MPS numeric smoke for {target}")


if __name__ == "__main__":
    main()
