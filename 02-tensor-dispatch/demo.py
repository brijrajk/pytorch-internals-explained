"""
demo.py — the PyTorch dispatcher, built from scratch

After building (see README), run from 02-tensor-dispatch/:
    python demo.py

The key idea: Python calls dispatch.scale(tensor, factor).
It never picks the kernel. The dispatcher reads the tensor's
DispatchKey and looks up the matching C++ function in the table.

This is exactly how torch.add, torch.matmul, and every other
PyTorch op selects its CPU vs CUDA vs Meta implementation.

GPU behaviour:
  - CUDA or ROCm GPU detected  → CUDA kernel runs on real hardware
  - No GPU / no GPU support    → CUDA kernel falls back to simulation,
                                  clearly labelled in the output
"""

import sys

import dispatch  # compiled C++ module


def section(title):
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")
    sys.stdout.flush()  # sync Python stdout with C++ stdout before kernel output


# ── Setup: register kernels ───────────────────────────────
print("\nRegistering kernels...")
sys.stdout.flush()
dispatch.register_ops()  # prints GPU mode: HARDWARE or SIMULATED


# ── 1. CPU tensor → CPU kernel ────────────────────────────
section("CPU tensor → CPU kernel")

x = dispatch.Tensor([1.0, 2.0, 3.0], dispatch.DispatchKey.CPU)
print(f"  x.device = {x.device()}")
result = dispatch.scale(x, 2.0)
print(f"  result   = {result.data}")
# dispatch.scale didn't pick scale_cpu — the dispatcher did.
# Python passed a tensor whose key == CPU; the table returned scale_cpu.


# ── 2. CUDA tensor → CUDA kernel (hardware or simulated) ──
if dispatch.cuda_available():
    section("CUDA tensor → CUDA kernel  [HARDWARE]")
else:
    section("CUDA tensor → CUDA kernel  [SIMULATED — no GPU detected]")

x = dispatch.Tensor([1.0, 2.0, 3.0], dispatch.DispatchKey.CUDA)
print(f"  x.device = {x.device()}")
result = dispatch.scale(x, 2.0)
print(f"  result   = {result.data}")
# Same Python call as section 1. Different key. Different kernel.
# On a machine with a GPU the output above shows device name + VRAM.
# Without a GPU the kernel prints SIMULATED and computes on CPU memory.


# ── 3. Meta tensor → Meta kernel (shape, no data) ─────────
section("Meta tensor → Meta kernel (shape inference only)")

x = dispatch.Tensor([1.0, 2.0, 3.0], dispatch.DispatchKey.Meta)
print(f"  x.device     = {x.device()}")
result = dispatch.scale(x, 2.0)
print(f"  result.numel = {result.numel()}  ← shape known, data never touched")
# torch.compile and torch.export use Meta tensors to trace the
# computation graph without running any actual math.


# ── 4. Unregistered key → runtime error ───────────────────
section("Unregistered key → runtime error")

x = dispatch.Tensor([1.0, 2.0, 3.0], dispatch.DispatchKey.XPU)
print(f"  x.device = {x.device()}")
try:
    dispatch.scale(x, 2.0)
except RuntimeError as e:
    print(f"  RuntimeError: {e}")
# This is the exact error you see in real PyTorch when a model runs
# on a device that hasn't registered kernels for an op.
# Adding XPU support = registering kernels. Nothing else.


# ── 5. The dispatch table ─────────────────────────────────
section("What the dispatch table looks like from Python")

gpu_mode = "HARDWARE" if dispatch.cuda_available() else "SIMULATED"
print("  Registered kernels for 'scale':")
print(f"    CPU  → scale_cpu          (hardware)")
print(f"    CUDA → scale_cuda         ({gpu_mode})")
print(f"    Meta → scale_meta         (shape only — no computation)")
print(f"    XPU  → (not registered)")
print()
print("  Real PyTorch equivalent:")
print("    python -c \"import torch; print(torch._C._dispatch_dump('aten::add.Tensor'))\"")
print("    → ~30 registered backends for a single op")
