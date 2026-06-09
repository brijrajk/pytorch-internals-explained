# 02 — tensor dispatch

This is the code behind the LinkedIn post:
**"torch.add doesn't know what a CPU is. The dispatcher does."**

We build a minimal PyTorch-style dispatcher from scratch using pybind11.
No PyTorch install needed — the concept is made explicit in ~150 lines of C++.

The CUDA kernel runs on **real GPU hardware** when a CUDA or ROCm device is
detected at runtime, and falls back to a clearly-labelled simulation otherwise.
The same demo works on any machine.

---

## What's in here

| File | What it does |
|------|-------------|
| `dispatch.cpp` | `DispatchKey`, `Tensor`, `KernelRegistry`, kernels, pybind11 binding |
| `dispatch_gpu.cu` | Real GPU kernel — compiles for CUDA and ROCm/HIP without changes |
| `dispatch_gpu.h` | Header declaring `scale_on_gpu` |
| `demo.py` | Dispatches `scale` to CPU / CUDA / Meta / XPU and shows the table |
| `CMakeLists.txt` | Detects CUDA → ROCm → no GPU; compiles accordingly |

---

## Prerequisites

- Python 3.11+, CMake 3.18+, a C++17 compiler — see the [root README](../README.md#prerequisites) for platform-specific install steps.
- [uv](https://docs.astral.sh/uv/) for managing the Python environment.
- **Optional:** CUDA Toolkit or ROCm — required only for the hardware GPU path. Without either, the CUDA kernel falls back to simulation automatically.

Create the venv and install pybind11:

```bash
uv venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
uv pip install pybind11
```

---

## Build

Run these commands from inside the `02-tensor-dispatch/` directory.
CMake auto-detects CUDA or ROCm and enables the GPU kernel if found.

```bash
cd 02-tensor-dispatch               # must be here — CMakeLists.txt lives in this folder
source ../.venv/bin/activate        # activate the venv so cmake finds the right Python + pybind11
mkdir build && cd build
cmake ..                            # detects CUDA → ROCm → no GPU automatically
make
```

CMake will print one of:

```
-- Found CUDA 12.x          — building GPU kernel
-- Found ROCm/HIP x.x       — building GPU kernel
-- No CUDA or ROCm found    — GPU kernel will run in simulation mode
```

A platform-specific extension module will appear next to `demo.py`, named after your Python version and architecture — for example:

```
dispatch.cpython-314-x86_64-linux-gnu.so   # Linux, Python 3.14
dispatch.cpython-311-arm64-darwin.so       # macOS Apple Silicon, Python 3.11
dispatch.cp311-win_amd64.pyd               # Windows, Python 3.11
```

**Cleaning up:**

```bash
make clean        # remove build tree artifacts (keeps the .so next to demo.py)
make clean-all    # remove everything, including the .so next to demo.py
```

---

## Run

From the `02-tensor-dispatch/` directory (with the venv active):

```bash
source ../.venv/bin/activate     # skip if already active
python demo.py
```

### On a machine **without** a GPU (simulation mode)

```
Registering kernels...
  [registry] no GPU detected — CUDA kernel → SIMULATED

──────────────────────────────────────────────────
  CPU tensor → CPU kernel
──────────────────────────────────────────────────
  [dispatcher] key=CPU → kernel selected
  [CPU kernel]  executing scale
  x.device = CPU
  result   = [2.0, 4.0, 6.0]

──────────────────────────────────────────────────
  CUDA tensor → CUDA kernel  [SIMULATED — no GPU detected]
──────────────────────────────────────────────────
  [dispatcher] key=CUDA → kernel selected
  [CUDA kernel] *** SIMULATED (no GPU hardware detected) ***
  [CUDA kernel] computing on CPU memory as stand-in
  x.device = CUDA
  result   = [2.0, 4.0, 6.0]
  ...
```

### On a machine **with** a GPU (hardware mode)

```
Registering kernels...
  [registry] GPU detected   — CUDA kernel → HARDWARE

──────────────────────────────────────────────────
  CUDA tensor → CUDA kernel  [HARDWARE]
──────────────────────────────────────────────────
  [dispatcher] key=CUDA → kernel selected
  [CUDA kernel] *** RUNNING ON HARDWARE GPU ***
  [CUDA kernel] device : NVIDIA GeForce RTX 4090
  [CUDA kernel] VRAM   : 24564 MB
  [CUDA kernel] compute: 8.9
  x.device = CUDA
  result   = [2.0, 4.0, 6.0]
  ...
```

> The device name, VRAM, and compute capability confirm that the kernel ran on real hardware.

---

## What to notice

**`dispatch.cpp` has three distinct layers:**

```
Python call        dispatch.scale(x, 2.0)
                         ↓
Dispatcher         reads x.key → O(1) table lookup → calls fn
                         ↓
Kernel             scale_cpu  /  scale_cuda  /  scale_meta
```

The kernel has no `if device == "cuda"` logic. The dispatcher handles routing entirely. This is the invariant that lets PyTorch add new backends without touching existing kernels.

**The CUDA kernel prefers hardware, falls back to simulation.**
If a CUDA or ROCm device is found at runtime, `dispatch_gpu.cu` runs the real
GPU kernel. Otherwise the same dispatch path is exercised with CPU math, and
every output line is clearly marked `SIMULATED` so there's no ambiguity.

**The Meta kernel runs no math.**
`scale_meta` returns an empty tensor. Its only job is to acknowledge the shape.
`torch.compile` and `torch.export` use this to trace graphs without running
any computation.

**The XPU error is the real error.**
When you run a PyTorch model on a device with no registered kernels for an op,
you get exactly this message. Adding device support means registering kernels —
that's all.

**The dispatch table is built at registration time.**
`register_ops()` fills the table once. Every subsequent `dispatch.scale()` call
is a single hash lookup. No runtime type checking, no Python overhead.

---

## How this relates to PyTorch

PyTorch's real dispatcher lives in `c10/core/Dispatch.h`. The mechanism is identical:

```cpp
// Real PyTorch — registering a CPU kernel
TORCH_LIBRARY_IMPL(aten, CPU, m) {
    m.impl("add.Tensor", &at::cpu::add);
}

// Real PyTorch — registering a CUDA kernel
TORCH_LIBRARY_IMPL(aten, CUDA, m) {
    m.impl("add.Tensor", &at::cuda::add);
}
```

Our `scale_registry.register_kernel(DispatchKey::CPU, scale_cpu)` is the same
thing, without the macro.

---

## Next

`03-autograd-internals` — how gradients flow backward through the computation graph. Coming soon.
