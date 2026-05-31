# 01 — pybind11 basics

This is the code behind the LinkedIn post:
**"You're writing Python. Your GPU is doing a billion operations a second. Something doesn't add up."**

The idea is simple: write a C++ function, wrap it with pybind11, call it from Python.
That's exactly what PyTorch does — at a much larger scale.

---

## What's in here

| File | What it does |
|------|-------------|
| `add.cpp` | Pure C++ functions (`add`, `multiply`, `dot`) + pybind11 binding |
| `demo.py` | Calls those functions from Python as if they were native |
| `CMakeLists.txt` | Build configuration |

---

## Prerequisites

- Python 3.11+, CMake 3.15+, a C++17 compiler — see the [root README](../README.md#prerequisites) for platform-specific install steps.
- [uv](https://docs.astral.sh/uv/) for managing the Python environment.

Create the venv and install pybind11:

```bash
uv venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
uv pip install pybind11
```

---

## Build

Run these commands from inside the `01-pybind11-basics/` directory:

```bash
cd 01-pybind11-basics               # must be here — CMakeLists.txt lives in this folder
source ../.venv/bin/activate        # activate the venv so cmake finds the right Python + pybind11
mkdir build && cd build
cmake ..                            # .. points to 01-pybind11-basics/, where CMakeLists.txt is
make
```

A file called `ops.so` (Linux/macOS) or `ops.pyd` (Windows) will appear next to `demo.py`.

**Cleaning up:**

```bash
# Remove only build tree artifacts (keeps the .so next to demo.py)
make clean

# Remove everything — build artifacts + the .so next to demo.py
make clean-all
```

---

## Run

From the `01-pybind11-basics/` directory (with the venv active):

```bash
source ../.venv/bin/activate     # skip if already active
python demo.py
```

Expected output:

```
──────────────────────────────────────────────────
  Element-wise addition
──────────────────────────────────────────────────
  a        = [1.0, 2.0, 3.0]
  b        = [4.0, 5.0, 6.0]
  add(a,b) = [5.0, 7.0, 9.0]

──────────────────────────────────────────────────
  Element-wise multiplication
──────────────────────────────────────────────────
  multiply(a,b) = [4.0, 10.0, 18.0]

──────────────────────────────────────────────────
  Dot product
──────────────────────────────────────────────────
  dot(a,b) = 32.0

──────────────────────────────────────────────────
  Inspecting the module
──────────────────────────────────────────────────
  module   : <module 'ops' from '.../ops.cpython-3XX-....so'>
  docstring: Simple C++ ops exposed to Python via pybind11 — mirrors how PyTorch works under the hood
  functions: ['add', 'dot', 'multiply']

──────────────────────────────────────────────────
  Error handling (mismatched lengths)
──────────────────────────────────────────────────
  Caught Python ValueError: Vectors must have the same length
```

> The `ValueError` on the last line is **intentional** — section 5 of the demo deliberately passes mismatched vectors to show that `std::invalid_argument` in C++ is automatically translated to a `ValueError` in Python by pybind11. This is expected output, not a crash.

---

## What to notice

**`add.cpp` has two distinct sections:**
- The top half is pure C++. No Python headers, no Python types.
- The bottom half is the pybind11 binding — the only part that knows about Python.

This mirrors exactly how PyTorch is structured. `ATen` (the tensor library) knows nothing about Python. pybind11 sits at the boundary and handles the translation.

**The binding is generated at compile time.**
By the time you run `import ops`, Python already knows exactly which C++ function maps to `ops.add`. There's no runtime lookup, no boxing, no overhead at the call site.

**Error handling is automatic.**
`std::invalid_argument` in C++ becomes `ValueError` in Python. pybind11 handles the translation. You don't write any conversion code.

---

## How this relates to PyTorch

When you write `a + b` in PyTorch, this is roughly what happens:

```
Python:  a + b
           ↓
pybind11:  ops.add(a, b)         ← boundary crossing
           ↓
C++:       at::add(a, b)         ← actual computation
           ↓
Kernels:   CUDA / CPU / MKL
```

The difference between this demo and PyTorch is scale — PyTorch has thousands of such bindings. The mechanism is identical.

---

## Next

`02-tensor-dispatch` — how PyTorch decides which kernel to call (CPU vs CUDA vs MPS) at runtime using the dispatcher. Coming soon.
