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

```bash
# Python 3.8+
pip install pybind11

# CMake 3.15+
# macOS:   brew install cmake
# Ubuntu:  sudo apt install cmake
# Windows: https://cmake.org/download
```

---

## Build

```bash
# From this folder
mkdir build && cd build
cmake ..
make

# A file called ops.so (or ops.pyd on Windows) appears next to demo.py
```

---

## Run

```bash
# Back in 01-pybind11-basics/
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
  module   : <module 'ops' from '...'>
  docstring: Simple C++ ops exposed to Python via pybind11
  functions: ['add', 'dot', 'multiply']

──────────────────────────────────────────────────
  Error handling (mismatched lengths)
──────────────────────────────────────────────────
  Caught Python ValueError: Vectors must have the same length
```

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
