# pytorch-internals-explained

A series of minimal, runnable demos that show what happens inside PyTorch — one layer at a time.

Each folder matches a LinkedIn post. The code is intentionally small. The goal is to make one concept obvious, not to be a complete implementation.

---

## Series

| # | Topic | Post |
|---|-------|------|
| [01 — pybind11 basics](./01-pybind11-basics/) | How Python calls C++ — the bridge PyTorch is built on | [LinkedIn ↗](#) |
| 02 — tensor dispatch | How PyTorch picks the right kernel (CPU / CUDA / MPS) at runtime | coming soon |
| 03 — autograd internals | How gradients flow backward through the computation graph | coming soon |

---

## Why this repo exists

PyTorch feels like Python. It is mostly C++.

Understanding that boundary — and the layers below it — changes how you debug, optimise, and reason about what your models are actually doing.

These demos are the shortest path I've found to making that concrete.

---

## Prerequisites

- Python 3.8+
- CMake 3.15+
- A C++17 compiler (gcc, clang, or MSVC)
- `pip install pybind11`

Each folder has its own README with build and run instructions.

---

## Author

**Brij Raj Kishore** — Senior SMTS at ZettaBolt
Working on PyTorch OOT device support and LLM acceleration.

[LinkedIn](https://linkedin.com/in/) · [GitHub](https://github.com/)
