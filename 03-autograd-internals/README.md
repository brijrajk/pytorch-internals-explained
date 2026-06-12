# 03 — autograd internals

This is the code behind the LinkedIn post:
**"loss.backward() is not magic. It's a graph walk your forward pass already paid for."**

We build a minimal PyTorch-style autograd engine from scratch using pybind11.
No PyTorch install needed — the whole mechanism is explicit in ~200 lines of C++.

---

## What's in here

| File | What it does |
|------|-------------|
| `autograd.cpp` | `Tensor` with `.grad`/`.grad_fn`, graph `Node`s, `add`/`mul` ops, `backward()`, `detach()` |
| `demo.py` | Builds graphs, runs backward, shows accumulation, no-grad mode, and detach |
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

Run these commands from inside the `03-autograd-internals/` directory:

```bash
cd 03-autograd-internals            # must be here — CMakeLists.txt lives in this folder
source ../.venv/bin/activate        # activate the venv so cmake finds the right Python + pybind11
mkdir build && cd build
cmake ..                            # .. points to 03-autograd-internals/, where CMakeLists.txt is
make
```

A platform-specific extension module will appear next to `demo.py`, named after your Python version and architecture — for example:

```
autograd.cpython-314-x86_64-linux-gnu.so   # Linux, Python 3.14
autograd.cpython-311-arm64-darwin.so       # macOS Apple Silicon, Python 3.11
autograd.cp311-win_amd64.pyd               # Windows, Python 3.11
```

**Cleaning up:**

```bash
make clean        # remove build tree artifacts (keeps the .so next to demo.py)
make clean-all    # remove everything, including the .so next to demo.py
```

---

## Run

From the `03-autograd-internals/` directory (with the venv active):

```bash
source ../.venv/bin/activate     # skip if already active
python demo.py
```

Expected output (abridged):

```
──────────────────────────────────────────────────
  Forward pass records the graph
──────────────────────────────────────────────────
  [graph] recorded MulBackward
  c.data    = [8.0, 15.0]
  c.grad_fn = MulBackward
  c.is_leaf = False

──────────────────────────────────────────────────
  backward() applies the chain rule
──────────────────────────────────────────────────
  [graph] recorded AddBackward
  y.data    = [10.0, 18.0]
  y.grad_fn = AddBackward

  [backward] seed gradient = all ones
  [backward] executing AddBackward
  [backward] executing MulBackward
  [backward] AccumulateGrad → leaf.grad updated
  [backward] AccumulateGrad → leaf.grad updated
  [backward] AccumulateGrad → leaf.grad updated

  a.grad = [5.0, 6.0]   ← dy/da = b + 1 = [5.0, 6.0]
  b.grad = [2.0, 3.0]   ← dy/db = a     = [2.0, 3.0]

──────────────────────────────────────────────────
  Gradients accumulate across backward calls
──────────────────────────────────────────────────
  after 1st backward: x.grad = [10.0, 20.0]
  after 2nd backward: x.grad = [20.0, 40.0]  ← doubled!

──────────────────────────────────────────────────
  requires_grad=False builds no graph
──────────────────────────────────────────────────
  r.grad_fn = None (leaf tensor)
  RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

---

## What to notice

**The forward pass does two jobs.**

```
c = mul(a, b)
      │
      ├── job 1: compute  c.data = a.data * b.data
      └── job 2: record   c.grad_fn = MulBackward
                          (with a.data and b.data SAVED for later)
```

This is why training uses more memory than inference: every op that needs its
inputs for the chain rule keeps them alive until backward runs. PyTorch calls
these *saved tensors* (`ctx.save_for_backward`).

**backward() is just a reverse graph walk.**
Seed the output gradient with ones, visit each node, apply its chain-rule
function, and pass the results down the edges. No symbolic math, no calculus
at runtime — every op registered its derivative as a closure during forward.

**Gradients accumulate — they never overwrite.**
`AccumulateGrad` does `leaf.grad += incoming`. Call backward twice and your
gradients double. This is the entire reason `optimizer.zero_grad()` exists.

**`requires_grad=False` means no graph, not a disabled graph.**
When no input requires grad, the op records nothing — zero memory, zero
bookkeeping. That's all `torch.no_grad()` does: temporarily stop recording.

**`detach()` cuts history, not data.**
Same values, no `grad_fn`. This is how you stop gradients at a boundary —
target networks in RL, frozen backbones, truncated BPTT.

---

## How this relates to PyTorch

| This demo | Real PyTorch |
|-----------|--------------|
| `Node` with `backward_fn` | `torch::autograd::Node` (`AddBackward0`, `MulBackward0`) |
| saved input copies | `ctx.save_for_backward` / saved-tensor hooks |
| `leaf.grad += g` | `AccumulateGrad` (literally the same name) |
| recursive `backward_impl` | `torch/csrc/autograd/engine.cpp` — queue-based, multi-threaded |
| `detach()` | `tensor.detach()` |

And remember post 02: **Autograd is just another DispatchKey.** When a tensor
requires grad, the dispatcher routes through `AutogradCPU` first — that wrapper
records the node, then re-dispatches to the real CPU kernel. The two posts
describe the same machine.

---

## Next

`04 — strides & views` — how tensors actually live in memory: zero-copy
transpose, slicing, broadcasting, and what `.contiguous()` really does.
Coming soon.
