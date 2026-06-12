"""
demo.py — a PyTorch-style autograd engine, built from scratch

After building (see README), run from 03-autograd-internals/:
    python demo.py

The key idea: the forward pass does two jobs, not one.
It computes the result AND records a graph of backward nodes.
loss.backward() just walks that graph in reverse, applying the
chain rule at every node and accumulating into leaf .grad fields.

This is exactly what happens between `loss.backward()` and
`optimizer.step()` in every PyTorch training loop.
"""

import sys

import autograd  # compiled C++ module

# C++ flushes its log lines immediately; make Python do the same so the
# two output streams interleave in true execution order.
sys.stdout.reconfigure(line_buffering=True)


def section(title):
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")


# ── 1. The forward pass records a graph ───────────────────
section("Forward pass records the graph")

a = autograd.Tensor([2.0, 3.0], requires_grad=True)
b = autograd.Tensor([4.0, 5.0], requires_grad=True)

c = autograd.mul(a, b)  # c = a * b
print(f"  c.data    = {c.data}")
print(f"  c.grad_fn = {c.grad_fn}")
print(f"  c.is_leaf = {c.is_leaf()}")
# c was created by an op, so it carries grad_fn=MulBackward.
# a and b were created by the user — they are leaves, grad_fn=None.


# ── 2. backward() walks the graph in reverse ──────────────
section("backward() applies the chain rule")

# y = a * b + a   (a is used twice — watch what happens to a.grad)
y = autograd.add(c, a)
print(f"  y.data    = {y.data}")
print(f"  y.grad_fn = {y.grad_fn}")
print()
autograd.backward(y)
print()
print(f"  a.grad = {a.grad}   ← dy/da = b + 1 = [5.0, 6.0]")
print(f"  b.grad = {b.grad}   ← dy/db = a     = [2.0, 3.0]")
# The chain rule, executed mechanically:
#   AddBackward passes the gradient through to c and a
#   MulBackward turns c's gradient into b-scaled and a-scaled gradients
#   AccumulateGrad sums everything arriving at each leaf


# ── 3. Gradients ACCUMULATE — they don't overwrite ────────
section("Gradients accumulate across backward calls")

x = autograd.Tensor([1.0, 1.0], requires_grad=True)
w = autograd.Tensor([10.0, 20.0], requires_grad=True)

out1 = autograd.mul(x, w)
autograd.backward(out1)
print(f"  after 1st backward: x.grad = {x.grad}")

out2 = autograd.mul(x, w)
autograd.backward(out2)
print(f"  after 2nd backward: x.grad = {x.grad}  ← doubled!")
# This is WHY optimizer.zero_grad() exists.
# PyTorch never resets .grad for you — it always does grad +=.


# ── 4. requires_grad=False → no graph, no overhead ────────
section("requires_grad=False builds no graph")

p = autograd.Tensor([1.0, 2.0])  # requires_grad defaults to False
q = autograd.Tensor([3.0, 4.0])
r = autograd.mul(p, q)  # note: no "[graph] recorded" line printed
print(f"  r.data    = {r.data}")
print(f"  r.grad_fn = {r.grad_fn}")
try:
    autograd.backward(r)
except RuntimeError as e:
    print(f"  RuntimeError: {e}")
# Same error message real PyTorch gives. This is also what
# torch.no_grad() does under the hood — it just stops recording.


# ── 5. detach() cuts the graph ────────────────────────────
section("detach() disconnects a tensor from the graph")

a2 = autograd.Tensor([2.0], requires_grad=True)
b2 = autograd.mul(a2, a2)  # b2 has grad_fn=MulBackward
b2_free = autograd.detach(b2)  # same data, no history
print(f"  b2.grad_fn      = {b2.grad_fn}")
print(f"  detached.grad_fn = {b2_free.grad_fn}")
print(f"  detached.requires_grad = {b2_free.requires_grad}")
# .detach() is how you stop gradients at a boundary —
# e.g. target networks in RL, or freezing parts of a model.


# ── 6. The big picture ────────────────────────────────────
section("What PyTorch does with this")

print("""  Every training loop runs this machinery:

    loss = model(x)            # forward: compute + record graph
    loss.backward()            # reverse walk + chain rule
    optimizer.step()           # use the .grad fields
    optimizer.zero_grad()      # because gradients ACCUMULATE

  Real PyTorch equivalent of what we built:
    Node          → torch::autograd::Node   (AddBackward0, MulBackward0)
    saved inputs  → ctx.save_for_backward
    AccumulateGrad→ same name in PyTorch!
    our backward()→ torch/csrc/autograd/engine.cpp (with a thread pool)""")
