"""
demo.py — calling C++ from Python via pybind11

After building (see README), run:
    python demo.py

You will be calling compiled C++ directly.
Python is just the caller. The computation happens in C++.
"""

import ops  # this is the compiled C++ module, not a Python file


def section(title):
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")


# ── 1. Basic add ──────────────────────────────────────────
section("Element-wise addition")

a = [1.0, 2.0, 3.0]
b = [4.0, 5.0, 6.0]

result = ops.add(a, b)
print(f"  a        = {a}")
print(f"  b        = {b}")
print(f"  add(a,b) = {result}")
# → [5.0, 7.0, 9.0]
# This called compiled C++. Python did not loop over anything.


# ── 2. Element-wise multiply ──────────────────────────────
section("Element-wise multiplication")

result = ops.multiply(a, b)
print(f"  multiply(a,b) = {result}")
# → [4.0, 10.0, 18.0]


# ── 3. Dot product ────────────────────────────────────────
section("Dot product")

result = ops.dot(a, b)
print(f"  dot(a,b) = {result}")
# → 32.0  (1*4 + 2*5 + 3*6)


# ── 4. What the module looks like from Python ─────────────
section("Inspecting the module")

print(f"  module   : {ops}")
print(f"  docstring: {ops.__doc__}")
print(f"  functions: {[f for f in dir(ops) if not f.startswith('_')]}")


# ── 5. Error handling crosses the boundary cleanly ────────
section("Error handling (mismatched lengths)")

try:
    ops.add([1.0, 2.0], [1.0, 2.0, 3.0])
except ValueError as e:
    print(f"  Caught Python ValueError: {e}")
# pybind11 automatically translates C++ std::invalid_argument
# into a Python ValueError. No boilerplate needed.
