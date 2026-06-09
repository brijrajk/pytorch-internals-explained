# pytorch-internals-explained

A series of minimal, runnable demos that show what happens inside PyTorch — one layer at a time.

Each folder matches a LinkedIn post. The code is intentionally small. The goal is to make one concept obvious, not to be a complete implementation.

---

## Contents

- [Series](#series)
- [Why this repo exists](#why-this-repo-exists)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Author](#author)

---

## Series

| # | Topic | Post |
|---|-------|------|
| [01 — pybind11 basics](./01-pybind11-basics/) | How Python calls C++ — the bridge PyTorch is built on | [LinkedIn ↗](#) |
| [02 — tensor dispatch](./02-tensor-dispatch/) | How PyTorch picks the right kernel (CPU / CUDA / MPS) at runtime | [LinkedIn ↗](#) |
| 03 — autograd internals | How gradients flow backward through the computation graph | coming soon |

---

## Why this repo exists

PyTorch feels like Python. It is mostly C++.

Understanding that boundary — and the layers below it — changes how you debug, optimise, and reason about what your models are actually doing.

These demos are the shortest path I've found to making that concrete.

---

## Prerequisites

### Python 3.11+

**Ubuntu / Debian**
```bash
sudo apt update && sudo apt install -y python3.11 python3.11-dev
```

**macOS (Homebrew)**
```bash
brew install python@3.11
```

**Windows** — download the installer from [python.org](https://www.python.org/downloads/).

---

### CMake 3.15+

**Ubuntu / Debian**
```bash
sudo apt install -y cmake
```

**macOS**
```bash
brew install cmake
```

**Windows** — download from [cmake.org](https://cmake.org/download/) and add to PATH during install.

Verify: `cmake --version`

---

### C++17 compiler

**Ubuntu / Debian — GCC**
```bash
sudo apt install -y build-essential
```

**macOS — Clang (ships with Xcode Command Line Tools)**
```bash
xcode-select --install
```

**Windows — MSVC** — install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and select the **Desktop development with C++** workload.

---

### uv

[uv](https://docs.astral.sh/uv/) is a fast Python package and project manager (replaces pip + venv).

**Linux / macOS**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell)**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify: `uv --version`

---

## Installation

**1. Install uv**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**2. Clone the repo**

```bash
git clone https://github.com/brijrajk/pytorch-internals-explained.git
cd pytorch-internals-explained
```

**3. Create a virtual environment and install dependencies**

```bash
uv venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv pip install pybind11
```

Each folder has its own README with build and run instructions.

---

## Author

**Brij Raj Kishore** — Senior SMTS at ZettaBolt
Working on PyTorch OOT device support and LLM acceleration.

[LinkedIn](https://linkedin.com/in/brijrajkishore) · [GitHub](https://github.com/brijrajk)
