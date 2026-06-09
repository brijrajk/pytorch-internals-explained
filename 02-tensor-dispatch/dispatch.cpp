#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef WITH_CUDA
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
#include <hip/hip_runtime.h>
#define gpuGetDeviceCount hipGetDeviceCount
using gpuError_t = hipError_t;
#define gpuSuccess hipSuccess
#else
#include <cuda_runtime.h>
#define gpuGetDeviceCount cudaGetDeviceCount
using gpuError_t = cudaError_t;
#define gpuSuccess cudaSuccess
#endif
#include "dispatch_gpu.h"
#endif

namespace py = pybind11;

// ─────────────────────────────────────────────────────────────────────────────
// DispatchKey — every tensor carries one.
// PyTorch has ~100 keys (CPU, CUDA, MPS, XPU, Meta, Autograd*, ...).
// We use four to make the concept concrete.
// ─────────────────────────────────────────────────────────────────────────────

enum class DispatchKey { CPU, CUDA, Meta, XPU };

std::string key_name(DispatchKey k) {
  switch (k) {
    case DispatchKey::CPU:  return "CPU";
    case DispatchKey::CUDA: return "CUDA";
    case DispatchKey::Meta: return "Meta";
    case DispatchKey::XPU:  return "XPU";
  }
  return "Unknown";
}

// ─────────────────────────────────────────────────────────────────────────────
// Tensor — data + device tag.
// The tag is the only thing the dispatcher reads.
// ─────────────────────────────────────────────────────────────────────────────

struct Tensor {
  std::vector<float> data;
  DispatchKey key;

  Tensor(std::vector<float> d, DispatchKey k) : data(std::move(d)), key(k) {}

  std::string device() const { return key_name(key); }
  size_t numel() const { return data.size(); }
};

// ─────────────────────────────────────────────────────────────────────────────
// KernelRegistry — the dispatch table.
// In PyTorch this is c10::OperatorEntry's per-key jump table.
// One entry per registered backend; lookup is O(1).
// ─────────────────────────────────────────────────────────────────────────────

using KernelFn = std::function<Tensor(const Tensor&, float)>;

struct KernelRegistry {
  std::unordered_map<int, KernelFn> table;

  void register_kernel(DispatchKey key, KernelFn fn) {
    table[static_cast<int>(key)] = std::move(fn);
  }

  Tensor dispatch(const Tensor& input, float factor) const {
    auto it = table.find(static_cast<int>(input.key));
    if (it == table.end()) {
      throw std::runtime_error(
          "Could not run 'scale': no kernel registered for DispatchKey::" +
          key_name(input.key) +
          "\n  (In real PyTorch this is the error you see with unsupported "
          "devices)");
    }
    std::cout << "  [dispatcher] key=" << key_name(input.key)
              << " → kernel selected" << std::endl;
    return it->second(input, factor);
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// Runtime GPU detection
// ─────────────────────────────────────────────────────────────────────────────

bool cuda_available() {
#ifdef WITH_CUDA
  int count = 0;
  gpuError_t err = gpuGetDeviceCount(&count);
  return (err == gpuSuccess && count > 0);
#else
  return false;
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernels — each knows about exactly one backend, nothing else.
// This separation is the whole point: business logic stays out of dispatch.
// ─────────────────────────────────────────────────────────────────────────────

Tensor scale_cpu(const Tensor& t, float factor) {
  std::cout << "  [CPU kernel]  executing scale" << std::endl;
  std::vector<float> out(t.numel());
  for (size_t i = 0; i < t.numel(); ++i) out[i] = t.data[i] * factor;
  return Tensor(out, DispatchKey::CPU);
}

Tensor scale_cuda(const Tensor& t, float factor) {
#ifdef WITH_CUDA
  if (cuda_available()) {
    // ── HARDWARE PATH ──────────────────────────────────────────────────────
    // scale_on_gpu: H2D → kernel launch → D2H.
    // Prints device name + VRAM so it is unambiguous that real GPU ran.
    std::cout << "  [CUDA kernel] *** RUNNING ON HARDWARE GPU ***" << std::endl;
    return Tensor(scale_on_gpu(t.data, factor), DispatchKey::CUDA);
  }
#endif
  // ── SIMULATION FALLBACK ───────────────────────────────────────────────────
  // No GPU hardware (or built without GPU support) — simulate on CPU memory
  // so the dispatch path is still exercised. Output is clearly labelled.
  std::cout << "  [CUDA kernel] *** SIMULATED (no GPU hardware detected) ***"
            << std::endl;
  std::cout << "  [CUDA kernel] computing on CPU memory as stand-in"
            << std::endl;
  std::vector<float> out(t.numel());
  for (size_t i = 0; i < t.numel(); ++i) out[i] = t.data[i] * factor;
  return Tensor(out, DispatchKey::CUDA);
}

// Meta kernel: shape inference only — no data touched.
// Used by torch.compile and torch.export to trace graphs without running them.
Tensor scale_meta(const Tensor& t, float factor) {
  (void)factor;
  std::cout << "  [Meta kernel] shape=" << t.numel() << " elems (no data)"
            << std::endl;
  return Tensor({}, DispatchKey::Meta);
}

// ─────────────────────────────────────────────────────────────────────────────
// Op: scale — one schema, multiple backends.
// TORCH_LIBRARY / TORCH_LIBRARY_IMPL is the real PyTorch equivalent.
// ─────────────────────────────────────────────────────────────────────────────

static KernelRegistry scale_registry;

void register_ops() {
  scale_registry.register_kernel(DispatchKey::CPU,  scale_cpu);
  scale_registry.register_kernel(DispatchKey::CUDA, scale_cuda);  // always registered
  scale_registry.register_kernel(DispatchKey::Meta, scale_meta);

  // Report GPU mode at registration time so the user knows what they'll get
  if (cuda_available()) {
    std::cout << "  [registry] GPU detected   — CUDA kernel → HARDWARE"
              << std::endl;
  } else {
    std::cout << "  [registry] no GPU detected — CUDA kernel → SIMULATED"
              << std::endl;
  }
  // XPU intentionally omitted — demonstrates the unregistered-key error path
}

Tensor scale(const Tensor& input, float factor) {
  return scale_registry.dispatch(input, factor);
}

// ─────────────────────────────────────────────────────────────────────────────
// pybind11 binding — the only part that knows about Python
// ─────────────────────────────────────────────────────────────────────────────

PYBIND11_MODULE(dispatch, m) {
  m.doc() =
      "Minimal PyTorch-style dispatcher — shows how kernels are selected at "
      "runtime based on a tensor's dispatch key";

  py::enum_<DispatchKey>(m, "DispatchKey")
      .value("CPU",  DispatchKey::CPU)
      .value("CUDA", DispatchKey::CUDA)
      .value("Meta", DispatchKey::Meta)
      .value("XPU",  DispatchKey::XPU)
      .export_values();

  py::class_<Tensor>(m, "Tensor")
      .def(py::init<std::vector<float>, DispatchKey>())
      .def_readonly("data",  &Tensor::data)
      .def_readonly("key",   &Tensor::key)
      .def("device",         &Tensor::device)
      .def("numel",          &Tensor::numel)
      .def("__repr__", [](const Tensor& t) {
        return "Tensor(" + std::to_string(t.numel()) +
               " elems, device=" + t.device() + ")";
      });

  m.def("cuda_available", &cuda_available,
        "True if a CUDA or ROCm GPU is detected at runtime");
  m.def("register_ops", &register_ops,
        "Register CPU / CUDA / Meta kernels (CUDA only if GPU detected)");
  m.def("scale", &scale, py::arg("input"), py::arg("factor"),
        "Dispatch scale to the kernel matching the tensor's DispatchKey");
}
