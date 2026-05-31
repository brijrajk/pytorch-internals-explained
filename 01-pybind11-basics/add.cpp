#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <stdexcept>

namespace py = pybind11;

// ------------------------------------------------------------
// A plain C++ function — no Python anywhere
// This is what actually runs when you write a + b in PyTorch
// ------------------------------------------------------------

std::vector<float> add(const std::vector<float>& a,
                        const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must have the same length");
    }

    std::vector<float> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}

std::vector<float> multiply(const std::vector<float>& a,
                             const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must have the same length");
    }

    std::vector<float> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] * b[i];
    }
    return result;
}

float dot(const std::vector<float>& a,
          const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must have the same length");
    }

    float result = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

// ------------------------------------------------------------
// The pybind11 binding
// This is the only part that knows about Python.
// Everything above is pure C++.
// ------------------------------------------------------------

PYBIND11_MODULE(ops, m) {
    m.doc() = "Simple C++ ops exposed to Python via pybind11 — mirrors how PyTorch works under the hood";

    m.def("add", &add,
          py::arg("a"), py::arg("b"),
          "Element-wise addition of two float vectors");

    m.def("multiply", &multiply,
          py::arg("a"), py::arg("b"),
          "Element-wise multiplication of two float vectors");

    m.def("dot", &dot,
          py::arg("a"), py::arg("b"),
          "Dot product of two float vectors");
}
