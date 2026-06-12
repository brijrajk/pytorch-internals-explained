#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace py = pybind11;

// ─────────────────────────────────────────────────────────────────────────────
// Node — one entry in the computation graph.
// In PyTorch this is torch::autograd::Node (AddBackward0, MulBackward0, ...).
// Each node knows:
//   - how to turn the output gradient into input gradients (chain rule)
//   - which nodes produced its inputs (edges to walk during backward)
// ─────────────────────────────────────────────────────────────────────────────

struct Tensor;  // forward declaration

struct Node {
  std::string name;  // "MulBackward", "AddBackward", ...

  // Given dL/d(output), return dL/d(input_i) for each input.
  std::function<std::vector<std::vector<float>>(const std::vector<float>&)>
      backward_fn;

  // Edges to the nodes that produced each input (nullptr edge = leaf input).
  std::vector<std::shared_ptr<Node>> next_nodes;

  // Leaf tensors to accumulate gradients into (AccumulateGrad in PyTorch).
  std::vector<std::shared_ptr<Tensor>> leaf_inputs;
};

// ─────────────────────────────────────────────────────────────────────────────
// Tensor — data + grad + the hook into the graph.
// grad_fn points at the Node that created this tensor.
// Leaf tensors (created by the user) have no grad_fn.
// ─────────────────────────────────────────────────────────────────────────────

struct Tensor : std::enable_shared_from_this<Tensor> {
  std::vector<float> data;
  std::vector<float> grad;
  bool requires_grad = false;
  std::shared_ptr<Node> grad_fn;  // null for leaf tensors

  Tensor(std::vector<float> d, bool rg)
      : data(std::move(d)), grad(data.size(), 0.0f), requires_grad(rg) {}

  size_t numel() const { return data.size(); }
  bool is_leaf() const { return grad_fn == nullptr; }

  std::string grad_fn_name() const {
    return grad_fn ? grad_fn->name : "None (leaf tensor)";
  }
};

using TensorPtr = std::shared_ptr<Tensor>;

// ─────────────────────────────────────────────────────────────────────────────
// Forward ops — each one does two jobs:
//   1. compute the result            (what the user asked for)
//   2. record a Node in the graph    (what backward will need)
// This is exactly what PyTorch ops do when requires_grad is set.
// ─────────────────────────────────────────────────────────────────────────────

static void check_same_size(const TensorPtr& a, const TensorPtr& b,
                            const char* op) {
  if (a->numel() != b->numel())
    throw std::invalid_argument(std::string(op) +
                                ": tensors must have the same length");
}

// Attach graph bookkeeping for a 2-input op.
static void record_node(
    const TensorPtr& out, const TensorPtr& a, const TensorPtr& b,
    const std::string& name,
    std::function<std::vector<std::vector<float>>(const std::vector<float>&)>
        backward_fn) {
  if (!a->requires_grad && !b->requires_grad) return;  // no graph needed

  auto node = std::make_shared<Node>();
  node->name = name;
  node->backward_fn = std::move(backward_fn);
  node->next_nodes = {a->grad_fn, b->grad_fn};
  node->leaf_inputs = {a->is_leaf() && a->requires_grad ? a : nullptr,
                       b->is_leaf() && b->requires_grad ? b : nullptr};

  out->requires_grad = true;
  out->grad_fn = node;
  std::cout << "  [graph] recorded " << name << std::endl;
}

TensorPtr add(const TensorPtr& a, const TensorPtr& b) {
  check_same_size(a, b, "add");
  std::vector<float> out(a->numel());
  for (size_t i = 0; i < out.size(); ++i) out[i] = a->data[i] + b->data[i];

  auto result = std::make_shared<Tensor>(out, false);
  // d(a+b)/da = 1,  d(a+b)/db = 1  →  pass the gradient through unchanged
  record_node(result, a, b, "AddBackward", [](const std::vector<float>& g) {
    return std::vector<std::vector<float>>{g, g};
  });
  return result;
}

TensorPtr mul(const TensorPtr& a, const TensorPtr& b) {
  check_same_size(a, b, "mul");
  std::vector<float> out(a->numel());
  for (size_t i = 0; i < out.size(); ++i) out[i] = a->data[i] * b->data[i];

  auto result = std::make_shared<Tensor>(out, false);
  // d(a*b)/da = b,  d(a*b)/db = a  →  the node must SAVE the input values.
  // PyTorch calls these "saved tensors" (ctx.save_for_backward).
  std::vector<float> a_saved = a->data;
  std::vector<float> b_saved = b->data;
  record_node(result, a, b, "MulBackward",
              [a_saved, b_saved](const std::vector<float>& g) {
                std::vector<float> ga(g.size()), gb(g.size());
                for (size_t i = 0; i < g.size(); ++i) {
                  ga[i] = g[i] * b_saved[i];  // chain rule
                  gb[i] = g[i] * a_saved[i];
                }
                return std::vector<std::vector<float>>{ga, gb};
              });
  return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// backward — walk the graph in reverse, applying the chain rule at each node.
// PyTorch's engine (torch/csrc/autograd/engine.cpp) does this with a
// ready-queue and multi-threading. The traversal logic is the same.
// ─────────────────────────────────────────────────────────────────────────────

namespace {

void backward_impl(const std::shared_ptr<Node>& node,
                   const std::vector<float>& grad_out,
                   std::unordered_set<Node*>& visited_log) {
  if (!node) return;

  if (visited_log.insert(node.get()).second)
    std::cout << "  [backward] executing " << node->name << std::endl;
  else
    std::cout << "  [backward] executing " << node->name << " (again)"
              << std::endl;

  // Chain rule: turn dL/d(output) into dL/d(input_i)
  auto input_grads = node->backward_fn(grad_out);

  for (size_t i = 0; i < input_grads.size(); ++i) {
    // Leaf input → accumulate into .grad (PyTorch's AccumulateGrad node)
    if (node->leaf_inputs[i]) {
      auto& leaf = node->leaf_inputs[i];
      for (size_t j = 0; j < leaf->grad.size(); ++j)
        leaf->grad[j] += input_grads[i][j];  // += matters: grads accumulate
      std::cout << "  [backward] AccumulateGrad → leaf.grad updated"
                << std::endl;
    }
    // Interior input → keep walking the graph
    backward_impl(node->next_nodes[i], input_grads[i], visited_log);
  }
}

}  // namespace

void backward(const TensorPtr& t) {
  if (!t->requires_grad)
    throw std::runtime_error(
        "element 0 of tensors does not require grad and does not have a "
        "grad_fn");

  // Seed: dL/dL = 1 (same default as loss.backward() in PyTorch)
  std::vector<float> seed(t->numel(), 1.0f);
  std::cout << "  [backward] seed gradient = all ones" << std::endl;
  std::unordered_set<Node*> visited_log;
  backward_impl(t->grad_fn, seed, visited_log);
}

// ─────────────────────────────────────────────────────────────────────────────
// detach — return a copy disconnected from the graph.
// ─────────────────────────────────────────────────────────────────────────────

TensorPtr detach(const TensorPtr& t) {
  return std::make_shared<Tensor>(t->data, false);
}

// ─────────────────────────────────────────────────────────────────────────────
// pybind11 binding — the only part that knows about Python
// ─────────────────────────────────────────────────────────────────────────────

PYBIND11_MODULE(autograd, m) {
  m.doc() =
      "Minimal PyTorch-style autograd engine — forward ops record a graph, "
      "backward walks it in reverse applying the chain rule";

  py::class_<Tensor, TensorPtr>(m, "Tensor")
      .def(py::init<std::vector<float>, bool>(), py::arg("data"),
           py::arg("requires_grad") = false)
      .def_readonly("data", &Tensor::data)
      .def_readonly("grad", &Tensor::grad)
      .def_readonly("requires_grad", &Tensor::requires_grad)
      .def_property_readonly("grad_fn", &Tensor::grad_fn_name)
      .def("is_leaf", &Tensor::is_leaf)
      .def("numel", &Tensor::numel)
      .def("__repr__", [](const Tensor& t) {
        std::string s = "Tensor(" + std::to_string(t.numel()) + " elems";
        if (t.requires_grad) s += ", requires_grad=True";
        if (t.grad_fn) s += ", grad_fn=<" + t.grad_fn->name + ">";
        return s + ")";
      });

  m.def("add", &add, py::arg("a"), py::arg("b"),
        "Element-wise add — records AddBackward if needed");
  m.def("mul", &mul, py::arg("a"), py::arg("b"),
        "Element-wise multiply — records MulBackward if needed");
  m.def("backward", &backward, py::arg("tensor"),
        "Walk the graph in reverse, accumulating gradients into leaves");
  m.def("detach", &detach, py::arg("tensor"),
        "Return a copy disconnected from the graph");
}
