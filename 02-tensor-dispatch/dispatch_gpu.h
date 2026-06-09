#pragma once
#include <vector>

// Implemented in dispatch_gpu.cu — compiled only when WITH_CUDA or WITH_ROCM
// is defined. The same source file builds for both backends via HIP compat.
std::vector<float> scale_on_gpu(const std::vector<float>& data, float factor);
