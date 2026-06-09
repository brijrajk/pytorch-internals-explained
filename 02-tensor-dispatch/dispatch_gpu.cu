// dispatch_gpu.cu — real GPU kernel for the scale op.
// Compiles for CUDA (nvcc) and ROCm (hipcc) without changes.
// ROCm ships a CUDA compatibility layer; the hip* functions below
// map directly to the CUDA equivalents when building with hipcc.

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
#include <hip/hip_runtime.h>
#define GPU_PREFIX "ROCm/HIP"
using gpuError_t = hipError_t;
#define gpuSuccess            hipSuccess
#define gpuGetDeviceCount     hipGetDeviceCount
#define gpuGetDeviceProperties hipGetDeviceProperties
#define gpuMalloc             hipMalloc
#define gpuFree               hipFree
#define gpuMemcpy             hipMemcpy
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuDeviceSynchronize  hipDeviceSynchronize
#define gpuGetLastError       hipGetLastError
#define gpuGetErrorString     hipGetErrorString
using gpuDeviceProp_t = hipDeviceProp_t;
#else
#include <cuda_runtime.h>
#define GPU_PREFIX "CUDA"
using gpuError_t = cudaError_t;
#define gpuSuccess            cudaSuccess
#define gpuGetDeviceCount     cudaGetDeviceCount
#define gpuGetDeviceProperties cudaGetDeviceProperties
#define gpuMalloc             cudaMalloc
#define gpuFree               cudaFree
#define gpuMemcpy             cudaMemcpy
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuDeviceSynchronize  cudaDeviceSynchronize
#define gpuGetLastError       cudaGetLastError
#define gpuGetErrorString     cudaGetErrorString
using gpuDeviceProp_t = cudaDeviceProp;
#endif

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "dispatch_gpu.h"

// ─────────────────────────────────────────────────────────────────────────────
// Error-checking helper
// ─────────────────────────────────────────────────────────────────────────────

static void gpu_check(gpuError_t err, const char* ctx) {
  if (err != gpuSuccess)
    throw std::runtime_error(std::string(GPU_PREFIX) + " error in " + ctx +
                             ": " + gpuGetErrorString(err));
}

// ─────────────────────────────────────────────────────────────────────────────
// GPU kernel — one thread per element
// ─────────────────────────────────────────────────────────────────────────────

__global__ void scale_kernel(const float* __restrict__ in,
                              float* __restrict__ out, float factor, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) out[idx] = in[idx] * factor;
}

// ─────────────────────────────────────────────────────────────────────────────
// Host-side launcher: H2D → kernel → D2H
// ─────────────────────────────────────────────────────────────────────────────

std::vector<float> scale_on_gpu(const std::vector<float>& data, float factor) {
  int n = static_cast<int>(data.size());

  // Print the device we're actually running on
  gpuDeviceProp_t prop;
  gpu_check(gpuGetDeviceProperties(&prop, 0), "gpuGetDeviceProperties");
  std::cout << "  [" << GPU_PREFIX << " kernel] device : " << prop.name
            << std::endl;
  std::cout << "  [" << GPU_PREFIX << " kernel] VRAM   : "
            << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
#ifndef __HIP_PLATFORM_AMD__
  // CUDA-only field; HIP exposes it differently
  std::cout << "  [" << GPU_PREFIX << " kernel] compute: "
            << prop.major << "." << prop.minor << std::endl;
#endif

  float *d_in, *d_out;
  gpu_check(gpuMalloc(&d_in, n * sizeof(float)), "gpuMalloc d_in");
  gpu_check(gpuMalloc(&d_out, n * sizeof(float)), "gpuMalloc d_out");

  gpu_check(gpuMemcpy(d_in, data.data(), n * sizeof(float),
                      gpuMemcpyHostToDevice),
            "H2D memcpy");

  constexpr int kThreads = 256;
  int blocks = (n + kThreads - 1) / kThreads;
  scale_kernel<<<blocks, kThreads>>>(d_in, d_out, factor, n);
  gpu_check(gpuGetLastError(), "kernel launch");
  gpu_check(gpuDeviceSynchronize(), "gpuDeviceSynchronize");

  std::vector<float> result(n);
  gpu_check(gpuMemcpy(result.data(), d_out, n * sizeof(float),
                      gpuMemcpyDeviceToHost),
            "D2H memcpy");

  gpuFree(d_in);
  gpuFree(d_out);
  return result;
}
