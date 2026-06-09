cmake_minimum_required(VERSION 3.18)

# Defer CUDA language enablement until after GPU detection
project(tensor_dispatch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Resolve pybind11's cmake dir from the active Python env (works with pip/uv/conda)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE pybind11_FOUND_VIA_PY
)
find_package(pybind11 CONFIG REQUIRED)

# ── GPU detection (CUDA / ROCm) ───────────────────────────────────────────────
# Try CUDA first; if not found, try ROCm/HIP.
# Both use the same dispatch_gpu.cu source — the .cu file includes HIP compat
# headers so hipcc compiles it unchanged.

find_package(CUDAToolkit QUIET)

if(CUDAToolkit_FOUND)
    message(STATUS "Found CUDA ${CUDAToolkit_VERSION} — building GPU kernel")
    enable_language(CUDA)
    set(GPU_FOUND TRUE)
    set(GPU_SOURCES dispatch.cpp dispatch_gpu.cu)
    set(GPU_DEFINE WITH_CUDA)
    set(GPU_LIBS CUDA::cudart)
else()
    # Try ROCm / HIP
    find_package(HIP QUIET)
    if(HIP_FOUND)
        message(STATUS "Found ROCm/HIP ${HIP_VERSION} — building GPU kernel")
        enable_language(HIP)
        set(GPU_FOUND TRUE)
        set(GPU_SOURCES dispatch.cpp dispatch_gpu.cu)
        set(GPU_DEFINE WITH_CUDA)           # same flag — dispatch_gpu.cu handles both
        set(GPU_LIBS hip::host)
    else()
        message(STATUS "No CUDA or ROCm found — GPU kernel will run in simulation mode")
        set(GPU_FOUND FALSE)
        set(GPU_SOURCES dispatch.cpp)
        set(GPU_DEFINE "")
        set(GPU_LIBS "")
    endif()
endif()

# ── Build ─────────────────────────────────────────────────────────────────────

pybind11_add_module(dispatch ${GPU_SOURCES})

if(GPU_FOUND)
    target_compile_definitions(dispatch PRIVATE ${GPU_DEFINE})
    target_link_libraries(dispatch PRIVATE ${GPU_LIBS})
endif()

# Output the .so next to demo.py
set_target_properties(dispatch PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# 'make clean-all' removes build artifacts AND the .so dropped next to demo.py.
add_custom_target(clean-all
    COMMAND ${CMAKE_BUILD_TOOL} clean
    COMMAND ${CMAKE_COMMAND} -E remove -f $<TARGET_FILE:dispatch>
    COMMENT "Removing build artifacts and output extension module"
    VERBATIM
)
