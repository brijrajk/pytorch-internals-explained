cmake_minimum_required(VERSION 3.15)
project(autograd_internals)

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

# Build the extension module
pybind11_add_module(autograd autograd.cpp)

# Output the .so next to demo.py
set_target_properties(autograd PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# 'make clean-all' removes build artifacts AND the .so dropped next to demo.py.
# Built-in 'make clean' only cleans the build tree, not the output copied to source.
add_custom_target(clean-all
    COMMAND ${CMAKE_BUILD_TOOL} clean
    COMMAND ${CMAKE_COMMAND} -E remove -f $<TARGET_FILE:autograd>
    COMMENT "Removing build artifacts and output extension module"
    VERBATIM
)
