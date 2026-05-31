cmake_minimum_required(VERSION 3.15)
project(pybind11_basics)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Build the extension module
pybind11_add_module(ops add.cpp)

# Output the .so next to demo.py
set_target_properties(ops PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
)
