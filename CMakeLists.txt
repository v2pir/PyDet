cmake_minimum_required(VERSION 3.10)

project(filters)

add_subdirectory(pybind11)
pybind11_add_module(filters filters.cpp)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(filters PRIVATE ${OpenCV_LIBS})