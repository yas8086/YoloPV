cmake_minimum_required(VERSION 3.5)
project(YoloPVInference VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# OpenCV（必需）
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# CUDA（可选，用于 GPU 加速）
option(ENABLE_CUDA "启用 CUDA 加速" OFF)
if(ENABLE_CUDA)
    find_package(CUDA 11 REQUIRED)
    add_definitions(-DUSE_CUDA)
    message(STATUS "CUDA 已启用")
else()
    message(STATUS "CUDA 未启用，使用 CPU 推理")
endif()

set(PROJECT_SOURCES
    main.cpp
    inference.h
    inference.cpp
)

add_executable(yolopv_infer ${PROJECT_SOURCES})
target_link_libraries(yolopv_infer ${OpenCV_LIBS})
