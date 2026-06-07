cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(tensorrt_demo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# TensorRT 安装路径（从 tensorrt/cpp/ 回到项目根目录，再进入 TensorRT/）
set(TRT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../TensorRT" CACHE PATH "TensorRT root directory")

# CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# TensorRT
include_directories(${TRT_ROOT}/include)
link_directories(${TRT_ROOT}/lib)

add_executable(trt_demo main.cpp)
target_link_libraries(trt_demo
    nvinfer
    nvinfer_plugin
    ${CUDA_LIBRARIES}
    ${CUDA_CUDART_LIBRARY}
)

# 运行时能找到 TensorRT 动态库
set_target_properties(trt_demo PROPERTIES
    BUILD_RPATH "${TRT_ROOT}/lib"
    INSTALL_RPATH "${TRT_ROOT}/lib"
)
