cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(onnx_demo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ONNX Runtime 预编译包路径（从 onnx/cpp/ 回到项目根目录，再进入 onnxruntime/）
set(ORT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../onnxruntime")

include_directories(${ORT_ROOT}/include)
find_library(ORT_LIB onnxruntime PATHS ${ORT_ROOT}/lib REQUIRED)

add_executable(onnx_demo main.cpp)
target_link_libraries(onnx_demo ${ORT_LIB})

# 运行时能找到动态库
set_target_properties(onnx_demo PROPERTIES
    BUILD_RPATH "${ORT_ROOT}/lib"
    INSTALL_RPATH "${ORT_ROOT}/lib"
)
