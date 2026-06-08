cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(onnx_demo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ONNX Runtime 预编译包路径（从 onnx/cpp/ 回到项目根目录，再进入 third_party/onnxruntime/）
set(ORT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/onnxruntime")

include_directories(${ORT_ROOT}/include)
find_library(ORT_LIB onnxruntime PATHS ${ORT_ROOT}/lib REQUIRED)

# CUDA provider 库（可选）
find_library(ORT_CUDA_LIB onnxruntime_providers_cuda PATHS ${ORT_ROOT}/lib)
find_library(ORT_SHARED_LIB onnxruntime_providers_shared PATHS ${ORT_ROOT}/lib)

add_executable(onnx_demo main.cpp)
target_link_libraries(onnx_demo ${ORT_LIB})

add_executable(onnx_bench benchmark.cpp)
if(ORT_CUDA_LIB AND ORT_SHARED_LIB)
    message(STATUS "ONNX Runtime CUDA provider found, building with GPU support")
    find_package(CUDA QUIET)
    if(CUDA_FOUND)
        target_link_libraries(onnx_bench ${ORT_LIB} ${ORT_CUDA_LIB} ${ORT_SHARED_LIB} ${CUDA_CUDART_LIBRARY})
        include_directories(${CUDA_INCLUDE_DIRS})
    else()
        target_link_libraries(onnx_bench ${ORT_LIB} ${ORT_CUDA_LIB} ${ORT_SHARED_LIB})
    endif()
    target_compile_definitions(onnx_bench PRIVATE ORT_CUDA_AVAILABLE)
else()
    message(STATUS "ONNX Runtime CUDA provider not found, CPU only")
    target_link_libraries(onnx_bench ${ORT_LIB})
endif()

# 运行时能找到动态库
set_target_properties(onnx_demo onnx_bench PROPERTIES
    BUILD_RPATH "${ORT_ROOT}/lib"
    INSTALL_RPATH "${ORT_ROOT}/lib"
)
