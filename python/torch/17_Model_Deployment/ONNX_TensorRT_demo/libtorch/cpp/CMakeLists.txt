cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(libtorch_demo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# LibTorch 路径
# 优先使用 conda 环境中的 GPU 版本，备选使用独立 CPU 版本
set(LIBTORCH_GPU_PATH "$ENV{CONDA_PREFIX}/lib/python3.9/site-packages/torch" CACHE PATH "GPU libtorch path")
set(LIBTORCH_CPU_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/Libtorch" CACHE PATH "CPU libtorch path")

# 检测 GPU libtorch 是否可用
if(EXISTS "${LIBTORCH_GPU_PATH}/share/cmake/Torch/TorchConfig.cmake")
    message(STATUS "Using GPU libtorch from: ${LIBTORCH_GPU_PATH}")
    set(CMAKE_PREFIX_PATH "${LIBTORCH_GPU_PATH}")
    set(USING_GPU_LIBTORCH TRUE)
elseif(EXISTS "${LIBTORCH_CPU_PATH}/share/cmake/Torch/TorchConfig.cmake")
    message(STATUS "Using CPU libtorch from: ${LIBTORCH_CPU_PATH}")
    set(CMAKE_PREFIX_PATH "${LIBTORCH_CPU_PATH}")
    set(USING_GPU_LIBTORCH FALSE)
else()
    message(FATAL_ERROR "Cannot find libtorch! Set CONDA_PREFIX or place libtorch at ${LIBTORCH_CPU_PATH}")
endif()

find_package(Torch REQUIRED)

# ========== libtorch_demo (推理示例) ==========
add_executable(libtorch_demo main.cpp)
target_link_libraries(libtorch_demo "${TORCH_LIBRARIES}")

# ========== libtorch_bench (基准测试) ==========
add_executable(libtorch_bench benchmark.cpp)
target_link_libraries(libtorch_bench "${TORCH_LIBRARIES}")

# 检测 CUDA 支持并传递宏定义
if(USING_GPU_LIBTORCH)
    target_compile_definitions(libtorch_bench PRIVATE AT_CUDA)
    target_compile_definitions(libtorch_demo PRIVATE AT_CUDA)
    message(STATUS "CUDA support enabled")
endif()

# 设置运行时库路径
if(USING_GPU_LIBTORCH)
    set_target_properties(libtorch_demo PROPERTIES
        BUILD_RPATH "${LIBTORCH_GPU_PATH}/lib"
        INSTALL_RPATH "${LIBTORCH_GPU_PATH}/lib"
    )
    set_target_properties(libtorch_bench PROPERTIES
        BUILD_RPATH "${LIBTORCH_GPU_PATH}/lib"
        INSTALL_RPATH "${LIBTORCH_GPU_PATH}/lib"
    )
else()
    set_target_properties(libtorch_demo PROPERTIES
        BUILD_RPATH "${LIBTORCH_CPU_PATH}/lib"
        INSTALL_RPATH "${LIBTORCH_CPU_PATH}/lib"
    )
    set_target_properties(libtorch_bench PROPERTIES
        BUILD_RPATH "${LIBTORCH_CPU_PATH}/lib"
        INSTALL_RPATH "${LIBTORCH_CPU_PATH}/lib"
    )
endif()

set_property(TARGET libtorch_demo libtorch_bench PROPERTY CXX_STANDARD 17)
