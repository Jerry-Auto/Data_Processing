cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(libtorch_demo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 指向 libtorch 的 cmake 配置目录
# 使用相对于 CMakeLists.txt 的路径: ../../libtorch
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../libtorch")

find_package(Torch REQUIRED)

add_executable(libtorch_demo main.cpp)
target_link_libraries(libtorch_demo "${TORCH_LIBRARIES}")
