cmake_minimum_required(VERSION 3.10)
project(MyProject)

# 设置编译模式Debug Realease
set(CMAKE_BUILD_TYPE Debug)

# 设置编译选项
# 设置C语言标准（例如C99）
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# 添加 -fno-elide-constructors 标志
add_compile_options(-fno-elide-constructors)
add_compile_options(-Wall -Wextra -Wpedantic)

# 输出目录设置
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

add_subdirectory(src)