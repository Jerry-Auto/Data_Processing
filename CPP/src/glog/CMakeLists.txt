# 设置为 Debug 模式，确保 NDEBUG 没有被定义
set(CMAKE_BUILD_TYPE Debug)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
find_package(glog REQUIRED)
add_executable(glog_learn main.cpp)
target_link_libraries(glog_learn PRIVATE glog::glog)