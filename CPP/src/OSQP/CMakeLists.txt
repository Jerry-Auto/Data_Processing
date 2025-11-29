# 查找所需的OSQP库
find_package(osqp REQUIRED)

# 如果CMake无法自动找到OSQP，您可以手动指定其路径（取消注释并修改下面的路径）
# set(OSQP_DIR "/home/zhangjinrui/anaconda3/envs/PNC/lib/cmake/osqp")  # 替换为您的实际路径
# find_package(osqp REQUIRED)

# 创建可执行文件，将osqp.c编译成名为"osqp_learn"的可执行程序
add_executable(osqp_learn osqp.c)

# 将OSQP库链接到您的可执行文件
target_link_libraries(osqp_learn PRIVATE osqp::osqp)

# set_target_properties(osqp_learn PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin/osqp_learn"
# )



# 查找所需的库
find_package(OsqpEigen REQUIRED)

# 创建可执行文件
add_executable(osqp_program osqp_cpp.cpp)

# 链接库，使用现代CMake的目标名称（Target Name）方式，能自动处理依赖
target_link_libraries(osqp_program PRIVATE OsqpEigen::OsqpEigen)
# set_target_properties(osqp_program PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin/osqp_program"
# )