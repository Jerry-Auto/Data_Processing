# 自动处理所有子目录，但排除指定的目录
file(GLOB subdirs LIST_DIRECTORIES true RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*")

# 设置需要排除的目录列表
set(EXCLUDE_DIRS
    bin
    build
    lib
    # 你可以在这里添加其他需要排除的目录，例如 .git
)

foreach(subdir ${subdirs})
    # 检查当前子目录是否在排除列表中
    list(FIND EXCLUDE_DIRS ${subdir} index_result)
    if(index_result EQUAL -1) # 如果不在排除列表中 (-1 表示未找到)
        # 拼接绝对路径
        set(full_subdir_path "${CMAKE_CURRENT_SOURCE_DIR}/${subdir}")
        if(IS_DIRECTORY ${full_subdir_path})
            message(STATUS "Processing subdirectory: ${subdir}")
            add_subdirectory(${full_subdir_path} ${CMAKE_BINARY_DIR}/${subdir})
        endif()
    else()
        message(STATUS "Skipping excluded directory: ${subdir}")
    endif()
endforeach()