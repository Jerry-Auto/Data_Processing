include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(IPOPT main.cpp IPOPT_demo.cpp)

target_link_libraries(IPOPT
    ipopt
)