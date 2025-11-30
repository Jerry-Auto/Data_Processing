include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(eigen_learn main.cpp eigen_learn.cpp)
find_package(Eigen3 REQUIRED)

target_link_libraries(eigen_learn
    Eigen3::Eigen
)

target_include_directories(eigen_learn INTERFACE
    ${EIGEN3_INCLUDE_DIR}
)