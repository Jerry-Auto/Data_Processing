include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(LQR LQR.cpp main.cpp)
find_package(Eigen3 REQUIRED)
find_package(casadi CONFIG REQUIRED)

target_link_libraries(LQR
    Eigen3::Eigen
	casadi::casadi
)

target_include_directories(LQR INTERFACE
    ${EIGEN3_INCLUDE_DIR}
)