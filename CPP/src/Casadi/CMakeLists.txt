option(BUILD_CASADI_DEMO "Build CasADi C++ demo" OFF)

if(BUILD_CASADI_DEMO)
	find_package(casadi CONFIG REQUIRED)

	add_executable(casadi_demo casadi_demo.cpp)

	if(TARGET casadi::casadi)
		target_link_libraries(casadi_demo PRIVATE casadi::casadi)
	elseif(TARGET casadi)
		target_link_libraries(casadi_demo PRIVATE casadi)
	else()
		message(FATAL_ERROR "CasADi package found, but no known target (casadi::casadi or casadi) was exported.")
	endif()
endif()
