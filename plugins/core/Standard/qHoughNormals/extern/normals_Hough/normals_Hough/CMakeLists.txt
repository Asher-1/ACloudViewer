cmake_minimum_required(VERSION ${CLOUDVIEWER_PLUGIN_CMAKE_MINIMUM_VERSION})

project(normalsHoughProject)

SET(CMAKE_BUILD_TYPE Release)

set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib )

set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} "/usr/lib/x86_64-linux-gnu/")

############################
## OPENMP
############################
if(UNIX)
	#use openMP
	find_package( OpenMP REQUIRED)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif(UNIX)

add_executable(Hough_Exec
main.cpp
Normals.h
pc_io.cpp
pc_io.h)
target_link_libraries(Hough_Exec ${EXTRA_LIBS} ${PROJECT_LINK_LIB} )
target_link_libraries(Hough_Exec 3rdparty_pcl)