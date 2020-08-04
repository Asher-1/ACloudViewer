cmake_minimum_required(VERSION 3.5)

# PYTHON ENGINE BACKEND
include_directories( ${CV_CORE_LIB_SOURCE_DIR}/include )
include_directories( ${ECV_DB_LIB_SOURCE_DIR} )
if( MSVC )
	   include_directories( ${ECV_DB_LIB_SOURCE_DIR}/msvc )
endif()
include_directories( ${ECV_IO_LIB_SOURCE_DIR} )

include_directories( ${OPENGL_ENGINE_LIB_SOURCE_DIR} )
if( MSVC )
   include_directories( ${OPENGL_ENGINE_LIB_SOURCE_DIR}/msvc )
endif()

# Option 1: Do not define "PYTHON_EXECUTABLE", but run `cmake ..` within your
#           virtual environment. CMake will pick up the python executable in the
#           virtual environment.
# Option 2: You can also define `cmake -DPYTHON_EXECUTABLE` to specify a python
#           executable.
if (NOT PYTHON_EXECUTABLE)
	# find_program will returns the python executable in current PATH, which
	# works with virtualenv
	find_program(PYTHON_IN_PATH "python")
	set(PYTHON_EXECUTABLE ${PYTHON_IN_PATH})
endif()
message(STATUS "Using Python executable: ${PYTHON_EXECUTABLE}")

# Detect whether `npm` is installed. Jupyter support will only be enabled if
# `npm` is found.
find_program(NPM "npm")
if (ENABLE_JUPYTER)
	if (NOT NPM)
		message(WARNING "Cannot find npm. Jupyter support will be disabled.")
		set(ENABLE_JUPYTER OFF)
	else()
		message(STATUS "npm found at: ${NPM}. Jupyter support will be enabled.")
	endif()
endif()
message(STATUS "ENABLE_JUPYTER is set to ${ENABLE_JUPYTER}")

# We need to get python version to configure some meta files
execute_process(
	COMMAND ${PYTHON_EXECUTABLE} -c "import sys; print('%d.%d' % (sys.version_info.major, sys.version_info.minor))"
	OUTPUT_VARIABLE PYTHON_VERSION
	OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Using Python version: ${PYTHON_VERSION}")

execute_process(
	COMMAND ${PYTHON_EXECUTABLE} -c "import sys; print('%d' % (sys.version_info.major,))"
	OUTPUT_VARIABLE PYTHON_VERSION_MAJOR
	OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Using Python version major: ${PYTHON_VERSION_MAJOR}")

project( ECV_PYTHON_LIB )

set(PACKAGE_NAME cloudViewer)

set( CMAKE_CXX_VISIBILITY_PRESET hidden )

# CURRENT DIR
file( GLOB header_list 	*.h)
file( GLOB_RECURSE PY_ALL_SOURCE_FILES "cloudViewer_pybind/*.cpp" "cloudViewer_pybind/*.h")

if (NOT BUILD_AZURE_KINECT)
	list(REMOVE_ITEM PY_ALL_SOURCE_FILES
		"${CMAKE_CURRENT_SOURCE_DIR}/cloudViewer_pybind/io/sensor.cpp")
endif()

pybind11_add_module(${PACKAGE_NAME} 
	${PY_ALL_SOURCE_FILES}
)

target_include_directories(${PACKAGE_NAME} PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}
)

# Suppress Pybind11 warnings
target_include_directories(${PACKAGE_NAME} SYSTEM PRIVATE
	${PYBIND11_INCLUDE_DIR}
)

# The Python headers won't be included outside the project, so it is okay to
# use the target_compile_definitions approach. Otherwise, use configure_file.
if (BUILD_AZURE_KINECT)
	target_compile_definitions(${PACKAGE_NAME} PRIVATE BUILD_AZURE_KINECT)
endif()

# CLOUDVIEWER PYBIND MODULE
SET( CLOUDVIEWER_PYBIND11 ${CMAKE_CURRENT_SOURCE_DIR}/cloudViewer_pybind )
file(GLOB basic_list 
	${CLOUDVIEWER_PYBIND11}/*.h
	${CLOUDVIEWER_PYBIND11}/*.cpp
	${CLOUDVIEWER_PYBIND11}/*.hpp
)

source_group("cloudViewer_pybind" FILES ${basic_list} )
#ADD_SOURCE_GROUP(cloudViewer_pybind)
file( GLOB_RECURSE python_camera_list     ${CLOUDVIEWER_PYBIND11}/camera/*)
source_group("cloudViewer_pybind\\camera" FILES ${python_camera_list} )
file( GLOB_RECURSE python_color_map_list     ${CLOUDVIEWER_PYBIND11}/color_map/*)
source_group("cloudViewer_pybind\\color_map" FILES ${python_color_map_list} )
file( GLOB_RECURSE python_geometry_list     ${CLOUDVIEWER_PYBIND11}/geometry/*)
source_group("cloudViewer_pybind\\geometry" FILES ${python_geometry_list} )
file( GLOB_RECURSE python_integration_list     ${CLOUDVIEWER_PYBIND11}/integration/*)
source_group("cloudViewer_pybind\\integration" FILES ${python_integration_list} )
file( GLOB_RECURSE python_io_list     ${CLOUDVIEWER_PYBIND11}/io/*)
source_group("cloudViewer_pybind\\io" FILES ${python_io_list} )
file( GLOB_RECURSE python_odometry_list     ${CLOUDVIEWER_PYBIND11}/odometry/*)
source_group("cloudViewer_pybind\\odometry" FILES ${python_odometry_list} )
file( GLOB_RECURSE python_utility_list 		${CLOUDVIEWER_PYBIND11}/utility/*)
source_group("cloudViewer_pybind\\utility" FILES ${python_utility_list} )
file( GLOB_RECURSE python_registration_list  ${CLOUDVIEWER_PYBIND11}/registration/*)
source_group("cloudViewer_pybind\\registration" FILES ${python_registration_list} )
file( GLOB_RECURSE python_visualization_list     ${CLOUDVIEWER_PYBIND11}/visualization/*)
source_group("cloudViewer_pybind\\visualization" FILES ${python_visualization_list} )

target_link_libraries(${PACKAGE_NAME} PRIVATE 
		${3RDPARTY_LIBRARIES}
		CV_CORE_LIB
		ECV_DB_LIB
		ECV_IO_LIB
)
target_link_libraries(${PACKAGE_NAME} PRIVATE OPENGL_ENGINE_LIB)

if (${PYTHON_VERSION_MAJOR} EQUAL 2)
	target_compile_definitions(${PACKAGE_NAME} PRIVATE PYTHON_2_FALLBACK)
endif ()

# At `make`: cloudViewer.so (or the equivalents) will be created at
# PYTHON_COMPILED_MODULE_DIR. The default locaiton is `build/lib/Python`
set(PYTHON_COMPILED_MODULE_DIR "${CMAKE_BINARY_DIR}/lib/Python")
set_target_properties(${PACKAGE_NAME} PROPERTIES
					  FOLDER "Python"
					  LIBRARY_OUTPUT_DIRECTORY "${PYTHON_COMPILED_MODULE_DIR}"
					  ARCHIVE_OUTPUT_DIRECTORY "${PYTHON_COMPILED_MODULE_DIR}")

# Use `make python-package` to create the python package in the build directory
# The python package will be created at PYTHON_PACKAGE_DIR. It contains:
# 1) Pure-python code and misc files, copied from src/Python/package
# 2) The compiled python-C++ module, i.e. cloudViewer.so (or the equivalents)
# 3) Configured files and supporting files
# Note: `make python-package` clears PYTHON_COMPILED_MODULE_DIR first every time
set(PYTHON_PACKAGE_DST_DIR "${CMAKE_BINARY_DIR}/lib/python_package")
message(STATUS "PYPI_PACKAGE_NAME: ${PYPI_PACKAGE_NAME}")
message("${PROJECT_VERSION_THREE_NUMBER}: " ${PROJECT_VERSION_THREE_NUMBER})
add_custom_target(python-package
COMMAND ${CMAKE_COMMAND}
		-DPYTHON_PACKAGE_SRC_DIR=${CMAKE_CURRENT_SOURCE_DIR}
		-DPYTHON_PACKAGE_DST_DIR=${PYTHON_PACKAGE_DST_DIR}
		-DPYTHON_VERSION=${PYTHON_VERSION}
		-DPYTHON_COMPILED_MODULE_PATH=$<TARGET_FILE:${PACKAGE_NAME}>
		-DENABLE_JUPYTER=${ENABLE_JUPYTER}
		-DPROJECT_EMAIL=${PROJECT_EMAIL}
		-DPROJECT_HOME=${PROJECT_HOME}
		-DPROJECT_DOCS=${PROJECT_DOCS}
		-DPROJECT_CODE=${PROJECT_CODE}
		-DPROJECT_ISSUES=${PROJECT_ISSUES}
		-DPROJECT_VERSION=${CLOUDVIEWER_VERSION}
		-DPROJECT_VERSION_THREE_NUMBER=${PROJECT_VERSION_THREE_NUMBER}
		-DPYPI_PACKAGE_NAME=${PYPI_PACKAGE_NAME}
		-P ${CMAKE_CURRENT_SOURCE_DIR}/make_python_package.cmake
)

# Use `make pip-package` to create the pip package in the build directory
add_custom_target(pip-package
	COMMAND ${PYTHON_EXECUTABLE} setup.py bdist_wheel --dist-dir pip_package
	COMMAND echo "pip wheel created at ${PYTHON_PACKAGE_DST_DIR}/pip_package"
	WORKING_DIRECTORY ${PYTHON_PACKAGE_DST_DIR}
	DEPENDS python-package
)

# Use `make install-pip-package` to install pip wheel package to the current
# python environment.
add_custom_target(install-pip-package
	COMMAND ${CMAKE_COMMAND}
			-DPYTHON_PACKAGE_DST_DIR=${PYTHON_PACKAGE_DST_DIR}
			-P ${CMAKE_CURRENT_SOURCE_DIR}/make_install_pip_package.cmake
	DEPENDS pip-package
)

# Use `make conda-package` to create conda package in the build directory
# Note that we don't provide `make install-conda-package` similar to pip. This
# is becuase:
#     1) `make install-pip-whell` works in conda environment for local build
#     2) `make conda-package` is mainly for internal use to distribute conda
add_custom_target(conda-package
	COMMAND ${CMAKE_COMMAND}
			-DPYTHON_PACKAGE_DST_DIR=${PYTHON_PACKAGE_DST_DIR}
			-DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}
			-P ${CMAKE_CURRENT_SOURCE_DIR}/check_and_install_conda_deps.cmake
	# If we put the following `conda-build` in check_and_install_conda_deps.cmake, it
	# causes broken pipe problem while running conda build. So we put it here.
	COMMAND conda-build conda_meta --output-folder conda_package
	COMMAND echo "conda package created at ${PYTHON_PACKAGE_DST_DIR}/conda_package"
	WORKING_DIRECTORY ${PYTHON_PACKAGE_DST_DIR}
	DEPENDS python-package
)

##############################################################################################################
##############################################################################################################
# UTILS PYBIND MODULE
file( GLOB_RECURSE utility_pybind_list ${CMAKE_CURRENT_SOURCE_DIR}/Utility/*)
source_group("Utility" FILES ${utility_pybind_list} )
# ADD_SOURCE_GROUP(Utility)
file( GLOB recognition_pybind_list
	${CMAKE_CURRENT_SOURCE_DIR}/Recognition/*h
	${CMAKE_CURRENT_SOURCE_DIR}/Recognition/*cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Recognition/*hpp
)
source_group("Recognition" FILES ${recognition_pybind_list} )

add_library(
	${PROJECT_NAME}  SHARED
	${header_list}
	${utility_pybind_list}
	${recognition_pybind_list}
)

# Qt
if(NOT CMAKE_VERSION VERSION_LESS 3.0)
	target_link_libraries(
		${PROJECT_NAME}
		pybind11::embed
		${PYTHON_LIBRARIES}
		${3RDPARTY_LIBRARIES}
		Qt5::Core
		Qt5::PrintSupport
		CV_CORE_LIB
		ECV_DB_LIB
		ECV_IO_LIB
	)
else()
	target_include_directories(
		${PROJECT_NAME}
		${PYBIND11_INCLUDE_DIR}
		${PYTHON_INCLUDE_DIRS}
	)

	target_compile_options(
		${PROJECT_NAME}
		${PYBIND11_CPP_STANDARD}
	)

	target_link_libraries(
		${PROJECT_NAME}
		pybind11::embed
		${PYTHON_LIBRARIES}
		${3RDPARTY_LIBRARIES}
		Qt5::Core
		Qt5::PrintSupport
		CV_CORE_LIB
		ECV_DB_LIB
		ECV_IO_LIB
	)

endif()

# Add custom preprocessor definitions
if (WIN32)
	# Do not treat includes from IMPORTED target as SYSTEM (Python headers in pybind11::embed).
	# This may be needed to resolve header conflicts, e.g. between Python release and debug headers.
	set_target_properties(${PROJECT_NAME} PROPERTIES NO_SYSTEM_FROM_IMPORTED ON)
else()
	# Do not treat includes from IMPORTED target as SYSTEM (Python headers in pybind11::embed).
	# This may be needed to resolve header conflicts, e.g. between Python release and debug headers.
	set_target_properties(${PROJECT_NAME} PROPERTIES NO_SYSTEM_FROM_IMPORTED ON COMPILE_DEFINITIONS ECV_PYTHON_USE_AS_DLL)
endif()

target_compile_definitions( ${PROJECT_NAME} PRIVATE ECV_PYTHON_LIBRARY_BUILD )

# install (shared) lib to specified destinations
if( WIN32)
	foreach( dest ${INSTALL_DESTINATIONS} )
		install_shared( ${PROJECT_NAME} ${dest} 1 )
	endforeach()
else()
	 install_shared( ${PROJECT_NAME} ${CMAKE_INSTALL_LIBDIR}/ErowCloudViewer 0 ) #default destination: /usr/lib
endif()
