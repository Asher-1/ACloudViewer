cmake_minimum_required( VERSION 3.10 )

include_directories( ${CV_CORE_LIB_SOURCE_DIR}/include )
include_directories( ${ECV_DB_LIB_SOURCE_DIR} )
if( MSVC )
   include_directories( ${ECV_DB_LIB_SOURCE_DIR}/msvc )
endif()
include_directories( ${ECV_IO_LIB_SOURCE_DIR} )

project( OPENGL_ENGINE_LIB )

add_subdirectory(Visualization)

file( GLOB basic_list ${CMAKE_CURRENT_SOURCE_DIR}/*.h )

file( GLOB_RECURSE camera_list     		${CMAKE_CURRENT_SOURCE_DIR}/Camera/*)
ADD_SOURCE_GROUP(Camera)
file( GLOB_RECURSE colorMap_list 		${CMAKE_CURRENT_SOURCE_DIR}/ColorMap/*)
ADD_SOURCE_GROUP(ColorMap)
file( GLOB_RECURSE integration_list 		${CMAKE_CURRENT_SOURCE_DIR}/Integration/*)
ADD_SOURCE_GROUP(Integration)
file( GLOB_RECURSE registration_list 		${CMAKE_CURRENT_SOURCE_DIR}/Registration/*)
ADD_SOURCE_GROUP(Registration)
file( GLOB_RECURSE odometry_list 		${CMAKE_CURRENT_SOURCE_DIR}/Odometry/*)
ADD_SOURCE_GROUP(Odometry)

file(GLOB Visualization_basic 					${CMAKE_CURRENT_SOURCE_DIR}/Visualization/*.txt)
source_group("Visualization" FILES 				${Visualization_basic} )
file( GLOB visualization_Shader    				
${CMAKE_CURRENT_SOURCE_DIR}/Visualization/Shader/*.h 
${CMAKE_CURRENT_SOURCE_DIR}/Visualization/Shader/*.cpp)
source_group("Visualization\\Shader" FILES 		${visualization_Shader} )
file( GLOB_RECURSE visualization_Utility    	${CMAKE_CURRENT_SOURCE_DIR}/Visualization/Utility/*)
source_group("Visualization\\Utility" FILES 	${visualization_Utility} )
file( GLOB_RECURSE visualization_Visualizer    	${CMAKE_CURRENT_SOURCE_DIR}/Visualization/Visualizer/*)
source_group("Visualization\\Visualizer" FILES 	${visualization_Visualizer} )

set(Visualization_list ${visualization_Shader} ${visualization_Utility} ${visualization_Visualizer})

#set( CMAKE_CXX_VISIBILITY_PRESET hidden )

# Install headers
install(DIRECTORY   "${CMAKE_CURRENT_SOURCE_DIR}"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/include"
        PATTERN     "Visualization/Shader/GLSL" EXCLUDE
        PATTERN     "*.cpp" EXCLUDE
        PATTERN     "*.in"  EXCLUDE
        PATTERN     "*.txt" EXCLUDE)

# note: adding at least one real source file to any target that references
# reference: https://cmake.org/cmake/help/v3.12/command/add_library.html#object-libraries

add_library(${PROJECT_NAME} SHARED
			${basic_list}
			${camera_list}
			${odometry_list}
			${colorMap_list}
			${integration_list}
			${registration_list}
			${Visualization_list})

# Enforce 3rd party dependencies
add_dependencies(${PROJECT_NAME} ShaderFileTarget)
add_dependencies(${PROJECT_NAME} build_all_3rd_party_libs)
# headless rendering
if (ENABLE_HEADLESS_RENDERING)
    find_package(OSMesa REQUIRED)
    include_directories(${OSMESA_INCLUDE_DIR})
    target_link_libraries(${PROJECT_NAME} ${OSMESA_LIBRARY})
else ()
    find_package(OpenGL REQUIRED)
    target_link_libraries(${PROJECT_NAME} ${OPENGL_LIBRARIES})
endif ()

target_link_libraries(${PROJECT_NAME}
                      ${3RDPARTY_LIBRARIES}
                      ${OMP_LIBRARIES})

# INTERNAL LIB
target_link_libraries(  ${PROJECT_NAME} 
						CV_CORE_LIB 
						ECV_DB_LIB
						ECV_IO_LIB)

# Qt
target_link_libraries(${PROJECT_NAME} Qt5::Core)

ShowAndAbortOnWarning(${PROJECT_NAME})
					  
# input_dirs: a list of absolute paths in the source dir
# output_dirs: a list of relative paths in the install dir
# input_base_dir: the base dir of input_dirs
function(SourcePath2InstallPath input_dirs output_dirs input_base_dir)
    get_filename_component(absolute_input_base_dir ${input_base_dir} ABSOLUTE)
    foreach(input_dir ${input_dirs})
        # We need to handle case where input_dir is not child subdirectory of
        # input_base_dir (or input_base_dir itself).
        # This could happen for pre-installed headers, e.g.
        #   input_dir == /usr/include/libpng16
        #   input_base_dir = /path-to-repo/cloudViewer
        if(${input_dir} MATCHES "^${absolute_input_base_dir}($|/.*)")
            # extract path relative to the project source dir
            set(relative_path "")
            file(RELATIVE_PATH relative_path ${absolute_input_base_dir} ${input_dir})

            # construct relative path in the install dir
            set(install_path "${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}/${relative_path}")

            # save
            list(APPEND converted ${install_path})
        else()
            # When input_dir is not child subdirectory of input_base_dir,
            # we simply use the input_dir.
            list(APPEND converted ${input_dir})
        endif()
    endforeach()
    set (${output_dirs} ${converted} PARENT_SCOPE)
endfunction()

# build a list of include folders
SourcePath2InstallPath("${3RDPARTY_INCLUDE_DIRS_AT_INSTALL}" INSTALL_3RDPARTY_INCLUDE_DIRS ${PROJECT_SOURCE_DIR})

# set cloudViewer include directories
list(APPEND CONFIG_CloudViewer_INCLUDE_DIRS
    ${CMAKE_INSTALL_PREFIX}/include
    ${INSTALL_3RDPARTY_INCLUDE_DIRS}
)
list(REMOVE_DUPLICATES CONFIG_CloudViewer_INCLUDE_DIRS)


# set cloudViewer libraries
# If cloudViewer is a shared library
#     - Built-from-source 3rd-party libraries is embedded in cloudViewer
#     - Built-from-source 3rd-party libraries will NOT be installed
#     - A user app needs to link
#           1) cloudViewer
#           2) Pre-installed 3rd-party libraries
# If cloudViewer is a static library
#     - Built-from-source 3rd-party libraries is NOT embedded in cloudViewer
#     - Built-from-source 3rd-party libraries will be installed
#     - A user app needs to link
#           1) cloudViewer
#           2) Pre-installed 3rd-party libraries
#           3) Built-from-source 3rd-party libraries
list(APPEND CONFIG_CloudViewer_LIBRARIES "${PROJECT_NAME}" ${OMP_LIBRARIES})
if (BUILD_SHARED_LIBS)
    list(APPEND CONFIG_CloudViewer_LIBRARIES ${PRE_BUILT_3RDPARTY_LIBRARIES})
else ()
    list(APPEND CONFIG_CloudViewer_LIBRARIES ${3RDPARTY_LIBRARIES})
endif ()

# hot-fix for glfw vs glfw3 issue for Windows
# TODO: we shall revisit this for a better fix
# glfw is the target_name i.e. there is add_library(glfw) in glfw's cmake
# glfw3.lib (or libglfw3.a) is the library name of the compiled glfw
function(list_replace src_list dst_list src_val dst_val)
    foreach(val ${src_list})
        if(${val} STREQUAL ${src_val})
            message(STATUS "Replacing ${val} with ${dst_val}")
            list(APPEND converted ${dst_val})
        else()
            list(APPEND converted ${val})
        endif()
    endforeach()
    set(${dst_list} ${converted} PARENT_SCOPE)
endfunction(list_replace)

if (${BUILD_GLFW})
    message(STATUS "${CONFIG_CloudViewer_LIBRARIES}")
    list_replace("${CONFIG_CloudViewer_LIBRARIES}" CONFIG_CloudViewer_LIBRARIES "glfw" "glfw3")
    message(STATUS "${CONFIG_CloudViewer_LIBRARIES}")
endif()

# set cloudViewer library directories
list(APPEND CONFIG_CloudViewer_LIBRARY_DIRS
    "${CMAKE_INSTALL_PREFIX}/lib"
    ${3RDPARTY_LIBRARY_DIRS}
    ${OMP_LIBRARY_DIRS})

include(CMakePackageConfigHelpers)

add_subdirectory(Tools)

# install
install(TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

file(GLOB HEADERS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${HEADERS_ROOT} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME})


# Add custom preprocessor definitions
target_compile_definitions( ${PROJECT_NAME} PRIVATE ECV_OPENGL_ENGINE_LIBRARY_BUILD )

# install (shared) lib to specified destinations
InstallSharedLibrary( TARGET ${PROJECT_NAME} )
