# cloudViewer_set_cloudViewer_lib_properties() sets properties for the CloudViewer lib itself.
# This should be used for all object libraries that make up the CloudViewer lib.
#
# In comparison, cloudViewer_set_global_properties() sets properties for the CloudViewer
# lib, and targets that links to the CloudViewer lib, e.g pybind, unit tests, etc.
function(cloudViewer_set_cloudViewer_lib_properties target)
    target_compile_definitions(${target} PRIVATE CLOUDVIEWER_ENABLE_DLL_EXPORTS)
    target_compile_definitions(${target} PUBLIC ECV_OPENGL_ENGINE_LIBRARY_BUILD)
endfunction()

project( OPENGL_ENGINE_LIB )

include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )

file( GLOB basic_list ${CMAKE_CURRENT_SOURCE_DIR}/*.h )

# Configure a header file to pass the version settings to the source code
configure_file("${PROJECT_SOURCE_DIR}/CloudViewer.h.in"
               "${PROJECT_SOURCE_DIR}/CloudViewer.h")
configure_file("${PROJECT_SOURCE_DIR}/CloudViewerConfig.h.in"
               "${PROJECT_SOURCE_DIR}/CloudViewerConfig.h")

# Subdirectories
add_subdirectory(core)
set_target_properties(core PROPERTIES FOLDER "Modules")
add_subdirectory(t/geometry)
set_target_properties(tgeometry PROPERTIES FOLDER "Modules")
add_subdirectory(t/io)
set_target_properties(tio PROPERTIES FOLDER "Modules")
add_subdirectory(io)
set_target_properties(io PROPERTIES FOLDER "Modules")

add_subdirectory(ml)
if (BUILD_TENSORFLOW_OPS)
	set_target_properties(cloudViewer_tf_ops PROPERTIES FOLDER "Modules")
endif ()
if (BUILD_PYTORCH_OPS)
	set_target_properties(cloudViewer_torch_ops PROPERTIES FOLDER "Modules")
endif ()
set_target_properties(ml_contrib PROPERTIES FOLDER "Modules")

add_subdirectory(pipelines)
set_target_properties(pipelines PROPERTIES FOLDER "Modules")
add_subdirectory(visualization)
set_target_properties(visualization PROPERTIES FOLDER "Modules")
if (BUILD_GUI)
    add_subdirectory(visualization/gui)
    set(GUI_OBJECTS "$<TARGET_OBJECTS:GUI>")
	set_target_properties(GUI PROPERTIES FOLDER "Modules")
	set_target_properties(GuiResources PROPERTIES FOLDER "Modules")
endif()

# Source group for Visual Studio
ADD_SOURCE_GROUP(core)
ADD_SOURCE_GROUP(tgeometry)
ADD_SOURCE_GROUP(io)
ADD_SOURCE_GROUP(tio)
ADD_SOURCE_GROUP(ml)
ADD_SOURCE_GROUP(pipelines)
ADD_SOURCE_GROUP(visualization)

# note: adding at least one real source file to any target that references
# reference: https://cmake.org/cmake/help/v3.12/command/add_library.html#object-libraries
add_library(${PROJECT_NAME}
    CloudViewerConfig.cpp
    $<TARGET_OBJECTS:core>
    $<TARGET_OBJECTS:tgeometry>
    $<TARGET_OBJECTS:io>
    $<TARGET_OBJECTS:tio>
    $<TARGET_OBJECTS:ml_contrib>
    $<TARGET_OBJECTS:pipelines>
    $<TARGET_OBJECTS:visualization>
    ${GUI_OBJECTS}
)

# INTERNAL LIB
target_link_libraries(  ${PROJECT_NAME} PUBLIC
						ECV_DB_LIB
						ECV_IO_LIB)

# Qt
#target_link_libraries(${PROJECT_NAME} PUBLIC Qt5::Core)

ShowAndAbortOnWarning(${PROJECT_NAME})
cloudViewer_set_global_properties(${PROJECT_NAME})
cloudViewer_set_cloudViewer_lib_properties(${PROJECT_NAME})
cloudViewer_link_3rdparty_libraries(${PROJECT_NAME})

if(X11_TARGET)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${X11_TARGET})
endif()

add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

include(CMakePackageConfigHelpers)

# find_package CloudViewer
configure_package_config_file(CloudViewerConfig.cmake.in
                              "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CloudViewerConfig.cmake"
                              INSTALL_DESTINATION ${CloudViewer_INSTALL_CMAKE_DIR}
                              PATH_VARS CloudViewer_INSTALL_INCLUDE_DIR
                              NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

# find_package CloudViewer Version
write_basic_package_version_file("${PROJECT_BINARY_DIR}/CloudViewerConfigVersion.cmake"
                                 VERSION ${PROJECT_VERSION_THREE_NUMBER}
                                 COMPATIBILITY ExactVersion)

# Installation
install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}Targets
        RUNTIME DESTINATION ${CloudViewer_INSTALL_BIN_DIR}
        LIBRARY DESTINATION ${CloudViewer_INSTALL_LIB_DIR}
        ARCHIVE DESTINATION ${CloudViewer_INSTALL_LIB_DIR}
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DESTINATION ${CloudViewer_INSTALL_INCLUDE_DIR}
    FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.cuh"
)

# Install the CloudViewerConfig.cmake and CloudViewerConfigVersion.cmake
install(FILES
        "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CloudViewerConfig.cmake"
        "${PROJECT_BINARY_DIR}/CloudViewerConfigVersion.cmake"
        DESTINATION "${CloudViewer_INSTALL_CMAKE_DIR}" COMPONENT dev)

# uninstall target
if(NOT TARGET uninstall)
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
                   "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
                   IMMEDIATE @ONLY)

    add_custom_target(uninstall COMMAND ${CMAKE_COMMAND} -P
                      ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif()

# Export GUI_RESOURCE_FILES to parent CMake context (python/cloudViewer/)
set(GUI_RESOURCE_FILES ${GUI_RESOURCE_FILES} PARENT_SCOPE)
set(GUI_RESOURCE_DIR ${GUI_RESOURCE_DIR} PARENT_SCOPE)

# Add custom preprocessor definitions
target_compile_definitions( ${PROJECT_NAME} PUBLIC ECV_OPENGL_ENGINE_LIBRARY_BUILD )

# install (shared) lib to specified destinations
InstallSharedLibrary( TARGET ${PROJECT_NAME} )