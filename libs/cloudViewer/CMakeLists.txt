cmake_minimum_required(VERSION ${CLOUDVIEWER_SUBPROJECT_CMAKE_MINIMUM_VERSION})

# cloudViewer_set_cloudViewer_lib_properties() sets properties for the CloudViewer lib itself.
# This should be used for all object libraries that make up the CloudViewer lib.
#
# In comparison, cloudViewer_set_global_properties() sets properties for the CloudViewer
# lib, and targets that links to the CloudViewer lib, e.g pybind, unit tests, etc.
function(cloudViewer_set_cloudViewer_lib_properties target)
    cmake_parse_arguments(arg "HIDDEN" "" "" ${ARGN})
    if(NOT BUILD_SHARED_LIBS)
        target_compile_definitions(${target} PUBLIC CLOUDVIEWER_STATIC)
    endif()
    if (arg_HIDDEN)
        set_target_properties(${target} PROPERTIES
                        CXX_VISIBILITY_PRESET hidden
                        VISIBILITY_INLINES_HIDDEN ON
                        )
    else ()
        target_compile_definitions(${target} PRIVATE CLOUDVIEWER_ENABLE_DLL_EXPORTS)
    endif()

    # Hide all 3rd party symbols, e.g. from header libraries.
    get_target_property(TARGET_TYPE ${target} TYPE)
    if (TARGET_TYPE STREQUAL SHARED_LIBRARY)
        if (APPLE)
            file(GENERATE OUTPUT libCloudViewer.map CONTENT
                [=[*cloudViewer*
                *CloudViewer*
                ]=])
            target_link_options(${target} PRIVATE $<$<CONFIG:Release>:
                -Wl,-exported_symbols_list
                "${CMAKE_CURRENT_BINARY_DIR}/libCloudViewer.map" >)
        elseif (UNIX)   # Linux
            file(GENERATE OUTPUT libCloudViewer.map CONTENT
                [=[{
                    global:
                        *cloudViewer*;
                        extern "C++" {
                            cloudViewer::*;
                        };
                    local: *;
                };]=])
            target_link_options(${target} PRIVATE $<$<CONFIG:Release>:
            "-Wl,--version-script=${CMAKE_CURRENT_BINARY_DIR}/libCloudViewer.map" >)
        elseif (WIN32)
            # TODO(Sameer): Only export cloudViewer symbols
        endif()
    endif()
endfunction()

project( ${CLOUDVIEWER_APP_NAME} )

add_library( ${CLOUDVIEWER_APP_NAME} )

include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )

# Configure a header file to pass the version settings to the source code
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/CloudViewer.h.in"
               "${CMAKE_CURRENT_SOURCE_DIR}/CloudViewer.h")

# Subdirectories
add_subdirectory(core)
set_target_properties(core PROPERTIES FOLDER "Modules")
add_subdirectory(data)
set_target_properties(data PROPERTIES FOLDER "Modules")
add_subdirectory(t/geometry)
set_target_properties(tgeometry PROPERTIES FOLDER "Modules")
add_subdirectory(t/io)
set_target_properties(tio PROPERTIES FOLDER "Modules")
add_subdirectory(t/pipelines)
set_target_properties(tpipelines PROPERTIES FOLDER "Modules")
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
add_subdirectory(utility)
set_target_properties(utility PROPERTIES FOLDER "Modules")

add_subdirectory(visualization)
set_target_properties(visualization PROPERTIES FOLDER "Modules")
if (BUILD_GUI)
    add_subdirectory(visualization/gui)
    set_target_properties(GUI PROPERTIES FOLDER "Modules")
endif()

if (BUILD_WEBRTC)
    add_subdirectory(visualization/webrtc_server)
    set_target_properties(webrtc_server PROPERTIES FOLDER "Modules")
endif()

# note: adding at least one real source file to any target that references
# reference: https://cmake.org/cmake/help/v3.12/command/add_library.html#object-libraries
target_sources(${CLOUDVIEWER_APP_NAME} PRIVATE
    $<TARGET_OBJECTS:core>
    $<TARGET_OBJECTS:core_impl>
    $<TARGET_OBJECTS:data>
    $<TARGET_OBJECTS:tgeometry>
    $<TARGET_OBJECTS:tgeometry_kernel>
    $<TARGET_OBJECTS:io>
    $<TARGET_OBJECTS:tio>
    $<TARGET_OBJECTS:ml_contrib>
    $<TARGET_OBJECTS:pipelines>
    $<TARGET_OBJECTS:tpipelines>
    $<TARGET_OBJECTS:tpipelines_kernel>
    $<TARGET_OBJECTS:utility>
    $<TARGET_OBJECTS:visualization>
)

if (BUILD_GUI)
    target_sources(${CLOUDVIEWER_APP_NAME} PRIVATE
        $<TARGET_OBJECTS:GUI>
    )
endif()

if (BUILD_WEBRTC)
    target_sources(${CLOUDVIEWER_APP_NAME} PRIVATE
        $<TARGET_OBJECTS:webrtc_server>
    )
endif()

# Source group for Visual Studio
add_source_group(core)
add_source_group(data)
add_source_group(tgeometry)
add_source_group(io)
add_source_group(tio)
add_source_group(ml)
add_source_group(pipelines)
add_source_group(tpipelines)
add_source_group(utility)
add_source_group(visualization)

cloudViewer_show_and_abort_on_warning(${CLOUDVIEWER_APP_NAME})
cloudViewer_set_global_properties(${CLOUDVIEWER_APP_NAME})
cloudViewer_set_cloudViewer_lib_properties(${CLOUDVIEWER_APP_NAME})
cloudViewer_link_3rdparty_libraries(${CLOUDVIEWER_APP_NAME})

# If we are building a STATIC_LIBRARY, hide symbols coming from 3rd party static
# libraries that are not hidden during compilation. Don't propagate beyond
# direct consumers of libCloudViewer.a
target_link_options(${CLOUDVIEWER_APP_NAME} INTERFACE
    $<$<STREQUAL:$<TARGET_PROPERTY:${CLOUDVIEWER_APP_NAME},TYPE>,STATIC_LIBRARY>:$<LINK_ONLY:${CLOUDVIEWER_HIDDEN_3RDPARTY_LINK_OPTIONS}>>)

add_library(${CLOUDVIEWER_APP_NAME}::${CLOUDVIEWER_APP_NAME} ALIAS ${CLOUDVIEWER_APP_NAME})

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
if (WIN32 OR APPLE) 
    install(TARGETS ${CLOUDVIEWER_APP_NAME} EXPORT CloudViewerTargets
            RUNTIME DESTINATION ${CloudViewer_INSTALL_BIN_DIR}
            LIBRARY DESTINATION ${CloudViewer_INSTALL_LIB_DIR}
            ARCHIVE DESTINATION ${CloudViewer_INSTALL_LIB_DIR}
    )
endif()

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

# install (shared) lib to specified destinations
if (WIN32 OR APPLE) 
    InstallSharedLibrary( TARGET ${CLOUDVIEWER_APP_NAME} )
endif()
