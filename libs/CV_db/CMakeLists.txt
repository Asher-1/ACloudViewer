cmake_minimum_required(VERSION ${CLOUDVIEWER_SUBPROJECT_CMAKE_MINIMUM_VERSION})

project(CV_DB_LIB)

add_library(${PROJECT_NAME} SHARED)

# Enable AUTOMOC and AUTOUIC for this target as we now use QObject based classes
set_target_properties(${PROJECT_NAME} PROPERTIES AUTOMOC ON AUTOUIC ON)

# UI files
file(GLOB_RECURSE UI_FILES ${CMAKE_CURRENT_SOURCE_DIR}/ui_templates/*.ui)
target_sources(${PROJECT_NAME} PRIVATE ${UI_FILES})

# Add ui_templates directory for AUTOUIC search path
set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY AUTOUIC_SEARCH_PATHS
    ${CMAKE_CURRENT_SOURCE_DIR}/ui_templates
)

# Include the generated UI headers directory
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

add_subdirectory(include)
add_subdirectory(src)

if (WIN32)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

if (MSVC)
    if (CVCORELIB_USE_TBB)
        include_directories(${TBB_INCLUDE_DIRS})
    endif ()
endif ()

target_include_directories(${PROJECT_NAME} PUBLIC ${NANOFLANN_INCLUDE_DIRS})

# INTERNAL LIB
target_link_libraries(${PROJECT_NAME} PUBLIC CVCoreLib)
# Qt
# Qt5/Qt6 compatibility - use unified Qt:: prefix
target_link_libraries(${PROJECT_NAME} PUBLIC Qt::OpenGL Qt::Widgets)

# Enforce 3rd party dependencies
if (USE_SIMD)
    target_compile_definitions(${PROJECT_NAME} PRIVATE ${CLOUDVIEWERCONFIG_SSE_DEFINITIONS})
    target_compile_options(${PROJECT_NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CLOUDVIEWERCONFIG_SSE_COMPILE_OPTIONS}>")
endif ()
cloudViewer_link_3rdparty_libraries(${PROJECT_NAME})

option(OPTION_USE_RANSAC_LIB "Build with Ransac Lib" ON)
if (${OPTION_USE_RANSAC_LIB})
    target_compile_definitions(${PROJECT_NAME} PUBLIC CV_RANSAC_SUPPORT POINTSWITHINDEX)
endif ()

# Add custom preprocessor definitions
target_compile_definitions(${PROJECT_NAME} PRIVATE CV_DB_LIBRARY_BUILD)

set_target_properties(${PROJECT_NAME} PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        )

if (NOT APPLE) # no need to copy targets on macos due to rpath
    cloudViewer_install_targets(${PROJECT_NAME})
    # install (shared) lib to specified destinations
    InstallSharedLibrary(TARGET ${PROJECT_NAME})
endif ()

include(cmake/InstallCGALDependencies.cmake)
