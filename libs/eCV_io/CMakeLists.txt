project(ECV_IO_LIB)

add_library(${PROJECT_NAME} SHARED)

include(cmake/GDALSupport.cmake)

add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(ui)

if (WIN32)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

option(OPTION_USE_DXF_LIB "Build with Ribbonsoft's DXF Lib (AutoCAD DXF files support)" ON)
if (${OPTION_USE_DXF_LIB})
    set(DXF_LIB_DIR "extern/dxflib" CACHE PATH "DXF lib path")
    add_subdirectory("${DXF_LIB_DIR}")
    target_link_libraries(${PROJECT_NAME} PUBLIC dxflib)
endif ()

option(OPTION_USE_SHAPE_LIB "Build with ShapeLib (SHP files support)" ON)
if (${OPTION_USE_SHAPE_LIB})
    set(SHAPELIB_LIB_DIR "extern/shapelib" CACHE PATH "shapelib lib path")
    add_subdirectory("${SHAPELIB_LIB_DIR}")
    target_link_libraries(${PROJECT_NAME} PUBLIC shapelib)
endif ()

# Enforce 3rd party dependencies
if (USE_SIMD)
    target_compile_definitions(${PROJECT_NAME} PRIVATE ${CLOUDVIEWERCONFIG_SSE_DEFINITIONS})
    target_compile_options(${PROJECT_NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CLOUDVIEWERCONFIG_SSE_COMPILE_OPTIONS}>")
endif ()

cloudViewer_link_3rdparty_libraries(${PROJECT_NAME})

# INTERNAL LIB
target_link_libraries(${PROJECT_NAME} PUBLIC ECV_DB_LIB)

if (OPTION_USE_GDAL)
    target_link_GDAL(${PROJECT_NAME})
endif ()

# contrib. libraries support
target_link_contrib(${PROJECT_NAME})

# Add custom preprocessor definitions
target_compile_definitions(${PROJECT_NAME} PRIVATE ECV_IO_LIBRARY_BUILD)

set_target_properties(${PROJECT_NAME} PROPERTIES
        AUTOUIC ON # FIXME Remove after everything has moved to targets and we can set it globally
        CXX_VISIBILITY_PRESET hidden
        )

if (BUILD_TESTING)
    add_subdirectory(test)
endif ()

if (NOT APPLE) # no need to copy targets on macos due to rpath
    cloudViewer_install_targets(${PROJECT_NAME})
    InstallSharedLibrary(TARGET ${PROJECT_NAME})
endif ()
