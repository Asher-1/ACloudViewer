# SPDX-License-Identifier: MIT
# Copyright © Andy Maloney <asmaloney@gmail.com>

cmake_minimum_required(VERSION 3.10)

project(CVCoreLib
        DESCRIPTION
        "Core CloudViewer data structures & algorithms for handling point clouds"
        LANGUAGES
        CXX
        VERSION
        1.0
        )

# Options
option(CVCORELIB_USE_CGAL
        "Compile ${PROJECT_NAME} with CGAL (to enable Delaunay 2.5D triangulation with a GPL-compliant licence)"
        OFF
        )
option(CVCORELIB_USE_TBB
        "Compile ${PROJECT_NAME} with Intel Threading Building Blocks lib (enables some parallel processing )"
        OFF
        )
option(CVCORELIB_USE_QT_CONCURRENT
        "Compile ${PROJECT_NAME} with QtConcurrent (to enable parallel processing)"
        ON
        )
option(CVCORELIB_SHARED
        "Compile ${PROJECT_NAME} as a shared library"
        ON
        )
option(CVCORELIB_SCALAR_DOUBLE
        "Define ScalarType as double (instead of float)"
        OFF
        )

# Add the library (shared or static)
if (CVCORELIB_SHARED)
    add_library(${PROJECT_NAME} SHARED)

    set_target_properties(${PROJECT_NAME}
            PROPERTIES
            CXX_VISIBILITY_PRESET hidden
            VISIBILITY_INLINES_HIDDEN 1
            )
else ()
    add_library(${PROJECT_NAME} STATIC)

    target_compile_definitions(${PROJECT_NAME}
            PRIVATE
            CV_CORE_LIB_STATIC_DEFINE
            )
endif ()

set_target_properties(${PROJECT_NAME}
        PROPERTIES
        DEBUG_POSTFIX d
        )

# ccache
# https://crascit.com/2016/04/09/using-ccache-with-cmake/
find_program(CCACHE_PROGRAM ccache)

if (CCACHE_PROGRAM)
    set_target_properties(${PROJECT_NAME}
            PROPERTIES
            CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}"
            C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}"
            )
endif ()

# Main sources and includes
add_subdirectory(include)
add_subdirectory(src)

# Compiler & definitions
target_compile_features(${PROJECT_NAME}
        PRIVATE
        cxx_std_14
        )

set_target_properties(${PROJECT_NAME}
        PROPERTIES
        CXX_EXTENSIONS OFF
        )

target_compile_definitions(${PROJECT_NAME}
        PRIVATE
        "$<$<CONFIG:DEBUG>:CV_DEBUG>"
        )

if (CVCORELIB_SCALAR_DOUBLE)
    target_compile_definitions(${PROJECT_NAME}
            PUBLIC
            CV_CORE_LIB_USES_DOUBLE
            )
else ()
    target_compile_definitions(${PROJECT_NAME}
            PUBLIC
            CV_CORE_LIB_USES_FLOAT
            )
endif ()

if (${USE_SIMD})
    target_compile_definitions(${PROJECT_NAME} PUBLIC SIMD_ENABLED)
    target_compile_definitions(${PROJECT_NAME} PRIVATE ${CLOUDVIEWERCONFIG_SSE_DEFINITIONS})
    target_compile_options(${PROJECT_NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CLOUDVIEWERCONFIG_SSE_COMPILE_OPTIONS}>")
endif ()

# Windows-specific flags
if (WIN32)
    # Disable SECURE_SCL
    # See https://channel9.msdn.com/shows/Going+Deep/STL-Iterator-Debugging-and-Secure-SCL/
    target_compile_definitions(${PROJECT_NAME}
            PRIVATE
            "$<$<CONFIG:RELEASE>:_SECURE_SCL=0>"
            )

    target_compile_definitions(${PROJECT_NAME}
            PRIVATE
            _CRT_SECURE_NO_WARNINGS
            __STDC_LIMIT_MACROS
            NOMINMAX
            )
endif ()

# TBB (optional)
# Must come before CGAL so it can use TBB properly
if (CVCORELIB_USE_TBB)
    if (WIN32)
        include(cmake/FindTBB.cmake)
        if (TBB_FOUND)
            include_directories(${TBB_INCLUDE_DIRS})
            target_link_libraries(${PROJECT_NAME} PUBLIC ${ALL_TBB_LIBRARIES})
            target_compile_definitions(${PROJECT_NAME} PUBLIC USE_TBB)
            target_compile_definitions(${PROJECT_NAME} PUBLIC CV_CORE_LIB_USES_TBB)
            #InstallFiles( FILES ${TBB_LIBRARIES} )
        endif ()
    elseif (UNIX OR APPLE)
        message("Do not recommend using TBB on Unix or Mac platform!")
        #include( cmake/FindTBB_linux.cmake )
        include_directories(${STATIC_TBB_INCLUDE_DIR})
        target_compile_definitions(${PROJECT_NAME} PUBLIC USE_TBB)
        target_compile_definitions(${PROJECT_NAME} PUBLIC CV_CORE_LIB_USES_TBB)
    endif ()

endif ()

# CGAL (optional)
if (CVCORELIB_USE_CGAL)
    if (WIN32)
        include(cmake/CGALSupport.cmake)
    else ()
        find_package(CGAL REQUIRED)
    endif ()

    if (WIN32)
        # Need to force the visibility of these variables so that we can use them later
        set(GMP_LIBRARIES ${GMP_LIBRARIES} PARENT_SCOPE)
        set(MPFR_LIBRARIES ${MPFR_LIBRARIES} PARENT_SCOPE)
    endif ()

    if (CVCORELIB_USE_TBB)
        if (WIN32)
            if (TBB_FOUND)
                # Once Linux libcgal-dev >= 5.0, target_compile_definitions replaced by:
                #  CGAL_target_use_TBB( ${PROJECT_NAME} )

                target_compile_definitions(${PROJECT_NAME}
                        PRIVATE
                        CGAL_LINKED_WITH_TBB
                        NOMINMAX
                        )
            else ()
                message(WARNING "CGAL cannot compile with TBB (TBB not found)")
            endif ()
        elseif (UNIX OR APPLE)
            target_compile_definitions(${PROJECT_NAME}
                    PRIVATE
                    CGAL_LINKED_WITH_TBB
                    NOMINMAX
                    )
        endif ()

    endif ()

    target_link_libraries(${PROJECT_NAME}
            PUBLIC
            CGAL::CGAL
            )
    #target_link_libraries( ${PROJECT_NAME} PUBLIC ${CGAL_LIBRARIES} )
    target_compile_definitions(${PROJECT_NAME} PUBLIC USE_CGAL_LIB)
endif ()

# QT (optional)
if (CVCORELIB_USE_QT_CONCURRENT)
    find_package(Qt5
            COMPONENTS
            Concurrent
            REQUIRED
            )

    set_target_properties(${PROJECT_NAME} PROPERTIES
            AUTOMOC OFF
            AUTORCC OFF
            AUTOUIC OFF
            )

    target_link_libraries(${PROJECT_NAME}
            PUBLIC
            Qt5::Concurrent
            )

    target_compile_definitions(${PROJECT_NAME}
            PUBLIC
            CV_CORE_LIB_USES_QT_CONCURRENT
            )
endif ()

# Enforce 3rd party dependencies
target_compile_definitions(${PROJECT_NAME} PUBLIC USE_EIGEN)
cloudViewer_link_3rdparty_libraries(${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME} PUBLIC ${INTERNAL_EIGEN3_TARGET})

cloudViewer_install_targets(${PROJECT_NAME})
target_compile_definitions(${PROJECT_NAME} PUBLIC CV_CORE_LIB_LIBRARY_BUILD)

if (CVCORELIB_SHARED)
    InstallSharedLibrary(TARGET ${PROJECT_NAME})
endif ()

if (WIN32)
    target_compile_definitions(${PROJECT_NAME} PUBLIC _CRT_SECURE_NO_WARNINGS)
endif ()

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -DCV_DEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DCV_DEBUG")
