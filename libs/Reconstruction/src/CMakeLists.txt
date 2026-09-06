# Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

if (IS_MSVC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3")
elseif (IS_GNU OR IS_CLANG)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
endif ()

set_property(GLOBAL PROPERTY COLMAP_SOURCES "")

add_subdirectory(base)
add_subdirectory(controllers)
add_subdirectory(estimators)
add_subdirectory(geometry)
add_subdirectory(exe)
add_subdirectory(feature)
add_subdirectory(mvs)
add_subdirectory(optim)
add_subdirectory(retrieval)
add_subdirectory(scene)
add_subdirectory(sfm)
add_subdirectory(tools)
add_subdirectory(util)
add_subdirectory(pipelines)
if (GUI_ENABLED)
    add_subdirectory(ui)
endif ()

get_property(COLMAP_SOURCES GLOBAL PROPERTY COLMAP_SOURCES)
COLMAP_ADD_STATIC_LIBRARY(${COLMAP_LIB_NAME} ${COLMAP_SOURCES})
target_link_libraries(${COLMAP_LIB_NAME} PUBLIC 3rdparty_meshoptimizer)
target_link_libraries(${COLMAP_LIB_NAME} PRIVATE 3rdparty_faiss)

if (CUDA_ENABLED)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
    # Use a separate stream per thread to allow for concurrent kernel execution
    # between multiple threads on the same device.
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --default-stream per-thread")

    # Suppress warnings:
    # ptxas warning : Stack size for entry function X cannot be statically determined.
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas=-suppress-stack-size-warning")

    # Fix for Ubuntu 16.04.
    add_compile_definitions(_MWAITXINTRIN_H_INCLUDED)
    if (MSVC)
        # Avoid pulling in too many header files through <windows.h>
        add_compile_definitions(WIN32_LEAN_AND_MEAN)
    endif ()

    COLMAP_ADD_STATIC_CUDA_LIBRARY(colmap_cuda ${COLMAP_CUDA_SOURCES})
    target_link_libraries(colmap_cuda PRIVATE ${COLMAP_LIB_NAME})
    target_link_libraries(${COLMAP_LIB_NAME} PRIVATE colmap_cuda ${COLMAP_INTERNAL_LIBRARIES})
    target_link_libraries(${COLMAP_LIB_NAME} PUBLIC  ${COLMAP_EXTERNAL_LIBRARIES})
elseif(HIP_ENABLED)
    COLMAP_ADD_STATIC_HIP_LIBRARY(colmap_hip ${COLMAP_HIP_SOURCES})
    target_link_libraries(colmap_hip PRIVATE ${COLMAP_LIB_NAME})
    target_link_libraries(${COLMAP_LIB_NAME} PRIVATE colmap_hip ${COLMAP_INTERNAL_LIBRARIES})
    target_link_libraries(${COLMAP_LIB_NAME} PUBLIC ${COLMAP_EXTERNAL_LIBRARIES})
else ()
    target_link_libraries(${COLMAP_LIB_NAME} PUBLIC ${COLMAP_INTERNAL_LIBRARIES} ${COLMAP_EXTERNAL_LIBRARIES})
endif ()

# DA3: dynamic link to libAICore.so (PUBLIC so static ColmapLib consumers resolve symbols)
if(AICore_ENABLED AND TARGET AICore)
    target_link_libraries(${COLMAP_LIB_NAME} PUBLIC AICore)
    # Propagate to ColmapLib translation units — PUBLIC link interface alone does not
    # always define AICore_ENABLED when compiling this static library's .cc files.
    target_compile_definitions(${COLMAP_LIB_NAME} PUBLIC AICore_ENABLED)
endif()

# PoseLib is enabled by default for reconstruction. Keep it scoped to ColmapLib
# so unrelated CloudViewer targets do not inherit a minimal-solver library.
if(RECONSTRUCTION_FETCH_POSELIB AND TARGET PoseLib::PoseLib)
    target_link_libraries(${COLMAP_LIB_NAME} PRIVATE PoseLib::PoseLib)
    target_compile_definitions(${COLMAP_LIB_NAME} PRIVATE RECONSTRUCTION_FETCH_POSELIB)
endif()

if(RECONSTRUCTION_CASPAR_ENABLED AND TARGET 3rdparty_caspar)
    target_link_libraries(${COLMAP_LIB_NAME} PRIVATE 3rdparty_caspar)
    # ColmapLib is static, so compile its sources with the backend selected by
    # the private Caspar interface dependency as well.
    target_compile_definitions(${COLMAP_LIB_NAME} PRIVATE CASPAR_ENABLED)
    if(RECONSTRUCTION_CASPAR_USE_DOUBLE)
        target_compile_definitions(${COLMAP_LIB_NAME} PRIVATE CASPAR_USE_DOUBLE)
    endif()
endif()

# The embedded reconstruction widgets use QFutureWatcher/QtConcurrent for
# cancellable model downloads. Keep this dependency explicit for both Qt 5
# and Qt 6 builds instead of relying on the application target transitively.
if (GUI_ENABLED AND TARGET Qt::Concurrent)
    target_link_libraries(${COLMAP_LIB_NAME} PUBLIC Qt::Concurrent)
endif()

# Keep the OIIO migration contract executable across all supported platforms.
# bitmap_test covers format I/O, numeric EXIF, and resampling; the other two
# exercise the downstream undistortion and texture callers using Bitmap.
if (TESTS_ENABLED)
    # All parity gates run ctest from the subtree where enable_testing() was
    # called: the build-tree root has no CTestTestfile.cmake, so a plain
    # ${CMAKE_BINARY_DIR} working directory would silently discover zero tests.
    add_custom_target(reconstruction_oiio_parity_gate
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
                -R "^(util/bitmap_test|base/undistortion_test|mvs/texture_mapping_test)$"
        DEPENDS bitmap_test undistortion_test texture_mapping_test
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/libs/Reconstruction"
        USES_TERMINAL
        VERBATIM)

    # This protects full-sphere projection and the legacy rig persistence API.
    # The optional PoseLib adapter appends its one-sided-focal tests only when
    # its targets exist in the configured build.
    set(RECONSTRUCTION_CAMERA_RIG_PARITY_TESTS
        base/camera_test
        base/projection_test
        base/camera_rig_test
        estimators/essential_matrix_test
        estimators/two_view_geometry_test)
    set(RECONSTRUCTION_CAMERA_RIG_PARITY_DEPS
        camera_test projection_test camera_rig_test essential_matrix_test
        two_view_geometry_test)
    if (RECONSTRUCTION_FETCH_POSELIB)
        list(APPEND RECONSTRUCTION_CAMERA_RIG_PARITY_TESTS
             estimators/relpose_one_sided_focal_test)
        list(APPEND RECONSTRUCTION_CAMERA_RIG_PARITY_DEPS
             relpose_one_sided_focal_test)
    endif()
    string(JOIN "|" RECONSTRUCTION_CAMERA_RIG_PARITY_REGEX
           ${RECONSTRUCTION_CAMERA_RIG_PARITY_TESTS})
    add_custom_target(reconstruction_camera_rig_parity_gate
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
                -R "^(${RECONSTRUCTION_CAMERA_RIG_PARITY_REGEX})$"
        DEPENDS ${RECONSTRUCTION_CAMERA_RIG_PARITY_DEPS}
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/libs/Reconstruction"
        USES_TERMINAL
        VERBATIM)

    # Database user_version migration machinery: proves that legacy fork and
    # legacy upstream databases are migrated (sentinel poses/matrices -> NULL,
    # descriptors.type) and that databases written by a newer schema are
    # refused instead of being silently misread.
    add_custom_target(reconstruction_db_migration_gate
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
                -R "^base/database_test$"
        DEPENDS database_test
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/libs/Reconstruction"
        USES_TERMINAL
        VERBATIM)
endif()

# Caspar is an optional CUDA backend. The gate intentionally only exists in a
# Caspar-enabled configuration, where it proves that the generated solver is
# executed directly and remains numerically close to the Ceres reference.
if (TESTS_ENABLED AND RECONSTRUCTION_CASPAR_ENABLED)
    add_custom_target(reconstruction_caspar_parity_gate
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
                -R "^optim/bundle_adjustment_caspar_test$"
        DEPENDS bundle_adjustment_caspar_test
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/libs/Reconstruction"
        USES_TERMINAL
        VERBATIM)
endif()
