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

add_subdirectory(calibration)
add_subdirectory(controllers)
add_subdirectory(estimators)
add_subdirectory(geometry)
add_subdirectory(image)
add_subdirectory(inertial)
add_subdirectory(exe)
add_subdirectory(feature)
add_subdirectory(math)
add_subdirectory(mvs)
add_subdirectory(optim)
add_subdirectory(retrieval)
add_subdirectory(scene)
add_subdirectory(sensor)
add_subdirectory(sfm)
add_subdirectory(tools)
add_subdirectory(util)
add_subdirectory(pipelines)
if (GUI_ENABLED)
    add_subdirectory(ui)
endif ()

get_property(COLMAP_SOURCES GLOBAL PROPERTY COLMAP_SOURCES)
COLMAP_ADD_STATIC_LIBRARY(${COLMAP_LIB_NAME} ${COLMAP_SOURCES})

# Upstream parity: upstream COLMAP compiles without an -march override, so
# its host-side pose math (mvs/image.cc: ComputeRelativePose,
# ComposeProjectionMatrix, ...) keeps the baseline non-FMA contraction. The
# global -march=x86-64-v3 (cmake/CMakeFindSSE.cmake) enables AVX2/FMA and
# shifts those 3x3 products by ~1 ULP, perturbing the patch_match pose
# texture and cascading into divergent depth maps. Restore the baseline for
# the whole COLMAP library, matching upstream codegen.
target_compile_options(ColmapLib PRIVATE "-march=x86-64")
target_link_libraries(${COLMAP_LIB_NAME} PUBLIC 3rdparty_meshoptimizer)
target_link_libraries(${COLMAP_LIB_NAME} PRIVATE 3rdparty_faiss)

# Vendored Boost 1.92 header-only subset = boost/unordered 1.92 plus its
# measured include closure (assert/config/container_hash/core/cstdint/mp11/
# predef/throw_exception/describe...) backing util/hash_containers.h. The
# FlatHashMap/NodeHashMap aliases are data members of public scene/SfM
# headers and their iteration order is observable SfM behavior (see
# util/hash_containers.h), so every C++ consumer of ColmapLib must see the
# same boost::unordered definition: the path is PUBLIC and searched BEFORE
# the system Boost headers (supported floor: 1.71). boost/move, boost/
# container and boost/intrusive are deliberately NOT vendored: the unordered
# closure does not include them, so system Boost resolves them everywhere
# and no TU ever mixes 1.74 container/pair.hpp with 1.92 move detail traits
# (the ODR redefinition that broke the CGAL TUs when the two trees were
# mixed). The one intentional local deviation inside the closure is the
# serialization/nvp guard handshake in core/serialization.hpp (plus the
# noexcept-less make_nvp in core/nvp.hpp): the vendor definition must yield
# to the legacy system serialization/nvp.hpp on Boost < 1.74 while still
# providing the complete nvp/make_nvp that Boost >= 1.84's own headers
# (multi_index via core/serialization.hpp) instantiate — see the NOTE in
# that header. Measured PASS on 1.71 (focal image), 1.74, 1.84.
# Compiled Boost libraries stay on the system/ext_boost version; none of the
# vendored libraries has a compiled component that ColmapLib uses. CUDA TUs
# never include the aliases and keep their existing Boost resolution.
target_include_directories(${COLMAP_LIB_NAME} BEFORE PUBLIC
        "$<$<COMPILE_LANGUAGE:CXX>:${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/boost_unordered/include>")


# Upstream-parity host math: the Image constructor computes P_/inv_P_ through
# Eigen's Matrix4f::inverse(), whose result is codegen-sensitive — with the
# fork-wide EIGEN_MAX_ALIGN_BYTES=0 (PCL compatibility, root CMakeLists) it
# produces last-bit-different inv_P values versus the upstream build. inv_P
# feeds the patch_match poses texture (channels 31-42) and is consumed by
# ComputeGeomConsistencyCost: a 1-ulp inv_P difference chaotically diverges
# the geometric-consistency pass while the photometric pass (raw K/R/T
# channels, no host arithmetic) stays bit-exact. Compile this TU with the
# default Eigen alignment so inv_P is bit-identical to upstream. All Eigen
# objects in this TU are function-local, so the alignment change cannot leak
# across translation units (no ODR/layout exposure).
# NOTE: must be set in THIS directory — source properties are scoped to the
# directory where they are set, and ColmapLib is defined here.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(
            ${CMAKE_CURRENT_SOURCE_DIR}/mvs/image.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/feature/utils.cc
            # Two-view geometry / RANSAC host math: the TVG verifier
            # (fundamental/homography/essential estimation, Sampson checks,
            # LORANSAC) runs the same Eigen expression graphs on both sides;
            # with the fork-wide EIGEN_MAX_ALIGN_BYTES=0 a handful of
            # threshold decisions flip by 1-ulp (observed: TVG pair (4,5)
            # inlier sets differ by exactly one match). Restore upstream
            # default alignment on the estimator path. All Eigen objects
            # involved are function-local (same ODR argument as image.cc).
            ${CMAKE_CURRENT_SOURCE_DIR}/estimators/two_view_geometry.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/estimators/fundamental_matrix_degensac.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/geometry/essential_matrix.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/geometry/homography_matrix.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/geometry/normalization.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/geometry/pose.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/optim/support_measurement.cc
            PROPERTIES COMPILE_OPTIONS "-UEIGEN_MAX_ALIGN_BYTES")
endif()


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

# Texturing/mvs integration consumes the C++ IO layer (AutoReadMesh) and
# ccMesh geometry helpers (CV_DB). Both must be PUBLIC: ColmapLib is STATIC,
# so its undefined symbols have to resolve in every executable that links
# it (colmap_test binaries were failing with DSO-missing-linker errors).
if(TARGET CV_IO_LIB)
    target_link_libraries(${COLMAP_LIB_NAME} PUBLIC CV_IO_LIB)
endif()
if(TARGET CV_DB_LIB)
    target_link_libraries(${COLMAP_LIB_NAME} PUBLIC CV_DB_LIB)
endif()

# PoseLib is enabled by default for reconstruction. Keep it scoped to ColmapLib
# so unrelated CloudViewer targets do not inherit a minimal-solver library.
if(RECONSTRUCTION_FETCH_POSELIB AND TARGET PoseLib::PoseLib)
    # PUBLIC: estimators/solvers/poselib_utils.h exposes poselib types in
    # its interface, so test targets including it need the include dirs.
    target_link_libraries(${COLMAP_LIB_NAME} PUBLIC PoseLib::PoseLib)
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
    # Every gate also passes --no-tests=error: ctest otherwise exits 0 when its
    # regex selects nothing, so a renamed or relocated test would turn the gate
    # into a green no-op.
    # Unified C++ test gate for the whole reconstruction subtree: runs
    # every COLMAP_ADD_TEST gtest (scene/estimators/geometry/mvs/image/
    # sensor/optim/math/util/sfm/controllers/feature/retrieval) except the
    # environment-bound ones (SiftGPU and opengl_utils need an OpenGL
    # context, which headless GPU-less CI runners do not have). The set is
    # pure CPU, passes in seconds locally, and satisfies the GitHub runner
    # timeout constraints. Conditional tests (caspar, PoseLib) are only in
    # the collected list when they were actually registered. The name regex
    # also matches device tests registered by COLMAP_ADD_CUDA_TEST /
    # COLMAP_ADD_HIP_TEST (e.g. mvs/gpu_mat_test), so their "gpu" label is
    # excluded to keep the selected set equal to the collected one.
    get_property(RECONSTRUCTION_ALL_TEST_TARGETS GLOBAL PROPERTY
                 COLMAP_RECONSTRUCTION_TESTS)
    add_custom_target(reconstruction_cpp_tests_gate
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --no-tests=error
                -R "^(scene|estimators|feature|geometry|image|math|mvs|optim|retrieval|sensor|sfm|util|controllers)/[a-z_0-9]+_test$"
                -E "^(feature/sift_test|util/opengl_utils_test)$"
                -LE "^gpu$"
        DEPENDS ${RECONSTRUCTION_ALL_TEST_TARGETS}
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/libs/Reconstruction"
        USES_TERMINAL
        VERBATIM)

    add_custom_target(reconstruction_oiio_parity_gate
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --no-tests=error
                -R "^(sensor/bitmap_test|image/undistortion_test|mvs/texture_mapping_test)$"
        DEPENDS bitmap_test undistortion_test texture_mapping_test
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/libs/Reconstruction"
        USES_TERMINAL
        VERBATIM)

    # This protects full-sphere projection and the legacy rig persistence API.
    # The optional PoseLib adapter appends its one-sided-focal tests only when
    # its targets exist in the configured build.
    set(RECONSTRUCTION_CAMERA_RIG_PARITY_TESTS
        scene/camera_test
        scene/projection_test
        scene/rig_test
        estimators/solvers_essential_matrix_test
        estimators/two_view_geometry_test)
    set(RECONSTRUCTION_CAMERA_RIG_PARITY_DEPS
        camera_test projection_test rig_test solvers_essential_matrix_test
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
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --no-tests=error
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
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --no-tests=error
                -R "^scene/database_test$"
        DEPENDS database_test
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/libs/Reconstruction"
        USES_TERMINAL
        VERBATIM)

    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(
        NAME scripts/run_loma_sfm_comparison_test
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/run_loma_sfm_comparison_test.py)

    # L3 is deliberately opt-in: it needs a pinned fixture, an executable
    # built from the exact upstream revision, and calibrated thresholds. An
    # absent prerequisite is a configuration error, never a green no-op.
    if (RECONSTRUCTION_L3_GATE_ENABLED)
        # Use the fork's real command registry unless an external candidate was
        # explicitly supplied. The target file generator expression selects
        # the platform's executable suffix and macOS bundle layout correctly.
        set(_L3_ACV_COLMAP "${RECONSTRUCTION_L3_ACV_COLMAP}")
        if ("${_L3_ACV_COLMAP}" STREQUAL "")
            if (APPLE)
                set(_L3_ACV_COLMAP "$<TARGET_FILE:ColmapApp>")
                set(_L3_ACV_TARGET ColmapApp)
            else ()
                set(_L3_ACV_COLMAP "$<TARGET_FILE:colmap_exe>")
                set(_L3_ACV_TARGET colmap_exe)
            endif ()
        endif ()
        set(_L3_RUNTIME_LIBRARY_ARGS
            --runtime-library-dir "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
            --runtime-library-dir "${CMAKE_BINARY_DIR}/external/lib")
        cloudviewer_resolve_qt_lib_dir(_L3_QT_LIBRARY_DIR)
        if (_L3_QT_LIBRARY_DIR)
            list(APPEND _L3_RUNTIME_LIBRARY_ARGS
                 --runtime-library-dir "${_L3_QT_LIBRARY_DIR}")
        endif ()

        foreach (L3_REQUIRED_VAR
                 RECONSTRUCTION_L3_IMAGES
                 RECONSTRUCTION_L3_UPSTREAM_COLMAP
                 RECONSTRUCTION_L3_UPSTREAM_SOURCE
                 RECONSTRUCTION_L3_UPSTREAM_REVISION
                 RECONSTRUCTION_L3_THRESHOLDS)
            if ("${${L3_REQUIRED_VAR}}" STREQUAL "")
                message(FATAL_ERROR
                    "${L3_REQUIRED_VAR} is required when RECONSTRUCTION_L3_GATE_ENABLED=ON")
            endif()
        endforeach()
        add_custom_target(reconstruction_l3_alignment_gate
            COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/run_loma_sfm_comparison.py
                    --l3
                    --images ${RECONSTRUCTION_L3_IMAGES}
                    --output ${RECONSTRUCTION_L3_OUTPUT}
                    --harness ${RECONSTRUCTION_L3_HARNESS}
                    --acv-colmap ${_L3_ACV_COLMAP}
                    ${_L3_RUNTIME_LIBRARY_ARGS}
                    --upstream-colmap ${RECONSTRUCTION_L3_UPSTREAM_COLMAP}
                    --upstream-source ${RECONSTRUCTION_L3_UPSTREAM_SOURCE}
                    --upstream-revision ${RECONSTRUCTION_L3_UPSTREAM_REVISION}
                    --seed ${RECONSTRUCTION_L3_SEED}
                    # Fixture options of the committed thresholds; the runner
                    # rejects a run whose configuration differs from them.
                    --single-camera
                    --mapper-option Mapper.num_threads=1
                    --mapper-option Mapper.min_num_matches=20
                    --l3-thresholds ${RECONSTRUCTION_L3_THRESHOLDS}
            DEPENDS ${_L3_ACV_TARGET}
            USES_TERMINAL
            VERBATIM)
    endif()
endif()

# Caspar is an optional CUDA backend. The gate intentionally only exists in a
# Caspar-enabled configuration, where it proves that the generated solver is
# executed directly and remains numerically close to the Ceres reference.
if (TESTS_ENABLED AND RECONSTRUCTION_CASPAR_ENABLED)
    # The test is registered from estimators/ (COLMAP_ADD_TEST names tests
    # after the registering directory), not from optim/.
    add_custom_target(reconstruction_caspar_parity_gate
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --no-tests=error
                -R "^estimators/bundle_adjustment_caspar_test$"
        DEPENDS bundle_adjustment_caspar_test
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/libs/Reconstruction"
        USES_TERMINAL
        VERBATIM)
endif()
