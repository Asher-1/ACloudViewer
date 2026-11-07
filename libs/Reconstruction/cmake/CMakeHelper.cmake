cmake_minimum_required(VERSION 3.24)

if (POLICY CMP0043)
    cmake_policy(SET CMP0043 NEW)
endif ()

if (POLICY CMP0054)
    cmake_policy(SET CMP0054 NEW)
endif ()

# Determine project compiler.
if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(IS_MSVC TRUE)
endif ()
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(IS_GNU TRUE)
endif ()
if (CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
    set(IS_CLANG TRUE)
endif ()

# Determine project architecture.
if (CMAKE_SYSTEM_PROCESSOR MATCHES "[ix].?86|amd64|AMD64")
    set(IS_X86 TRUE)
endif ()

# Determine project operating system.
string(REGEX MATCH "Linux" IS_LINUX ${CMAKE_SYSTEM_NAME})
string(REGEX MATCH "DragonFly|BSD" IS_BSD ${CMAKE_SYSTEM_NAME})
string(REGEX MATCH "SunOS" IS_SOLARIS ${CMAKE_SYSTEM_NAME})
if (WIN32)
    SET(IS_WINDOWS TRUE BOOL INTERNAL)
endif ()
if (APPLE)
    SET(IS_MACOS TRUE BOOL INTERNAL)
endif ()

string(TOLOWER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_LOWER)
if (CMAKE_BUILD_TYPE_LOWER STREQUAL "debug"
        OR CMAKE_BUILD_TYPE_LOWER STREQUAL "relwithdebinfo")
    set(IS_DEBUG TRUE)
endif ()

# Enable solution folders.
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_TARGETS_ROOT_FOLDER "cmake")
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER
        ${CMAKE_TARGETS_ROOT_FOLDER})
set(COLMAP_TARGETS_ROOT_FOLDER "colmap_targets")
set(COLMAP_SRC_ROOT_FOLDER "colmap_sources")

set(CUDA_LINK_LIBRARIES_KEYWORD PRIVATE)

# Test executables are grouped into bin/colmap_test/ (mirroring bin/aicore_tests/)
# so the main bin/ only holds the app and shared libraries. CTest resolves each
# binary through $<TARGET_FILE>, so the CTest registry stays valid, and runtime
# libraries are located via the build-tree absolute RPATH.
set(COLMAP_TEST_RUNTIME_DIR "${CMAKE_BINARY_DIR}/bin/colmap_test")
file(MAKE_DIRECTORY "${COLMAP_TEST_RUNTIME_DIR}")

# When AICore is linked, omit static CUDA archives from 3rdparty deps to avoid
# embedding ~400MB nv_fatbin into shared libs / executables (CUDA lives in libAICore.so).
macro(COLMAP_LINK_3RDPARTY TARGET_NAME)
    if(AICore_ENABLED AND TARGET AICore)
        cloudViewer_link_3rdparty_libraries(${TARGET_NAME} EXCLUDE_CUDA_STATIC)
    else()
        cloudViewer_link_3rdparty_libraries(${TARGET_NAME})
    endif()
endmacro(COLMAP_LINK_3RDPARTY)

# This macro will search for source files in a given directory, will add them
# to a source group (folder within a project), and will then return paths to
# each of the found files. The usage of the macro is as follows:
# COLMAP_ADD_SOURCE_DIR(
#     <source directory to search>
#     <output variable with found source files>
#     <search expressions such as *.h *.cc>)
macro(COLMAP_ADD_SOURCE_DIR SRC_DIR SRC_VAR)
    # Create the list of expressions to be used in the search.
    set(GLOB_EXPRESSIONS "")
    foreach (ARG ${ARGN})
        list(APPEND GLOB_EXPRESSIONS ${SRC_DIR}/${ARG})
    endforeach ()
    # Perform the search for the source files.
    file(GLOB ${SRC_VAR} RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
            ${GLOB_EXPRESSIONS})
    # Create the source group.
    string(REPLACE "/" "\\" GROUP_NAME ${SRC_DIR})
    source_group(${GROUP_NAME} FILES ${${SRC_VAR}})
    # Clean-up.
    unset(GLOB_EXPRESSIONS)
    unset(ARG)
    unset(GROUP_NAME)
endmacro(COLMAP_ADD_SOURCE_DIR)

# Macro to add source files to COLMAP library.
macro(COLMAP_ADD_SOURCES)
    set(SOURCE_FILES "")
    foreach (SOURCE_FILE ${ARGN})
        if (SOURCE_FILE MATCHES "^/.*")
            list(APPEND SOURCE_FILES ${SOURCE_FILE})
        else ()
            list(APPEND SOURCE_FILES
                    "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE}")
        endif ()
    endforeach ()
    # Source directories are nested below src/. PARENT_SCOPE loses entries as
    # soon as a nested directory writes its own inherited list, leaving parts
    # of COLMAP (notably estimators) out of ColmapLib. A global property keeps
    # collection independent of the directory depth.
    set_property(GLOBAL APPEND PROPERTY COLMAP_SOURCES ${SOURCE_FILES})
endmacro(COLMAP_ADD_SOURCES)

# Macro to add CUDA source files to COLMAP library.
macro(COLMAP_ADD_CUDA_SOURCES)
    set(SOURCE_FILES "")
    foreach (SOURCE_FILE ${ARGN})
        if (SOURCE_FILE MATCHES "^/.*")
            # Absolute path.
            list(APPEND SOURCE_FILES ${SOURCE_FILE})
        else ()
            # Relative path.
            list(APPEND SOURCE_FILES
                    "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE}")
        endif ()
    endforeach ()

    set(COLMAP_CUDA_SOURCES
            ${COLMAP_CUDA_SOURCES}
            ${SOURCE_FILES}
            PARENT_SCOPE)
endmacro(COLMAP_ADD_CUDA_SOURCES)

# HIP consumes the same CUDA-spelled MVS sources through cuda_to_hip.h. Keep a
# separate source list so CUDA and HIP libraries can never be linked together.
macro(COLMAP_ADD_HIP_SOURCES)
    set(SOURCE_FILES "")
    foreach (SOURCE_FILE ${ARGN})
        if (SOURCE_FILE MATCHES "^/.*")
            list(APPEND SOURCE_FILES ${SOURCE_FILE})
        else ()
            list(APPEND SOURCE_FILES
                    "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE}")
        endif ()
    endforeach ()
    foreach (SOURCE_FILE ${SOURCE_FILES})
        if (SOURCE_FILE MATCHES "\\.cu$")
            set_source_files_properties(${SOURCE_FILE} PROPERTIES LANGUAGE HIP)
        endif ()
    endforeach ()
    set(COLMAP_HIP_SOURCES
            ${COLMAP_HIP_SOURCES}
            ${SOURCE_FILES}
            PARENT_SCOPE)
endmacro(COLMAP_ADD_HIP_SOURCES)

# Replacement for the normal add_library() command. The syntax remains the same
# in that the first argument is the target name, and the following arguments
# are the source files to use when building the target.
macro(COLMAP_ADD_LIBRARY TARGET_NAME)
    # ${ARGN} will store the list of source files passed to this function.
    add_library(${TARGET_NAME} ${ARGN})
    set_target_properties(${TARGET_NAME} PROPERTIES FOLDER
            ${COLMAP_TARGETS_ROOT_FOLDER}/${FOLDER_NAME})

    if (MSVC)
        # fix compiling error on windows platform
        target_compile_options(${TARGET_NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/sdl->")
    else ()
        target_compile_options(${TARGET_NAME} PRIVATE "-Wno-deprecated-declarations")
    endif ()

    # Enforce 3rd party dependencies
    cloudViewer_show_and_abort_on_warning(${TARGET_NAME})
    cloudViewer_set_global_properties(${TARGET_NAME})
    cloudViewer_set_cloudViewer_lib_properties(${TARGET_NAME})
    COLMAP_LINK_3RDPARTY(${TARGET_NAME})

    # install
    install(TARGETS ${TARGET_NAME} DESTINATION ${CloudViewer_INSTALL_LIB_DIR}/${COLMAP_APP_NAME}/)
endmacro(COLMAP_ADD_LIBRARY)
macro(COLMAP_ADD_STATIC_LIBRARY TARGET_NAME)
    # ${ARGN} will store the list of source files passed to this function.
    add_library(${TARGET_NAME} STATIC ${ARGN})
    set_target_properties(${TARGET_NAME} PROPERTIES FOLDER
            ${COLMAP_TARGETS_ROOT_FOLDER}/${FOLDER_NAME})

    if (MSVC)
        # fix compiling error on windows platform
        target_compile_options(${TARGET_NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/sdl->")
    else ()
        target_compile_options(${TARGET_NAME} PRIVATE "-Wno-deprecated-declarations")
    endif ()

    # Enforce 3rd party dependencies
    cloudViewer_show_and_abort_on_warning(${TARGET_NAME})
    cloudViewer_set_global_properties(${TARGET_NAME})
    cloudViewer_set_cloudViewer_lib_properties(${TARGET_NAME})
    COLMAP_LINK_3RDPARTY(${TARGET_NAME})

    # install
    # install(TARGETS ${TARGET_NAME} DESTINATION ${CloudViewer_INSTALL_LIB_DIR}/${COLMAP_APP_NAME})
endmacro(COLMAP_ADD_STATIC_LIBRARY)

# Replacement for the normal cuda_add_library() command. The syntax remains the
# same in that the first argument is the target name, and the following
# arguments are the source files to use when building the target.
macro(COLMAP_ADD_CUDA_LIBRARY TARGET_NAME)
    # ${ARGN} will store the list of source files passed to this function.
    add_library(${TARGET_NAME} ${ARGN})
    set_target_properties(${TARGET_NAME} PROPERTIES FOLDER
            ${COLMAP_TARGETS_ROOT_FOLDER}/${FOLDER_NAME})

    if (MSVC)
        # fix compiling error on windows platform
        target_compile_options(${TARGET_NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/sdl->")
    else ()
        target_compile_options(${TARGET_NAME} PRIVATE "-Wno-deprecated-declarations")
    endif ()

    # Enforce 3rd party dependencies
    cloudViewer_show_and_abort_on_warning(${TARGET_NAME})
    cloudViewer_set_global_properties(${TARGET_NAME})
    cloudViewer_set_cloudViewer_lib_properties(${TARGET_NAME})
    COLMAP_LINK_3RDPARTY(${TARGET_NAME})
    target_include_directories(${TARGET_NAME} SYSTEM PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

    # install
    install(TARGETS ${TARGET_NAME} DESTINATION ${CloudViewer_INSTALL_LIB_DIR}/${COLMAP_APP_NAME}/)
endmacro(COLMAP_ADD_CUDA_LIBRARY)
macro(COLMAP_ADD_STATIC_CUDA_LIBRARY TARGET_NAME)
    # ${ARGN} will store the list of source files passed to this function.
    add_library(${TARGET_NAME} STATIC ${ARGN})
    target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

    set_target_properties(${TARGET_NAME} PROPERTIES FOLDER
            ${COLMAP_TARGETS_ROOT_FOLDER}/${FOLDER_NAME})

    if (MSVC)
        # fix compiling error on windows platform
        target_compile_options(${TARGET_NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/sdl->")
    else ()
        target_compile_options(${TARGET_NAME} PRIVATE "-Wno-deprecated-declarations")
    endif ()

    # Enforce 3rd party dependencies
    cloudViewer_show_and_abort_on_warning(${TARGET_NAME})
    cloudViewer_set_global_properties(${TARGET_NAME})
    cloudViewer_set_cloudViewer_lib_properties(${TARGET_NAME})
    COLMAP_LINK_3RDPARTY(${TARGET_NAME})
    target_include_directories(${TARGET_NAME} SYSTEM PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

    # install
    # install(TARGETS ${TARGET_NAME} DESTINATION ${CloudViewer_INSTALL_LIB_DIR}/${COLMAP_APP_NAME}/)
endmacro(COLMAP_ADD_STATIC_CUDA_LIBRARY)

macro(COLMAP_ADD_STATIC_HIP_LIBRARY TARGET_NAME)
    add_library(${TARGET_NAME} STATIC ${ARGN})
    set_target_properties(${TARGET_NAME} PROPERTIES FOLDER
            ${COLMAP_TARGETS_ROOT_FOLDER}/${FOLDER_NAME})

    if (MSVC)
        target_compile_options(${TARGET_NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/sdl->")
    else ()
        target_compile_options(${TARGET_NAME} PRIVATE "-Wno-deprecated-declarations")
    endif ()

    cloudViewer_show_and_abort_on_warning(${TARGET_NAME})
    cloudViewer_set_global_properties(${TARGET_NAME})
    cloudViewer_set_cloudViewer_lib_properties(${TARGET_NAME})
    COLMAP_LINK_3RDPARTY(${TARGET_NAME})
    target_link_libraries(${TARGET_NAME} PRIVATE 3rdparty_rocm)
endmacro(COLMAP_ADD_STATIC_HIP_LIBRARY)

# Replacement for the normal add_executable() command. The syntax remains the
# same in that the first argument is the target name, and the following
# arguments are the source files to use when building the target.
macro(COLMAP_ADD_EXECUTABLE TARGET_NAME)
    # ${ARGN} will store the list of source files passed to this function.
    if (MSVC)
        set(rc_list ${COLOUDVIEWER_PROJECT_PATH}/app/reconstruction/media/reconstruction_icon.rc)
        add_executable(${TARGET_NAME} ${ARGN} ${rc_list})
    else ()
        add_executable(${TARGET_NAME} ${ARGN})
    endif ()

    set_target_properties(${TARGET_NAME} PROPERTIES FOLDER ${COLMAP_TARGETS_ROOT_FOLDER}/${FOLDER_NAME})
    target_link_libraries(${TARGET_NAME} PRIVATE ${COLMAP_LIB_NAME})

    # Enforce 3rd party dependencies
    cloudViewer_show_and_abort_on_warning(${TARGET_NAME})
    cloudViewer_set_global_properties(${TARGET_NAME})
    cloudViewer_set_cloudViewer_lib_properties(${TARGET_NAME})
    COLMAP_LINK_3RDPARTY(${TARGET_NAME})

    # fix that You must build your code with position independent code if Qt was built with -reduce-relocations
    cloudViewer_set_targets_independent(${TARGET_NAME})

    # The standalone COLMAP-compatible executable is also the L3 candidate.
    # Its build-tree dependencies live outside the executable directory, so
    # record their directories instead of requiring a caller-specific shell
    # environment. Windows resolves deployed DLLs beside the executable and
    # therefore intentionally does not receive Unix RPATH properties here.
    if (UNIX)
        # A CMake list: CMake joins it with ':' for GNU ld and emits one
        # -rpath per entry for Apple ld64, which has no ':' list syntax.
        set(_COLMAP_EXECUTABLE_BUILD_RPATH
            "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}" "${CMAKE_BINARY_DIR}/external/lib")
        cloudviewer_resolve_qt_lib_dir(_COLMAP_EXECUTABLE_QT_LIB_DIR)
        if (_COLMAP_EXECUTABLE_QT_LIB_DIR)
            list(APPEND _COLMAP_EXECUTABLE_BUILD_RPATH
                 "${_COLMAP_EXECUTABLE_QT_LIB_DIR}")
        endif ()
        set_target_properties(${TARGET_NAME} PROPERTIES
            BUILD_RPATH "${_COLMAP_EXECUTABLE_BUILD_RPATH}")
        if (NOT APPLE)
            # Linux RUNPATH does not apply to indirect dependencies such as
            # OpenImageIO loaded through the static Reconstruction archive.
            target_link_options(${TARGET_NAME} PRIVATE "-Wl,--disable-new-dtags")
        endif ()
    endif ()

    if (MSVC)
        # fix compiling error on windows platform
        target_compile_options(${TARGET_NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/sdl->")
        # fix gflags library bugs on windows
        target_link_libraries(${TARGET_NAME} PRIVATE shlwapi.lib)
    else ()
        target_compile_options(${TARGET_NAME} PRIVATE "-Wno-deprecated-declarations")
    endif ()

    # install
    if (VCPKG_BUILD)
        install(TARGETS ${TARGET_NAME} DESTINATION tools/)
    elseif (NOT APPLE)
        # On macOs, the DeployQt step will install the bundle that contains the executable with
        # library paths properly set, reinstalling the executable here would break the bundle.
        install(TARGETS ${TARGET_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/${CloudViewer_INSTALL_BIN_DIR}/${COLMAP_APP_NAME}/)
    endif ()
endmacro(COLMAP_ADD_EXECUTABLE)

# Wrapper for test executables.
# Optional trailing "LIBS <libs...>" section (upstream parity with the
# upstream COLMAP_ADD_TEST LINK_LIBS argument): test-private link libraries
# are consumed INSIDE the TESTS_ENABLED block, so a target's link lifetime is
# identical to its creation lifetime. The bare post-macro
# target_link_libraries() calls added in 30e9fdd981 broke every
# TESTS_ENABLED=OFF configure (the Windows app matrix passes
# -DBUILD_UNIT_TESTS=OFF; set(TESTS_ENABLED ${BUILD_UNIT_TESTS}) in
# libs/Reconstruction/CMakeLists.txt): "Cannot specify link libraries for
# target ... which is not built by this project".
macro(COLMAP_ADD_TEST TARGET_NAME)
    # ARGN is only a string-substitution pseudo-variable inside a macro, so
    # list() commands cannot operate on it by name; capture it first.
    set(_COLMAP_TEST_ALL_ARGS ${ARGN})
    list(FIND _COLMAP_TEST_ALL_ARGS "LIBS" _COLMAP_TEST_LIBS_IDX)
    if(_COLMAP_TEST_LIBS_IDX EQUAL -1)
        set(_COLMAP_TEST_SRCS ${_COLMAP_TEST_ALL_ARGS})
        set(_COLMAP_TEST_LIBS)
    else()
        list(SUBLIST _COLMAP_TEST_ALL_ARGS 0 ${_COLMAP_TEST_LIBS_IDX} _COLMAP_TEST_SRCS)
        list(SUBLIST _COLMAP_TEST_ALL_ARGS ${_COLMAP_TEST_LIBS_IDX} -1 _COLMAP_TEST_LIBS)
        list(REMOVE_AT _COLMAP_TEST_LIBS 0)
    endif()
    if (TESTS_ENABLED)
        # ${_COLMAP_TEST_SRCS} stores the list of source files passed to this
        # function (everything before the optional LIBS keyword).
        add_executable(${TARGET_NAME} ${_COLMAP_TEST_SRCS})
        set_target_properties(${TARGET_NAME} PROPERTIES
                FOLDER ${COLMAP_TARGETS_ROOT_FOLDER}/${FOLDER_NAME}
                RUNTIME_OUTPUT_DIRECTORY "${COLMAP_TEST_RUNTIME_DIR}")
        # gmock provides the matcher infrastructure used by upstream test
        # files (e.g. util/eigen_matchers.h).
        target_link_libraries(${TARGET_NAME} PRIVATE ${COLMAP_LIB_NAME} gtest_main gmock ${_COLMAP_TEST_LIBS})

        # Batch C test-layout fix: resolve shared-library directories at
        # configure time (Qt, vendored opencv) into the build RPATH so the
        # relocated tests under bin/colmap_test/ stay runnable on all three
        # platforms without relying on the caller's shell environment.
        # A CMake list (see COLMAP_ADD_EXECUTABLE): ':'-joined for GNU ld,
        # one -rpath per entry for Apple ld64.
        set(_COLMAP_TEST_BUILD_RPATH "${CMAKE_BINARY_DIR}/external/lib" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
        # Version-agnostic Qt runtime directory (Qt5/Qt6/unprefixed alias).
        cloudviewer_resolve_qt_lib_dir(_COLMAP_QT_LIB_DIR)
        if (_COLMAP_QT_LIB_DIR)
            list(APPEND _COLMAP_TEST_BUILD_RPATH "${_COLMAP_QT_LIB_DIR}")
        endif ()
        set_target_properties(${TARGET_NAME} PROPERTIES
                BUILD_RPATH "${_COLMAP_TEST_BUILD_RPATH}"
                INSTALL_RPATH "${_COLMAP_TEST_BUILD_RPATH}")
        if (UNIX AND NOT APPLE)
            # Emit DT_RPATH instead of DT_RUNPATH on Linux: RPATH is honored
            # for the whole dependency tree (indirect deps such as opencv
            # pulled in through CVCoreLib), while RUNPATH is not. macOS rpath
            # lists already apply globally per executable, and Apple ld64
            # rejects the GNU-only --disable-new-dtags option.
            target_link_options(${TARGET_NAME} PRIVATE "-Wl,--disable-new-dtags")
        endif ()

        if (MSVC)
            # fix compiling error on windows platform
            target_compile_options(${TARGET_NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/sdl->")
        else ()
            target_compile_options(${TARGET_NAME} PRIVATE "-Wno-deprecated-declarations")
        endif ()

        add_test(NAME "${FOLDER_NAME}/${TARGET_NAME}"
                 COMMAND $<TARGET_FILE:${TARGET_NAME}>)
        # Collect for the unified reconstruction_cpp_tests_gate (CUDA tests
        # use COLMAP_ADD_CUDA_TEST and are intentionally not collected: the
        # gate must stay GPU-independent for GPU-less CI runners).
        set_property(GLOBAL APPEND PROPERTY COLMAP_RECONSTRUCTION_TESTS
                     "${TARGET_NAME}")
        if (MSVC)
            # Tests live under bin/colmap_test/ but the shared DLLs are
            # centralized in bin/<Config>/; Windows has no RPATH, so prepend
            # the DLL directory for every test (requires CMake >= 3.22 for
            # generator expressions in ENVIRONMENT).
            set_tests_properties("${FOLDER_NAME}/${TARGET_NAME}" PROPERTIES
                    ENVIRONMENT "PATH=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>\;$ENV{PATH}")
        else ()
            # Linux/macOS: same reasoning for shared libraries that live in
            # bin/ (e.g. libAICore.so, whose own $ORIGIN RUNPATH deliberately
            # hides build paths, so its vendored-opencv dependency must be
            # resolved from the environment) plus the Qt and vendored-opencv
            # directories resolved above (batch C test-layout fix).
            set_tests_properties("${FOLDER_NAME}/${TARGET_NAME}" PROPERTIES
                    ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}:${_COLMAP_QT_LIB_DIR}:${CMAKE_BINARY_DIR}/external/lib:${COLMAP_TEST_RUNTIME_DIR}:$ENV{LD_LIBRARY_PATH}")
        endif ()
        # Tests are build-tree diagnostics, not release artifacts. Skipping
        # the post-link strip avoids a second writer touching the executable
        # while highly parallel Make builds finalize test targets.
        set_property(TARGET ${TARGET_NAME} PROPERTY
                     CLOUDVIEWER_SKIP_POST_BUILD_STRIP TRUE)
        # Enforce 3rd party dependencies
        cloudViewer_show_and_abort_on_warning(${TARGET_NAME})
        cloudViewer_set_global_properties(${TARGET_NAME})
        cloudViewer_set_cloudViewer_lib_properties(${TARGET_NAME})
        cloudViewer_set_targets_independent(${TARGET_NAME})
        COLMAP_LINK_3RDPARTY(${TARGET_NAME})

        # install
        if (IS_MSVC)
            install(TARGETS ${TARGET_NAME} DESTINATION bin/colmap_test/)
        endif ()
    endif ()
endmacro(COLMAP_ADD_TEST)

# Wrapper for CUDA test executables.
macro(COLMAP_ADD_CUDA_TEST TARGET_NAME)
    if (TESTS_ENABLED)
        # ${ARGN} will store the list of source files passed to this function.
        add_executable(${TARGET_NAME} ${ARGN})
        set_target_properties(${TARGET_NAME} PROPERTIES
                FOLDER ${COLMAP_TARGETS_ROOT_FOLDER}/${FOLDER_NAME}
                RUNTIME_OUTPUT_DIRECTORY "${COLMAP_TEST_RUNTIME_DIR}")
        target_link_libraries(${TARGET_NAME} PRIVATE ${COLMAP_LIB_NAME} gtest_main)

        if (MSVC)
            # fix compiling error on windows platform
            target_compile_options(${TARGET_NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/sdl->")
        else ()
            target_compile_options(${TARGET_NAME} PRIVATE "-Wno-deprecated-declarations")
        endif ()

        add_test(NAME "${FOLDER_NAME}/${TARGET_NAME}"
                 COMMAND $<TARGET_FILE:${TARGET_NAME}>)
        # Needs a CUDA device and driver: the GPU-independent gates select
        # their tests by name, so exclude this one explicitly by label.
        set_tests_properties("${FOLDER_NAME}/${TARGET_NAME}" PROPERTIES
                LABELS gpu)
        if (MSVC)
            # Same DLL-path guarantee as COLMAP_ADD_TEST above.
            set_tests_properties("${FOLDER_NAME}/${TARGET_NAME}" PROPERTIES
                    ENVIRONMENT "PATH=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>\;$ENV{PATH}")
        endif ()

        # Enforce 3rd party dependencies
        cloudViewer_show_and_abort_on_warning(${TARGET_NAME})
        cloudViewer_set_global_properties(${TARGET_NAME})
        cloudViewer_set_cloudViewer_lib_properties(${TARGET_NAME})
        COLMAP_LINK_3RDPARTY(${TARGET_NAME})
        target_include_directories(${TARGET_NAME} SYSTEM PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

        # install
        if (IS_MSVC)
            install(TARGETS ${TARGET_NAME} DESTINATION bin/colmap_test/)
        endif ()
    endif ()
endmacro(COLMAP_ADD_CUDA_TEST)

macro(COLMAP_ADD_HIP_TEST TARGET_NAME)
    if (TESTS_ENABLED)
        set(HIP_TEST_SOURCES "")
        foreach (SOURCE_FILE ${ARGN})
            if (SOURCE_FILE MATCHES "^/.*")
                list(APPEND HIP_TEST_SOURCES ${SOURCE_FILE})
            else ()
                list(APPEND HIP_TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE}")
            endif ()
        endforeach ()
        set_source_files_properties(${HIP_TEST_SOURCES} PROPERTIES LANGUAGE HIP)
        add_executable(${TARGET_NAME} ${HIP_TEST_SOURCES})
        set_target_properties(${TARGET_NAME} PROPERTIES
                FOLDER ${COLMAP_TARGETS_ROOT_FOLDER}/${FOLDER_NAME}
                RUNTIME_OUTPUT_DIRECTORY "${COLMAP_TEST_RUNTIME_DIR}")
        target_link_libraries(${TARGET_NAME} PRIVATE ${COLMAP_LIB_NAME}
                              gtest_main 3rdparty_rocm)

        if (MSVC)
            target_compile_options(${TARGET_NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/sdl->")
        else ()
            target_compile_options(${TARGET_NAME} PRIVATE "-Wno-deprecated-declarations")
        endif ()

        add_test(NAME "${FOLDER_NAME}/${TARGET_NAME}" COMMAND $<TARGET_FILE:${TARGET_NAME}>)
        # Needs a ROCm device; see COLMAP_ADD_CUDA_TEST.
        set_tests_properties("${FOLDER_NAME}/${TARGET_NAME}" PROPERTIES
                LABELS gpu)
        if (MSVC)
            # Same DLL-path guarantee as COLMAP_ADD_TEST above.
            set_tests_properties("${FOLDER_NAME}/${TARGET_NAME}" PROPERTIES
                    ENVIRONMENT "PATH=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>\;$ENV{PATH}")
        endif ()
        cloudViewer_show_and_abort_on_warning(${TARGET_NAME})
        cloudViewer_set_global_properties(${TARGET_NAME})
        cloudViewer_set_cloudViewer_lib_properties(${TARGET_NAME})
        cloudViewer_set_targets_independent(${TARGET_NAME})
        COLMAP_LINK_3RDPARTY(${TARGET_NAME})
    endif ()
endmacro(COLMAP_ADD_HIP_TEST)
