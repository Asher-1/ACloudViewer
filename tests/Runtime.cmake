include_guard(GLOBAL)

# Runtime paths and target helpers are owned by the root C++ test module.
function(cloudviewer_set_bin_runtime_output_directory target)
    if(MSVC)
        set(_bin_root "${CMAKE_BINARY_DIR}/bin/plugin_test")
        set_target_properties(${target} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY_DEBUG "${_bin_root}/Debug"
            RUNTIME_OUTPUT_DIRECTORY_RELEASE "${_bin_root}/Release"
            RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${_bin_root}/RelWithDebInfo"
        )
    else()
        set_target_properties(${target} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/plugin_test"
        )
    endif()
endfunction()

# Standard boilerplate for a plugin unit-test executable, shared by the
# plugins' tests/ subdirectories: PIC objects everywhere, GNU ld -no-pie on
# Linux (Apple ld64 warns "argument unused" for it), and the shared
# bin/plugin_test/ runtime layout. Qt AUTOMOC/AUTORCC come from the global
# directory settings in CMakeExternalLibs.cmake. Register the ctest row at
# the call site (COMMAND may need wrappers such as xvfb-run or extra
# arguments) and pair with cloudviewer_set_plugin_test_env[_offscreen] for
# the runtime environment.
function(cloudviewer_add_plugin_test target)
    cmake_parse_arguments(PARG "" "" "SOURCES;LIBS;INCLUDES;DEFINES" ${ARGN})
    add_executable(${target} ${PARG_SOURCES})
    if(PARG_INCLUDES)
        target_include_directories(${target} PRIVATE ${PARG_INCLUDES})
    endif()
    if(PARG_LIBS)
        target_link_libraries(${target} PRIVATE ${PARG_LIBS})
    endif()
    if(PARG_DEFINES)
        target_compile_definitions(${target} PRIVATE ${PARG_DEFINES})
    endif()
    if(NOT MSVC)
        set_target_properties(${target} PROPERTIES
            POSITION_INDEPENDENT_CODE ON)
        target_compile_options(${target} PRIVATE -fPIC)
        if(UNIX AND NOT APPLE)
            target_link_options(${target} PRIVATE "-no-pie")
        endif()
    endif()
    cloudviewer_set_bin_runtime_output_directory(${target})
endfunction()


# CTest runtime environment for tests relocated under bin/plugin_test/ or one
# of the other shared test directories.  The helper below prepends build-tree
# loader paths without serialising the host PATH as a CMake list.
function(cloudviewer_resolve_python_lib_dir out_var)
    # pybind11::embed consumers resolve libpython at runtime from
    # LD_LIBRARY_PATH, not from the link line: DT_NEEDED carries no path, so
    # a build configured against a non-system interpreter (pyenv shim,
    # conda) can be silently served the DISTRO libpython of the same ABI
    # version by ldconfig at test time. The two builds share the ABI tag
    # (e.g. two 3.12's) but not their site-packages, so the embedded
    # interpreter reports "No module named 'numpy'" for packages that are
    # installed in the build interpreter. Pin the interpreter the build
    # actually linked by putting its library directory first.
    # FindPython publishes the per-config library under internal names
    # (_Python3_LIBRARY_RELEASE / _DEBUG) and only fills Python3_LIBRARY
    # itself for single-config lookups; accept either spelling.
    # Cache variables must be read with get_property to be visible inside
    # functions.
    set(_cv_python_lib "")
    foreach (_cv_var_name
             Python3_LIBRARY
             _Python3_LIBRARY_RELEASE
             _Python3_LIBRARY_DEBUG
             _Python3_LIBRARY)
        get_property(_cv_candidate CACHE ${_cv_var_name} PROPERTY VALUE)
        if (_cv_candidate)
            set(_cv_python_lib "${_cv_candidate}")
            break()
        endif ()
    endforeach ()
    if (_cv_python_lib)
        get_filename_component(_cv_python_lib_dir "${_cv_python_lib}" DIRECTORY)
        set(${out_var} "${_cv_python_lib_dir}" PARENT_SCOPE)
    else ()
        set(${out_var} "" PARENT_SCOPE)
    endif ()
endfunction()

function(cloudviewer_set_plugin_test_env_offscreen test_name)
    cloudviewer_set_test_runtime_environment(${test_name})
    # ENVIRONMENT is intentionally kept for scalar values.  The runtime path
    # uses ENVIRONMENT_MODIFICATION so CTest preserves the runner's complete
    # PATH/LD_LIBRARY_PATH and applies the host-native path-list separator.
    set_property(TEST ${test_name} APPEND PROPERTY ENVIRONMENT
        "QT_QPA_PLATFORM=offscreen")
endfunction()

function(cloudviewer_set_plugin_test_env test_name)
    cloudviewer_set_test_runtime_environment(${test_name})
endfunction()

# Add the build-tree runtime directories to a registered CTest row without
# serialising the configure-time PATH.  ENVIRONMENT values are CMake lists,
# so embedding a Windows PATH there loses every entry after the first `;`.
# ENVIRONMENT_MODIFICATION (CMake >= 3.22, project minimum 3.24) performs the
# operation at test time with the host-native separator and leaves the runner
# environment intact for every other test. Additional arguments are runtime
# directories owned by the caller (for example COLMAP's existing test path).
function(cloudviewer_set_test_runtime_environment test_name)
    if (NOT TEST ${test_name})
        return()
    endif()

    cloudviewer_resolve_qt_lib_dir(_qt_lib_dir)
    cloudviewer_resolve_python_lib_dir(_cv_python_lib_dir)
    if (MSVC)
        set(_cv_loader_var PATH)
    elseif (APPLE)
        # Preserve absolute install names and LC_RPATH resolution. A global
        # DYLD_LIBRARY_PATH also replaces system dependencies (including libc++)
        # with same-named Conda libraries, even when they were not linked here.
        set(_cv_loader_var DYLD_FALLBACK_LIBRARY_PATH)
    else()
        set(_cv_loader_var LD_LIBRARY_PATH)
    endif()
    set(_cv_runtime_modifications "")
    if(APPLE AND IS_DIRECTORY "$ENV{TMPDIR}")
        # macOS exposes /var through /private/var, while getcwd is physical.
        file(REAL_PATH "$ENV{TMPDIR}" _cv_tmp_dir)
        list(APPEND _cv_runtime_modifications "TMPDIR=set:${_cv_tmp_dir}")
    endif()
    foreach (_cv_runtime_directory IN LISTS ARGN)
        list(APPEND _cv_runtime_modifications
            "${_cv_loader_var}=path_list_prepend:${_cv_runtime_directory}")
    endforeach()
    if (NOT MSVC)
        list(APPEND _cv_runtime_modifications
            "${_cv_loader_var}=path_list_prepend:${CMAKE_BINARY_DIR}/external/lib")
    endif()
    if (NOT MSVC AND _qt_lib_dir)
        list(APPEND _cv_runtime_modifications
            "${_cv_loader_var}=path_list_prepend:${_qt_lib_dir}")
    endif()
    if (MSVC)
        list(APPEND _cv_runtime_modifications
            "PATH=path_list_prepend:${CMAKE_BINARY_DIR}/bin/$<CONFIG>")
    else()
        list(APPEND _cv_runtime_modifications
            "${_cv_loader_var}=path_list_prepend:${CMAKE_BINARY_DIR}/bin")
    endif()
    # Prepend Python last so it wins over same-SONAME distro libraries in
    # the Qt directory (often /usr/lib/<arch>). CTest applies these in order.
    if (NOT MSVC AND _cv_python_lib_dir)
        list(APPEND _cv_runtime_modifications
            "${_cv_loader_var}=path_list_prepend:${_cv_python_lib_dir}")
    endif()
    get_property(_cv_existing_modifications TEST ${test_name}
        PROPERTY ENVIRONMENT_MODIFICATION)
    foreach (_cv_modification IN LISTS _cv_runtime_modifications)
        if (NOT _cv_modification IN_LIST _cv_existing_modifications)
            set_property(TEST ${test_name} APPEND PROPERTY
                ENVIRONMENT_MODIFICATION "${_cv_modification}")
        endif()
    endforeach()
endfunction()

# AICore unit tests share ggml backends already copied to bin/ by the AICore target.
function(cloudviewer_set_aicore_test_runtime_layout target)
    if(MSVC)
        set(_bin_root "${CMAKE_BINARY_DIR}/bin")
        set_target_properties(${target} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${_bin_root}/aicore_tests"
        )
    else()
        set_target_properties(${target} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/aicore_tests"
        )
        if(APPLE)
            set(_rpath "@loader_path/..")
            # Test binaries inherit Qt through AICore's PUBLIC Qt link. Qt
            # dylibs use @rpath install names, so dyld also needs Qt's lib
            # directory on the search path (BUILD_WITH_INSTALL_RPATH skips
            # CMake's automatic rpath computation). $<TARGET_FILE_DIR:...>
            # resolves Qt's IMPORTED_LOCATION(_<CONFIG>) whichever form the
            # Qt package ships (plain LOCATION is NOTFOUND for config-only
            # imports, e.g. conda-forge Qt5).  The versioned targets
            # (Qt5::Gui/Qt6::Gui) are the real shared libraries and must be
            # checked first: conda-forge Qt5 defines Qt::Gui as an INTERFACE
            # aggregate target with no file location, which TARGET_FILE_DIR
            # rejects.  The unversioned Qt::Gui fallback is only usable when
            # it resolves to a real library via ALIASED_TARGET.
            if(TARGET Qt5::Gui)
                list(APPEND _rpath "$<TARGET_FILE_DIR:Qt5::Gui>")
            elseif(TARGET Qt6::Gui)
                list(APPEND _rpath "$<TARGET_FILE_DIR:Qt6::Gui>")
            elseif(TARGET Qt::Gui)
                get_target_property(_qt_gui_alias Qt::Gui ALIASED_TARGET)
                if(_qt_gui_alias)
                    list(APPEND _rpath "$<TARGET_FILE_DIR:${_qt_gui_alias}>")
                endif()
            endif()
        else()
            set(_rpath "\$ORIGIN/..")
            # Build-tree-only addition (batch C test-layout fix): the vendored
            # opencv (libopencv_world.so.<abi>) lives under external/lib and is
            # a transitive dependency of libAICore; without it, link-time and
            # run-time resolution fail once tests moved out of bin/.
            list(APPEND _rpath "${CMAKE_BINARY_DIR}/external/lib")
            set_target_properties(${target} PROPERTIES
                BUILD_RPATH "${_rpath}"
                INSTALL_RPATH "\$ORIGIN/.."
            )
            # DT_RPATH (not RUNPATH) so the executable's search list also
            # covers indirect dependencies such as libAICore's opencv.
            if (NOT MSVC)
                target_link_options(${target} PRIVATE "-Wl,--disable-new-dtags")
            endif ()
            return()
        endif()
        set_target_properties(${target} PROPERTIES
            BUILD_RPATH "${_rpath}"
            INSTALL_RPATH "${_rpath}"
            BUILD_WITH_INSTALL_RPATH TRUE
        )
    endif()

    if(GGML_DYNAMIC_BACKENDS AND TARGET AICore)
        add_dependencies(${target} AICore)
    endif()
endfunction()
