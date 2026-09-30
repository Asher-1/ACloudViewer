# Centralized build output layout for CHANGE_TARGET_GENERATION_PATH_FOR_DEBUGGING.
#
# Design:
#   - Globals are switched only through the cloudviewer_apply_dev_* macros below.
#   - Deliverables with special destinations use per-target helpers (plugins, examples,
#     AICore tests, plugin CLIs).
#   - MSVC uses bin/<Config>/…; Unix/macOS use flat bin/ (except plugin .so → bin/plugins).

macro(cloudviewer_apply_dev_runtime_bin_layout)
    if(CHANGE_TARGET_GENERATION_PATH_FOR_DEBUGGING)
        if(MSVC)
            set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin/Debug")
            set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin/Release")
            set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/bin/RelWithDebInfo")
        else()
            set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
            set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
        endif()
    endif()
endmacro()

# cloudViewer static/object libs — keep under lib/$<CONFIG> during dev builds.
macro(cloudviewer_apply_dev_internal_lib_layout)
    if(CHANGE_TARGET_GENERATION_PATH_FOR_DEBUGGING)
        if(MSVC)
            unset(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG)
            unset(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE)
            unset(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO)
        else()
            set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/$<CONFIG>")
        endif()
    endif()
endmacro()

# CVViewer / CV_db shared libs — runtime in bin/, libraries still under lib/$<CONFIG> on Unix.
macro(cloudviewer_apply_dev_cvviewer_lib_layout)
    if(CHANGE_TARGET_GENERATION_PATH_FOR_DEBUGGING)
        if(MSVC)
            set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin/Debug")
            set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin/Release")
            set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/bin/RelWithDebInfo")
        else()
            set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/$<CONFIG>")
        endif()
    endif()
endmacro()

# Default CMAKE_LIBRARY_OUTPUT_DIRECTORY for targets created while plugins/ is processed.
# Plugin modules themselves override via cloudviewer_set_plugin_module_output_directory().
macro(cloudviewer_apply_dev_plugin_lib_layout)
    if(CHANGE_TARGET_GENERATION_PATH_FOR_DEBUGGING AND NOT MSVC)
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/plugins")
    endif()
endmacro()

# Python extension modules (pybind / tf_ops / torch_ops) are packaging artifacts,
# not GUI runtime deliverables. Keep them under lib/<Config>/Python/{cpu|cuda}
# regardless of CHANGE_TARGET_GENERATION_PATH_FOR_DEBUGGING.
function(cloudviewer_get_python_compiled_module_dir out_var)
    set(${out_var}
        "${CMAKE_BINARY_DIR}/lib/$<CONFIG>/Python/$<IF:$<BOOL:${BUILD_CUDA_MODULE}>,cuda,cpu>"
        PARENT_SCOPE)
endfunction()

function(cloudviewer_set_python_extension_output_directory target)
    cloudviewer_get_python_compiled_module_dir(_py_mod_dir)
    set_target_properties(${target} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${_py_mod_dir}"
        ARCHIVE_OUTPUT_DIRECTORY "${_py_mod_dir}"
    )
endfunction()

# ML custom-op libraries (tf_ops / torch_ops) follow Open3D layout:
#   Unix .so/.dylib → lib/<Config>/{cpu|cuda}
#   Windows .dll    → lib/<Config>/Python/{cpu|cuda}  (RUNTIME artifact)
# python-package copies base_dir/{cpu,cuda} into cloudViewer/{cpu,cuda}/.
function(cloudviewer_set_ml_ops_output_directory target)
    set(_ops_base "${CMAKE_BINARY_DIR}/lib/$<CONFIG>")
    set(_ops_arch_dir "${_ops_base}/$<IF:$<BOOL:${BUILD_CUDA_MODULE}>,cuda,cpu>")
    set(_ops_runtime_dir "${_ops_base}/Python/$<IF:$<BOOL:${BUILD_CUDA_MODULE}>,cuda,cpu>")
    set_target_properties(${target} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${_ops_arch_dir}"
        ARCHIVE_OUTPUT_DIRECTORY "${_ops_arch_dir}"
        RUNTIME_OUTPUT_DIRECTORY "${_ops_runtime_dir}"
    )
endfunction()

function(cloudviewer_set_plugin_module_output_directory target)
    if(MSVC)
        set(_plugin_bin "${CMAKE_BINARY_DIR}/bin")
        set_target_properties(${target} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY_DEBUG "${_plugin_bin}/Debug/plugins"
            RUNTIME_OUTPUT_DIRECTORY_RELEASE "${_plugin_bin}/Release/plugins"
            RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${_plugin_bin}/RelWithDebInfo/plugins"
        )
    else()
        set_target_properties(${target} PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/plugins"
        )
    endif()
endfunction()

# Plugin / plugin-API unit tests are grouped under bin/plugin_test/
# (mirroring bin/colmap_test/ and bin/aicore_tests/) so the main bin/ only
# holds the app, plugins, and shared libraries. ctest resolves each binary
# through the target name / $<TARGET_FILE>, and build-tree RPATHs are
# absolute, so the subfolder does not affect test discovery or loading.
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


# Batch C test-layout companion: ctest environment for tests relocated under
# bin/plugin_test/ (PATH for MSVC DLLs; LD_LIBRARY_PATH for Linux/macOS with
# Qt and the vendored opencv resolved from their configure-time locations).
function(cloudviewer_set_plugin_test_env test_name)
    set(_qt_lib_dir "")
    if (TARGET Qt5::Core)
        get_target_property(_qt_core_loc Qt5::Core LOCATION)
        get_filename_component(_qt_lib_dir "${_qt_core_loc}" DIRECTORY)
    endif ()
    if (MSVC)
        set_tests_properties(${test_name} PROPERTIES
                ENVIRONMENT "PATH=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>\;$ENV{PATH}")
    else ()
        set_tests_properties(${test_name} PROPERTIES
                ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}:${_qt_lib_dir}:${CMAKE_BINARY_DIR}/external/lib:$ENV{LD_LIBRARY_PATH}")
    endif ()
endfunction()

function(cloudviewer_set_examples_runtime_output_directory target)
    # Per-target override matches legacy EXAMPLE_BIN_DIR = bin/examples on all platforms.
    set_target_properties(${target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/examples"
    )
endfunction()

# AICore unit tests share ggml backends already copied to bin/ by the AICore target.
function(cloudviewer_set_aicore_test_runtime_layout target)
    if(MSVC)
        set(_bin_root "${CMAKE_BINARY_DIR}/bin")
        # Legacy layout used a single RUNTIME_OUTPUT_DIRECTORY under aicore_tests/.
        set_target_properties(${target} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${_bin_root}/aicore_tests"
        )
        # Manual benchmarks (bench_rfdetr_perf, bench_sam3_backend_acceptance,
        # cmp_sam3_vit_stages) are plain executables without add_test();
        # set_tests_properties on a name that is not a registered test is a
        # fatal configure error, so guard the ctest-only ENVIRONMENT_<CONFIG>
        # properties with if(TEST).
        if(TEST ${target})
            foreach(_cfg IN ITEMS Debug Release RelWithDebInfo)
                string(TOUPPER "${_cfg}" _cfg_upper)
                set_tests_properties(${target} PROPERTIES
                    "ENVIRONMENT_${_cfg_upper}"
                        "PATH=${_bin_root}/${_cfg};$ENV{PATH}"
                )
            endforeach()
        endif()
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
