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

function(cloudviewer_set_bin_runtime_output_directory target)
    if(MSVC)
        set(_bin_root "${CMAKE_BINARY_DIR}/bin")
        set_target_properties(${target} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY_DEBUG "${_bin_root}/Debug"
            RUNTIME_OUTPUT_DIRECTORY_RELEASE "${_bin_root}/Release"
            RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${_bin_root}/RelWithDebInfo"
        )
    else()
        set_target_properties(${target} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        )
    endif()
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
        foreach(_cfg IN ITEMS Debug Release RelWithDebInfo)
            string(TOUPPER "${_cfg}" _cfg_upper)
            set_tests_properties(${target} PROPERTIES
                "ENVIRONMENT_${_cfg_upper}"
                    "PATH=${_bin_root}/${_cfg};$ENV{PATH}"
            )
        endforeach()
    else()
        set_target_properties(${target} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/aicore_tests"
        )
        if(APPLE)
            set(_rpath "@loader_path/..")
        else()
            set(_rpath "\$ORIGIN/..")
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
