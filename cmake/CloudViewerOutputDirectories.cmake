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
# Resolve the directory holding the Qt runtime libraries for whichever Qt
# major version this build links (Qt5 / Qt6 / unprefixed alias). Batch C
# test-layout consumers (RPATH / ctest ENVIRONMENT) must not hardcode a
# version: Qt6 exports Qt6::<mod> plus a Qt::<mod> alias, while Qt5 only
# exports Qt5::<mod>; conda-forge Qt5 additionally defines Qt::<mod> as an
# INTERFACE aggregate without LOCATION, so the alias is only usable through
# ALIASED_TARGET (same caveat as the AICore layout's Qt5/6::Gui chain).
function(cloudviewer_resolve_qt_lib_dir out_var)
    set(_qt_core_target "")
    if (TARGET Qt5::Core)
        set(_qt_core_target Qt5::Core)
    elseif (TARGET Qt6::Core)
        set(_qt_core_target Qt6::Core)
    elseif (TARGET Qt::Core)
        get_target_property(_qt_core_alias Qt::Core ALIASED_TARGET)
        if (_qt_core_alias)
            set(_qt_core_target ${_qt_core_alias})
        endif ()
    endif ()
    set(${out_var} "")
    if (_qt_core_target)
        get_target_property(_qt_core_loc ${_qt_core_target} LOCATION)
        if (_qt_core_loc)
            get_filename_component(_qt_dir "${_qt_core_loc}" DIRECTORY)
            set(${out_var} "${_qt_dir}")
        endif ()
    endif ()
    set(${out_var} "${${out_var}}" PARENT_SCOPE)
endfunction()

function(cloudviewer_set_examples_runtime_output_directory target)
    # Per-target override matches legacy EXAMPLE_BIN_DIR = bin/examples on all platforms.
    # MSVC needs two extra steps to make that override actually stick and to keep
    # the example links out of the archive directory:
    # 1. The dev-layout macros export CMAKE_RUNTIME_OUTPUT_DIRECTORY_<CONFIG>
    #    (e.g. bin/Release); CMake snapshots those into per-target
    #    RUNTIME_OUTPUT_DIRECTORY_<CONFIG> properties at add_executable() time,
    #    and a per-config property beats the generic one set here — the example
    #    would land in bin/Release/ instead of bin/examples/. Re-derive the
    #    override with $<CONFIG> so one property covers every configuration.
    # 2. On MSVC an executable link also writes an import library into
    #    ARCHIVE_OUTPUT_DIRECTORY (lib/<CONFIG> in CI), where every static
    #    archive lives. Under Ninja Multi-Config the last wave of example
    #    links runs concurrently and a real-time AV scan holding the freshly
    #    written files made one link fail with LNK1114 ERROR_SHARING_VIOLATION
    #    (Flann.exe -> lib\Release\Flann.lib, run 39273926318). Keeping the
    #    import libs beside the example binaries isolates them from the
    #    archive directory entirely.
    if (MSVC)
        set_target_properties(${target} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/examples/$<CONFIG>"
            ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/examples/$<CONFIG>"
            RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin/examples/Debug"
            RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin/examples/Release"
            RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/bin/examples/RelWithDebInfo"
            RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_BINARY_DIR}/bin/examples/MinSizeRel"
        )
    else ()
        set_target_properties(${target} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/examples"
        )
    endif ()
endfunction()

# Test-specific layout and loader contracts have one owner in tests/.
include("${CMAKE_CURRENT_LIST_DIR}/../tests/Runtime.cmake")
