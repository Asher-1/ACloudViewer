set(MIN_NODE_VERSION "14.00.0")

# Clean up directory
file(REMOVE_RECURSE ${PYTHON_PACKAGE_DST_DIR})
file(MAKE_DIRECTORY ${PYTHON_PACKAGE_DST_DIR}/cloudViewer)
file(MAKE_DIRECTORY ${PYTHON_PACKAGE_DST_DIR}/cloudViewer/lib)
set(PYTHON_INSTALL_LIB_DESTINATION "${PYTHON_PACKAGE_DST_DIR}/cloudViewer/lib")
set(CUSTOM_SO_NAME ".so.${PROJECT_VERSION_THREE_NUMBER}")

# Create python package. It contains:
# 1) Pure-python code and misc files, copied from ${PYTHON_PACKAGE_SRC_DIR}
# 2) The compiled python-C++ module, i.e. cloudViewer.so (or the equivalents)
#    Optionally other modules e.g. cloudViewer_tf_ops.so may be included.
# 3) Configured files and supporting files

# 1) Pure-python code and misc files, copied from ${PYTHON_PACKAGE_SRC_DIR}
file(COPY ${PYTHON_PACKAGE_SRC_DIR}/
        DESTINATION ${PYTHON_PACKAGE_DST_DIR}
        )

# 2) The compiled python-C++ module, i.e. cloudViewer.so (or the equivalents)
#    Optionally other modules e.g. cloudViewer_tf_ops.so may be included.
# pybind and libCloudViewer are copied flat into cloudViewer/. The ML ops
# (cloudViewer_{torch,tf}_ops) are built into base_dir/{cpu,cuda} subdirs; copy
# every arch subdir that exists so a CUDA wheel bundles both CPU- and CUDA-linked
# ops (cloudViewer/cpu and cloudViewer/cuda), selected by device at runtime.
set(_copied_pybind FALSE)
foreach(COMPILED_MODULE_PATH ${COMPILED_MODULE_PATH_LIST})
    get_filename_component(COMPILED_MODULE_NAME ${COMPILED_MODULE_PATH} NAME)
    get_filename_component(COMPILED_MODULE_DIR ${COMPILED_MODULE_PATH} DIRECTORY)
    get_filename_component(COMPILED_MODULE_PARENT ${COMPILED_MODULE_DIR} NAME)
    if(COMPILED_MODULE_PARENT STREQUAL "cpu" OR COMPILED_MODULE_PARENT STREQUAL "cuda")
        get_filename_component(COMPILED_MODULE_BASE_DIR ${COMPILED_MODULE_DIR} DIRECTORY)
        foreach(ARCH cpu cuda)
            if(EXISTS "${COMPILED_MODULE_BASE_DIR}/${ARCH}/${COMPILED_MODULE_NAME}")
                file(COPY "${COMPILED_MODULE_BASE_DIR}/${ARCH}/${COMPILED_MODULE_NAME}"
                     DESTINATION "${PYTHON_PACKAGE_DST_DIR}/cloudViewer/${ARCH}/"
                     FOLLOW_SYMLINK_CHAIN)
                set(_copied_pybind TRUE)
            endif()
        endforeach()
    elseif(EXISTS "${COMPILED_MODULE_PATH}")
        file(COPY ${COMPILED_MODULE_PATH}
             DESTINATION ${PYTHON_PACKAGE_DST_DIR}/cloudViewer/
             FOLLOW_SYMLINK_CHAIN)
        set(_copied_pybind TRUE)
    endif()
endforeach()
if(NOT _copied_pybind)
    message(FATAL_ERROR
        "python-package: pybind module was not copied. "
        "COMPILED_MODULE_PATH_LIST=${COMPILED_MODULE_PATH_LIST}; "
        "PYTHON_COMPILED_MODULE_DIR=${PYTHON_COMPILED_MODULE_DIR}")
endif()

# cloudViewer/__init__.py resolves the device API at import time and falls
# back to cloudViewer.cpu.pybind whenever no CUDA device is available (GPU-less
# machines, CI runners). A CUDA wheel without the CPU-linked bindings therefore
# cannot import at all on such hosts; fail the packaging here instead of
# shipping a wheel that breaks at import. The CPU variant is normally left
# behind by the CPU pass (BUILD_CUDA_MODULE=OFF) of the two-pass wheel build
# (see build_pip_package in util/ci_utils.sh).
file(GLOB _cpu_pybind_modules "${PYTHON_PACKAGE_DST_DIR}/cloudViewer/cpu/pybind*")
file(GLOB _cuda_pybind_modules "${PYTHON_PACKAGE_DST_DIR}/cloudViewer/cuda/pybind*")
if(BUILD_CUDA_MODULE AND NOT _cuda_pybind_modules)
    message(FATAL_ERROR
        "python-package: cloudViewer/cuda/pybind module missing from the "
        "staging tree. BUILD_CUDA_MODULE=ON builds the pybind module into "
        "${PYTHON_COMPILED_MODULE_DIR}; check that the CUDA build succeeded.")
endif()
if(NOT _cpu_pybind_modules)
    if(BUILD_CUDA_MODULE)
        set(_pybind_guard_hint
            "The CPU variant is produced by the CPU pass (BUILD_CUDA_MODULE=OFF) "
            "of the two-pass wheel build (see build_pip_package in util/ci_utils.sh); "
            "make sure that pass completed successfully.")
    else()
        set(_pybind_guard_hint
            "Check that the pybind target built into "
            "${PYTHON_COMPILED_MODULE_DIR}.")
    endif()
    message(FATAL_ERROR
        "python-package: cloudViewer/cpu/pybind module missing from the "
        "staging tree. cloudViewer/__init__.py requires it as the runtime "
        "fallback when no CUDA device is available. ${_pybind_guard_hint}")
endif()
unset(_pybind_guard_hint)

# Mark cloudViewer.cpu / cloudViewer.cuda as regular Python packages when the
# staging copy created them. Without __init__.py, `find_packages()` in
# setup.py prunes these dirs and setuptools reports "Package would be ignored";
# the compiled modules then only ship via the MANIFEST.in graft, which is
# ambiguous and fragile across setuptools versions.
foreach(_arch cpu cuda)
    if(IS_DIRECTORY "${PYTHON_PACKAGE_DST_DIR}/cloudViewer/${_arch}")
        file(WRITE
             "${PYTHON_PACKAGE_DST_DIR}/cloudViewer/${_arch}/__init__.py"
             "# Created by make_python_package.cmake so that setuptools ships the "
             "compiled ${_arch} modules in this subpackage.\n")
    endif()
endforeach()
# Include additional libraries that may be absent from the user system
# eg: libc++.so and libc++abi.so (needed by filament)
# The linker recognizes only library.so.MAJOR, so remove .MINOR from the filename
foreach (PYTHON_EXTRA_LIB ${PYTHON_EXTRA_LIBRARIES})
    get_filename_component(PYTHON_EXTRA_LIB_REAL ${PYTHON_EXTRA_LIB} REALPATH)
    get_filename_component(SO_VER_NAME ${PYTHON_EXTRA_LIB_REAL} NAME)
    if (APPLE)
        string(REGEX REPLACE "\\.([0-9]+)\\..*.dylib" ".\\1.dylib" SO_1_NAME ${SO_VER_NAME})
    elseif (UNIX)
        string(REGEX REPLACE "\\.so\\.([0-9]+)\\..*" ".so.\\1" SO_1_NAME ${SO_VER_NAME})
    endif()
    configure_file(${PYTHON_EXTRA_LIB_REAL} ${PYTHON_PACKAGE_DST_DIR}/cloudViewer/${SO_1_NAME} COPYONLY)
endforeach ()

# 3) Configured files and supporting files
configure_file("${PYTHON_PACKAGE_SRC_DIR}/setup.py"
               "${PYTHON_PACKAGE_DST_DIR}/setup.py")
configure_file("${PYTHON_PACKAGE_SRC_DIR}/cloudViewer/__init__.py"
               "${PYTHON_PACKAGE_DST_DIR}/cloudViewer/__init__.py")
configure_file("${PYTHON_PACKAGE_SRC_DIR}/tools/cli.py"
               "${PYTHON_PACKAGE_DST_DIR}/cloudViewer/tools/cli.py")
configure_file("${PYTHON_PACKAGE_SRC_DIR}/tools/app.py"
               "${PYTHON_PACKAGE_DST_DIR}/cloudViewer/app.py")
configure_file("${PYTHON_PACKAGE_SRC_DIR}/cloudViewer/web_visualizer.py"
               "${PYTHON_PACKAGE_DST_DIR}/cloudViewer/web_visualizer.py")
configure_file("${PYTHON_PACKAGE_SRC_DIR}/js/lib/web_visualizer.js"
               "${PYTHON_PACKAGE_DST_DIR}/js/lib/web_visualizer.js")
configure_file("${PYTHON_PACKAGE_SRC_DIR}/conda_meta/conda_build_config.yaml"
               "${PYTHON_PACKAGE_DST_DIR}/conda_meta/conda_build_config.yaml")
configure_file("${PYTHON_PACKAGE_SRC_DIR}/conda_meta/meta.yaml"
               "${PYTHON_PACKAGE_DST_DIR}/conda_meta/meta.yaml")
configure_file("${PYTHON_PACKAGE_SRC_DIR}/js/package.json"
               "${PYTHON_PACKAGE_DST_DIR}/js/package.json")
configure_file("${PYTHON_PACKAGE_SRC_DIR}/../libs/cloudViewer/visualization/webrtc_server/html/webrtcstreamer.js"
               "${PYTHON_PACKAGE_DST_DIR}/js/lib/webrtcstreamer.js")
file(COPY "${PYTHON_COMPILED_MODULE_DIR}/_build_config.py"
     DESTINATION "${PYTHON_PACKAGE_DST_DIR}/cloudViewer/")


# 4) Add extra dependencies.
message(STATUS "QT_PLUGINS_PATH_LIST: " ${QT_PLUGINS_PATH_LIST})
foreach( qt_plugins_folder ${QT_PLUGINS_PATH_LIST} )
    file(COPY "${qt_plugins_folder}"
         DESTINATION "${PYTHON_INSTALL_LIB_DESTINATION}/")
endforeach()

# Copy ggml dynamic backend modules into the wheel's lib directory BEFORE
# any platform-specific packaging runs. These are loaded at runtime by
# ggml_backend_load_all() and allow GPU backends to be optional (graceful
# fallback to CPU if drivers are missing).
#
# ggml dynamic-backend runtime layout (internal paths from ggml.cmake — see
# cmake/AICoreRuntimeLayout.cmake). Required to bundle libggml*.so + ggml-cpu*.so
# next to libAICore in the Python wheel; ordinary libs do not split core vs module.
if(AICore_ENABLED AND GGML_DYNAMIC_BACKENDS)
    if(NOT AICORE_COPY_GGML_SCRIPT)
        get_filename_component(_ggml_copy_script
            "${CMAKE_CURRENT_LIST_DIR}/../../../core/AICore/cmake/CopyGgmlBackends.cmake"
            ABSOLUTE)
    else()
        set(_ggml_copy_script "${AICORE_COPY_GGML_SCRIPT}")
    endif()
    if(NOT EXISTS "${_ggml_copy_script}")
        message(FATAL_ERROR
            "AICore packaging: CopyGgmlBackends.cmake not found: ${_ggml_copy_script}")
    endif()
    if(NOT IS_DIRECTORY "${GGML_RUNTIME_LIB_DIR}")
        message(FATAL_ERROR
            "AICore packaging: ggml runtime dir missing: ${GGML_RUNTIME_LIB_DIR}")
    endif()
    if(NOT IS_DIRECTORY "${GGML_BACKEND_DIR}")
        message(FATAL_ERROR
            "AICore packaging: ggml backend dir missing: ${GGML_BACKEND_DIR}")
    endif()
    get_filename_component(_ggml_runtime_lib_dir "${GGML_RUNTIME_LIB_DIR}" ABSOLUTE)
    get_filename_component(_ggml_backend_dir "${GGML_BACKEND_DIR}" ABSOLUTE)
    get_filename_component(_ggml_wheel_lib_dir "${PYTHON_INSTALL_LIB_DESTINATION}" ABSOLUTE)
    # execute_process passes each COMMAND argument literally; do not wrap -D values
    # in quotes or nested cmake -P sees quote characters inside GGML_BACKEND_DIR.
    execute_process(COMMAND ${CMAKE_COMMAND}
        -DCOPY_CORE_SHARED=ON
        -DGGML_BACKEND_DIR=${_ggml_runtime_lib_dir}
        -DDEST_DIR=${_ggml_wheel_lib_dir}
        -DLIB_PREFIX=${CMAKE_SHARED_LIBRARY_PREFIX}
        -DLIB_SUFFIX=${CMAKE_SHARED_LIBRARY_SUFFIX}
        -P ${_ggml_copy_script}
        RESULT_VARIABLE _ggml_core_copy_result
        ERROR_VARIABLE _ggml_core_copy_error)
    if(_ggml_core_copy_result)
        message(FATAL_ERROR
            "AICore packaging: failed to copy ggml core libs: ${_ggml_core_copy_error}")
    endif()
    string(REPLACE ";" "|" _ggml_gpu_backend_files_pipe "${GGML_GPU_BACKEND_MODULE_FILES}")
    set(_ggml_backend_copy_args
        -DCOPY_BACKEND_MODULES=ON
        -DGGML_BACKEND_DIR=${_ggml_backend_dir}
        -DDEST_DIR=${_ggml_wheel_lib_dir}
        -DLIB_PREFIX=${CMAKE_SHARED_LIBRARY_PREFIX}
        -DLIB_SUFFIX=${GGML_MODULE_SUFFIX}
        -DGPU_FILES_PIPE=${_ggml_gpu_backend_files_pipe})
    if(GGML_CPU_ALL_VARIANTS)
        list(APPEND _ggml_backend_copy_args -DCPU_ALL_VARIANTS=ON)
    endif()
    list(APPEND _ggml_backend_copy_args -P ${_ggml_copy_script})
    execute_process(COMMAND ${CMAKE_COMMAND} ${_ggml_backend_copy_args}
        RESULT_VARIABLE _ggml_backend_copy_result
        ERROR_VARIABLE _ggml_backend_copy_error)
    if(_ggml_backend_copy_result)
        message(FATAL_ERROR
            "AICore packaging: failed to copy ggml backend modules: ${_ggml_backend_copy_error}")
    endif()
endif()

if (WIN32)
   SET(PACK_SCRIPTS "windows/pack_windows.ps1")
elseif (UNIX AND NOT APPLE)
   SET(PACK_SCRIPTS "linux/pack_ubuntu.sh")
elseif (APPLE)
   SET(PACK_SCRIPTS "mac/bundle/lib_bundle_wheel.py")
endif ()
set(PACKAGE_TOOL "${PYTHON_PACKAGE_SRC_DIR}/../scripts/platforms/${PACK_SCRIPTS}")

if (APPLE)
    set(DEST_PATH "${PYTHON_PACKAGE_DST_DIR}/cloudViewer")
    execute_process(COMMAND python ${PACKAGE_TOOL} ${DEST_PATH} 
                    WORKING_DIRECTORY ${PYTHON_PACKAGE_DST_DIR}
                    RESULT_VARIABLE _mac_wheel_bundle_result)
    # The bundler raises when an @rpath dependency cannot be resolved
    # (e.g. a missing source-built dylib). Without this check the
    # traceback is only logged and a wheel that fails at import ships.
    if(NOT _mac_wheel_bundle_result EQUAL 0)
        message(FATAL_ERROR
            "macOS wheel bundling failed (exit ${_mac_wheel_bundle_result}); "
            "see the CCWheelBundler traceback above. Shipping the wheel "
            "anyway would break at import.")
    endif()
elseif (UNIX)
    set(CPU_FOLDER_PATH "${PYTHON_PACKAGE_DST_DIR}/../${CMAKE_BUILD_TYPE}/Python/cpu")
    set(CUDA_FOLDER_PATH "${PYTHON_PACKAGE_DST_DIR}/../${CMAKE_BUILD_TYPE}/Python/cuda")
    if (EXISTS "${CPU_FOLDER_PATH}")
        execute_process(COMMAND bash ${PACKAGE_TOOL}
                        "${CPU_FOLDER_PATH}" ${PYTHON_INSTALL_LIB_DESTINATION}
                        "${CLOUDVIEWER_EXTERNAL_INSTALL_DIR}/lib"
                        WORKING_DIRECTORY ${PYTHON_PACKAGE_DST_DIR})
    endif()
    if (BUILD_CUDA_MODULE)
        execute_process(COMMAND bash ${PACKAGE_TOOL}
                        "${CUDA_FOLDER_PATH}" ${PYTHON_INSTALL_LIB_DESTINATION}
                        "${CLOUDVIEWER_EXTERNAL_INSTALL_DIR}/lib"
                        WORKING_DIRECTORY ${PYTHON_PACKAGE_DST_DIR})
    endif()

    set(QXCB_LIB_PATH "${PYTHON_INSTALL_LIB_DESTINATION}/platforms/libqxcb.so")
    if(EXISTS ${QXCB_LIB_PATH})
        execute_process(COMMAND bash ${PACKAGE_TOOL}
                        ${QXCB_LIB_PATH} ${PYTHON_INSTALL_LIB_DESTINATION}
                        WORKING_DIRECTORY ${PYTHON_PACKAGE_DST_DIR})
    else()
        message(WARNING "File ${QXCB_LIB_PATH} does not exist.")
    endif()

    set(SVGICON_LIB_PATH "${PYTHON_INSTALL_LIB_DESTINATION}/iconengines/libqsvgicon.so")
    if(EXISTS "${SVGICON_LIB_PATH}")
        execute_process(COMMAND bash ${PACKAGE_TOOL}
                        ${SVGICON_LIB_PATH} ${PYTHON_INSTALL_LIB_DESTINATION}
                        WORKING_DIRECTORY ${PYTHON_PACKAGE_DST_DIR})
    else()
        message(WARNING "File ${SVGICON_LIB_PATH} does not exist.")
    endif()

    execute_process(COMMAND bash ${PACKAGE_TOOL}
                    ${PYTHON_INSTALL_LIB_DESTINATION} ${PYTHON_INSTALL_LIB_DESTINATION}
                    "${CLOUDVIEWER_EXTERNAL_INSTALL_DIR}/lib"
                    WORKING_DIRECTORY ${PYTHON_PACKAGE_DST_DIR})
elseif (WIN32) # for windows
    set(CPU_FOLDER_PATH "${PYTHON_PACKAGE_DST_DIR}/cloudViewer/cpu")
    set(CUDA_FOLDER_PATH "${PYTHON_PACKAGE_DST_DIR}/cloudViewer/cuda")
    # prepare search path for powershell
    set(EXTERNAL_DLL_DIR ${CLOUDVIEWER_EXTERNAL_INSTALL_DIR}/bin ${BUILD_BIN_PATH}/${CUSTOM_BUILD_TYPE})
    if (BUILD_WITH_CONDA)
        list(APPEND EXTERNAL_DLL_DIR "${CONDA_PREFIX}/Library/bin")
    endif()
    message(STATUS "Start search dependency from path: ${EXTERNAL_DLL_DIR}")
    string(REPLACE ";" "\",\"" PS_SEARCH_PATHS "${EXTERNAL_DLL_DIR}")
    set(PS_SEARCH_PATHS "\"${PS_SEARCH_PATHS}\"")
    message(STATUS "PS_SEARCH_PATHS: ${PS_SEARCH_PATHS}")

    # find powershell program
    find_program(POWERSHELL_PATH NAMES powershell pwsh)
    if(NOT POWERSHELL_PATH)
        message(FATAL_ERROR "PowerShell not found!")
    endif()

    # search dependency for ACloudViewer, CloudViewer and Colmap
    if (EXISTS "${CPU_FOLDER_PATH}")
        execute_process(
            COMMAND ${POWERSHELL_PATH} -ExecutionPolicy Bypass 
                    -Command "& '${PACKAGE_TOOL}' '${CPU_FOLDER_PATH}' '${PYTHON_INSTALL_LIB_DESTINATION}' @(${PS_SEARCH_PATHS}) -Recursive"
                    WORKING_DIRECTORY ${PYTHON_PACKAGE_DST_DIR}
        )
    endif()
    if (BUILD_CUDA_MODULE)
        execute_process(
            COMMAND ${POWERSHELL_PATH} -ExecutionPolicy Bypass 
                    -Command "& '${PACKAGE_TOOL}' '${CUDA_FOLDER_PATH}' '${PYTHON_INSTALL_LIB_DESTINATION}' @(${PS_SEARCH_PATHS}) -Recursive"
                    WORKING_DIRECTORY ${PYTHON_PACKAGE_DST_DIR}
        )
    endif()
endif()

if (BUILD_TENSORFLOW_OPS OR BUILD_PYTORCH_OPS)
    # copy generated files
    file(COPY "${PYTHON_PACKAGE_DST_DIR}/../ml"
            DESTINATION "${PYTHON_PACKAGE_DST_DIR}/cloudViewer/")
endif ()

if (BUNDLE_CLOUDVIEWER_ML)
    file(COPY "${PYTHON_PACKAGE_DST_DIR}/../../cloudViewer_ml/src/cloudViewer_ml/ml3d"
            DESTINATION "${PYTHON_PACKAGE_DST_DIR}/cloudViewer/")
    file(RENAME "${PYTHON_PACKAGE_DST_DIR}/cloudViewer/ml3d" "${PYTHON_PACKAGE_DST_DIR}/cloudViewer/_ml3d")
endif ()

# Build Jupyter plugin.
if (BUILD_JUPYTER_EXTENSION)
    if (WIN32 OR UNIX AND NOT LINUX_AARCH64)
        message(STATUS "Jupyter support is enabled, building Jupyter plugin now.")
    else ()
        message(FATAL_ERROR "Jupyter plugin is not supported on ARM.")
    endif ()

    find_program(NODE node)
    if (NODE)
        message(STATUS "node found at: ${NODE}")
    else()
        message(STATUS "node not found.")
        message(FATAL_ERROR "Please install Node.js."
                            "Visit https://nodejs.org/en/download/package-manager/ for details."
                            "For ubuntu, we recommend getting the latest version of Node.js from"
                            "https://github.com/nodesource/distributions/blob/master/README.md#installation-instructions.")
    endif()
    execute_process(COMMAND "${NODE}" --version
                    OUTPUT_VARIABLE NODE_VERSION
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    STRING(REGEX REPLACE "v" "" NODE_VERSION ${NODE_VERSION})
    message(STATUS "node version: ${NODE_VERSION}")
    if (NODE_VERSION VERSION_LESS ${MIN_NODE_VERSION})
        message(FATAL_ERROR "node version ${NODE_VERSION} is too old. "
                            "Please upgrade to ${MIN_NODE_VERSION} or higher.")
    endif()

    find_program(YARN yarn)
    if (YARN)
        message(STATUS "YARN found at: ${YARN}")
    else ()
        message(FATAL_ERROR "yarn not found. You may install yarn globally by "
                "npm install -g yarn.")
    endif ()

    # Append requirements_jupyter.txt to requirements.txt
    execute_process(COMMAND ${CMAKE_COMMAND} -E cat
            ${PYTHON_PACKAGE_SRC_DIR}/requirements.txt
            ${PYTHON_PACKAGE_SRC_DIR}/requirements_jupyter_install.txt
            OUTPUT_VARIABLE ALL_REQUIREMENTS
            )
    # The double-quote "" is important as it keeps the semicolons.
    file(WRITE ${PYTHON_PACKAGE_DST_DIR}/requirements.txt "${ALL_REQUIREMENTS}")
endif ()

if (BUILD_GUI)
    file(MAKE_DIRECTORY "${PYTHON_PACKAGE_DST_DIR}/cloudViewer/resources/")
    file(COPY ${GUI_RESOURCE_DIR}
            DESTINATION "${PYTHON_PACKAGE_DST_DIR}/cloudViewer/")
endif ()

# Add all examples to installation directory.
file(MAKE_DIRECTORY "${PYTHON_PACKAGE_DST_DIR}/cloudViewer/examples/")
file(COPY "${PYTHON_PACKAGE_SRC_DIR}/../examples/Python/"
     DESTINATION "${PYTHON_PACKAGE_DST_DIR}/cloudViewer/examples")
file(COPY "${PYTHON_PACKAGE_SRC_DIR}/../examples/Python/"
     DESTINATION "${PYTHON_PACKAGE_DST_DIR}/cloudViewer/examples")
