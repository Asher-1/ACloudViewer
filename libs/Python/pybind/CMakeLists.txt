if (NOT Python3_EXECUTABLE)
    message(FATAL_ERROR "Python 3 not found in top level file")
endif ()

# We need to get python version to configure some meta files
set(PYTHON_VERSION "${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}")

message(STATUS "PYTHON VERSION: " ${PYTHON_VERSION})

include_directories(${CloudViewer_SOURCE_DIR})
pybind11_add_module(pybind)

add_subdirectory(camera)
add_subdirectory(core)
add_subdirectory(data)
add_subdirectory(geometry)
add_subdirectory(io)
add_subdirectory(ml)
add_subdirectory(pipelines)
add_subdirectory(t)
add_subdirectory(t/geometry)
add_subdirectory(t/io)
add_subdirectory(t/pipelines)
add_subdirectory(utility)
add_subdirectory(visualization)

if (BUILD_RECONSTRUCTION)
    target_include_directories(pybind PRIVATE ${CLOUDVIEWER_EXTERNAL_INSTALL_DIR}/include)
    target_include_directories(pybind PRIVATE ${COLMAP_SOURCE_DIR}/src)
    target_include_directories(pybind PRIVATE ${COLMAP_SOURCE_DIR}/lib)
    add_subdirectory(reconstruction)
    # Some fixes for the Glog library.
    target_compile_definitions(pybind PRIVATE -DGLOG_NO_ABBREVIATED_SEVERITIES)
    target_compile_definitions(pybind PRIVATE -DGOOGLE_GLOG_DLL_DECL=)
endif ()

target_sources(pybind PRIVATE
        docstring.cpp
        cloudViewer_pybind.cpp
        pybind_utils.cpp
)

# Include with `#include "pybind/geometry/xxx.h`
target_include_directories(pybind PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/.."
)

# Suppress Pybind11 warnings
target_include_directories(pybind SYSTEM PRIVATE
        ${PYBIND11_INCLUDE_DIR}
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        )

cloudViewer_show_and_abort_on_warning(pybind)
cloudViewer_set_global_properties(pybind)
target_link_libraries(pybind PRIVATE CloudViewer::CloudViewer)

set_target_properties(pybind PROPERTIES DEBUG_POSTFIX "")
if (WIN32)
    target_link_options(pybind PUBLIC "/force:multiple")
    # TODO(Sameer): Only export PyInit_pybind, cloudViewer_core_cuda_device_count
elseif (APPLE)
    file(GENERATE OUTPUT pybind.map CONTENT
        [=[_PyInit_pybind
_cloudViewer_core_cuda_device_count
        ]=])
    target_link_options(pybind PRIVATE $<$<CONFIG:Release>:
        -Wl,-exported_symbols_list
        "${CMAKE_CURRENT_BINARY_DIR}/pybind.map" >)
elseif (UNIX)   # Linux
    file(GENERATE OUTPUT pybind.map CONTENT
        [=[{
    global:
        PyInit_pybind;
        cloudViewer_core_cuda_device_count;
    local:
        *;
};]=])
    target_link_options(pybind PRIVATE $<$<CONFIG:Release>:
        "-Wl,--version-script=${CMAKE_CURRENT_BINARY_DIR}/pybind.map" >)
    target_link_options(pybind PRIVATE "-flto=auto")
endif()

if (BUILD_RECONSTRUCTION)
    target_link_libraries(pybind PRIVATE ${COLMAP_LIB_NAME})
    # Add custom preprocessor definitions
    target_compile_definitions(pybind PRIVATE BUILD_RECONSTRUCTION)
endif ()

# At `make`: cloudViewer.so (or the equivalents) will be created at
# PYTHON_COMPILED_MODULE_DIR. The default location is
# `build/lib/${CMAKE_BUILD_TYPE}/Python/{cpu|cuda}`
set(PYTHON_COMPILED_MODULE_DIR
        "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/Python/$<IF:$<BOOL:${BUILD_CUDA_MODULE}>,cuda,cpu>")

# Set BUILD_RPATH to find tbb (and a shared libCloudViewer). We don't install through cmake.
# if (APPLE)
#         set_target_properties(pybind PROPERTIES BUILD_RPATH "@loader_path;@loader_path/..;@loader_path/../lib")
# elseif (UNIX)
#         # Use RPATH instead of RUNPATH in pybind so that needed libc++.so can find child dependant libc++abi.so in RPATH
#         # https://stackoverflow.com/questions/69662319/managing-secondary-dependencies-of-shared-libraries
#         target_link_options(pybind PRIVATE "LINKER:--disable-new-dtags")
#         set_target_properties(pybind PROPERTIES BUILD_RPATH "$ORIGIN;$ORIGIN/..;$ORIGIN/../lib")
# endif()

set_target_properties(pybind PROPERTIES
        FOLDER "Python"
        LIBRARY_OUTPUT_DIRECTORY "${PYTHON_COMPILED_MODULE_DIR}"
        ARCHIVE_OUTPUT_DIRECTORY "${PYTHON_COMPILED_MODULE_DIR}")
# Include additional libraries that may be absent from the user system
# eg: libc++.so and libc++abi.so (needed by filament) for Linux.
# libc++.so is a linker script including libc++.so.1 and libc++abi.so, so append 1 to libc++.so
set(PYTHON_EXTRA_LIBRARIES $<TARGET_FILE:TBB::tbb>)
if (BUILD_GUI AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    list(APPEND PYTHON_EXTRA_LIBRARIES ${CPP_LIBRARY}.1 ${CPPABI_LIBRARY})
endif ()

# Use `make python-package` to create the python package in the build directory
# The python package will be created at PYTHON_PACKAGE_DIR. It contains:
# 1) Pure-python code and misc files, copied from python/package
# 2) The compiled python-C++ module, i.e. cloudViewer.so (or the equivalents)
# 3) Configured files and supporting files
# Note: `make python-package` clears PYTHON_COMPILED_MODULE_DIR first every time
set(PYTHON_PACKAGE_SRC_DIR "${PROJECT_SOURCE_DIR}/python")
set(PYTHON_PACKAGE_DST_DIR "${CMAKE_BINARY_DIR}/lib/python_package")
message(STATUS "PYPI_PACKAGE_NAME: ${PYPI_PACKAGE_NAME}")

# add the cloudViewer python module first
set(COMPILED_MODULE_PATH_LIST $<TARGET_FILE:pybind>)

set(GENERATED_OUTPUTS "")

# add additional optional compiled modules
if (BUILD_TENSORFLOW_OPS)
    list(APPEND COMPILED_MODULE_PATH_LIST $<TARGET_FILE:cloudViewer_tf_ops>)
    add_custom_command(OUTPUT "${CMAKE_BINARY_DIR}/lib/ml/tf/python/ops/ops.py"
            COMMAND ${Python3_EXECUTABLE} generate_tf_ops_wrapper.py --input "${PYTHON_PACKAGE_SRC_DIR}/cloudViewer/ml/tf/python/ops/ops.py.in" --output "${CMAKE_BINARY_DIR}/lib/ml/tf/python/ops/ops.py" --lib $<TARGET_FILE:cloudViewer_tf_ops>
            DEPENDS cloudViewer_tf_ops
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Generating python ops.py")

    list(APPEND GENERATED_OUTPUTS "${CMAKE_BINARY_DIR}/lib/ml/tf/python/ops/ops.py")
    # find tensorflow to get some info for the _build_config.py
    find_package(Tensorflow)
endif ()

# add additional optional compiled modules
if (BUILD_PYTORCH_OPS)
    list(APPEND COMPILED_MODULE_PATH_LIST $<TARGET_FILE:cloudViewer_torch_ops>)
    set(DEPENDENCIES_DIR "$<TARGET_FILE_DIR:TBB::tbb>")
    # Set KMP_DUPLICATE_LIB_OK to avoid OpenMP initialization errors when loading torch ops library
    add_custom_command(OUTPUT "${CMAKE_BINARY_DIR}/lib/ml/torch/python/ops.py" "${CMAKE_BINARY_DIR}/lib/ml/torch/python/return_types.py"
            COMMAND ${CMAKE_COMMAND} -E env KMP_DUPLICATE_LIB_OK=TRUE ${Python3_EXECUTABLE} generate_torch_ops_wrapper.py --input_ops_py_in "${PYTHON_PACKAGE_SRC_DIR}/cloudViewer/ml/torch/python/ops.py.in" --input_return_types_py_in "${PYTHON_PACKAGE_SRC_DIR}/cloudViewer/ml/torch/python/return_types.py.in" --output_dir "${CMAKE_BINARY_DIR}/lib/ml/torch/python/" --lib $<TARGET_FILE:cloudViewer_torch_ops> --tensorflow_ops_dir "${CMAKE_CURRENT_SOURCE_DIR}/../../cloudViewer/ml/tensorflow" --dependencies_dir "${DEPENDENCIES_DIR}"
            DEPENDS cloudViewer_torch_ops
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Generating python ops.py and return_types.py")

    list(APPEND GENERATED_OUTPUTS "${CMAKE_BINARY_DIR}/lib/ml/torch/python/ops.py" "${CMAKE_BINARY_DIR}/lib/ml/torch/python/return_types.py")
    
    # Pytorch_VERSION should be available from FindPytorch.cmake (set as cache variable)
    # Verify it's set correctly for _build_config.py
    if(NOT DEFINED Pytorch_VERSION OR Pytorch_VERSION STREQUAL "")
        message(WARNING "Pytorch_VERSION is not set or empty!")
        message(WARNING "This should have been set by FindPytorch.cmake")
        message(WARNING "Attempting to get PyTorch version directly...")
        
        execute_process(
                COMMAND ${Python3_EXECUTABLE} "-c"
                "import torch; print(torch.__version__, end='')"
                OUTPUT_VARIABLE Pytorch_VERSION
                RESULT_VARIABLE Pytorch_VERSION_RESULT
                ERROR_VARIABLE Pytorch_VERSION_ERROR
                OUTPUT_STRIP_TRAILING_WHITESPACE)
        
        if(NOT Pytorch_VERSION_RESULT EQUAL "0" OR Pytorch_VERSION STREQUAL "")
            message(FATAL_ERROR "Cannot determine PyTorch version. Error: ${Pytorch_VERSION_ERROR}")
        endif()
        message(STATUS "PyTorch version (from Python): ${Pytorch_VERSION}")
    else()
        message(STATUS "PyTorch version (from FindPytorch): ${Pytorch_VERSION}")
    endif()
endif ()

if (BUNDLE_CLOUDVIEWER_ML)
    find_path(
            CLOUDVIEWER_ML_ROOT
            NAMES set_cloudViewer_ml_root.sh
            HINTS $ENV{CLOUDVIEWER_ML_ROOT}
            DOC "Path to the CloudViewer-ML repo. This should be set if BUNDLE_CLOUDVIEWER_ML is enabled. Alternatively set an env var with the same name to populate this var."
            REQUIRED
            NO_DEFAULT_PATH
    )

    ExternalProject_Add(
            cloudViewer_ml
            PREFIX "${CMAKE_BINARY_DIR}/cloudViewer_ml"
            GIT_REPOSITORY "${CLOUDVIEWER_ML_ROOT}"
            GIT_TAG main
            GIT_SHALLOW
            BUILD_IN_SOURCE ON
            # do not configure
            CONFIGURE_COMMAND ""
            # do not build
            BUILD_COMMAND ""
            # do not install
            INSTALL_COMMAND ""
    )
    list(APPEND GENERATED_OUTPUTS cloudViewer_ml)
endif ()

configure_file("_build_config.py.in"
        "${CMAKE_BINARY_DIR}/lib/_build_config.py.in")

file(GENERATE
        OUTPUT "${PYTHON_COMPILED_MODULE_DIR}/_build_config.py"
        INPUT "${CMAKE_BINARY_DIR}/lib/_build_config.py.in"
        )

set(BUILD_BIN_PATH "${CMAKE_BINARY_DIR}/bin")
set(CUSTOM_BUILD_TYPE "${CMAKE_BUILD_TYPE}")
add_custom_target(python-package
        COMMAND ${CMAKE_COMMAND}
        -DPYTHON_PACKAGE_SRC_DIR=${PYTHON_PACKAGE_SRC_DIR}
        -DPYTHON_PACKAGE_DST_DIR=${PYTHON_PACKAGE_DST_DIR}
        -DPYTHON_COMPILED_MODULE_DIR=${PYTHON_COMPILED_MODULE_DIR}
        -DPYTHON_VERSION=${PYTHON_VERSION}
        "-DCOMPILED_MODULE_PATH_LIST=${COMPILED_MODULE_PATH_LIST}"
        "-DPYTHON_EXTRA_LIBRARIES=${PYTHON_EXTRA_LIBRARIES}"
        "-DQT5_PLUGINS_PATH_LIST=${QT5_PLUGINS_PATH_LIST}"
        "-DCLOUDVIEWER_EXTERNAL_INSTALL_DIR=${CLOUDVIEWER_EXTERNAL_INSTALL_DIR}"
        "-DBUILD_BIN_PATH=${BUILD_BIN_PATH}"
        "-DCUSTOM_BUILD_TYPE=${CUSTOM_BUILD_TYPE}"
        "-DCONDA_PREFIX=${CONDA_PREFIX}"
        -DBUILD_WITH_CONDA=${BUILD_WITH_CONDA}
        -DBUILD_JUPYTER_EXTENSION=${BUILD_JUPYTER_EXTENSION}
        -DBUILD_TENSORFLOW_OPS=${BUILD_TENSORFLOW_OPS}
        -DBUILD_PYTORCH_OPS=${BUILD_PYTORCH_OPS}
        -DBUNDLE_CLOUDVIEWER_ML=${BUNDLE_CLOUDVIEWER_ML}
        -DCLOUDVIEWER_ML_ROOT=${CLOUDVIEWER_ML_ROOT}
        -DBUILD_CUDA_MODULE=${BUILD_CUDA_MODULE}
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DBUILD_GUI=${BUILD_GUI}
        -DBUILD_RECONSTRUCTION=${BUILD_RECONSTRUCTION}
        -DGUI_RESOURCE_DIR=${GUI_RESOURCE_DIR}
        -DPROJECT_EMAIL=${PROJECT_EMAIL}
        -DPROJECT_HOME=${PROJECT_HOME}
        -DPROJECT_DOCS=${PROJECT_DOCS}
        -DPROJECT_CODE=${PROJECT_CODE}
        -DPROJECT_ISSUES=${PROJECT_ISSUES}
        -DPROJECT_VERSION=${CLOUDVIEWER_VERSION_FULL}
        -DPROJECT_DESCRIPTION=${PROJECT_DESCRIPTION}
        -DPROJECT_VERSION_THREE_NUMBER=${PROJECT_VERSION_THREE_NUMBER}
        -DPYPI_PACKAGE_NAME=${PYPI_PACKAGE_NAME}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/make_python_package.cmake
        VERBATIM
        DEPENDS ${GENERATED_OUTPUTS}
        )

# Use `make pip-package` to create the pip package in the build directory
add_custom_target(pip-package
        COMMAND ${Python3_EXECUTABLE} setup.py bdist_wheel --dist-dir pip_package
        COMMAND echo "pip wheel created at ${PYTHON_PACKAGE_DST_DIR}/pip_package"
        WORKING_DIRECTORY ${PYTHON_PACKAGE_DST_DIR}
        DEPENDS python-package
        )

# Use `make install-pip-package` to install pip wheel package to the current
# python environment.
add_custom_target(install-pip-package
        COMMAND ${CMAKE_COMMAND}
        -DPYTHON_PACKAGE_DST_DIR=${PYTHON_PACKAGE_DST_DIR}
        -DPython3_EXECUTABLE=${Python3_EXECUTABLE}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/make_install_pip_package.cmake
        DEPENDS pip-package
        )

# FOR DEBUGGING ONLY Use `make install-python-package` to build and install
# python package in the current python environment. This is substantially
# faster than `make install-pip-package`. However this approach does not create
# wheel or egg files and does not take care of dependencies thus not suitable
# for deployment.
# Ref: https://stackoverflow.com/a/33791008/1255535
add_custom_target(install-python-package
        COMMAND ${Python3_EXECUTABLE} setup.py install --single-version-externally-managed --root=/
        WORKING_DIRECTORY ${PYTHON_PACKAGE_DST_DIR}
        DEPENDS python-package
        )

# Use `make conda-package` to create conda package in the build directory
# Note that we don't provide `make install-conda-package` similar to pip. This
# is becuase:
#     1) `make install-pip-package` works in conda environment for local build
#     2) `make conda-package` is mainly for internal use to distribute conda
add_custom_target(conda-package
        COMMAND conda-build conda_meta --channel conda-forge --output-folder conda_package
        COMMAND echo "conda package created at ${PYTHON_PACKAGE_DST_DIR}/conda_package"
        WORKING_DIRECTORY ${PYTHON_PACKAGE_DST_DIR}
        DEPENDS python-package
        )

# Make pip and conda package in one command. The python-package target clears
# the ${PYTHON_PACKAGE_DST_DIR}, so if we make pip and conda package one-by-one,
# the previous one will be deleted.
add_custom_target(pip-conda-package
        COMMAND ${Python3_EXECUTABLE} setup.py bdist_wheel --dist-dir pip_package
        COMMAND conda-build conda_meta --channel conda-forge --output-folder conda_package
        COMMAND echo "pip wheel created at ${PYTHON_PACKAGE_DST_DIR}/pip_package"
        COMMAND echo "conda package created at ${PYTHON_PACKAGE_DST_DIR}/conda_package"
        WORKING_DIRECTORY ${PYTHON_PACKAGE_DST_DIR}
        DEPENDS python-package
        )