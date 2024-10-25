set(MIN_NODE_VERSION "14.00.0")

# Clean up directory
file(REMOVE_RECURSE ${PYTHON_PACKAGE_DST_DIR})
file(MAKE_DIRECTORY ${PYTHON_PACKAGE_DST_DIR}/cloudViewer)
file(MAKE_DIRECTORY ${PYTHON_PACKAGE_DST_DIR}/cloudViewer/lib)
set(PYTHON_INSTALL_LIB_DESTINATION "${PYTHON_PACKAGE_DST_DIR}/cloudViewer/lib")
set(CUSTOM_SO_NAME ".so.${PROJECT_VERSION}")

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
# Folder structure is base_dir/{cpu|cuda}/{pybind*.so|cloudViewer_{torch|tf}_ops.so},
# so copy base_dir directly to ${PYTHON_PACKAGE_DST_DIR}/cloudViewer
foreach (COMPILED_MODULE_PATH ${COMPILED_MODULE_PATH_LIST})
    get_filename_component(COMPILED_MODULE_NAME ${COMPILED_MODULE_PATH} NAME)
    get_filename_component(COMPILED_MODULE_ARCH_DIR ${COMPILED_MODULE_PATH} DIRECTORY)
    get_filename_component(COMPILED_MODULE_BASE_DIR ${COMPILED_MODULE_ARCH_DIR} DIRECTORY)
    foreach (ARCH cpu cuda)
        if (IS_DIRECTORY "${COMPILED_MODULE_BASE_DIR}/${ARCH}")
            file(INSTALL "${COMPILED_MODULE_BASE_DIR}/${ARCH}/" DESTINATION
                    "${PYTHON_PACKAGE_DST_DIR}/cloudViewer/${ARCH}"
                    FILES_MATCHING PATTERN "${COMPILED_MODULE_NAME}")
        endif ()
    endforeach ()
endforeach ()
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
configure_file("${PYTHON_PACKAGE_SRC_DIR}/cloudViewer/visualization/__init__.py"
        "${PYTHON_PACKAGE_DST_DIR}/cloudViewer/visualization/__init__.py")
configure_file("${PYTHON_PACKAGE_SRC_DIR}/cloudViewer/visualization/gui/__init__.py"
        "${PYTHON_PACKAGE_DST_DIR}/cloudViewer/visualization/gui/__init__.py")
configure_file("${PYTHON_PACKAGE_SRC_DIR}/cloudViewer/visualization/rendering/__init__.py"
        "${PYTHON_PACKAGE_DST_DIR}/cloudViewer/visualization/rendering/__init__.py")
configure_file("${PYTHON_PACKAGE_SRC_DIR}/cloudViewer/web_visualizer.py"
        "${PYTHON_PACKAGE_DST_DIR}/cloudViewer/web_visualizer.py")
configure_file("${PYTHON_PACKAGE_SRC_DIR}/conda_meta/conda_build_config.yaml"
        "${PYTHON_PACKAGE_DST_DIR}/conda_meta/conda_build_config.yaml")
configure_file("${PYTHON_PACKAGE_SRC_DIR}/conda_meta/meta.yaml"
        "${PYTHON_PACKAGE_DST_DIR}/conda_meta/meta.yaml")
configure_file("${PYTHON_PACKAGE_SRC_DIR}/js/lib/web_visualizer.js"
        "${PYTHON_PACKAGE_DST_DIR}/js/lib/web_visualizer.js")
configure_file("${PYTHON_PACKAGE_SRC_DIR}/js/package.json"
        "${PYTHON_PACKAGE_DST_DIR}/js/package.json")
configure_file("${PYTHON_PACKAGE_SRC_DIR}/../libs/cloudViewer/visualization/webrtc_server/html/webrtcstreamer.js"
        "${PYTHON_PACKAGE_DST_DIR}/js/lib/webrtcstreamer.js")
file(COPY "${PYTHON_COMPILED_MODULE_DIR}/_build_config.py"
        DESTINATION "${PYTHON_PACKAGE_DST_DIR}/cloudViewer/")


# 4) Add extra dependencies.
message(STATUS "QT5_PLUGINS_PATH_LIST: " ${QT5_PLUGINS_PATH_LIST})
foreach( qt5_plugins_folder ${QT5_PLUGINS_PATH_LIST} )
    file(COPY "${qt5_plugins_folder}"
         DESTINATION "${PYTHON_INSTALL_LIB_DESTINATION}/")
endforeach()

if (WIN32)
   SET(PACK_SCRIPTS "windows/pack_windows.bat")
elseif (UNIX AND NOT APPLE)
   SET(PACK_SCRIPTS "linux/pack_ubuntu.sh")
elseif (APPLE)
   SET(PACK_SCRIPTS "mac/pack_macos_wheel.sh")
endif ()
set(PACKAGE_TOOL "${PYTHON_PACKAGE_SRC_DIR}/../scripts/platforms/${PACK_SCRIPTS}")

if (APPLE)
    execute_process(COMMAND bash ${PACKAGE_TOOL}
                    "${PYTHON_PACKAGE_DST_DIR}/cloudViewer/cpu" ${PYTHON_INSTALL_LIB_DESTINATION}
                    WORKING_DIRECTORY ${PYTHON_PACKAGE_DST_DIR})
    execute_process(COMMAND bash ${PACKAGE_TOOL} ${PYTHON_INSTALL_LIB_DESTINATION} ${PYTHON_INSTALL_LIB_DESTINATION}
                    WORKING_DIRECTORY ${PYTHON_PACKAGE_DST_DIR})
elseif (UNIX)
    if (BUILD_CUDA_MODULE)
        execute_process(COMMAND bash ${PACKAGE_TOOL}
                        "${PYTHON_PACKAGE_DST_DIR}/../${CMAKE_BUILD_TYPE}/Python/cuda" ${PYTHON_INSTALL_LIB_DESTINATION}
                        WORKING_DIRECTORY ${PYTHON_PACKAGE_DST_DIR})
    else ()
        execute_process(COMMAND bash ${PACKAGE_TOOL}
                        "${PYTHON_PACKAGE_DST_DIR}/../${CMAKE_BUILD_TYPE}/Python/cpu" ${PYTHON_INSTALL_LIB_DESTINATION}
                        WORKING_DIRECTORY ${PYTHON_PACKAGE_DST_DIR})
    endif()
    execute_process(COMMAND bash ${PACKAGE_TOOL}
                    ${PYTHON_INSTALL_LIB_DESTINATION}/platforms/libqxcb.so ${PYTHON_INSTALL_LIB_DESTINATION}
                    WORKING_DIRECTORY ${PYTHON_PACKAGE_DST_DIR})

    # rename ldd lib to the format like "${CUSTOM_SO_NAME}"
    file(GLOB ldd_libs_list "${PYTHON_INSTALL_LIB_DESTINATION}/*.so*" )
    foreach (filename ${ldd_libs_list})
        get_filename_component(EXTRA_LIB_REAL ${filename} REALPATH)
        get_filename_component(SO_VER_NAME ${EXTRA_LIB_REAL} NAME)
        string(SUBSTRING ${SO_VER_NAME} 0 6 LIB_SBU_STR)
        if(NOT (${LIB_SBU_STR} STREQUAL "libicu")) # fix cannot found libicuuc.so.56: cannot open shared object file
            string(REGEX REPLACE "\\.so\\.[0-9.]+$" "${CUSTOM_SO_NAME}" NEW_SO_NAME ${SO_VER_NAME})
            message(STATUS "Copy ldd lib: " ${NEW_SO_NAME})
            file(RENAME ${EXTRA_LIB_REAL} ${PYTHON_INSTALL_LIB_DESTINATION}/${NEW_SO_NAME})
        endif()
    endforeach ()

    message(STATUS "CLOUDVIEWER_EXTERNAL_INSTALL_LIB_DIR: " ${CLOUDVIEWER_EXTERNAL_INSTALL_LIB_DIR})
    file(GLOB external_libs_list "${CLOUDVIEWER_EXTERNAL_INSTALL_LIB_DIR}/*.so*" )
    # rename external lib to the format like "${CUSTOM_SO_NAME}"
    foreach (filename ${external_libs_list})
        get_filename_component(EXTRA_LIB_REAL ${filename} REALPATH)
        get_filename_component(SO_VER_NAME ${EXTRA_LIB_REAL} NAME)
        string(SUBSTRING ${SO_VER_NAME} 0 9 LIB_SBU_STR)
        if(${LIB_SBU_STR} STREQUAL "libgflags") # fix libgflags.so.2.2: cannot open shared object file
            set(NEW_SO_NAME ${SO_VER_NAME})
        else()
            string(REGEX REPLACE "\\.so\\.[0-9.]+$" "${CUSTOM_SO_NAME}" NEW_SO_NAME ${SO_VER_NAME})
        endif()
        message(STATUS "Copy external lib: " ${NEW_SO_NAME})
        configure_file(${EXTRA_LIB_REAL} ${PYTHON_INSTALL_LIB_DESTINATION}/${NEW_SO_NAME} COPYONLY)
    endforeach ()
elseif (WIN32)
    # for windows
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