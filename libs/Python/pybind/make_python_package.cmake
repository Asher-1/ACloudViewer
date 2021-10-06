# Clean up directory
file(REMOVE_RECURSE ${PYTHON_PACKAGE_DST_DIR})
file(MAKE_DIRECTORY ${PYTHON_PACKAGE_DST_DIR}/cloudViewer)

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
foreach(COMPILED_MODULE_PATH ${COMPILED_MODULE_PATH_LIST})
    get_filename_component(COMPILED_MODULE_NAME ${COMPILED_MODULE_PATH} NAME)
    get_filename_component(COMPILED_MODULE_ARCH_DIR ${COMPILED_MODULE_PATH} DIRECTORY)
    get_filename_component(COMPILED_MODULE_BASE_DIR ${COMPILED_MODULE_ARCH_DIR} DIRECTORY)
    foreach(ARCH cpu cuda)
        if(IS_DIRECTORY "${COMPILED_MODULE_BASE_DIR}/${ARCH}")
            file(INSTALL "${COMPILED_MODULE_BASE_DIR}/${ARCH}/" DESTINATION
                "${PYTHON_PACKAGE_DST_DIR}/cloudViewer/${ARCH}"
                FILES_MATCHING PATTERN "${COMPILED_MODULE_NAME}")
        endif()
    endforeach()
endforeach()
# Include additional libraries that may be absent from the user system
# eg: libc++.so and libc++abi.so (needed by filament)
# The linker recognizes only library.so.MAJOR, so remove .MINOR from the filename
foreach(PYTHON_EXTRA_LIB ${PYTHON_EXTRA_LIBRARIES})
    get_filename_component(PYTHON_EXTRA_LIB_REAL ${PYTHON_EXTRA_LIB} REALPATH)
    get_filename_component(SO_VER_NAME ${PYTHON_EXTRA_LIB_REAL} NAME)
    string(REGEX REPLACE "\\.so\\.1\\..*" ".so.1" SO_1_NAME ${SO_VER_NAME})
    configure_file(${PYTHON_EXTRA_LIB_REAL} ${PYTHON_PACKAGE_DST_DIR}/cloudViewer/${SO_1_NAME} COPYONLY)
endforeach()

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
     DESTINATION "${PYTHON_PACKAGE_DST_DIR}/cloudViewer/" )

if (BUILD_TENSORFLOW_OPS OR BUILD_PYTORCH_OPS)
    # copy generated files
    file(COPY "${PYTHON_PACKAGE_DST_DIR}/../ml"
         DESTINATION "${PYTHON_PACKAGE_DST_DIR}/cloudViewer/" )
endif()

if (BUNDLE_CLOUDVIEWER_ML)
    file(COPY "${PYTHON_PACKAGE_DST_DIR}/../../cloudViewer_ml/src/cloudViewer_ml/ml3d"
         DESTINATION "${PYTHON_PACKAGE_DST_DIR}/cloudViewer/" )
    file(RENAME "${PYTHON_PACKAGE_DST_DIR}/cloudViewer/ml3d" "${PYTHON_PACKAGE_DST_DIR}/cloudViewer/_ml3d")
endif()

# Build Jupyter plugin.
if (BUILD_JUPYTER_EXTENSION)
    if(WIN32 OR UNIX AND NOT LINUX_AARCH64)
        message(STATUS "Jupyter support is enabled, building Jupyter plugin now.")
    else()
        message(FATAL_ERROR "Jupyter plugin is not supported on ARM.")
    endif()

    find_program(NPM npm)
    if(NPM)
        message(STATUS "NPM found at: ${NPM}")
    else()
        message(FATAL_ERROR "npm not found. Please install Node.js and npm."
                            "Visit https://www.npmjs.com/get-npm for details.")
    endif()

    find_program(YARN yarn)
    if(YARN)
        message(STATUS "YARN found at: ${YARN}")
    else()
        message(FATAL_ERROR "yarn not found. You may install yarn globally by "
                            "npm install -g yarn.")
    endif()

    # Append requirements_jupyter.txt to requirements.txt
    execute_process(COMMAND ${CMAKE_COMMAND} -E cat
        ${PYTHON_PACKAGE_SRC_DIR}/requirements.txt
        ${PYTHON_PACKAGE_SRC_DIR}/requirements_jupyter.txt
        OUTPUT_VARIABLE ALL_REQUIREMENTS
    )
    file(WRITE ${PYTHON_PACKAGE_DST_DIR}/requirements.txt ${ALL_REQUIREMENTS})
endif()

if (BUILD_GUI)
    file(MAKE_DIRECTORY "${PYTHON_PACKAGE_DST_DIR}/cloudViewer/resources/")
    file(COPY ${GUI_RESOURCE_DIR}
         DESTINATION "${PYTHON_PACKAGE_DST_DIR}/cloudViewer/")
endif()
