# This function sets a global PYTHON_BASE_PREFIX cache variable
function(setup_python_env)
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" "-c"
            "import sys;print(sys.base_prefix.replace('\\\\', '/'), end='')"
    OUTPUT_VARIABLE PYTHON_BASE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  set(PYTHON_BASE_PREFIX
    "${PYTHON_BASE_PREFIX}"
    PARENT_SCOPE
  )

  if (WIN32)
    execute_process(
      COMMAND "${PYTHON_EXECUTABLE}" "-c"
              "import site; print(site.getsitepackages()[1])"
      OUTPUT_VARIABLE PYTHON_SITELIB
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    get_filename_component(PYTHON_LIB_DIR "${PYTHON_SITELIB}" DIRECTORY)
    get_filename_component(PYTHON_LIB_NAME "${PYTHON_LIB_DIR}" NAME)
  else ()
    execute_process(
      COMMAND "${PYTHON_EXECUTABLE}" "-c"
              "import site; print(site.getsitepackages()[0])"
      OUTPUT_VARIABLE PYTHON_SITELIB
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    get_filename_component(PYTHON_VERSION_DIR "${PYTHON_SITELIB}" DIRECTORY)
    get_filename_component(PYTHON_NAME "${PYTHON_VERSION_DIR}" NAME)
    get_filename_component(PYTHON_LIB_DIR "${PYTHON_VERSION_DIR}" DIRECTORY)
    get_filename_component(PYTHON_LIB_NAME "${PYTHON_LIB_DIR}" NAME)
    set(PYTHON_NAME
      "${PYTHON_NAME}"
      PARENT_SCOPE
    )
    set(PYTHON_VERSION_DIR
      "${PYTHON_VERSION_DIR}"
      PARENT_SCOPE
    )
  endif()

  set(PYTHON_LIB_NAME
    "${PYTHON_LIB_NAME}"
    PARENT_SCOPE
  )

  # For pyenv-based Python, we need to find the actual library directory
  # which may be different from the site-packages parent directory
  if (UNIX AND NOT WIN32)
    # Try to find the lib directory containing libpython and other shared libraries
    set(PYTHON_LIB_SEARCH_PATHS
      "${PYTHON_BASE_PREFIX}/lib"
      "${PYTHON_LIB_DIR}"
    )

    foreach(search_path ${PYTHON_LIB_SEARCH_PATHS})
      if(EXISTS "${search_path}")
        set(PYTHON_RUNTIME_LIBRARY_DIRS "${search_path}")
        break()
      endif()
    endforeach()
  else()
    set(PYTHON_RUNTIME_LIBRARY_DIRS "${PYTHON_LIB_DIR}")
  endif()

  set(PYTHON_RUNTIME_LIBRARY_DIRS
    "${PYTHON_RUNTIME_LIBRARY_DIRS}"
    PARENT_SCOPE
  )
  set(PYTHON_SITELIB
      "${PYTHON_SITELIB}"
      PARENT_SCOPE
  )

endfunction()


function(install_python_libraries DES_LIB_DIR)
    foreach(lib IN LISTS ARGN)
        if(EXISTS "${lib}")
            message(STATUS "Installing python dep: ${lib} to ${DES_LIB_DIR}")
            cloudViewer_install_ext(FILES "${lib}" "${DES_LIB_DIR}" "")
        else()
            message(WARNING "Library ${lib} does not exist.")
        endif()
    endforeach()
endfunction()

# Install Python directory with exclusions to reduce package size
# This function excludes __pycache__, *.pyc, *.pyo, test files, docs, etc.
# It reuses cloudViewer_install_ext for consistency
# Note: We avoid excluding directories that might be Python module names (e.g., "build", "test", "dist")
# to prevent breaking packages like pip which uses "build" as a module name
function(cloudViewer_install_python_dir SOURCE_DIR DEST_DIR)
    set(_extra_patterns "")
    if(${ARGC} GREATER 2)
        set(_extra_patterns ${ARGN})
    endif()

    # Define exclusion patterns for Python environment
    set(PYTHON_EXCLUSION_PATTERNS
        # Exclude Python cache files and directories (most important for size reduction)
        PATTERN "__pycache__" EXCLUDE
        PATTERN "*.pyc" EXCLUDE
        PATTERN "*.pyo" EXCLUDE
        # Note: *.pyd files are Windows extension modules, should be kept
        # Exclude test files (but not test directories/modules to avoid breaking packages)
        PATTERN "*_test.py" EXCLUDE
        PATTERN "*_tests.py" EXCLUDE
        PATTERN "test_*.py" EXCLUDE
        PATTERN "tests_*.py" EXCLUDE
        # Exclude example files (but not example directories that might be modules)
        PATTERN "*_example.py" EXCLUDE
        PATTERN "*_examples.py" EXCLUDE
        PATTERN "example_*.py" EXCLUDE
        PATTERN "examples_*.py" EXCLUDE
        # Exclude development and build artifacts (only files, not directories)
        PATTERN ".pytest_cache" EXCLUDE
        PATTERN ".mypy_cache" EXCLUDE
        PATTERN ".tox" EXCLUDE
        PATTERN ".coverage" EXCLUDE
        PATTERN "htmlcov" EXCLUDE
        PATTERN ".eggs" EXCLUDE
        PATTERN ".git" EXCLUDE
        PATTERN ".gitignore" EXCLUDE
        PATTERN ".github" EXCLUDE
        PATTERN ".travis.yml" EXCLUDE
        PATTERN ".appveyor.yml" EXCLUDE
        PATTERN ".coveragerc" EXCLUDE
        # Exclude IDE and editor files
        PATTERN ".vscode" EXCLUDE
        PATTERN ".idea" EXCLUDE
        PATTERN "*.swp" EXCLUDE
        PATTERN "*.swo" EXCLUDE
        PATTERN "*~" EXCLUDE
        PATTERN ".DS_Store" EXCLUDE
        # Note: We do NOT exclude "build", "dist", "test", "tests", "doc", "docs", 
        # "example", "examples" directories as they might be legitimate Python module names
        # (e.g., pip._internal.operations.build, some packages have test/ as a module)
        ${_extra_patterns}
    )

    # Reuse cloudViewer_install_ext with exclusion patterns
    cloudViewer_install_ext(DIRECTORY "${SOURCE_DIR}/" "${DEST_DIR}" "" "${PYTHON_EXCLUSION_PATTERNS}")
endfunction()

# Packages commonly present in dev pyenv/conda envs but not needed in the GUI plugin runtime.
function(cloudViewer_python_bloat_exclusion_patterns OUT_VAR)
    set(_patterns
        PATTERN "torch" EXCLUDE
        PATTERN "torchvision" EXCLUDE
        PATTERN "torchaudio" EXCLUDE
        PATTERN "tensorflow" EXCLUDE
        PATTERN "tensorboard" EXCLUDE
        PATTERN "keras" EXCLUDE
        PATTERN "jax" EXCLUDE
        PATTERN "jupyter" EXCLUDE
        PATTERN "jupyterlab" EXCLUDE
        PATTERN "notebook" EXCLUDE
        PATTERN "ipython" EXCLUDE
        PATTERN "scipy" EXCLUDE
        PATTERN "pandas" EXCLUDE
        PATTERN "matplotlib" EXCLUDE
        PATTERN "sklearn" EXCLUDE
        PATTERN "scikit_learn" EXCLUDE
        PATTERN "cv2" EXCLUDE
        PATTERN "opencv_python" EXCLUDE
        PATTERN "cloudViewer" EXCLUDE
        PATTERN "cloudviewer" EXCLUDE
        PATTERN "nvidia" EXCLUDE
        PATTERN "triton" EXCLUDE
    )
    set(${OUT_VAR} "${_patterns}" PARENT_SCOPE)
endfunction()

function(get_python_release_packages OUT_VAR)
    set(_req_file "${CMAKE_CURRENT_SOURCE_DIR}/requirements-release.txt")
    set(_packages "")
    if(EXISTS "${_req_file}")
        file(READ "${_req_file}" _content)
        string(REPLACE "\n" ";" _lines "${_content}")
        foreach(_line IN LISTS _lines)
            string(STRIP "${_line}" _line)
            if(_line STREQUAL "" OR _line MATCHES "^#")
                continue()
            endif()
            if(_line MATCHES "^([A-Za-z0-9_]+)")
                list(APPEND _packages "${CMAKE_MATCH_1}")
            endif()
        endforeach()
    endif()
    set(${OUT_VAR} "${_packages}" PARENT_SCOPE)
endfunction()

# Packages that must exist in the build Python env when minimal copy is enabled.
function(get_python_release_required_imports OUT_VAR)
    set(_imports numpy pip setuptools tqdm invoke typing_extensions)
    set(${OUT_VAR} "${_imports}" PARENT_SCOPE)
endfunction()

function(verify_python_release_packages_available)
    if(NOT PYTHON_EXECUTABLE)
        message(FATAL_ERROR
            "PLUGIN_PYTHON_COPY_MINIMAL_ENV=ON requires Python3_EXECUTABLE. "
            "Create a venv and install plugins/core/Standard/qPythonRuntime/requirements-release.txt")
    endif()
    get_python_release_required_imports(_required_imports)
    foreach(_mod IN LISTS _required_imports)
        execute_process(
            COMMAND "${PYTHON_EXECUTABLE}" -c "import ${_mod}"
            RESULT_VARIABLE _import_result
            ERROR_VARIABLE _import_error
        )
        if(NOT _import_result EQUAL 0)
            message(FATAL_ERROR
                "Python module '${_mod}' is missing from ${PYTHON_EXECUTABLE}. "
                "Install release deps before install:\n"
                "  python -m pip install -r plugins/core/Standard/qPythonRuntime/requirements-release.txt\n"
                "Details: ${_import_error}")
        endif()
    endforeach()
    message(STATUS "Python release packages verified for minimal env: ${_required_imports}")
endfunction()

function(copy_python_release_site_packages SOURCE_SITELIB DEST_SITELIB)
    get_python_release_packages(_pkgs)
    if(NOT _pkgs)
        message(FATAL_ERROR "requirements-release.txt is empty; cannot build minimal python env.")
    endif()

    set(_missing "")
    foreach(_pkg IN LISTS _pkgs)
        if(_pkg STREQUAL "pybind11")
            continue()
        endif()

        set(_found FALSE)
        set(_src_dir "${SOURCE_SITELIB}/${_pkg}")
        set(_src_py "${SOURCE_SITELIB}/${_pkg}.py")

        if(IS_DIRECTORY "${_src_dir}")
            message(STATUS "Installing minimal python package: ${_pkg}/")
            cloudViewer_install_python_dir("${_src_dir}" "${DEST_SITELIB}/${_pkg}")
            set(_found TRUE)
        elseif(EXISTS "${_src_py}")
            message(STATUS "Installing minimal python module: ${_pkg}.py")
            install(FILES "${_src_py}" DESTINATION "${DEST_SITELIB}")
            set(_found TRUE)
        endif()

        file(GLOB _dist_entries "${SOURCE_SITELIB}/${_pkg}-*.dist-info")
        foreach(_dist IN LISTS _dist_entries)
            get_filename_component(_dist_name "${_dist}" NAME)
            cloudViewer_install_python_dir("${_dist}" "${DEST_SITELIB}/${_dist_name}")
            set(_found TRUE)
        endforeach()

        if(NOT _found)
            if(_pkg STREQUAL "wheel")
                message(STATUS "Optional python package not installed (skipped): wheel (bundled with pip)")
            else()
                list(APPEND _missing "${_pkg}")
            endif()
        endif()
    endforeach()

    if(_missing)
        message(FATAL_ERROR
            "Missing site-packages for minimal python install: ${_missing}\n"
            "Run: python -m pip install -r plugins/core/Standard/qPythonRuntime/requirements-release.txt")
    endif()

    if(EXISTS "${SOURCE_SITELIB}/numpy.libs")
        cloudViewer_install_python_dir("${SOURCE_SITELIB}/numpy.libs" "${DEST_SITELIB}/numpy.libs")
    endif()
endfunction()

function(install_win_python_ssl_dlls INSTALL_DLL_DIR)
    file(GLOB LIBEXPAT_LIBS "${PYTHON_BASE_PREFIX}/Library/bin/libexpat*.dll")
    file(GLOB FFI_LIBS "${PYTHON_BASE_PREFIX}/Library/bin/ffi*.dll")
    install_python_libraries("${INSTALL_DLL_DIR}" ${LIBEXPAT_LIBS})
    install_python_libraries("${INSTALL_DLL_DIR}" ${FFI_LIBS})
    file(GLOB LIBSSL_LIBS "${PYTHON_BASE_PREFIX}/Library/bin/libssl*.dll")
    file(GLOB LIBCRYPTO_LIBS "${PYTHON_BASE_PREFIX}/Library/bin/libcrypto*.dll")
    install_python_libraries("${INSTALL_DLL_DIR}" ${LIBSSL_LIBS})
    install_python_libraries("${INSTALL_DLL_DIR}" ${LIBCRYPTO_LIBS})
endfunction()

function(install_linux_python_ssl_libs INSTALL_PYTHON_LIB_PATH)
    file(GLOB LIBSSL_LIBS "${PYTHON_RUNTIME_LIBRARY_DIRS}/libssl.so*")
    file(GLOB LIBCRYPTO_LIBS "${PYTHON_RUNTIME_LIBRARY_DIRS}/libcrypto.so*")
    foreach(lib ${LIBSSL_LIBS} ${LIBCRYPTO_LIBS})
        if(EXISTS ${lib})
            message(STATUS "Installing ${lib}")
            cloudViewer_install_ext(FILES "${lib}" "${INSTALL_PYTHON_LIB_PATH}/" "")
        endif()
    endforeach()
endfunction()

function(install_linux_python_binary INSTALL_DIR)
    execute_process(
        COMMAND "${PYTHON_EXECUTABLE}" "-c" "import sys; print(sys.executable)"
        OUTPUT_VARIABLE REAL_PYTHON_EXECUTABLE
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(EXISTS "${REAL_PYTHON_EXECUTABLE}")
        message(STATUS "Installing real Python executable: ${REAL_PYTHON_EXECUTABLE}")
        install(FILES "${REAL_PYTHON_EXECUTABLE}" DESTINATION "${INSTALL_DIR}/bin" RENAME "python"
                PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
        )
    else()
        message(WARNING "Real Python executable not found, falling back to: ${PYTHON_EXECUTABLE}")
        install(FILES "${PYTHON_EXECUTABLE}" DESTINATION "${INSTALL_DIR}/bin" RENAME "python"
                PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
        )
    endif()
endfunction()

function(install_linux_python_site_packages_symlink INSTALL_PYTHON_LIB_PATH)
    set(SOURCE_RELATIVE_SITELIB "${PYTHON_NAME}/site-packages")
    set(TARGET_RELATIVE_SITELIB "site-packages")
    install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${SOURCE_RELATIVE_SITELIB} ${TARGET_RELATIVE_SITELIB}
            WORKING_DIRECTORY ${INSTALL_PYTHON_LIB_PATH})")
endfunction()

function(copy_linux_python_env_minimal INSTALL_DIR)
    set(INSTALL_PYTHON_LIB_PATH "${INSTALL_DIR}/${PYTHON_LIB_NAME}")
    set(INSTALL_PYVER_DIR "${INSTALL_PYTHON_LIB_PATH}/${PYTHON_NAME}")
    message(
        STATUS
        "Minimal python env copy (stdlib + requirements-release.txt only):
            PYTHON_NAME:        ${PYTHON_NAME}
            PYTHON_SITELIB:     ${PYTHON_SITELIB}
            INSTALL_DIR:        ${INSTALL_DIR}"
    )

    cloudViewer_install_python_dir(
        "${PYTHON_VERSION_DIR}/"
        "${INSTALL_PYVER_DIR}/"
        PATTERN "site-packages" EXCLUDE
    )

    copy_python_release_site_packages(
        "${PYTHON_SITELIB}"
        "${INSTALL_PYVER_DIR}/site-packages"
    )

    file(GLOB PYTHON_SO "${PYTHON_RUNTIME_LIBRARY_DIRS}/libpython*.so*")
    install_python_libraries("${INSTALL_PYTHON_LIB_PATH}/" ${PYTHON_SO})

    install_linux_python_ssl_libs("${INSTALL_PYTHON_LIB_PATH}/")
    install_linux_python_binary("${INSTALL_DIR}")
    install_linux_python_site_packages_symlink("${INSTALL_PYTHON_LIB_PATH}")
endfunction()

function(copy_win_python_env_minimal INSTALL_DIR)
    set(INSTALL_PYTHON_LIB "${INSTALL_DIR}/${PYTHON_LIB_NAME}")
    message(
        STATUS
        "Minimal python env copy (Lib stdlib + requirements-release.txt only):
            PYTHON_BASE_PREFIX: ${PYTHON_BASE_PREFIX}
            PYTHON_SITELIB:     ${PYTHON_SITELIB}"
    )

    message(STATUS "COPYING python exe from ${PYTHON_EXECUTABLE} to ${INSTALL_DIR}")
    cloudViewer_install_ext(FILES "${PYTHON_EXECUTABLE}" "${INSTALL_DIR}" "" )

    cloudViewer_install_python_dir(
        "${PYTHON_BASE_PREFIX}/Lib/"
        "${INSTALL_PYTHON_LIB}/"
        PATTERN "site-packages" EXCLUDE
    )

    copy_python_release_site_packages(
        "${PYTHON_SITELIB}"
        "${INSTALL_PYTHON_LIB}/site-packages"
    )

    if(EXISTS "${PYTHON_BASE_PREFIX}/DLLs")
        cloudViewer_install_ext(DIRECTORY "${PYTHON_BASE_PREFIX}/DLLs" "${INSTALL_DIR}/" "")
    endif()
    install_win_python_ssl_dlls("${INSTALL_DIR}/DLLs")
endfunction()

function(copy_win_python_env INSTALL_DIR)
  set(INSTALL_PYTHON_LIB "${CC_PYTHON_INSTALL_DIR}/${PYTHON_LIB_NAME}")
  cloudViewer_python_bloat_exclusion_patterns(_bloat_patterns)
  message(
    STATUS
    "Full python env copy (Lib + filtered site-packages):
        PYTHON_BASE_PREFIX:     ${PYTHON_BASE_PREFIX}
        PYTHON_RUNTIME_LIBRARY_DIRS: ${PYTHON_RUNTIME_LIBRARY_DIRS}"
  )
  message(STATUS "COPYING python exe from ${PYTHON_EXECUTABLE} to ${INSTALL_DIR}")
  cloudViewer_install_ext(FILES "${PYTHON_EXECUTABLE}" "${INSTALL_DIR}" "" )

  cloudViewer_install_python_dir(
      "${PYTHON_BASE_PREFIX}/Lib/"
      "${INSTALL_PYTHON_LIB}/"
      PATTERN "site-packages" EXCLUDE
  )
  cloudViewer_install_python_dir(
      "${PYTHON_SITELIB}/"
      "${INSTALL_PYTHON_LIB}/site-packages/"
      ${_bloat_patterns}
  )

  if(EXISTS "${PYTHON_BASE_PREFIX}/DLLs")
    message(STATUS "COPYING DLLs from ${PYTHON_BASE_PREFIX}/DLLs to ${CC_PYTHON_INSTALL_DIR}/")
    cloudViewer_install_ext(DIRECTORY "${PYTHON_BASE_PREFIX}/DLLs" "${CC_PYTHON_INSTALL_DIR}/" "")
  endif()
  # fix pip dependency
  file(GLOB LIBEXPAT_LIBS "${PYTHON_BASE_PREFIX}/Library/bin/libexpat*.dll")
  file(GLOB FFI_LIBS "${PYTHON_BASE_PREFIX}/Library/bin/ffi*.dll")
  install_python_libraries("${CC_PYTHON_INSTALL_DIR}/DLLs" ${LIBEXPAT_LIBS})
  install_python_libraries("${CC_PYTHON_INSTALL_DIR}/DLLs" ${FFI_LIBS})
  install_win_python_ssl_dlls("${CC_PYTHON_INSTALL_DIR}/DLLs")
endfunction()

function(copy_linux_python_env INSTALL_DIR)
  set(INSTALL_PYTHON_LIB_PATH "${INSTALL_DIR}/${PYTHON_LIB_NAME}")
  cloudViewer_python_bloat_exclusion_patterns(_bloat_patterns)
  message(
    STATUS
    "Full python env copy (stdlib + filtered site-packages):
        PYTHON_NAME:            ${PYTHON_NAME}
        PYTHON_LIB_NAME:        ${PYTHON_LIB_NAME}
        PYTHON_BASE_PREFIX:     ${PYTHON_BASE_PREFIX}
        PYTHON_VERSION_DIR:     ${PYTHON_VERSION_DIR}
        PYTHON_SITELIB:         ${PYTHON_SITELIB}
        PYTHON_LIBRARIES:       ${PYTHON_RUNTIME_LIBRARY_DIRS}"
  )

  cloudViewer_install_python_dir(
      "${PYTHON_VERSION_DIR}/"
      "${INSTALL_PYTHON_LIB_PATH}/${PYTHON_NAME}/"
      PATTERN "site-packages" EXCLUDE
  )

  cloudViewer_install_python_dir(
      "${PYTHON_SITELIB}/"
      "${INSTALL_PYTHON_LIB_PATH}/${PYTHON_NAME}/site-packages/"
      ${_bloat_patterns}
  )

  install_linux_python_ssl_libs("${INSTALL_PYTHON_LIB_PATH}/")
  install_linux_python_binary("${INSTALL_DIR}")
  install_linux_python_site_packages_symlink("${INSTALL_PYTHON_LIB_PATH}")
endfunction()

function(copy_python_dll)
  message(
    STATUS
    "Python DLL: = ${PYTHON_BASE_PREFIX}/python${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}.dll"
  )
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Debug build")
    install(
      FILES "${PYTHON_BASE_PREFIX}/python${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}_d.dll"
      # install the python3 base dll as well because some libs will try to
      # find it (PySide and PyQT for example)
      "${PYTHON_BASE_PREFIX}/python${Python_VERSION_MAJOR}_d.dll"
      DESTINATION ${CC_PYTHON_INSTALL_DIR}
    )
  else()
    install(
      FILES "${PYTHON_BASE_PREFIX}/python${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}.dll"
      # install the python3 base dll as well because some libs will try to
      # find it (PySide and PyQT for example)
      "${PYTHON_BASE_PREFIX}/python${PYTHON_VERSION_MAJOR}.dll"
      DESTINATION ${CC_PYTHON_INSTALL_DIR}
    )
  endif()
endfunction()

function(manage_windows_install)

  setup_python_env()
  set(INSTALL_PYTHON_SITELIB "${CC_PYTHON_INSTALL_DIR}/${PYTHON_LIB_NAME}/site-packages")

  if(PLUGIN_PYTHON_COPY_MINIMAL_ENV AND NOT PLUGIN_PYTHON_COPY_ENV)
    verify_python_release_packages_available()
  endif()

  if(PLUGIN_PYTHON_COPY_ENV)
    copy_win_python_env(${CC_PYTHON_INSTALL_DIR})
  elseif(PLUGIN_PYTHON_COPY_MINIMAL_ENV)
    copy_win_python_env_minimal(${CC_PYTHON_INSTALL_DIR})
  endif()

  copy_python_dll()

  install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/docs/stubfiles/pycc.pyi"
                "${CMAKE_CURRENT_SOURCE_DIR}/docs/stubfiles/cccorelib.pyi"
          DESTINATION "${INSTALL_PYTHON_SITELIB}"
  )

  if(NOT PLUGIN_PYTHON_USE_EMBEDDED_MODULES)
    install(TARGETS pycc cccorelib
            DESTINATION "${INSTALL_PYTHON_SITELIB}"
    )
  endif()
endfunction()

function(manage_linux_install)

  setup_python_env()
  set(INSTALL_PYTHON_SITELIB "${CC_PYTHON_INSTALL_DIR}/${PYTHON_LIB_NAME}/site-packages")

  if(PLUGIN_PYTHON_COPY_MINIMAL_ENV AND NOT PLUGIN_PYTHON_COPY_ENV)
    verify_python_release_packages_available()
  endif()

  if(PLUGIN_PYTHON_COPY_ENV)
    copy_linux_python_env(${CC_PYTHON_INSTALL_DIR})
  elseif(PLUGIN_PYTHON_COPY_MINIMAL_ENV)
    copy_linux_python_env_minimal(${CC_PYTHON_INSTALL_DIR})
  endif()

  install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/docs/stubfiles/pycc.pyi"
                "${CMAKE_CURRENT_SOURCE_DIR}/docs/stubfiles/cccorelib.pyi"
          DESTINATION "${INSTALL_PYTHON_SITELIB}"
  )

  if(NOT PLUGIN_PYTHON_USE_EMBEDDED_MODULES)
    install(TARGETS pycc cccorelib
            DESTINATION "${INSTALL_PYTHON_SITELIB}"
    )
  endif()
endfunction()

function(run_windeployqt TARGET_NAME FILE_PATH)
  # Qt5/Qt6 compatibility
  # Note: Core is already found in CMakeExternalLibs.cmake
  # Use unified Qt:: prefix
  get_target_property(QMAKE_EXE Qt::qmake IMPORTED_LOCATION)
  
  get_filename_component(QT_BIN_DIR "${QMAKE_EXE}" DIRECTORY)

  find_program(WINDEPLOYQT_EXECUTABLE windeployqt HINTS "${QT_BIN_DIR}")

  add_custom_command(
    TARGET ${TARGET_NAME}
    POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E make_directory deployqt
    COMMAND "${WINDEPLOYQT_EXECUTABLE}" "${FILE_PATH}" --dir
            "${CMAKE_CURRENT_BINARY_DIR}/deployqt"
  )
endfunction()
