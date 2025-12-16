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


function(install_python_libraries LIBS DES_LIB_DIR)
    foreach(lib ${LIBS})
        if(EXISTS ${lib})
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
    )

    # Reuse cloudViewer_install_ext with exclusion patterns
    cloudViewer_install_ext(DIRECTORY "${SOURCE_DIR}/" "${DEST_DIR}" "" "${PYTHON_EXCLUSION_PATTERNS}")
endfunction()

function(copy_win_python_env INSTALL_DIR)
  set(INSTALL_PYTHON_LIB "${CC_PYTHON_INSTALL_DIR}/${PYTHON_LIB_NAME}")
  message(
    STATUS
    "ENV copy:
        PYTHON_BASE_PREFIX:     ${PYTHON_BASE_PREFIX}
        PYTHON_RUNTIME_LIBRARY_DIRS:         ${PYTHON_RUNTIME_LIBRARY_DIRS}"
  )
  message(STATUS "COPYING python exe from ${PYTHON_EXECUTABLE} to ${INSTALL_DIR}")
  cloudViewer_install_ext( FILES "${PYTHON_EXECUTABLE}" "${INSTALL_DIR}" "" )
  message(STATUS "COPYING site-packages from ${PYTHON_RUNTIME_LIBRARY_DIRS}/ to ${INSTALL_PYTHON_LIB}/ (excluding cache and unnecessary files)")
  cloudViewer_install_python_dir( "${PYTHON_RUNTIME_LIBRARY_DIRS}/" "${INSTALL_PYTHON_LIB}/" )
  message(STATUS "COPYING site-packages from ${PYTHON_BASE_PREFIX}/DLLs to ${CC_PYTHON_INSTALL_DIR}/")
  cloudViewer_install_ext( DIRECTORY "${PYTHON_BASE_PREFIX}/DLLs" "${CC_PYTHON_INSTALL_DIR}/" "" )
  # fix pip dependency
  file(GLOB LIBEXPAT_LIBS "${PYTHON_BASE_PREFIX}/Library/bin/libexpat*.dll")
  file(GLOB FFI_LIBS "${PYTHON_BASE_PREFIX}/Library/bin/ffi*.dll")
  install_python_libraries(${LIBEXPAT_LIBS} "${CC_PYTHON_INSTALL_DIR}/DLLs")
  install_python_libraries(${FFI_LIBS} "${CC_PYTHON_INSTALL_DIR}/DLLs")
  # fix pip install issues with missing libssl*.dll and libcrypto*.dll
  file(GLOB LIBSSL_LIBS "${PYTHON_BASE_PREFIX}/Library/bin/libssl*.dll")
  file(GLOB LIBCRYPTO_LIBS "${PYTHON_BASE_PREFIX}/Library/bin/libcrypto*.dll")
  install_python_libraries(${LIBSSL_LIBS} "${CC_PYTHON_INSTALL_DIR}/DLLs")
  install_python_libraries(${LIBCRYPTO_LIBS} "${CC_PYTHON_INSTALL_DIR}/DLLs")
endfunction()

function(copy_linux_python_env INSTALL_DIR)
  set(INSTALL_PYTHON_LIB_PATH "${INSTALL_DIR}/${PYTHON_LIB_NAME}")
  message(
    STATUS
    "ENV copy:
        PYTHON_NAME:            ${PYTHON_NAME}
        PYTHON_LIB_NAME:        ${PYTHON_LIB_NAME}
        PYTHON_BASE_PREFIX:     ${PYTHON_BASE_PREFIX}
        PYTHON_VERSION_DIR:     ${PYTHON_VERSION_DIR}
        PYTHON_SITELIB:         ${PYTHON_SITELIB}
        PYTHON_LIBRARIES:       ${PYTHON_RUNTIME_LIBRARY_DIRS}"
  )
  message(STATUS "COPYING python env from ${PYTHON_VERSION_DIR}/ to ${INSTALL_PYTHON_LIB_PATH}/${PYTHON_NAME}/ (excluding cache and unnecessary files)")
  cloudViewer_install_python_dir( "${PYTHON_VERSION_DIR}/" "${INSTALL_PYTHON_LIB_PATH}/${PYTHON_NAME}/" )
  # fix pip install issues with missing libssl.so* and libcrypto.so*
  file(GLOB LIBSSL_LIBS "${PYTHON_RUNTIME_LIBRARY_DIRS}/libssl.so*")
  file(GLOB LIBCRYPTO_LIBS "${PYTHON_RUNTIME_LIBRARY_DIRS}/libcrypto.so*")
  foreach(lib ${LIBSSL_LIBS})
      if(EXISTS ${lib})
          message(STATUS "Installing ${lib}")
          cloudViewer_install_ext( FILES "${lib}" "${INSTALL_PYTHON_LIB_PATH}/" "" )
      else()
          message(WARNING "Library ${lib} does not exist.")
      endif()
  endforeach()
  foreach(lib ${LIBCRYPTO_LIBS})
      if(EXISTS ${lib})
          message(STATUS "Installing ${lib}")
          cloudViewer_install_ext( FILES "${lib}" "${INSTALL_PYTHON_LIB_PATH}/" "" )
      else()
          message(WARNING "Library ${lib} does not exist.")
      endif()
  endforeach()

  # install python executable and symlink site-packages
  # For pyenv environments, PYTHON_EXECUTABLE might be a shim, we need the real binary
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

  # Create symlink for site-packages for easier access
  set(SOURCE_RELATIVE_SITELIB "${PYTHON_NAME}/site-packages")
  set(TARGET_RELATIVE_SITELIB "site-packages")
  install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${SOURCE_RELATIVE_SITELIB} ${TARGET_RELATIVE_SITELIB}
          WORKING_DIRECTORY ${INSTALL_PYTHON_LIB_PATH})")
endfunction()

function(copy_python_dll)
  message(
    STATUS
    "Python DLL: = ${PYTHON_BASE_PREFIX}/python${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}.dll"
  )
  install(
    FILES "${PYTHON_BASE_PREFIX}/python${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}.dll"
          # install the python3 base dll as well because some libs will try to
          # find it (PySide and PyQT for example)
          "${PYTHON_BASE_PREFIX}/python${PYTHON_VERSION_MAJOR}.dll"
    DESTINATION ${CC_PYTHON_INSTALL_DIR}
  )
endfunction()

function(manage_windows_install)

  setup_python_env()
  set(INSTALL_PYTHON_SITELIB "${CC_PYTHON_INSTALL_DIR}/${PYTHON_LIB_NAME}/site-packages")

  if(PLUGIN_PYTHON_COPY_ENV)
    copy_win_python_env(${CC_PYTHON_INSTALL_DIR})
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

  if(PLUGIN_PYTHON_COPY_ENV)
    copy_linux_python_env(${CC_PYTHON_INSTALL_DIR})
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
