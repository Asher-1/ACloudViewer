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

  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" "-c"
            "import site; print(site.getsitepackages()[0])"
    OUTPUT_VARIABLE PYTHON_SITELIB
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  get_filename_component(PARENT_DIR "${PYTHON_SITELIB}" DIRECTORY)
  get_filename_component(PARENT_DIR_ABSOLUTE "${PARENT_DIR}" ABSOLUTE)
  get_filename_component(PYTHON_NAME "${PARENT_DIR}" NAME)
  set(PYTHON_NAME
    "${PYTHON_NAME}"
    PARENT_SCOPE
  )
  set(Python_STDARCH
    "${PARENT_DIR_ABSOLUTE}"
    PARENT_SCOPE
  )
  set(PYTHON_SITELIB
      "${PYTHON_SITELIB}"
      PARENT_SCOPE
  )

endfunction()

function(copy_python_env INSTALL_DIR LIB_NAME)
  set(DEPLOY_PYTHON_SITELIB "${INSTALL_DIR}/${LIB_NAME}/site-packages")
  message(
    STATUS
    "ENV copy:
        PYTHON_NAME:            ${PYTHON_NAME}
        PYTHON_BASE_PREFIX:     ${PYTHON_BASE_PREFIX}
        PYTHON_SITELIB:         ${PYTHON_SITELIB}
        DEPLOY_PYTHON_SITELIB:  ${DEPLOY_PYTHON_SITELIB}
        PYTHON_LIBRARIES:       ${Python_RUNTIME_LIBRARY_DIRS}"
  )
  message(DEBUG "COPYING venv from ${PYTHON_BASE_PREFIX}/ to ${INSTALL_DIR}")
  cloudViewer_install_ext( DIRECTORY "${PYTHON_BASE_PREFIX}/" "${INSTALL_DIR}" "" )
  if (NOT WIN32)
    set(SOURCE_RELATIVE_SITELIB "${PYTHON_NAME}/site-packages")
    set(TARGET_RELATIVE_SITELIB "site-packages")
    install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${SOURCE_RELATIVE_SITELIB} ${TARGET_RELATIVE_SITELIB}
            WORKING_DIRECTORY ${INSTALL_DIR}/${LIB_NAME})")
  endif()

  set(DEPLOY_PYTHON_SITELIB "${DEPLOY_PYTHON_SITELIB}" PARENT_SCOPE)

  # message(DEBUG "COPYING venv from ${PYTHON_SITELIB}/ to ${INSTALL_DIR}/${LIB_NAME}/site-packages/")
  # cloudViewer_install_ext( DIRECTORY "${PYTHON_SITELIB}/" "${INSTALL_DIR}/${LIB_NAME}/site-packages/" "" )
endfunction()

function(copy_python_dll)
  message(
    DEBUG
    "Python DLL: = ${Python_RUNTIME_LIBRARY_DIRS}/python${Python_VERSION_MAJOR}${Python_VERSION_MINOR}.dll"
  )
  install(
    FILES "${Python_RUNTIME_LIBRARY_DIRS}/python${Python_VERSION_MAJOR}${Python_VERSION_MINOR}.dll"
          # install the python3 base dll as well because some libs will try to
          # find it (PySide and PyQT for example)
          "${Python_RUNTIME_LIBRARY_DIRS}/python${Python_VERSION_MAJOR}.dll"
    DESTINATION ${ACloudViewer_DEST_FOLDER}
  )
endfunction()

function(manage_windows_install)

  setup_python_env()

  if(PLUGIN_PYTHON_COPY_ENV)
    copy_python_env(${CC_PYTHON_INSTALL_DIR} "Lib")
  else ()
    set(DEPLOY_PYTHON_SITELIB "${CC_PYTHON_INSTALL_DIR}/Lib/site-packages")
  endif()

  copy_python_dll()

  install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/docs/stubfiles/pycc.pyi"
                "${CMAKE_CURRENT_SOURCE_DIR}/docs/stubfiles/cccorelib.pyi"
          DESTINATION "${DEPLOY_PYTHON_SITELIB}"
  )

  if(NOT PLUGIN_PYTHON_USE_EMBEDDED_MODULES)
    install(TARGETS pycc cccorelib
            DESTINATION "${DEPLOY_PYTHON_SITELIB}"
    )
  endif()
endfunction()

function(manage_linux_install)

  setup_python_env()

  if(PLUGIN_PYTHON_COPY_ENV)
    copy_python_env(${CC_PYTHON_INSTALL_DIR} "lib")
  else ()
    set(DEPLOY_PYTHON_SITELIB "${CC_PYTHON_INSTALL_DIR}/lib/site-packages")
  endif()

  install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/docs/stubfiles/pycc.pyi"
                "${CMAKE_CURRENT_SOURCE_DIR}/docs/stubfiles/cccorelib.pyi"
          DESTINATION "${DEPLOY_PYTHON_SITELIB}"
  )

  if(NOT PLUGIN_PYTHON_USE_EMBEDDED_MODULES)
    install(TARGETS pycc cccorelib
            DESTINATION "${DEPLOY_PYTHON_SITELIB}"
    )
  endif()
endfunction()

function(run_windeployqt TARGET_NAME FILE_PATH)
  # Force finding Qt5 to have the Qt5::qmake thing later
  find_package(
    Qt5
    COMPONENTS Core
    REQUIRED
  )

  get_target_property(QMAKE_EXE Qt5::qmake IMPORTED_LOCATION)
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
