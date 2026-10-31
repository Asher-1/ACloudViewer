if(BUILD_UNIT_TESTS)
  add_executable(test_qpythonruntime_binding_types
    test_binding_types.cpp test_binding_types_consumer.cpp)
  set_target_properties(test_qpythonruntime_binding_types PROPERTIES
    AUTOMOC OFF AUTOUIC OFF AUTORCC OFF POSITION_INDEPENDENT_CODE ON)
  qpythonruntime_configure_bindings(test_qpythonruntime_binding_types)
  if(UNIX AND NOT APPLE)
    # Qt reduce-relocations requires PIC without disabling executable PIE protection.
    target_compile_options(test_qpythonruntime_binding_types PRIVATE -fPIC)
  endif()
  target_link_libraries(test_qpythonruntime_binding_types PRIVATE
    pybind11::embed CVCoreLib CV_DB_LIB Qt${QT_VERSION_MAJOR}::Core)
  target_include_directories(test_qpythonruntime_binding_types PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../wrapper/cccorelib/src)
  cloudviewer_set_bin_runtime_output_directory(test_qpythonruntime_binding_types)
  add_test(NAME test_qpythonruntime_binding_types COMMAND test_qpythonruntime_binding_types)
  cloudviewer_set_plugin_test_env(test_qpythonruntime_binding_types)

  add_executable(test_qpythonruntime_plugin_loading test_plugin_loading.cpp)
  set_target_properties(test_qpythonruntime_plugin_loading PROPERTIES
    AUTOMOC OFF AUTOUIC OFF AUTORCC OFF POSITION_INDEPENDENT_CODE ON)
  if(UNIX AND NOT APPLE)
    target_compile_options(test_qpythonruntime_plugin_loading PRIVATE -fPIC)
  endif()
  target_link_libraries(test_qpythonruntime_plugin_loading PRIVATE
    pybind11::embed CVPluginAPI CVPluginStub Qt::Widgets)
  cloudviewer_set_bin_runtime_output_directory(test_qpythonruntime_plugin_loading)
  add_test(NAME test_qpythonruntime_plugin_loading
    COMMAND test_qpythonruntime_plugin_loading $<TARGET_FILE:QPYTHONRUNTIME_PLUGIN>)
  cloudviewer_set_plugin_test_env(test_qpythonruntime_plugin_loading)
  set_property(TEST test_qpythonruntime_plugin_loading APPEND PROPERTY
    ENVIRONMENT "QT_QPA_PLATFORM=offscreen")
  set_tests_properties(test_qpythonruntime_plugin_loading PROPERTIES TIMEOUT 60)
  add_test(NAME test_qpythonruntime_binding_contracts
    COMMAND ${PYTHON_EXECUTABLE} -m pytest -p no:cacheprovider -q
      --acloudviewer_exe=$<TARGET_FILE:ACloudViewer>
      ${CMAKE_CURRENT_LIST_DIR})
  cloudviewer_set_plugin_test_env(test_qpythonruntime_binding_contracts)
  set_tests_properties(test_qpythonruntime_binding_contracts PROPERTIES
    TIMEOUT 180 RUN_SERIAL TRUE)

  # Source-identity attribution needs a real libclang tokenizer. It is a
  # provenance check, not a runtime binding contract, so it lives outside the
  # pytest-collected tests directory and is only registered when libclang exists.
  file(GLOB _qpythonruntime_llvm_lib_dirs LIST_DIRECTORIES true /usr/lib/llvm-*/lib)
  find_library(QPYTHONRUNTIME_LIBCLANG NAMES clang libclang.so.1
    libclang-18.so.1 libclang-17.so.1 libclang-14.so.1 libclang-10.so.1
    HINTS ${_qpythonruntime_llvm_lib_dirs})
  if(QPYTHONRUNTIME_LIBCLANG)
    add_test(NAME test_qpythonruntime_source_identity
      COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/../tools/test_source_identity.py)
    set_tests_properties(test_qpythonruntime_source_identity PROPERTIES
      ENVIRONMENT "ACV_LIBCLANG=${QPYTHONRUNTIME_LIBCLANG}"
      LABELS "fast;provenance" TIMEOUT 120)
  endif()
endif()

if(WIN32 OR APPLE)
  add_custom_target(
    pytest
    COMMAND ${PYTHON_EXECUTABLE} -m pytest --verbose --acloudviewer_exe
            $<TARGET_FILE:ACloudViewer> ${CMAKE_CURRENT_LIST_DIR}
  )
else()
  add_custom_target(
    pytest
    COMMAND
      ${CMAKE_COMMAND} -E env CC_PLUGIN_PATH=${CC_PLUGIN_PATH_VALUE}
      ${PYTHON_EXECUTABLE} -m pytest --verbose --acloudviewer_exe
      $<TARGET_FILE:ACloudViewer> ${CMAKE_CURRENT_LIST_DIR}
  )
endif()
