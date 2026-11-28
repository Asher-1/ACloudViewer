# Keep binary object/metadata regressions with their serialization owner.
add_executable(test_object_serialization test_object_serialization.cpp)
target_link_libraries(test_object_serialization PRIVATE CV_DB_LIB gtest_main)
if(NOT MSVC)
    target_compile_options(test_object_serialization PRIVATE -fPIC)
endif()
cloudviewer_add_test(NAME test_object_serialization
    COMMAND test_object_serialization LABELS "unit;db;serialization")
cloudviewer_set_test_runtime_environment(test_object_serialization)
set_tests_properties(test_object_serialization PROPERTIES TIMEOUT 30)
