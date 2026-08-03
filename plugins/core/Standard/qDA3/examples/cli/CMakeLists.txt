add_executable(da3-cli main.cpp)
cloudviewer_set_bin_runtime_output_directory(da3-cli)
target_link_libraries(da3-cli PRIVATE depthanything)
target_include_directories(da3-cli PRIVATE ${DA_ROOT}/src ${DA_ROOT}/include)
