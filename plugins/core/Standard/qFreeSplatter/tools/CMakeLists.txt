cmake_minimum_required(VERSION ${CLOUDVIEWER_PLUGIN_CMAKE_MINIMUM_VERSION})

if(NOT TARGET 3rdparty_opencv)
    message(WARNING "qFreeSplatter tools: 3rdparty_opencv not available — skipping free_splatter-cli. "
                    "Enable BUILD_OPENCV=ON and re-run CMake.")
    return()
endif()

add_executable(free_splatter-cli
    free_splatter-cli.cpp
    free_splatter_cli_api.cpp
)

set_target_properties(free_splatter-cli PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

target_include_directories(free_splatter-cli PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(free_splatter-cli PRIVATE AICore 3rdparty_opencv)

if(DEFINED OpenCV_INCLUDE_DIRS)
    target_include_directories(free_splatter-cli PRIVATE ${OpenCV_INCLUDE_DIRS})
endif()

if(NOT MSVC)
    target_compile_options(free_splatter-cli PRIVATE -Wall -Wextra)
endif()

message(STATUS "qFreeSplatter: building free_splatter-cli (OpenCV image I/O)")
