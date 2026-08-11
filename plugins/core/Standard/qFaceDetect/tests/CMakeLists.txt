if(NOT TARGET Qt::Core)
    find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Core Gui Sql Test)
endif()

add_executable(test_qfacedetect_embed_helpers test_embed_helpers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/FaceDetectEmbedHelpers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/FaceRegistryStore.cpp
)

target_include_directories(test_qfacedetect_embed_helpers PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/core/AICore/include
)

find_package(Threads REQUIRED)

target_link_libraries(test_qfacedetect_embed_helpers PRIVATE
    CVCoreLib
    AICore
    Qt${QT_VERSION_MAJOR}::Core
    Qt${QT_VERSION_MAJOR}::Gui
    Qt${QT_VERSION_MAJOR}::Sql
    Threads::Threads
)

if(NOT MSVC)
    target_compile_options(test_qfacedetect_embed_helpers PRIVATE -Wall -Wextra -fPIC)
    set_target_properties(test_qfacedetect_embed_helpers PROPERTIES
        POSITION_INDEPENDENT_CODE ON)
    target_link_options(test_qfacedetect_embed_helpers PRIVATE "-no-pie")
endif()

cloudviewer_set_bin_runtime_output_directory(test_qfacedetect_embed_helpers)
add_test(NAME test_qfacedetect_embed_helpers COMMAND test_qfacedetect_embed_helpers)
