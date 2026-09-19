if(NOT TARGET 3rdparty_googletest)
    message(STATUS "qYOLO tests: googletest not available — skipping")
    return()
endif()

add_executable(test_qyolo_helpers test_yolo_helpers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/YOLOModelCatalog.cpp
)

target_include_directories(test_qyolo_helpers PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/core/AICore/include
)

find_package(Threads REQUIRED)

target_link_libraries(test_qyolo_helpers PRIVATE
    CVCoreLib
    AICore
    3rdparty_googletest
    Qt${QT_VERSION_MAJOR}::Core
    Qt${QT_VERSION_MAJOR}::Gui
    Qt${QT_VERSION_MAJOR}::Sql
    Threads::Threads
)

if(NOT MSVC)
    target_compile_options(test_qyolo_helpers PRIVATE -Wall -Wextra -fPIC)
    set_target_properties(test_qyolo_helpers PROPERTIES
        POSITION_INDEPENDENT_CODE ON)
    # Apple ld64 warns "argument unused" for -no-pie; keep it GNU-only.
    if(UNIX AND NOT APPLE)
        target_link_options(test_qyolo_helpers PRIVATE "-no-pie")
    endif()
endif()

cloudviewer_set_bin_runtime_output_directory(test_qyolo_helpers)
add_test(NAME test_qyolo_helpers COMMAND test_qyolo_helpers)

# Multi-object tracker port: pure algorithm tests (no AICore, no Qt).
# GMC alignment gate: with OpenCV every method is the same cv2 call as
# utils/gmc.py; without it the self-contained sparseOptFlow fallback is
# exercised and orb/sift/ecc rejection is asserted.
add_executable(test_qyolo_tracker test_qyolo_tracker.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/tracking/tracker.cpp
)

target_include_directories(test_qyolo_tracker PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(test_qyolo_tracker PRIVATE
    3rdparty_googletest
    Threads::Threads
)

if(TARGET 3rdparty_opencv)
    target_link_libraries(test_qyolo_tracker PRIVATE 3rdparty_opencv)
    target_compile_definitions(test_qyolo_tracker PRIVATE QYOLO_WITH_OPENCV)
endif()

if(NOT MSVC)
    target_compile_options(test_qyolo_tracker PRIVATE -Wall -Wextra -fPIC)
    set_target_properties(test_qyolo_tracker PROPERTIES
        POSITION_INDEPENDENT_CODE ON)
    if(UNIX AND NOT APPLE)
        target_link_options(test_qyolo_tracker PRIVATE "-no-pie")
    endif()
endif()

cloudviewer_set_bin_runtime_output_directory(test_qyolo_tracker)
add_test(NAME test_qyolo_tracker COMMAND test_qyolo_tracker)

# Track-parity harness (official-runtime field-by-field comparison; driven by
# tests/track_parity_check.py, not a ctest).
add_executable(track_parity_harness track_parity_harness.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/tracking/tracker.cpp)
target_include_directories(track_parity_harness PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(track_parity_harness PRIVATE 3rdparty_googletest
    Threads::Threads)
if(TARGET 3rdparty_opencv)
    target_link_libraries(track_parity_harness PRIVATE 3rdparty_opencv)
    target_compile_definitions(track_parity_harness PRIVATE QYOLO_WITH_OPENCV)
endif()
if(NOT MSVC)
    target_compile_options(track_parity_harness PRIVATE -Wall -Wextra -fPIC)
    set_target_properties(track_parity_harness PROPERTIES
        POSITION_INDEPENDENT_CODE ON)
    if(UNIX AND NOT APPLE)
        target_link_options(track_parity_harness PRIVATE "-no-pie")
    endif()
endif()
cloudviewer_set_bin_runtime_output_directory(track_parity_harness)
