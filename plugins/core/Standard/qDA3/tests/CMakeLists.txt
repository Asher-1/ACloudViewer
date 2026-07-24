# Legacy parity harnesses only. See README.md — depth tests moved to core/AICore/tests/.

set(DA_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../..)

if(NOT TARGET AICore)
    return()
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/glb_parity_dump.cpp)
    add_executable(glb_parity_dump glb_parity_dump.cpp)
    target_link_libraries(glb_parity_dump PRIVATE AICore)
    target_include_directories(glb_parity_dump PRIVATE
        ${DA_ROOT}/core/AICore/src/depth ${DA_ROOT}/core/AICore/include)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/colmap_parity_dump.cpp)
    add_executable(colmap_parity_dump colmap_parity_dump.cpp)
    target_link_libraries(colmap_parity_dump PRIVATE AICore)
    target_include_directories(colmap_parity_dump PRIVATE
        ${DA_ROOT}/core/AICore/src/depth ${DA_ROOT}/core/AICore/include)
endif()
