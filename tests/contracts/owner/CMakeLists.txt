if(CASE STREQUAL graphics-only)
    add_executable(graphics_target ../pass.cpp)
    cloudviewer_add_test(NAME fixture-graphics COMMAND graphics_target
        LABELS "graphics;cpp-ci;test-ci")
    return()
elseif(CASE STREQUAL script-only)
    cloudviewer_add_test(NAME fixture-script-only COMMAND ${CMAKE_COMMAND} -E true)
    return()
endif()

if(CASE STREQUAL empty)
    add_executable(network_target ../pass.cpp)
    cloudviewer_add_test(NAME fixture-network COMMAND network_target
        LABELS "network")
    return()
endif()

add_executable(direct_target ../pass.cpp)
cloudviewer_add_test(NAME fixture-direct COMMAND direct_target
    LABELS "suite-aicore")
# Registration finalization must see labels set later in this owning directory.
set_tests_properties(fixture-direct PROPERTIES LABELS "suite-aicore;capi;model")

add_executable(wrapper_target ../pass.cpp)
cloudviewer_add_test(NAME fixture-wrapper
    COMMAND ${CMAKE_COMMAND} -E env
        "FIXTURE_TARGET=$<TARGET_FILE:wrapper_target>"
        "$<TARGET_FILE:wrapper_target>"
    LABELS "suite-aicore;capi")

# The target is declared by the parent directory only after this directory
# returns, exercising forward references in the shared build closure.
cloudviewer_add_test(NAME fixture-forward
    COMMAND "$<TARGET_FILE:forward_target>"
    LABELS "suite-aicore;gpu")

# A non-native wrapper row still participates in AICore classification.
cloudviewer_add_test(NAME fixture-script
    COMMAND ${CMAKE_COMMAND} -E echo "script fixture"
    LABELS "suite-aicore;capi;network;aicore-yolo-matrix")

cloudviewer_add_test(NAME fixture-cvplugin
    COMMAND wrapper_target
    LABELS "suite-aicore;cvpluginapi")

cloudviewer_add_test(NAME fixture-network
    COMMAND wrapper_target
    LABELS "suite-aicore;network")

# Late graphics labels and stale derived labels must never leak into headless CI.
add_executable(graphics_target ../pass.cpp)
cloudviewer_add_test(NAME fixture-graphics COMMAND graphics_target)
set_tests_properties(fixture-graphics PROPERTIES LABELS "graphics;cpp-ci;test-ci")
cloudviewer_add_test(NAME fixture-shared-graphics COMMAND wrapper_target LABELS graphics)
cloudviewer_add_test(NAME fixture-extra-script COMMAND ${CMAKE_COMMAND} -E true)

if(CASE STREQUAL duplicate)
    cloudviewer_add_test(NAME fixture-direct COMMAND direct_target)
elseif(CASE STREQUAL missing)
    cloudviewer_add_test(NAME fixture-missing
        COMMAND "$<TARGET_FILE:forward_target>"
        LABELS "cpp")
endif()
