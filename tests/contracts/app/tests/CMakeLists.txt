# Synthetic app owner: exercise registration, prerequisite discovery, and
# rejection of raw add_test() without depending on Qt or production modules.
add_executable(app_registration_probe ../../pass.cpp)
if(CASE STREQUAL bypass)
    add_test(NAME fixture-bypass COMMAND app_registration_probe)
else()
    cloudviewer_add_test(NAME fixture-app-registration
        COMMAND ${CMAKE_COMMAND} -E true BUILD_TARGETS app_registration_probe LABELS contract)
endif()
