# Smoke gate for the Spirula Studio component (see 3rdparty/spirula).
# Re-run after every SPIRULA_TAG bump: the tests need no GPU, no display and
# no network, so they are the cheap part of the "tag bump + smoke" upstream
# sync contract. Device enumeration needs a Vulkan runtime and therefore
# stays a manual check on machines that have one.

# Register whenever ext_spirula is configured (SPIRULA_EXECUTABLE is set);
# the executable may not exist yet at configure time on a first build, and
# ctest reports a clear failure for a missing command, which is the signal we
# want after a broken tag bump.
if(NOT SPIRULA_EXECUTABLE)
    message(STATUS "qSpirula tests: ext_spirula not configured — skipping")
    return()
endif()

add_test(NAME spirula_smoke_version
         COMMAND "${SPIRULA_EXECUTABLE}" --version)
add_test(NAME spirula_smoke_train_help
         COMMAND "${SPIRULA_EXECUTABLE}" train --help)
set_tests_properties(spirula_smoke_version spirula_smoke_train_help
                     PROPERTIES LABELS "spirula;smoke")
