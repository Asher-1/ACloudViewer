include(FetchContent)

FetchContent_Declare(
    ext_googletest
    PREFIX googletest
    URL https://github.com/google/googletest/releases/download/v1.16.0/googletest-1.16.0.tar.gz
    URL_HASH SHA256=78c676fc63881529bf97bf9d45948d905a66833fbfa5318ea2cd7478cb98f399
    DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/googletest"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

if (WIN32)
    # Disable installation of googletest (it's only used for testing, not needed in install)
    # This prevents "file INSTALL cannot find" errors on Windows when gmock.lib is locked
    # Set options before MakeAvailable so they take effect during configuration
    set(INSTALL_GTEST OFF CACHE BOOL "Enable installation of googletest" FORCE)
    set(gtest_force_shared_crt ON CACHE BOOL "Use shared (DLL) run-time lib" FORCE)
endif()

FetchContent_MakeAvailable(ext_googletest)
FetchContent_GetProperties(ext_googletest SOURCE_DIR GOOGLETEST_SOURCE_DIR)

if (WIN32)
    # Explicitly exclude googletest targets from installation
    # This is a safety measure in case INSTALL_GTEST setting didn't take effect
    # Similar to how tbb.cmake handles tbbbind targets
    foreach(target_name gtest gtest_main gmock gmock_main)
        if(TARGET ${target_name})
            set_target_properties(${target_name} PROPERTIES 
                EXCLUDE_FROM_ALL TRUE 
                EXCLUDE_FROM_DEFAULT_BUILD TRUE
            )
            message(STATUS "Disabled ${target_name} installation (googletest is only used for testing)")
        endif()
    endforeach()
endif()