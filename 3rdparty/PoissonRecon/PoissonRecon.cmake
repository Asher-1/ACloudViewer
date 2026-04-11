include(ExternalProject)

# PoissonRecon v12 has a known race condition in _SetSliceIsoCorners when using
# ThreadPool::Parallel_for (upstream issue #136). On macOS the race causes
# "Edge not set" / "Failed to close loop" errors leading to SIGABRT.
# The patch serializes that loop on Apple platforms (same fix as CloudCompare).
if(APPLE)
    find_program(PATCH_EXECUTABLE patch)
    if(PATCH_EXECUTABLE)
        # Use cmake -P to run the patch and tolerate "already applied" (exit 1
        # from patch -N).  ExternalProject treats any non-zero exit as fatal, so
        # we wrap the call in a CMake script that checks the result.
        set(_POISSON_PATCH_SCRIPT ${CMAKE_CURRENT_LIST_DIR}/apply_patch.cmake)
        set(_POISSON_PATCH_ARGS
            PATCH_COMMAND ${CMAKE_COMMAND}
                -DPATCH_EXECUTABLE=${PATCH_EXECUTABLE}
                -DPATCH_FILE=${CMAKE_CURRENT_LIST_DIR}/fix-macos-race-condition.patch
                -P ${_POISSON_PATCH_SCRIPT}
        )
    else()
        message(WARNING "PoissonRecon: 'patch' not found; macOS race-condition fix will NOT be applied")
        set(_POISSON_PATCH_ARGS)
    endif()
else()
    set(_POISSON_PATCH_ARGS)
endif()

ExternalProject_Add(ext_poisson
    PREFIX poisson
    URL https://github.com/Asher-1/cloudViewer_downloads/releases/download/poisson/Open3D-PoissonRecon-6ddec9a69f4aeb7a715e8f496310929d9f493041.tar.gz
    URL_HASH SHA256=3e02bebd1b22f76bb8874be6ff7ab60a0f74ed690829befe0f90e0f2b70bbbe6
    DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/poisson"
    SOURCE_DIR "poisson/src/ext_poisson/PoissonRecon" # Add extra directory level for POISSON_INCLUDE_DIRS.
    UPDATE_COMMAND ""
    ${_POISSON_PATCH_ARGS}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

ExternalProject_Get_Property(ext_poisson SOURCE_DIR)
set(POISSON_INCLUDE_DIRS ${SOURCE_DIR}) # Not using "/" is critical.
