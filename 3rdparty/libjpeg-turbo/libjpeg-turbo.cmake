include(ExternalProject)

# The version and configuration MUST match the copy OpenImageIO builds
# internally (its build_libjpeg-turbo.cmake: 3.1.2, WITH_JPEG8=1, static).
# azure_kinect's TurboJPEG archive and OIIO's libjpeg.a land in the same
# static link (pybind module, app); two different libjpeg-turbo versions
# export overlapping jpeg_* symbols with gaps, which pulls BOTH
# implementations into the link - multiple definition (ubuntu-jammy CI).
# With identical symbol sets, whichever archive the linker scans first
# satisfies every jpeg_* reference and the second pulls no members.

# Set compiler flags
if (MSVC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /D_CRT_SECURE_NO_WARNINGS")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_CRT_SECURE_NO_WARNINGS")
endif()

# Set WITH_SIMD
include(CheckLanguage)
check_language(ASM_NASM)
if (CMAKE_ASM_NASM_COMPILER)
    if (APPLE)
        # macOS might have /usr/bin/nasm but it cannot be used
        # https://stackoverflow.com/q/53974320
        # To fix this, run `brew install nasm`
        execute_process(COMMAND nasm --version RESULT_VARIABLE return_code)
        if("${return_code}" STREQUAL "0")
            enable_language(ASM_NASM)
            option(WITH_SIMD "" ON)
        else()
            message(STATUS "nasm found but can't be used, run `brew install nasm`")
            option(WITH_SIMD "" OFF)
        endif()
    else()
        enable_language(ASM_NASM)
        option(WITH_SIMD "" ON)
    endif()
else()
    option(WITH_SIMD "" OFF)
endif()
if (WITH_SIMD)
    message(STATUS "NASM assembler enabled")
else()
    message(STATUS "NASM assembler not found - libjpeg-turbo performance may suffer")
endif()

if (STATIC_WINDOWS_RUNTIME)
    set(WITH_CRT_DLL OFF)
else()
    set(WITH_CRT_DLL ON)
endif()
message(STATUS "libturbojpeg: WITH_CRT_DLL=${WITH_CRT_DLL}")

# If MSVC, the OUTPUT_NAME was set to turbojpeg-static
if(MSVC)
    set(lib_name "turbojpeg-static")
else()
    set(lib_name "turbojpeg")
endif()

ExternalProject_Add(ext_turbojpeg
    PREFIX turbojpeg
    GIT_REPOSITORY https://github.com/libjpeg-turbo/libjpeg-turbo
    # 3.1.2 = the exact commit OpenImageIO's internal build pins
    # (build_libjpeg-turbo.cmake); pin the commit, not the mutable tag.
    GIT_TAG 4e151a4ad91001b3aa8c2ece2205c15f487ce320
    UPDATE_COMMAND ""
    CMAKE_ARGS
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DWITH_CRT_DLL=${WITH_CRT_DLL}
        -DENABLE_STATIC=ON
        -DENABLE_SHARED=OFF
        -DWITH_SIMD=${WITH_SIMD}
        # Match OIIO's internal configuration: without WITH_JPEG8 the
        # jpeg_mem_* API is absent and the gap pulls OIIO's libjpeg.a into
        # links that already carry this archive (see header comment).
        -DWITH_JPEG8=1
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        ${ExternalProject_CMAKE_ARGS_hidden}
        # WARNING: libjpeg-turbo uses its own old version of
        # GNUInstallDirs.cmake and leads to invalid installs with
        # CMAKE_INSTALL_LIBDIR, so we undefine it and set
        # CMAKE_INSTALL_DEFAULT_LIBDIR instead.
        -UCMAKE_INSTALL_LIBDIR
        -DCMAKE_INSTALL_DEFAULT_LIBDIR=${CloudViewer_INSTALL_LIB_DIR}
    BUILD_BYPRODUCTS
        <INSTALL_DIR>/${CloudViewer_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${lib_name}${CMAKE_STATIC_LIBRARY_SUFFIX})

ExternalProject_Get_Property(ext_turbojpeg INSTALL_DIR)
set(JPEG_TURBO_INCLUDE_DIRS ${INSTALL_DIR}/include/) # "/" is critical.
set(JPEG_TURBO_LIB_DIR ${INSTALL_DIR}/${CloudViewer_INSTALL_LIB_DIR})
set(JPEG_TURBO_LIBRARIES ${lib_name})
