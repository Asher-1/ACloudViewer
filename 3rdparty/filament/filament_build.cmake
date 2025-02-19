include(ExternalProject)

set(FILAMENT_ROOT "${CMAKE_BINARY_DIR}/filament-binaries")

# Handle build type for single and multi-config generators.
get_property(is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
set(FILAMENT_BUILD_TYPE ${CMAKE_BUILD_TYPE})
if(NOT is_multi_config)
    # Do not mix debug/release CRT on Windows.
    if (NOT MSVC)
        set(FILAMENT_BUILD_TYPE "Release")
    endif()
endif()

set(filament_LIBRARIES
    filameshio
    filament
    filamat_lite
    filamat
    filaflat
    filabridge
    geometry
    backend
    bluegl
    ibl
    image
    meshoptimizer
    smol-v
    utils
)

# Locate byproducts
set(lib_dir lib)
if(APPLE)
    if(APPLE_AARCH64)
        set(lib_dir lib/arm64)
    else()
        set(lib_dir lib/x86_64)
    endif()
endif()

set(lib_byproducts ${filament_LIBRARIES})
list(TRANSFORM lib_byproducts PREPEND ${FILAMENT_ROOT}/${lib_dir}/${CMAKE_STATIC_LIBRARY_PREFIX})
list(TRANSFORM lib_byproducts APPEND ${CMAKE_STATIC_LIBRARY_SUFFIX})

# fix compiling issues with filament
set(filament_cxx_flags "${CMAKE_CXX_FLAGS} -Wno-deprecated -Wno-error=unused-but-set-variable -Wno-error=deprecated-copy")
if(NOT WIN32)
    # Issue Open3D#1909, filament#2146
    set(filament_cxx_flags "${filament_cxx_flags} -fno-builtin")
endif()

ExternalProject_Add(
    ext_filament
    PREFIX filament
    URL https://github.com/isl-org/filament/archive/d1d873d27f43ba0cee1674a555cc0f18daac3008.tar.gz
    URL_HASH SHA256=00c3f41af0fcfb2df904e1f77934f2678d943ddac5eb889788a5e22590e497bd
    DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/filament"
    UPDATE_COMMAND ""
    BUILD_ALWAYS 0
    # Patch for filament/libs/image/src/ImageSampler.cpp:41:17: error: expected unqualified-id replace M_PIf with M_PI_f
    COMMAND sed -i $<$<PLATFORM_ID:Darwin>:""> "s/M_PIf/M_PI_f/g" <SOURCE_DIR>/libs/image/src/ImageSampler.cpp
    CMAKE_ARGS
        # ${ExternalProject_CMAKE_ARGS}
        -DCMAKE_BUILD_TYPE=${FILAMENT_BUILD_TYPE}
        -DCCACHE_PROGRAM=OFF  # Enables ccache, "launch-cxx" is not working.
        -DFILAMENT_ENABLE_JAVA=OFF
        -DCMAKE_C_COMPILER=${FILAMENT_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${FILAMENT_CXX_COMPILER}
        -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
        -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
        -DCMAKE_CXX_FLAGS:STRING=${filament_cxx_flags}
        -DCMAKE_INSTALL_PREFIX=${FILAMENT_ROOT}
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DUSE_STATIC_CRT=${STATIC_WINDOWS_RUNTIME}
        -DUSE_STATIC_LIBCXX=ON
        -DFILAMENT_SUPPORTS_VULKAN=OFF
        -DFILAMENT_SKIP_SAMPLES=ON
        -DFILAMENT_OPENGL_HANDLE_ARENA_SIZE_IN_MB=20 # to support many small entities
        -DSPIRV_WERROR=OFF
    BUILD_BYPRODUCTS ${lib_byproducts}
)
