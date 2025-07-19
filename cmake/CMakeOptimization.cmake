# ----------------------------------------------------------------------------
# -                   CloudViewer Performance Optimization                   -
# ----------------------------------------------------------------------------
# This file contains performance optimization settings for CloudViewer
# Centralized location for all compiler optimization flags

# Enable advanced compiler optimizations
function(cloudviewer_set_performance_optimization target)
    # Enable Link Time Optimization (LTO) for better performance
    if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
        set_target_properties(${target} PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION TRUE
        )
        
        # Check if LTO is supported
        include(CheckIPOSupported)
        check_ipo_supported(RESULT lto_supported OUTPUT lto_error)
        if(lto_supported)
            message(STATUS "LTO enabled for ${target}")
        else()
            message(WARNING "LTO not supported: ${lto_error}")
        endif()
    endif()
    
    # Platform-specific optimizations
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE
            # Enable aggressive optimization for release builds
            $<$<CONFIG:Release>:-O3>
            $<$<CONFIG:Release>:-DNDEBUG>
            $<$<CONFIG:Release>:-march=native>
            $<$<CONFIG:Release>:-mtune=native>
            $<$<CONFIG:Release>:-ffast-math>
            $<$<CONFIG:Release>:-funroll-loops>
            $<$<CONFIG:Release>:-fomit-frame-pointer>
            $<$<CONFIG:Release>:-flto>
            
            # Enable vectorization hints
            $<$<CONFIG:Release>:-ftree-vectorize>
            $<$<CONFIG:Release>:-fvect-cost-model=unlimited>
            
            # RelWithDebInfo optimizations (debugging + performance)
            $<$<CONFIG:RelWithDebInfo>:-O2>
            $<$<CONFIG:RelWithDebInfo>:-DNDEBUG>
            $<$<CONFIG:RelWithDebInfo>:-march=native>
            $<$<CONFIG:RelWithDebInfo>:-mtune=native>
            
            # Debug optimizations (minimal impact on debugging)
            $<$<CONFIG:Debug>:-Og>
            $<$<CONFIG:Debug>:-g3>
        )
        
        # Link-time optimizations
        target_link_options(${target} PRIVATE
            $<$<CONFIG:Release>:-flto>
            $<$<CONFIG:Release>:-Wl,--gc-sections>
            $<$<CONFIG:Release>:-Wl,--strip-all>
        )
        
    elseif(MSVC)
        target_compile_options(${target} PRIVATE
            # Enable maximum optimization for release
            $<$<CONFIG:Release>:/O2>
            $<$<CONFIG:Release>:/Ob2>
            $<$<CONFIG:Release>:/Oi>
            $<$<CONFIG:Release>:/Ot>
            $<$<CONFIG:Release>:/Oy>
            $<$<CONFIG:Release>:/GL>
            $<$<CONFIG:Release>:/DNDEBUG>
            
            # Enable vectorization
            $<$<CONFIG:Release>:/arch:AVX2>
            
            # RelWithDebInfo optimizations
            $<$<CONFIG:RelWithDebInfo>:/O2>
            $<$<CONFIG:RelWithDebInfo>:/Zi>
            $<$<CONFIG:RelWithDebInfo>:/DNDEBUG>
        )
        
        # Link-time optimizations for MSVC
        target_link_options(${target} PRIVATE
            $<$<CONFIG:Release>:/LTCG>
            $<$<CONFIG:Release>:/OPT:REF>
            $<$<CONFIG:Release>:/OPT:ICF>
        )
    endif()
    
    # Enable CPU-specific optimizations if USE_SIMD is ON
    if(USE_SIMD)
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(${target} PRIVATE
                -msse4.2
                -mavx
                -mavx2
                -mfma
            )
        elseif(MSVC)
            target_compile_options(${target} PRIVATE
                /arch:AVX2
            )
        endif()
    endif()
    
    # Memory optimization flags
    target_compile_definitions(${target} PRIVATE
        $<$<CONFIG:Release>:EIGEN_NO_DEBUG>
        $<$<CONFIG:Release>:EIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT>
        $<$<CONFIG:Release>:EIGEN_MAX_ALIGN_BYTES=32>
    )
endfunction()

# Apply optimization to core libraries
function(cloudviewer_apply_core_optimizations)
    # Set global optimization flags for all targets
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        # Enable whole program optimization globally
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE PARENT_SCOPE)
        
        # Set optimized default flags
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native -ffast-math -flto" PARENT_SCOPE)
            set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native -ffast-math -flto" PARENT_SCOPE)
        elseif(MSVC)
            set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob2 /Oi /Ot /Oy /GL /DNDEBUG" PARENT_SCOPE)
            set(CMAKE_C_FLAGS_RELEASE "/O2 /Ob2 /Oi /Ot /Oy /GL /DNDEBUG" PARENT_SCOPE)
        endif()
    endif()
endfunction()

# Bundle size optimization for Python wheels
function(cloudviewer_optimize_python_bundle)
    # Strip debug symbols from shared libraries in release builds
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        if(UNIX AND NOT APPLE)
            add_custom_target(strip_symbols ALL
                COMMAND find ${CMAKE_BINARY_DIR} -name "*.so" -exec strip --strip-unneeded {} \;
                COMMENT "Stripping debug symbols from shared libraries"
                VERBATIM
            )
        elseif(APPLE)
            add_custom_target(strip_symbols ALL
                COMMAND find ${CMAKE_BINARY_DIR} -name "*.dylib" -exec strip -x {} \;
                COMMENT "Stripping debug symbols from dynamic libraries"
                VERBATIM
            )
        endif()
    endif()
endfunction()

# Memory pool optimization
function(cloudviewer_enable_memory_optimization target)
    # Enable TCMalloc or jemalloc for better memory performance
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(TCMALLOC QUIET libtcmalloc)
        if(TCMALLOC_FOUND)
            target_link_libraries(${target} PRIVATE ${TCMALLOC_LIBRARIES})
            target_compile_definitions(${target} PRIVATE USE_TCMALLOC)
            message(STATUS "Enabled TCMalloc for ${target}")
        else()
            pkg_check_modules(JEMALLOC QUIET jemalloc)
            if(JEMALLOC_FOUND)
                target_link_libraries(${target} PRIVATE ${JEMALLOC_LIBRARIES})
                target_compile_definitions(${target} PRIVATE USE_JEMALLOC)
                message(STATUS "Enabled jemalloc for ${target}")
            endif()
        endif()
    endif()
endfunction()

# Cache optimization
function(cloudviewer_enable_cache_optimization target)
    # Enable CPU cache-friendly data structures
    target_compile_definitions(${target} PRIVATE
        # Enable cache line optimization
        CACHE_LINE_SIZE=64
        # Prefer cache-friendly algorithms
        USE_CACHE_FRIENDLY_ALGORITHMS=1
    )
    
    # Enable prefetch hints
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE
            -fprefetch-loop-arrays
        )
    endif()
endfunction()