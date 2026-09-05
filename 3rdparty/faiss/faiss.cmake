include(FetchContent)

# Match COLMAP's CPU-only FAISS configuration.  It is intentionally kept out
# of the global dependency lists: only reconstruction's descriptor index and
# visual vocabulary consumers need its headers and static library.
FetchContent_Declare(
    faiss
    URL https://github.com/facebookresearch/faiss/archive/refs/tags/v1.14.1.zip
    URL_HASH SHA256=4b1ae7e7a0a46385b4084f0e3945623a15fcf99d793bf44d82aae8e24f11e5f5
    DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/faiss"
)

if (NOT MSVC)
    set(FAISS_OPT_LEVEL dd CACHE STRING "" FORCE)
endif ()
set(FAISS_ENABLE_GPU OFF CACHE BOOL "" FORCE)
set(FAISS_ENABLE_PYTHON OFF CACHE BOOL "" FORCE)
set(FAISS_ENABLE_MKL OFF CACHE BOOL "" FORCE)
set(FAISS_ENABLE_EXTRAS OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)

# faiss resolves BLAS/LAPACK with CMake's FindBLAS/FindLAPACK inside its own
# subdirectory. Windows ships no system BLAS, and the pinned static MKL is
# installed by ext_mkl at build time, so a configure-time find_library() scan
# cannot see it. Preset the result with the same pinned static MKL that
# 3rdparty_blas uses: a preset cache value makes FindBLAS/FindLAPACK skip
# their vendor probing and link these libraries directly. faiss indexes with
# 32-bit ints, so it must use the LP64 interface (mkl_intel_lp64) rather than
# the ILP64 one selected on UNIX - see mkl.cmake. Library order follows
# Intel's link-line advisor (interface -> threading -> core) so every
# dependency resolves downwards during the single-pass MSVC scan.
#
# mkl_tbb_thread (Release) carries ~6k undefined references into oneTBB, so
# the tbb import library is appended after it, taken from whichever target
# provides TBB: the FetchContent shared `tbb` target (USE_SYSTEM_TBB=OFF,
# default) or the imported `TBB::tbb` from the system package
# (USE_SYSTEM_TBB=ON). $<TARGET_LINKER_FILE> cannot be evaluated on the
# INTERFACE target 3rdparty_tbb, hence the two-way dispatch.
if (WIN32 AND NOT USE_BLAS AND TARGET ext_mkl)
    set(FAISS_MKL_LIBS
        "${STATIC_MKL_LIB_DIR}/mkl_intel_lp64.lib"
        "$<$<CONFIG:Debug>:${STATIC_MKL_LIB_DIR}/mkl_sequential.lib>"
        "$<$<CONFIG:Release>:${STATIC_MKL_LIB_DIR}/mkl_tbb_thread.lib>"
        "${STATIC_MKL_LIB_DIR}/mkl_core.lib")
    if(TARGET tbb)
        list(APPEND FAISS_MKL_LIBS "$<$<CONFIG:Release>:$<TARGET_LINKER_FILE:tbb>>")
    elseif(TARGET TBB::tbb)
        list(APPEND FAISS_MKL_LIBS "$<$<CONFIG:Release>:$<TARGET_LINKER_FILE:TBB::tbb>>")
    endif()
    set(BLAS_LIBRARIES ${FAISS_MKL_LIBS} CACHE STRING "" FORCE)
    set(LAPACK_LIBRARIES ${FAISS_MKL_LIBS} CACHE STRING "" FORCE)
elseif (UNIX AND NOT APPLE AND NOT BUILD_WITH_CONDA)
    # faiss calls both the CBLAS interface (cblas_sgemm & friends) and the
    # Fortran BLAS/LAPACK entry points (sgemm_, dpotrf_, ...). The netlib
    # BLAS/LAPACK built for Ceres/SuiteSparse only exports the Fortran
    # symbols from libblas.so.3/liblapack.so.3, and a CBLAS-capable system
    # BLAS sharing those sonames loses to the bundled netlib copies at wheel
    # load time (pack_ubuntu.sh dedups by basename, so the first same-name
    # library processed wins), which surfaces as "undefined symbol:
    # cblas_sgemm" at import. Pin faiss to the full netlib trio from
    # ext_lapack: distinct sonames, all bundled by pack_ubuntu.sh and
    # preloaded by cloudViewer/__init__.py.
    # Conda builds are excluded: conda's libcblas.so.3/liblapack.so.3 are
    # symlinks to libopenblas.so.0, so copying their real files under the
    # cblas/lapack names would record a mismatched SONAME in the wheel.
    set(FAISS_NETLIB_BLAS_LIBS
        "${CLOUDVIEWER_EXTERNAL_INSTALL_DIR}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}cblas${CMAKE_SHARED_LIBRARY_SUFFIX}"
        "${CLOUDVIEWER_EXTERNAL_INSTALL_DIR}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}blas${CMAKE_SHARED_LIBRARY_SUFFIX}")
    set(FAISS_NETLIB_LAPACK_LIB
        "${CLOUDVIEWER_EXTERNAL_INSTALL_DIR}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}lapack${CMAKE_SHARED_LIBRARY_SUFFIX}")
    set(BLAS_LIBRARIES ${FAISS_NETLIB_BLAS_LIBS} CACHE STRING "" FORCE)
    set(LAPACK_LIBRARIES ${FAISS_NETLIB_LAPACK_LIB} CACHE STRING "" FORCE)
else ()
    # Drop any previously forced pin (e.g. after toggling BUILD_WITH_CONDA)
    # so faiss falls back to its own discovery instead of reusing a stale
    # cache entry pointing at a lib directory that no longer matches.
    unset(BLAS_LIBRARIES CACHE)
    unset(LAPACK_LIBRARIES CACHE)
endif ()

FetchContent_MakeAvailable(faiss)

if (WIN32)
    target_compile_definitions(faiss PUBLIC FAISS_MAIN_LIB)
endif ()

add_library(3rdparty_faiss INTERFACE)
target_link_libraries(3rdparty_faiss INTERFACE faiss)
