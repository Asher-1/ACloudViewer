include(ExternalProject)

# Consumers link the installed lapack/blas (and faiss the cblas wrapper) by
# absolute path, which Ninja only accepts for files a rule generates.
if (UNIX AND NOT APPLE)
    set(LAPACK_LIB_PREFIX ${CMAKE_SHARED_LIBRARY_PREFIX})
    set(LAPACK_LIB_SUFFIX ${CMAKE_SHARED_LIBRARY_SUFFIX})
else ()
    set(LAPACK_LIB_PREFIX ${CMAKE_STATIC_LIBRARY_PREFIX})
    set(LAPACK_LIB_SUFFIX ${CMAKE_STATIC_LIBRARY_SUFFIX})
endif ()
set(LAPACK_BUILD_BYPRODUCTS)
foreach (item lapack blas cblas)
    list(APPEND LAPACK_BUILD_BYPRODUCTS <INSTALL_DIR>/lib/${LAPACK_LIB_PREFIX}${item}${LAPACK_LIB_SUFFIX})
endforeach ()

ExternalProject_Add(ext_lapack
        PREFIX lapack
        #   http://www.netlib.org/lapack/lapack-3.9.0.tar.gz
        URL https://github.com/Reference-LAPACK/lapack/archive/v3.9.0.tar.gz
        URL_HASH MD5=0b251e2a8d5f949f99b50dd5e2200ee2
        DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/lapack"
        BUILD_IN_SOURCE 0
        BUILD_ALWAYS 0
        INSTALL_DIR ${CLOUDVIEWER_EXTERNAL_INSTALL_DIR}
        UPDATE_COMMAND ""
        PATCH_COMMAND ${CMAKE_COMMAND} -DINPUT_DIR=<SOURCE_DIR> -P ${CloudViewer_3RDPARTY_DIR}/CMake/fix_all_cmake_minimum.cmake
        CMAKE_ARGS
            -DCMAKE_POLICY_VERSION_MINIMUM=3.5
            -DBUILD_SHARED_LIBS=$<$<PLATFORM_ID:Linux>:ON:OFF>
            # Installed shared lapack must resolve its own NEEDED libblas:
            # consumer RUNPATHs are non-transitive on Linux.
            -DCMAKE_INSTALL_RPATH=$ORIGIN
            -DCMAKE_BUILD_TYPE=$<IF:$<PLATFORM_ID:Windows>,${CMAKE_BUILD_TYPE},Release>
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
            -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
            # The netlib CBLAS wrapper (libcblas.so.3) exposes cblas_sgemm and
            # friends over the Fortran BLAS. faiss needs this C interface; the
            # plain libblas.so.3 above only exports the Fortran symbols, and
            # any same-soname system BLAS loses to this bundled copy at wheel
            # load time (undefined symbol: cblas_sgemm).
            -DCBLAS=ON
            -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
        DEPENDS 3rdparty_glog
        BUILD_BYPRODUCTS ${LAPACK_BUILD_BYPRODUCTS}
)

ExternalProject_Get_Property(ext_lapack INSTALL_DIR)
set(LAPACK_LIB_DIR ${INSTALL_DIR}/lib)
set(LAPACKBLAS_LIBRARIES lapack blas)

set(BLAS_LIBRARIES ${INSTALL_DIR}/lib/${LAPACK_LIB_PREFIX}blas${LAPACK_LIB_SUFFIX})
set(LAPACK_LIBRARIES ${INSTALL_DIR}/lib/${LAPACK_LIB_PREFIX}lapack${LAPACK_LIB_SUFFIX})

set(LAPACK_CMAKE_FLAGS -DBLAS=${BLAS_LIBRARIES} -DLAPACK=${LAPACK_LIBRARIES} -DBLAS_LIBRARIES=${BLAS_LIBRARIES} -DLAPACK_LIBRARIES=${LAPACK_LIBRARIES})