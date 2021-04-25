include(ExternalProject)

set_local_or_remote_url(
    DOWNLOAD_URL_PRIMARY
    LOCAL_URL   "${THIRD_PARTY_DOWNLOAD_DIR}/suitesparse-metis-1.5.0.zip"
    REMOTE_URLS "https://github.com/jlblancoc/suitesparse-metis-for-windows/archive/refs/tags/v1.5.0.zip"
)

ExternalProject_Add(
       ext_suitesparse
       PREFIX suitesparse
       URL ${DOWNLOAD_URL_PRIMARY} ${DOWNLOAD_URL_FALLBACK}
       BUILD_IN_SOURCE 0
       BUILD_ALWAYS 0
       UPDATE_COMMAND ""
       CONFIGURE_COMMAND ${CMAKE_COMMAND}
           -DOPENMP=ON
           -DBUILD_SHARED_LIBS=OFF
           -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
           -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
           -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
           -DCMAKE_CUDA_FLAGS=${CUDA_GENCODES}
           -DCUDA_INCLUDE_DIRS=${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
           -DWITH_CUDA=${BUILD_CUDA_MODULE}
           -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR> <SOURCE_DIR>
       BUILD_COMMAND $(MAKE)
       )
ExternalProject_Get_Property(ext_suitesparse INSTALL_DIR)
set(SUITESPARSE_INCLUDE_DIRS ${INSTALL_DIR}/include/suitesparse/)
set(SUITESPARSE_LIB_DIR ${INSTALL_DIR}/lib)
set(SUITESPARSE_LIBRARIES   suitesparseconfig$<$<CONFIG:Debug>:d>
                            amd$<$<CONFIG:Debug>:d>
                            btf$<$<CONFIG:Debug>:d>
                            camd$<$<CONFIG:Debug>:d>
                            ccolamd$<$<CONFIG:Debug>:d>
                            colamd$<$<CONFIG:Debug>:d>
                            cholmod$<$<CONFIG:Debug>:d>
                            cxsparse$<$<CONFIG:Debug>:d>
                            ldl$<$<CONFIG:Debug>:d>
                            klu$<$<CONFIG:Debug>:d>
                            umfpack$<$<CONFIG:Debug>:d>
                            spqr$<$<CONFIG:Debug>:d>
#                            rbio$<$<CONFIG:Debug>:d>
#                            graphblas$<$<CONFIG:Debug>:d>
#                            sliplu$<$<CONFIG:Debug>:d>
                            metis$<$<CONFIG:Debug>:d>)
if (BUILD_CUDA_MODULE)
    list(APPEND SUITESPARSE_LIBRARIES SuiteSparse_GPURuntime$<$<CONFIG:Debug>:d> GPUQREngine$<$<CONFIG:Debug>:d>)
endif()

if (WIN32)
    set(LAPACK_LIBRARIES ${INSTALL_DIR}/lib64/lapack_blas_windows/liblapack.lib)
    set(BLAS_LIBRARIES ${INSTALL_DIR}/lib64/lapack_blas_windows/libblas.lib)
endif()

if(WIN32)
    set(SUITESPARSE_CMAKE_FLAGS -DLAPACK_LIBRARIES=${LAPACK_LIBRARIES} -DBLAS_LIBRARIES=${BLAS_LIBRARIES} -DSuiteSparse_DIR=${INSTALL_DIR}/lib/cmake -DSUITESPARSE_LIBRARY_DIR=${INSTALL_DIR}/lib -DSUITESPARSE_INCLUDE_DIR_HINTS=${INSTALL_DIR}/include -DSUITESPARSE_LIBRARY_DIR_HINTS=${INSTALL_DIR}/lib)
else()
    set(SUITESPARSE_CMAKE_FLAGS ${LAPACK_CMAKE_FLAGS} -DSuiteSparse_DIR=${INSTALL_DIR}/lib/cmake/suitesparse-5.4.0/ -DSUITESPARSE_INCLUDE_DIR_HINTS=${INSTALL_DIR}/include -DSUITESPARSE_LIBRARY_DIR_HINTS=${INSTALL_DIR}/lib)
endif()

