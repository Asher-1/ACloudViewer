include(ExternalProject)

set_local_or_remote_url(
    DOWNLOAD_URL_PRIMARY
    LOCAL_URL   "${THIRD_PARTY_DOWNLOAD_DIR}/suitesparse-metis-1.5.0.tar"
    REMOTE_URLS "https://codeload.github.com/jlblancoc/suitesparse-metis-for-windows/zip/7bc503bfa2c4f1be9176147d36daf9e18340780a"
#    REMOTE_URLS "https://github.com/jlblancoc/suitesparse-metis-for-windows/archive/refs/tags/v1.5.0.zip"
)

ExternalProject_Add(
       ext_suitesparse
       PREFIX suitesparse
       URL ${DOWNLOAD_URL_PRIMARY} ${DOWNLOAD_URL_FALLBACK}
       URL_HASH MD5=e7c27075e8e0afc9d2cf188630090946
       BUILD_ALWAYS 0
       UPDATE_COMMAND ""
       CMAKE_ARGS
#           ${LAPACK_CMAKE_FLAGS}
           -DOPENMP=ON
           -DBUILD_SHARED_LIBS=OFF
           -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
           -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
           -DCUDA_INCLUDE_DIRS=${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
           -DWITH_CUDA=${BUILD_CUDA_MODULE}
           -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
           DEPENDS ${LAPACK_TARGET}
       )
ExternalProject_Get_Property(ext_suitesparse INSTALL_DIR)
set(SUITESPARSE_INCLUDE_DIRS ${INSTALL_DIR}/include/suitesparse/)
set(SUITESPARSE_LIB_DIR ${INSTALL_DIR}/lib)
set(EXT_SUITESPARSE_LIBRARIES   suitesparseconfig
                                amd
                                btf
                                camd
                                ccolamd
                                colamd
                                cholmod
                                cxsparse
                                klu
                                ldl
                                umfpack
                                spqr
#                                rbio
#                                graphblas
#                                sliplu
                                metis)
if (BUILD_CUDA_MODULE)
	if(NOT WIN32)
		list(APPEND SUITESPARSE_LIBRARIES SuiteSparse_GPURuntime GPUQREngine)
	endif()
endif()

if(WIN32)
    set(LAPACK_LIBRARIES ${INSTALL_DIR}/lib64/lapack_blas_windows/liblapack.lib)
    set(BLAS_LIBRARIES ${INSTALL_DIR}/lib64/lapack_blas_windows/libblas.lib)
    set(LAPACK_CMAKE_FLAGS -DBLAS_LIBRARIES=${BLAS_LIBRARIES} -DLAPACK_LIBRARIES=${LAPACK_LIBRARIES})
	set(LAPACK_INCLUDE_DIRS ${INSTALL_DIR}/include/)
	set(LAPACK_LIB_DIR ${INSTALL_DIR}/lib64/lapack_blas_windows)
	set(LAPACKBLAS_LIBRARIES liblapack libblas libgcc_s_sjlj-1 libgfortran-3 libquadmath-0)
endif()

# fix cxsparse issue!
set(CXSPARSE_CMAKE_FLAGS -DCXSPARSE=ON -DCXSPARSE_INCLUDE_DIR_HINTS=${SUITESPARSE_INCLUDE_DIRS} -DCXSPARSE_LIBRARY_DIR_HINTS=${SUITESPARSE_LIB_DIR})
set(SUITESPARSE_CMAKE_FLAGS ${LAPACK_CMAKE_FLAGS} ${CXSPARSE_CMAKE_FLAGS} -DSuiteSparse_DIR=${SUITESPARSE_LIB_DIR}/cmake/suitesparse-5.4.0 -DSUITESPARSE_INCLUDE_DIR_HINTS=${SUITESPARSE_INCLUDE_DIRS} -DSUITESPARSE_LIBRARY_DIR_HINTS=${SUITESPARSE_LIB_DIR})


