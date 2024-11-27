include(ExternalProject)

if (${GLIBCXX_USE_CXX11_ABI})
    set(CUSTOM_GLIBCXX_USE_CXX11_ABI 1)
    message(STATUS "add -D_GLIBCXX_USE_CXX11_ABI=${CUSTOM_GLIBCXX_USE_CXX11_ABI} support for suitesparse")
else ()
    set(CUSTOM_GLIBCXX_USE_CXX11_ABI 0)
    message(STATUS "add -D_GLIBCXX_USE_CXX11_ABI=${CUSTOM_GLIBCXX_USE_CXX11_ABI} support for suitesparse")
endif ()

ExternalProject_Add(
       ext_suitesparse
       PREFIX suitesparse
       URL https://codeload.github.com/jlblancoc/suitesparse-metis-for-windows/zip/7bc503bfa2c4f1be9176147d36daf9e18340780a
       URL_HASH MD5=e7c27075e8e0afc9d2cf188630090946
       DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/suitesparse"
       BUILD_IN_SOURCE 0
       BUILD_ALWAYS 0
       UPDATE_COMMAND ""
       INSTALL_DIR ${CLOUDVIEWER_EXTERNAL_INSTALL_DIR}
	   # fix compiling bugs on windows
	   PATCH_COMMAND ${CMAKE_COMMAND} -E copy ${CloudViewer_3RDPARTY_DIR}/suitesparse/CMakeLists.txt <SOURCE_DIR>
       CMAKE_ARGS
            ${MAC_OMP_FLAGS}
            -DOPENMP=ON # fix fatal error: 'omp.h' file not found on macos
            -DBUILD_SHARED_LIBS=OFF
            -DCMAKE_BUILD_TYPE=$<IF:$<PLATFORM_ID:Windows>,${CMAKE_BUILD_TYPE},Release>
            $<IF:$<PLATFORM_ID:Windows>,"",-DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=${CUSTOM_GLIBCXX_USE_CXX11_ABI}>
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
            -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
            # -DCUDA_INCLUDE_DIRS=${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
            # -DWITH_CUDA=${BUILD_CUDA_MODULE}
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
                                metis
                                )

if (BUILD_CUDA_MODULE)
    if(NOT WIN32)
        set(SUITESPARSE_LIBRARIES ${SUITESPARSE_LIBRARIES} SuiteSparse_GPURuntime GPUQREngine)
    endif()
endif()

if(WIN32)				
    # for compiling ceres-solver
    set(LAPACK_LIBRARIES ${INSTALL_DIR}/lib64/lapack_blas_windows/liblapack.lib)
    set(BLAS_LIBRARIES ${INSTALL_DIR}/lib64/lapack_blas_windows/libblas.lib)
    set(LAPACK_CMAKE_FLAGS -DBLAS_LIBRARIES=${BLAS_LIBRARIES} -DLAPACK_LIBRARIES=${LAPACK_LIBRARIES})

    # just for deploy dll on windows
    set(LAPACK_INCLUDE_DIRS ${INSTALL_DIR}/include/)
    set(LAPACK_LIB_DIR ${INSTALL_DIR}/lib64/lapack_blas_windows)
    set(LAPACKBLAS_LIBRARIES liblapack libblas)
    set(LAPACKBLAS_DLL ${LAPACKBLAS_LIBRARIES} libgcc_s_sjlj-1 libgfortran-3 libquadmath-0)

    # for debugging
    copy_shared_library(ext_suitesparse
        LIB_DIR      ${LAPACK_LIB_DIR}
        LIBRARIES    ${LAPACKBLAS_DLL})
	
    foreach( filename ${LAPACKBLAS_DLL} )
        set(library_filename ${CMAKE_SHARED_LIBRARY_PREFIX}${filename}${CMAKE_SHARED_LIBRARY_SUFFIX})
        cloudViewer_install_ext( FILES ${LAPACK_LIB_DIR}/${library_filename} ${INSTALL_DESTINATIONS} "")
    endforeach()
	
endif()

# fix cxsparse issue for ceres-solver!
set(CXSPARSE_CMAKE_FLAGS -DCXSPARSE=ON -DCXSPARSE_INCLUDE_DIR_HINTS=${SUITESPARSE_INCLUDE_DIRS} -DCXSPARSE_LIBRARY_DIR_HINTS=${SUITESPARSE_LIB_DIR})

# for compiling ceres-solver
set(SUITESPARSE_CMAKE_FLAGS ${LAPACK_CMAKE_FLAGS} ${CXSPARSE_CMAKE_FLAGS} -DSuiteSparse_DIR=${SUITESPARSE_LIB_DIR}/cmake/suitesparse-5.4.0 -DSUITESPARSE_INCLUDE_DIR_HINTS=${SUITESPARSE_INCLUDE_DIRS} -DSUITESPARSE_LIBRARY_DIR_HINTS=${SUITESPARSE_LIB_DIR})


