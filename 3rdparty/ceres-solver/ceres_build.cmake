include(ExternalProject)
#if (WIN32)
#    set( EIGEN_ROOT "" CACHE PATH "Eigen library root directory" )
#	if ( NOT EIGEN_ROOT )
#		message( SEND_ERROR "No Eigen library root specified (EIGEN_ROOT)" )
#	else()
#		include_directories( ${EIGEN_ROOT} )
#	endif()
#    set(EIGEN_CMAKE_FLAGS -DEigen3_DIR:PATH=${EIGEN_ROOT})
#else()
#    set(EIGEN_CMAKE_FLAGS -DEigen3_DIR:PATH=${EIGEN_ROOT_DIR}/share/eigen3/cmake -DEIGEN3_INCLUDE_DIR=${EIGEN_INCLUDE_DIR} -DEIGEN3_INCLUDE_DIRS=${EIGEN_INCLUDE_DIR} -DEIGEN_INCLUDE_DIR=${EIGEN_INCLUDE_DIR} -DEigen_INCLUDE_DIR=${EIGEN_INCLUDE_DIR})
#endif()

set_local_or_remote_url(
    DOWNLOAD_URL_PRIMARY
    LOCAL_URL   "${THIRD_PARTY_DOWNLOAD_DIR}/ceres-solver-1.14.0.zip"
    REMOTE_URLS "https://github.com/ceres-solver/ceres-solver/archive/1.14.0.zip"
)

if (WIN32)
    ExternalProject_Add(
       ext_ceres
       PREFIX ceres-solver
       URL ${DOWNLOAD_URL_PRIMARY} ${DOWNLOAD_URL_FALLBACK}
       URL_HASH MD5=26b255b7a9f330bbc1def3b839724a2a
       UPDATE_COMMAND ""
       CMAKE_ARGS
              ${EIGEN_CMAKE_FLAGS}
              ${GLOG_CMAKE_FLAGS}
              ${SUITESPARSE_CMAKE_FLAGS}
              -DBUILD_SHARED_LIBS=OFF
              -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
              -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
              -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
              -DGFLAGS=ON
              -DCMAKE_CXX_FLAGS=/DGOOGLE_GLOG_DLL_DECL=
              -DLAPACK=ON
              -DSUITESPARSE=ON
              -DOPENMP=ON
              -DBUILD_TESTING=OFF
              -DBUILD_EXAMPLES=OFF
              -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
              DEPENDS ${EIGEN3_TARGET} ${SUITESPARSE_TARGET}
       )
else()
    ExternalProject_Add(
       ext_ceres
       PREFIX ceres-solver
       URL ${DOWNLOAD_URL_PRIMARY} ${DOWNLOAD_URL_FALLBACK}
       URL_HASH MD5=26b255b7a9f330bbc1def3b839724a2a
       UPDATE_COMMAND ""
       CMAKE_ARGS
              ${EIGEN_CMAKE_FLAGS}
              ${GLOG_CMAKE_FLAGS}
              ${SUITESPARSE_CMAKE_FLAGS}
              -DBUILD_SHARED_LIBS=ON
              -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
              -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
              -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
              -DGFLAGS=ON
              -DLAPACK=ON
              -DSUITESPARSE=ON
              -DOPENMP=ON
              -DBUILD_TESTING=OFF
              -DBUILD_EXAMPLES=OFF
              -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
              DEPENDS ${INTERNAL_EIGEN3_TARGET} ${SUITESPARSE_TARGET}
       )
endif()

ExternalProject_Get_Property(ext_ceres INSTALL_DIR)
set(CERES_INCLUDE_DIRS ${INSTALL_DIR}/include/) # "/" is critical.
set(CERES_LIB_DIR ${INSTALL_DIR}/lib)
set(EXT_CERES_LIBRARIES ceres$<$<CONFIG:Debug>:-debug>)
set(CERES_INSTALL_DIR ${INSTALL_DIR}/lib/cmake)
