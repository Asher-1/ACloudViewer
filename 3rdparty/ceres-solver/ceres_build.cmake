include(ExternalProject)

ExternalProject_Add(
   ext_ceres
   PREFIX ceres
   URL https://github.com/ceres-solver/ceres-solver/archive/1.14.0.zip
   URL_HASH MD5=26b255b7a9f330bbc1def3b839724a2a
   DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/ceres"
   BUILD_IN_SOURCE 0
   BUILD_ALWAYS 0
   INSTALL_DIR ${CLOUDVIEWER_EXTERNAL_INSTALL_DIR}
   UPDATE_COMMAND ""
   CMAKE_ARGS
          ${EIGEN_CMAKE_FLAGS}
          ${GLOG_CMAKE_FLAGS}
          ${SUITESPARSE_CMAKE_FLAGS}
          -DBUILD_SHARED_LIBS=$<IF:$<PLATFORM_ID:Linux>,ON,OFF>
          -DCMAKE_BUILD_TYPE=$<IF:$<PLATFORM_ID:Windows>,${CMAKE_BUILD_TYPE},Release>
          -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
          -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
          -DGFLAGS=ON
          -DLAPACK=ON
          -DSUITESPARSE=ON
          -DOPENMP=ON
          -DBUILD_TESTING=OFF
          -DBUILD_EXAMPLES=OFF
          -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
          DEPENDS ${INTERNAL_EIGEN3_TARGET} ${SUITESPARSE_TARGET} ${GLOG_TARGET}
)

ExternalProject_Get_Property(ext_ceres INSTALL_DIR)
set(CERES_INCLUDE_DIRS ${INSTALL_DIR}/include/) # "/" is critical.
set(CERES_LIB_DIR ${INSTALL_DIR}/lib)
if (WIN32)
    set(EXT_CERES_LIBRARIES ceres$<$<CONFIG:Debug>:-debug>)
else()
    set(EXT_CERES_LIBRARIES ceres)
    set(library_filename ${CMAKE_SHARED_LIBRARY_PREFIX}${EXT_CERES_LIBRARIES}${CMAKE_SHARED_LIBRARY_SUFFIX})
    install_ext( FILES ${CERES_LIB_DIR}/${library_filename} ${INSTALL_DESTINATIONS} "")
endif()

set(CERES_CMAKE_FLAGS ${SUITESPARSE_CMAKE_FLAGS} ${EIGEN_CMAKE_FLAGS} ${GLOG_CMAKE_FLAGS} -DCeres_DIR=${CERES_LIB_DIR}/cmake/Ceres)
