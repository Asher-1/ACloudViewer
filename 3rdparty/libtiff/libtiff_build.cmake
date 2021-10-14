include(ExternalProject)

ExternalProject_Add(
       ext_libtiff
       PREFIX libtiff
       URL http://download.osgeo.org/libtiff/tiff-4.2.0.tar.gz
       URL_HASH MD5=2bbf6db1ddc4a59c89d6986b368fc063
       BUILD_IN_SOURCE ON
       BUILD_ALWAYS 0
       UPDATE_COMMAND ""
       CONFIGURE_COMMAND <SOURCE_DIR>/configure
                --prefix=<INSTALL_DIR>
                --enable-static
                --disable-shared
       BUILD_COMMAND $(MAKE)
       DEPENDS ${ZLIB_TARGET}
       )

ExternalProject_Get_Property(ext_libtiff INSTALL_DIR)
set(TIFF_INCLUDE_DIRS ${INSTALL_DIR}/include/)
set(TIFF_LIB_DIR ${INSTALL_DIR}/lib)
set(TIFF_LIBRARIES tiff tiffxx)

SET(TIFF_CMAKE_FLAGS -DTIFF_LIBRARY=${INSTALL_DIR}/lib/libtiff.so -DTIFF_INCLUDE_DIR=${INSTALL_DIR}/include)
