include(ExternalProject)

# Setup download links
if(WIN32)
    set_local_or_remote_url(
        DOWNLOAD_URL_PRIMARY
        LOCAL_URL   "${THIRD_PARTY_DOWNLOAD_DIR}/FreeImage3180Win32Win64.zip"
        REMOTE_URLS "https://kent.dl.sourceforge.net/project/freeimage/Binary%20Distribution/3.18.0/FreeImage3180Win32Win64.zip"
    )
else() # Linux or Mac
    set_local_or_remote_url(
        DOWNLOAD_URL_PRIMARY
        LOCAL_URL   "${THIRD_PARTY_DOWNLOAD_DIR}/FreeImage3180Linux.zip"
        REMOTE_URLS "https://kent.dl.sourceforge.net/project/freeimage/Source%20Distribution/3.18.0/FreeImage3180.zip"
    )
endif()

if(WIN32)
set(FREEIMAGE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/freeimage_install)
set(STATIC_FREEIMAGE_INCLUDE_DIR "${FREEIMAGE_INSTALL_PREFIX}/include/")
set(STATIC_FREEIMAGE_LIB_DIR "${FREEIMAGE_INSTALL_PREFIX}/lib")
    ExternalProject_Add(
        ext_freeimage
        PREFIX freeimage
        URL ${DOWNLOAD_URL_PRIMARY} ${DOWNLOAD_URL_FALLBACK}
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ""
        BUILD_IN_SOURCE ON
        PATCH_COMMAND ""
        BUILD_COMMAND  ""
        INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/Dist/x64 ${FREEIMAGE_INSTALL_PREFIX}/include &&
        ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/Dist/x64 ${FREEIMAGE_INSTALL_PREFIX}/lib
    )

    set(FREEIMAGE_INCLUDE_DIRS ${STATIC_FREEIMAGE_INCLUDE_DIR}) # "/" is critical.
    set(FREEIMAGE_LIB_DIR ${STATIC_FREEIMAGE_LIB_DIR})
    set(FREEIMAGE_LIBRARIES FreeImage)
else()
    ExternalProject_Add(
        ext_freeimage
        PREFIX freeimage
        URL ${DOWNLOAD_URL_PRIMARY} ${DOWNLOAD_URL_FALLBACK}
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ""
        BUILD_IN_SOURCE ON
        PATCH_COMMAND ${CMAKE_COMMAND} -E copy ${CloudViewer_3RDPARTY_DIR}/freeimage/Makefile.gnu <SOURCE_DIR>
        BUILD_COMMAND  cd <SOURCE_DIR> && $(MAKE) CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER}
        INSTALL_COMMAND cd <SOURCE_DIR> && $(MAKE) install DESTDIR=<INSTALL_DIR> CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER}
    )

    ExternalProject_Get_Property(ext_freeimage INSTALL_DIR)
    set(FREEIMAGE_INCLUDE_DIRS ${INSTALL_DIR}/include/) # "/" is critical.
    set(FREEIMAGE_LIB_DIR ${INSTALL_DIR}/lib)
    set(FREEIMAGE_LIBRARIES freeimage)
endif()
