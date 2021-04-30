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
    set(STATIC_FREEIMAGE_INCLUDE_DIR "${CLOUDVIEWER_EXTERNAL_INSTALL_DIR}/include/")
    set(STATIC_FREEIMAGE_LIB_DIR "${CLOUDVIEWER_EXTERNAL_INSTALL_DIR}/lib")
    ExternalProject_Add(
        ext_freeimage
        PREFIX ${CLOUDVIEWER_EXTERNAL_BUILD_DIR}
        URL ${DOWNLOAD_URL_PRIMARY} ${DOWNLOAD_URL_FALLBACK}
        URL_HASH MD5=393d3df75b14cbcb4887da1c395596e2
        DOWNLOAD_DIR ${CLOUDVIEWER_EXTERNAL_BUILD_DIR}/download/freeimage
        BUILD_IN_SOURCE 0
        BUILD_ALWAYS 0
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ""
        SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/freeimage
        BINARY_DIR ${CLOUDVIEWER_EXTERNAL_BUILD_DIR}/freeimage_build
        INSTALL_DIR ${CLOUDVIEWER_EXTERNAL_INSTALL_DIR}
        PATCH_COMMAND ""
        BUILD_COMMAND  ""
        INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/Dist/x64 ${STATIC_FREEIMAGE_INCLUDE_DIR}
        && ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/Dist/x64 ${STATIC_FREEIMAGE_LIB_DIR}
    )

    set(FREEIMAGE_INCLUDE_DIRS ${STATIC_FREEIMAGE_INCLUDE_DIR}) # "/" is critical.
    set(FREEIMAGE_LIB_DIR ${STATIC_FREEIMAGE_LIB_DIR})
    set(EXT_FREEIMAGE_LIBRARIES FreeImage)

    copy_shared_library(ext_freeimage
            LIB_DIR      ${FREEIMAGE_LIB_DIR}
            LIBRARIES    ${EXT_FREEIMAGE_LIBRARIES}
    )
else()
    ExternalProject_Add(
        ext_freeimage
        PREFIX ${CLOUDVIEWER_EXTERNAL_BUILD_DIR}
        URL ${DOWNLOAD_URL_PRIMARY} ${DOWNLOAD_URL_FALLBACK}
        URL_HASH MD5=f8ba138a3be233a3eed9c456e42e2578
        DOWNLOAD_DIR ${CLOUDVIEWER_EXTERNAL_BUILD_DIR}/download/freeimage
        BUILD_IN_SOURCE 0
        BUILD_ALWAYS 0
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ""
        SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/freeimage
        BINARY_DIR ${CLOUDVIEWER_EXTERNAL_BUILD_DIR}/freeimage_build
        INSTALL_DIR ${CLOUDVIEWER_EXTERNAL_INSTALL_DIR}
        PATCH_COMMAND ${CMAKE_COMMAND} -E copy ${CloudViewer_3RDPARTY_DIR}/freeimage/Makefile.gnu <SOURCE_DIR>
        BUILD_COMMAND  cd <SOURCE_DIR> && $(MAKE) CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER}
        INSTALL_COMMAND cd <SOURCE_DIR> && $(MAKE) install DESTDIR=<INSTALL_DIR> CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER}
    )

    ExternalProject_Get_Property(ext_freeimage INSTALL_DIR)
    set(FREEIMAGE_INCLUDE_DIRS ${INSTALL_DIR}/include/) # "/" is critical.
    set(FREEIMAGE_LIB_DIR ${INSTALL_DIR}/lib)
    set(EXT_FREEIMAGE_LIBRARIES freeimage)
endif()
