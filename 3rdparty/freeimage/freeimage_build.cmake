include(ExternalProject)

find_package(Git QUIET REQUIRED)

if (WIN32)
    set(FREEIMAGE_URL https://github.com/Asher-1/cloudViewer_downloads/releases/download/1.9.2/FreeImage3180Win32Win64.zip)
    set(FREEIMAGE_HASH 393d3df75b14cbcb4887da1c395596e2)
else () # Linux or Mac
    set(FREEIMAGE_URL https://sourceforge.net/projects/freeimage/files/Source%20Distribution/3.18.0/FreeImage3180.zip/download)
    set(FREEIMAGE_HASH f8ba138a3be233a3eed9c456e42e2578)
endif ()

if(WIN32)
    set(STATIC_FREEIMAGE_INCLUDE_DIR "${CLOUDVIEWER_EXTERNAL_INSTALL_DIR}/include/")
    set(STATIC_FREEIMAGE_LIB_DIR "${CLOUDVIEWER_EXTERNAL_INSTALL_DIR}/lib")
	set(EX_FREEIMAGE_LIBRARIES FreeImage)
    ExternalProject_Add(
        ext_freeimage
        PREFIX freeimage
        URL ${FREEIMAGE_URL}
        URL_HASH MD5=${FREEIMAGE_HASH}
        DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/freeimage"
        BUILD_IN_SOURCE 0
        BUILD_ALWAYS 0
        INSTALL_DIR ${CLOUDVIEWER_EXTERNAL_INSTALL_DIR}
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ""
        PATCH_COMMAND ""
        BUILD_COMMAND  ""
        INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_if_different <SOURCE_DIR>/Dist/x64/${EX_FREEIMAGE_LIBRARIES}.h ${STATIC_FREEIMAGE_INCLUDE_DIR}
        && ${CMAKE_COMMAND} -E copy_if_different <SOURCE_DIR>/Dist/x64/${EX_FREEIMAGE_LIBRARIES}.lib ${STATIC_FREEIMAGE_LIB_DIR}
    )

    ExternalProject_Get_Property(ext_freeimage SOURCE_DIR)
    set(FREEIMAGE_INCLUDE_DIRS ${STATIC_FREEIMAGE_INCLUDE_DIR}) # "/" is critical.
    set(FREEIMAGE_LIB_DIR ${STATIC_FREEIMAGE_LIB_DIR})

    # for debugging
    copy_shared_library(ext_freeimage
            LIB_DIR      ${SOURCE_DIR}/Dist/x64
            LIBRARIES    ${EX_FREEIMAGE_LIBRARIES})
elseif (UNIX AND NOT APPLE)
    ExternalProject_Add(
        ext_freeimage
        PREFIX freeimage
        URL ${FREEIMAGE_URL}
        URL_HASH MD5=${FREEIMAGE_HASH}
        DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/freeimage"
        BUILD_IN_SOURCE 0
        BUILD_ALWAYS 0
        INSTALL_DIR ${CLOUDVIEWER_EXTERNAL_INSTALL_DIR}
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ""
        PATCH_COMMAND ${CMAKE_COMMAND} -E copy ${CloudViewer_3RDPARTY_DIR}/freeimage/Makefile.gnu <SOURCE_DIR>
        BUILD_COMMAND  cd <SOURCE_DIR> && $(MAKE) CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER}
        INSTALL_COMMAND cd <SOURCE_DIR> && $(MAKE) install DESTDIR=<INSTALL_DIR> CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER}
    )

    ExternalProject_Get_Property(ext_freeimage INSTALL_DIR)
    set(FREEIMAGE_INCLUDE_DIRS ${INSTALL_DIR}/include/) # "/" is critical.
    set(FREEIMAGE_LIB_DIR ${INSTALL_DIR}/lib)
    set(EX_FREEIMAGE_LIBRARIES freeimage)
elseif(APPLE)
    ExternalProject_Add(
        ext_freeimage
        PREFIX freeimage
        URL ${FREEIMAGE_URL}
        URL_HASH MD5=${FREEIMAGE_HASH}
        DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/freeimage"
        BUILD_IN_SOURCE ON
        INSTALL_DIR ${CLOUDVIEWER_EXTERNAL_INSTALL_DIR}
        CONFIGURE_COMMAND cp ${CloudViewer_3RDPARTY_DIR}/freeimage/Makefile.osx <SOURCE_DIR>
        BUILD_COMMAND make -f Makefile.osx
        UPDATE_COMMAND ""
        INSTALL_COMMAND make -f Makefile.osx install PREFIX=<INSTALL_DIR>
    )

    ExternalProject_Get_Property(ext_freeimage INSTALL_DIR)
    set(FREEIMAGE_INCLUDE_DIRS ${INSTALL_DIR}/include/) # "/" is critical.
    set(FREEIMAGE_LIB_DIR ${INSTALL_DIR}/lib)
    set(EX_FREEIMAGE_LIBRARIES freeimage)
endif()

if(WIN32)
    set(library_filename ${CMAKE_SHARED_LIBRARY_PREFIX}${EX_FREEIMAGE_LIBRARIES}${CMAKE_SHARED_LIBRARY_SUFFIX})
    cloudViewer_install_ext( FILES ${SOURCE_DIR}/Dist/x64/${library_filename} ${INSTALL_DESTINATIONS} "")
endif()

