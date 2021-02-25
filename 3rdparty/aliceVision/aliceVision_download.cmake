include(FetchContent)

# Setup download links
if(WIN32)
    set_local_or_remote_url(
        DOWNLOAD_URL_PRIMARY
        LOCAL_URL   "${THIRD_PARTY_DOWNLOAD_DIR}/AliceVision-2.2.0-win64.zip"
        REMOTE_URLS "https://github.com/alicevision/AliceVision/releases/download/v2.2.0/AliceVision-2.2.0-win64.zip"
    )
elseif(APPLE)
    set_local_or_remote_url(
        DOWNLOAD_URL_PRIMARY
        LOCAL_URL   "${THIRD_PARTY_DOWNLOAD_DIR}/AliceVision-2.2.0-linux.tar.gz"
        REMOTE_URLS "https://github.com/alicevision/AliceVision/releases/download/v2.2.0/AliceVision-2.2.0-linux.tar.gz"
    )
else() # Linux
    set_local_or_remote_url(
        DOWNLOAD_URL_PRIMARY
        LOCAL_URL   "${THIRD_PARTY_DOWNLOAD_DIR}/AliceVision-2.2.0-linux.tar.gz"
        REMOTE_URLS "https://github.com/alicevision/AliceVision/releases/download/v2.2.0/AliceVision-2.2.0-linux.tar.gz"
    )
endif()

# ExternalProject_Add happens at build time.
ExternalProject_Add(
    ext_alicevision
    PREFIX alicevision
    URL ${DOWNLOAD_URL_PRIMARY}
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_IN_SOURCE ON
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)
ExternalProject_Get_Property(ext_alicevision SOURCE_DIR)
set(AliceVision_ROOT ${SOURCE_DIR})
set(AliceVision_DIR ${SOURCE_DIR}/share/aliceVision/cmake)

message(STATUS "AliceVision is located at ${AliceVision_ROOT}")

set(AliceVision_LIBRARIES aliceVision_system aliceVision_sfmDataIO )
if (UNIX OR WIN32)
    set(AliceVision_LIBRARIES ${AliceVision_LIBRARIES} aliceVision_sfmDataIO)
endif()
