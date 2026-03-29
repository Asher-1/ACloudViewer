include(ExternalProject)

set(NANOFLANN_VERSION "v1.5.0")
set(NANOFLANN_SHA256 "89aecfef1a956ccba7e40f24561846d064f309bc547cc184af7f4426e42f8e65")

ExternalProject_Add(ext_nanoflann
    PREFIX nanoflann
    URL https://github.com/jlblancoc/nanoflann/archive/refs/tags/${NANOFLANN_VERSION}.tar.gz
    URL_HASH SHA256=${NANOFLANN_SHA256}
    DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/nanoflann"
    TIMEOUT 120
    INACTIVITY_TIMEOUT 60
    TLS_VERIFY ON
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

ExternalProject_Get_Property(ext_nanoflann SOURCE_DIR)
set(NANOFLANN_INCLUDE_DIRS ${SOURCE_DIR}/include/) # "/" is critical.