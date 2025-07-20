include(ExternalProject)

ExternalProject_Add(ext_fbx
    PREFIX fbx
    URL https://github.com/Asher-1/cloudViewer_downloads/releases/download/1.7.0/FBX-2020.0.1.7z
    URL_HASH SHA256=361d69857201dc4b7bf237b95dc820a3e9814d11078875705c9fc31578cc3ecd
    DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/fbx"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

ExternalProject_Get_Property(ext_fbx SOURCE_DIR)
set(FBX_DIR ${SOURCE_DIR}) # Not using "/" is critical.
copy_shared_library(ext_fbx
	LIB_DIR      ${FBX_DIR}/lib/X64/release
	LIBRARIES    libfbxsdk)