include(FetchContent)

# download the SDK and set XERCES_ROOT for later find_package() call
set(DOWNLOAD_RDB_URL https://github.com/Asher-1/cloudViewer_downloads/releases/download/1.5.0/xerces-c-3.2.0.7z)
message(STATUS "xerces-c: downloading from URL '${DOWNLOAD_RDB_URL}'")
FetchContent_Declare(ext_xerces
					 URL "${DOWNLOAD_RDB_URL}" 
					 URL_HASH MD5=9177486A21C7706204C2AED3B7C3B6E7
					 DOWNLOAD_DIR ${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/xerces)
FetchContent_GetProperties(ext_xerces)
if (NOT ext_xerces_POPULATED)
	FetchContent_Populate(ext_xerces)
	FetchContent_MakeAvailable(ext_xerces)
	set(XERCES_ROOT ${ext_xerces_SOURCE_DIR})
	message(STATUS "ext_xerces: populated xerces-c and setting 'XERCES_ROOT' to '${XERCES_ROOT}'")
endif()


# download the SDK and set E57FORMAT_ROOT for later find_package() call
set(DOWNLOAD_RDB_URL https://github.com/Asher-1/cloudViewer_downloads/releases/download/1.5.0/E57Format-2.1.7z)
message(STATUS "libE57Format: downloading from URL '${DOWNLOAD_RDB_URL}'")
FetchContent_Declare(ext_e57format
					 URL "${DOWNLOAD_RDB_URL}"
					 URL_HASH MD5=EB15A80B87A15AEDF1F39816F70A0998
					 DOWNLOAD_DIR ${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/e57format)
FetchContent_GetProperties(ext_e57format)
if (NOT ext_e57format_POPULATED)
	FetchContent_Populate(ext_e57format)
	FetchContent_MakeAvailable(ext_e57format)
	set(E57FORMAT_ROOT ${ext_e57format_SOURCE_DIR})
	message(STATUS "ext_E57format: populated libE57Format and setting 'E57FORMAT_ROOT' to '${E57FORMAT_ROOT}'")
endif()