include(FetchContent)

# Keep the exact revision used by the COLMAP one-sided focal solver. PoseLib is
# populated by default with reconstruction so the one-sided focal solver has
# the same pinned minimal-solver dependency as COLMAP.
FetchContent_Declare(
    poselib
    URL https://github.com/PoseLib/PoseLib/archive/fa7280fee27f97aff31ae7f98bab7f583fac7d08.zip
    URL_HASH SHA256=5408d4ae8ce367cb2f076bc6c5f0f6f78abd3573d2c015304b04e46f23455f5b
    DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/poselib"
)

set(MARCH_NATIVE OFF CACHE BOOL "" FORCE)
set(POSELIB_BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(poselib)
