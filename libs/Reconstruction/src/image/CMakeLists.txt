set(FOLDER_NAME "image")
# Moved from base/ (upstream image/ parity).
COLMAP_ADD_SOURCES(
    line.h line.cc
    undistortion.h undistortion.cc
    warp.h warp.cc
)
if (TESTS_ENABLED)
    COLMAP_ADD_TEST(line_test line_test.cc)
    COLMAP_ADD_TEST(undistortion_test undistortion_test.cc)
    COLMAP_ADD_TEST(warp_test warp_test.cc)
endif ()
