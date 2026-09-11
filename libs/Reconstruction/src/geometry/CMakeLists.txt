# Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

set(FOLDER_NAME "geometry")

COLMAP_ADD_SOURCES(
    pose_prior.h pose_prior.cc
    rigid3.h rigid3.cc
    sim3.h sim3.cc
    bbox.h bbox.cc
    normalization.h normalization.cc
)

# rigid3_test lands with the test-infrastructure port (gtest -> Boost adaptation).
# GPS ENU parity cases live in base/gps_test.cc (upstream geometry/gps_test.cc
# dbb41680 subset; UTM cases deferred with the UTM API until W7).

if (TESTS_ENABLED)
    COLMAP_ADD_TEST(rigid3_test rigid3_test.cc)
    COLMAP_ADD_TEST(rigid3_matchers_test rigid3_matchers_test.cc)
    COLMAP_ADD_TEST(sim3_test sim3_test.cc)
    COLMAP_ADD_TEST(sim3_matchers_test sim3_matchers_test.cc)
    COLMAP_ADD_TEST(pose_prior_test pose_prior_test.cc)
    COLMAP_ADD_TEST(bbox_test bbox_test.cc)
    COLMAP_ADD_TEST(normalization_test normalization_test.cc)
endif ()
