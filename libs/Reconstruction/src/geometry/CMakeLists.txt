# Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

set(FOLDER_NAME "geometry")

COLMAP_ADD_SOURCES(
    pose_prior.h pose_prior.cc
    rigid3.h rigid3.cc
)

# rigid3_test lands with the test-infrastructure port (gtest -> Boost adaptation).
# GPS ENU parity cases live in base/gps_test.cc (upstream geometry/gps_test.cc
# dbb41680 subset; UTM cases deferred with the UTM API until W7).
