# Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

set(FOLDER_NAME "geometry")

COLMAP_ADD_SOURCES(
    pose_prior.h pose_prior.cc
    rigid3.h rigid3.cc
)

# rigid3_test lands with the test-infrastructure port (gtest -> Boost adaptation).
