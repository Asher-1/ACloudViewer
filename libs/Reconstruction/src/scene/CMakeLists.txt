# Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

set(FOLDER_NAME "scene")

# W9 (synthetic dataset generator): enabled with the W3-2a frame-aware data
# model. The remaining fork adaptations against upstream dbb41680 are marked
# inline in synthetic.cc and recorded in colmap_alignment_manifest.json
# (synthetic_dataset).
COLMAP_ADD_SOURCES(
    synthetic.h synthetic.cc
)

if (TESTS_ENABLED)
    COLMAP_ADD_TEST(synthetic_test synthetic_test.cc)
endif ()
