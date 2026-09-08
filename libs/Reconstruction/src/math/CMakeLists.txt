# Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

set(FOLDER_NAME "math")

# W4 (GLomap global SfM stack): upstream math helpers. union_find.h and
# connected_components.h are header-only; spanning_tree carries a .cc.
COLMAP_ADD_SOURCES(
    spanning_tree.h spanning_tree.cc
    connected_components.h
)

if (TESTS_ENABLED)
    COLMAP_ADD_TEST(connected_components_test connected_components_test.cc)
    COLMAP_ADD_TEST(spanning_tree_test spanning_tree_test.cc)
endif ()
