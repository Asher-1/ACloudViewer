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
    COLMAP_ADD_TEST(random_eigen_test random_eigen_test.cc)
    COLMAP_ADD_TEST(union_find_test union_find_test.cc)
endif ()

# Moved from base/ (graph_cut/polynomial) and util/ (math/matrix/random,
# upstream math/ parity).
COLMAP_ADD_SOURCES(
    graph_cut.h graph_cut.cc
    polynomial.h polynomial.cc
    math.h math.cc
    matrix.h
    random.h random.cc
)
if (TESTS_ENABLED)
    COLMAP_ADD_TEST(graph_cut_test graph_cut_test.cc)
    COLMAP_ADD_TEST(polynomial_test polynomial_test.cc)
    COLMAP_ADD_TEST(math_test math_test.cc)
    COLMAP_ADD_TEST(matrix_test matrix_test.cc)
    COLMAP_ADD_TEST(random_test random_test.cc)
endif ()
