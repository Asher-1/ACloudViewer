set(FOLDER_NAME "inertial")

COLMAP_ADD_SOURCES(
    preintegration.h preintegration.cc
    preintegration_cost.h
)

COLMAP_ADD_TEST(preintegration_test preintegration_test.cc)
COLMAP_ADD_TEST(preintegration_cost_test preintegration_cost_test.cc)
