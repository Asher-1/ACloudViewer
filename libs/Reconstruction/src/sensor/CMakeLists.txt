set(FOLDER_NAME "sensor")

COLMAP_ADD_SOURCES(
    database.h database.cc
)

COLMAP_ADD_TEST(sensor_database_test database_test.cc)
