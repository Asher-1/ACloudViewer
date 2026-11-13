set(FOLDER_NAME "sensor")

COLMAP_ADD_SOURCES(
    database.h database.cc
    imu.h imu.cc
    models.h models.cc
    models/base.h
    models/pinhole.h
    models/radial.h
    models/fisheye_radial.h
    models/opencv.h
    models/fov.h
    models/spherical.h
    models/thin_prism.h
    models/division.h
    models/eucm.h
    models/fisheye.h
    models/jacobian.h
    models/runtime.h
    specs.h specs.cc
    bitmap.h bitmap.cc
)

COLMAP_ADD_TEST(sensor_database_test database_test.cc)
COLMAP_ADD_TEST(imu_test imu_test.cc)
COLMAP_ADD_TEST(bitmap_test bitmap_test.cc)
COLMAP_ADD_TEST(models_test models_test.cc)
COLMAP_ADD_TEST(jacobian_test models/jacobian_test.cc)
