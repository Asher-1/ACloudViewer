# Upstream parity (d3ccaf35 calibration/CMakeLists.txt): single-view
# calibration module. The fork builds it into the monolithic ColmapLib via
# COLMAP_ADD_SOURCES; the ONNX inference backend is excluded per the fork
# adjudication (ggml/AICore replaces onnx). Asset-dependent tests use the
# shared AICore catalog/cache; preprocessing and argument checks stay offline.

set(FOLDER_NAME "calibration")

COLMAP_ADD_SOURCES(
    anycalib.h anycalib.cc
    single_view_calibrator.h single_view_calibrator.cc
    ray_fitting.h ray_fitting.cc
    resources.h
)

if (TESTS_ENABLED)
    COLMAP_ADD_TEST(single_view_calibrator_test single_view_calibrator_test.cc)
    COLMAP_ADD_TEST(ray_fitting_test ray_fitting_test.cc)
    COLMAP_ADD_TEST(anycalib_test anycalib_test.cc)
    set_property(TEST calibration/anycalib_test APPEND PROPERTY LABELS model)
    add_test(NAME calibration/anycalib_unit_test
        COMMAND $<TARGET_FILE:anycalib_test>
            "--gtest_filter=PrepareAnyCalibInputTest.*:CreateAnyCalibCalibratorTest.NullAnyCalibOptionsThrows:CreateAnyCalibCalibratorTest.MissingModelFileThrows")
    cloudviewer_set_test_runtime_environment(calibration/anycalib_unit_test
        "${COLMAP_TEST_RUNTIME_DIR}")
endif ()
