# Upstream parity (d3ccaf35 calibration/CMakeLists.txt): single-view
# calibration module. The fork builds it into the monolithic ColmapLib via
# COLMAP_ADD_SOURCES; the ONNX inference backend is excluded per the fork
# adjudication (ggml/AICore replaces onnx), so the anycalib_test inference
# cases GTEST_SKIP and CreateAnyCalibCalibrator throws at runtime.

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
endif ()
