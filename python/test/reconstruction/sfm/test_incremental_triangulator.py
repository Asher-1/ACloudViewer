"""Port of upstream pycolmap sfm/incremental_triangulator_test.py (Open3D
style)."""


def test_incremental_triangulator_options_init(sfm):
    assert sfm.IncrementalTriangulatorOptions() is not None


def test_incremental_triangulator_options_max_transitivity(sfm):
    options = sfm.IncrementalTriangulatorOptions()
    options.max_transitivity = 2
    assert options.max_transitivity == 2


def test_incremental_triangulator_options_create_max_angle_error(sfm):
    options = sfm.IncrementalTriangulatorOptions()
    options.create_max_angle_error = 3.0
    assert options.create_max_angle_error == 3.0


def test_incremental_triangulator_options_check(sfm):
    assert sfm.IncrementalTriangulatorOptions().check()


def test_incremental_triangulator_class_exists(sfm):
    assert hasattr(sfm, "IncrementalTriangulator")
