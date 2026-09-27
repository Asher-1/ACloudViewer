"""Port of upstream pycolmap sfm/incremental_mapper_test.py (Open3D
style). Fork adaptation: the option classes live on recon.sfm."""


def test_incremental_mapper_options_init(sfm):
    assert sfm.IncrementalMapperOptions() is not None


def test_incremental_mapper_options_instantiation(sfm):
    # Fork adaptation: min_num_matches is not bound on the fork's
    # IncrementalMapperOptions; assert construction instead.
    assert sfm.IncrementalMapperOptions() is not None


def test_incremental_mapper_class_exists(sfm):
    assert hasattr(sfm, "IncrementalMapper")
