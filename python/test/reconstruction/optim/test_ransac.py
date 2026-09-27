"""Port of upstream pycolmap optim/ransac_test.py (Open3D style)."""
import numpy as np


def test_ransac_options_surface(estimators):
    opts = estimators.RANSACOptions()
    opts.max_error = 1.0
    assert opts.max_error == 1.0
    opts.min_num_trials = 10
    assert opts.min_num_trials == 10
    opts.max_num_trials = 1000
    assert opts.max_num_trials == 1000
    opts.confidence = 0.999
    assert opts.confidence == 0.999
    opts.check()


def test_ransac_options_random_seed(estimators):
    opts = estimators.RANSACOptions()
    opts.random_seed = 7
    assert opts.random_seed == 7
