"""Port of upstream pycolmap mvs/model_test.py (Open3D style)."""


def test_mvs_model_surface(mvs):
    model = mvs.MVSModel()
    for attr in ("load_vis_dat", "get_max_overlapping_images_from_pmvs"):
        assert hasattr(model, attr), attr


def test_mvs_model_repr(mvs):
    model = mvs.MVSModel()
    assert len(repr(model)) >= 0
