"""Port of upstream pycolmap scene/track_test.py (Open3D style)."""


def test_track_default(scene):
    t = scene.Track()
    assert t.length == 0


def test_track_init_elements(scene):
    t = scene.Track(elements=[scene.TrackElement(1, 0),
                              scene.TrackElement(2, 3)])
    assert t.length == 2


def test_track_add_delete_element(scene):
    t = scene.Track()
    t.add_element(1, 0)
    t.add_element(scene.TrackElement(2, 3))
    assert t.length == 2
    t.delete_element(image_id=1, point2D_idx=0)
    assert t.length == 1
    t.delete_element(index=0)
    assert t.length == 0


def test_track_element_access(scene):
    t = scene.Track(elements=[scene.TrackElement(3, 5)])
    element = t.element(0)
    assert element.image_id == 3
    assert element.point2D_idx == 5
    t.set_element(0, scene.TrackElement(9, 9))
    assert t.element(0).image_id == 9


def test_track_reserve_compress(scene):
    t = scene.Track()
    t.reserve(16)
    t.compress()
    assert t.length == 0
