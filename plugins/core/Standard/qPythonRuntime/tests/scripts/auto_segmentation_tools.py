import cvcorelib
import pycc


def test_enlarge_box():
    dimMin = cvcorelib.CCVector3(0, 0, 0)
    dimMax = cvcorelib.CCVector3(1, 1, 1)

    cvcorelib.CCMiscTools.EnlargeBox(dimMin, dimMax, 2.0)
    assert dimMin.x == -1.0
    assert dimMin.y == -1.0
    assert dimMin.z == -1.0
    assert dimMax.x == 2.0
    assert dimMax.y == 2.0
    assert dimMax.z == 2.0


def test_make_min_and_max_cubical():
    dimMin = cvcorelib.CCVector3(0, 0, 0)
    dimMax = cvcorelib.CCVector3(1, 0.5, 0.75)

    cvcorelib.CCMiscTools.MakeMinAndMaxCubical(dimMin, dimMax, 1.0)
    print(dimMin, dimMax)
    assert dimMin.x == -0.5
    assert dimMin.y == -0.75
    assert dimMin.z == -0.625
    assert dimMax.x == 1.5
    assert dimMax.y == 1.25
    assert dimMax.z == 1.375


def test_tribox_overlap():
    boxCenter = cvcorelib.CCVector3(0, 0, 0)
    boxHalfSize = cvcorelib.CCVector3(0.5, 0.5, 0.5)

    triangle = [
        cvcorelib.CCVector3(0.25, 0.25, 0.25),
        cvcorelib.CCVector3(2.0, 2.0, 0.25),
        cvcorelib.CCVector3(0.75, 1.25, 1.0)
    ]

    assert cvcorelib.CCMiscTools.TriBoxOverlap(boxCenter, boxHalfSize, triangle)

    # TODO: needs CCVector3D to be defined in cvcorelib to be uncommented
    # boxCenter = cvcorelib.CCVector3d(0, 0, 0)
    # boxHalfSize = cvcorelib.CCVector3d(0.5, 0.5, 0.5)
    #
    # triangle = [
    #     cvcorelib.CCVector3d(0.25, 0.25, 0.25),
    #     cvcorelib.CCVector3d(2.0, 2.0, 0.25),
    #     cvcorelib.CCVector3d(0.75, 1.25, 1.0)
    # ]
    #
    # assert cvcorelib.CCMiscTools.TriBoxOverlapd(boxCenter, boxHalfSize, triangle)


def test_connected_components(cloud):
    _numLabels = cvcorelib.AutoSegmentationTools.labelConnectedComponents(cloud, level=1)

    reference_clouds = cvcorelib.ReferenceCloudContainer()
    success = cvcorelib.AutoSegmentationTools.extractConnectedComponents(cloud, reference_clouds)
    assert success


def main():
    CC = pycc.GetCmdLineInstance()
    assert CC is not None

    cloudDescs = CC.clouds()
    assert len(cloudDescs) == 1
    cloud = cloudDescs[0].pc

    test_enlarge_box()
    test_make_min_and_max_cubical()
    test_tribox_overlap()
    test_connected_components(cloud)


if __name__ == '__main__':
    main()
