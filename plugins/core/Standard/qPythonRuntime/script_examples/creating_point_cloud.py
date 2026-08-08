import pycc
import laspy
import numpy as np
import os

# This example shows how to create a ccPointCloud from your custom data
# here, we will load data from a las file using laspy.

# Note that the resulting ccPointCloud's point values and scalar field values
# will be copies.

# Try to find a LAS file from env var; otherwise generate synthetic data.
# NOTE: Do NOT use tkinter here — Tk 8.6 calls [NSApp macOSVersion]
# which crashes inside Qt's QNSApplication subclass.
path_to_las = os.environ.get("ACV_DEMO_LAS_PATH", "")


def _show_in_viewer(point_cloud):
    """Add point cloud to the ACloudViewer DB if running in embedded mode."""
    try:
        cc = pycc.GetInstance()
    except AttributeError:
        return  # not in embedded mode, nothing to do
    # Compute min/max for all scalar fields so colour maps display correctly
    for i in range(point_cloud.getNumberOfScalarFields()):
        point_cloud.getScalarField(i).computeMinAndMax()
    point_cloud.showSF(True)
    cc.addToDB(point_cloud)
    cc.updateUI()


if path_to_las and os.path.isfile(path_to_las):
    # --- LAS file processing ---

    # Be aware that ACloudViewer stores coordinates on 32 bit floats.
    # To avoid losing too much precision you should 'shift' your coordinates
    # if they are 64 bit floats (which is the default in python land)
    las = laspy.read(path_to_las)

    xs = (las.x - las.header.x_min).astype(pycc.PointCoordinateType)
    ys = (las.y - las.header.y_min).astype(pycc.PointCoordinateType)
    zs = (las.z - las.header.z_min).astype(pycc.PointCoordinateType)

    point_cloud = pycc.ccPointCloud(xs, ys, zs)
    # Add the global shift to ACloudViewer so that it can use it,
    # for example to display the real coordinates in point picking tool
    point_cloud.setGlobalShift(-las.header.x_min, -las.header.y_min, -las.header.z_min)
    point_cloud.setName(path_to_las)
    print(point_cloud.size())

    assert np.all(xs == point_cloud.points()[..., 0])

    # Adding scalar field & copying values the manual way
    idx = point_cloud.addScalarField("classification")

    classification_array = point_cloud.getScalarField(idx).asArray()
    classification_array[:] = las.classification[:]
    print(classification_array)

    # Or give the values directly
    idx = point_cloud.addScalarField("intensity", las.intensity)
    intensity_array = point_cloud.getScalarField(idx).asArray()
    print(intensity_array[:])

    _show_in_viewer(point_cloud)

else:
    # --- Synthetic data ---
    print("No LAS file selected. Generating synthetic point cloud instead.")
    n = 10000
    xs = np.random.uniform(0, 100, n).astype(pycc.PointCoordinateType)
    ys = np.random.uniform(0, 100, n).astype(pycc.PointCoordinateType)
    zs = (np.sin(xs / 10) * 20 + np.cos(ys / 10) * 20 + 50).astype(pycc.PointCoordinateType)
    point_cloud = pycc.ccPointCloud(xs, ys, zs)
    point_cloud.setName("synthetic_wave")
    idx = point_cloud.addScalarField("height")
    sf = point_cloud.getScalarField(idx).asArray()
    sf[:] = zs
    print(f"Generated synthetic cloud with {point_cloud.size()} points")

    point_cloud.getScalarField(idx).computeMinAndMax()
    point_cloud.setCurrentDisplayedScalarField(idx)
    point_cloud.showSF(True)

    _show_in_viewer(point_cloud)
