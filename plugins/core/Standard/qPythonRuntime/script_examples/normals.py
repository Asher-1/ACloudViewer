import pycc
import cccorelib

CC = pycc.GetInstance()

entities = CC.getSelectedEntities()
if not entities:
    raise RuntimeError("No entity selected. Select a point cloud first.")

pc = entities[0]


progress = pycc.ccProgressDialog()
progress.start()

success = pc.computeNormalsWithOctree(
    cccorelib.CV_LOCAL_MODEL_TYPES.LS,
    pycc.ccNormalVectors.Orientation.UNDEFINED,
    5.0,
    pDlg=progress,
)
if not success:
    raise RuntimeError("Failed to compute normals")

kNN = 6
success = pc.orientNormalsWithMST(6, progress)
if not success:
    raise RuntimeError("Failed to orient normals")

pycc.GetInstance().updateUI()
pycc.GetInstance().redrawAll()