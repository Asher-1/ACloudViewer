import pycc
import cccorelib
from pycc.plugins import qM3C2

CC = pycc.GetInstance()
entities = CC.getSelectedEntities()
if not entities or len(entities) == 0:
		raise RuntimeError("Please select at one point cloud")
cloud = entities[0]
if not isinstance(cloud, pycc.ccPointCloud):
		raise RuntimeError("Selected entity should be a point cloud")
      
bbMin, bbMax = cccorelib.CCVector3(), cccorelib.CCVector3()
cloud.getBoundingBox(bbMin, bbMax)
print(f"Min {bbMin}, Max: {bbMax}")
diag = bbMax - bbMin

m3c2_dialog = qM3C2.qM3C2Dialog(cloud, cloud)
m3c2_dialog.setCorePointsCloud(cloud)

allowsDialog = False
result = qM3C2.qM3C2Process.Compute(m3c2_dialog, allowsDialog)
CC.addToDB(result)
