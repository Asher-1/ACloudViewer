import pycc
import cccorelib

CC = pycc.GetInstance()

entities = CC.getSelectedEntities()
if not entities:
    raise RuntimeError("No entity selected. Select a point cloud first.")

pc = entities[0]

bbMin, bbMax = cccorelib.CCVector3(), cccorelib.CCVector3()
pc.getBoundingBox(bbMin, bbMax)
center = (bbMax - bbMin) / 2
center = cccorelib.CCVector3d(*center)

vp = pycc.ccViewportParameters()
vp.setCameraCenter(center, False)
vp.setFocalDistance(150)

pycc.ccDisplayTools.setViewportParameters(vp)
pycc.ccDisplayTools.redrawDisplay()
