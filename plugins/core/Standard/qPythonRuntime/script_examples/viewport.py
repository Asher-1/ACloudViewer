import pycc
import cvcorelib

CC = pycc.GetInstance()

pc = CC.getSelectedEntities()[0]

bbMin, bbMax = cvcorelib.CCVector3(), cvcorelib.CCVector3()
pc.getBoundingBox(bbMin, bbMax)
center = (bbMax - bbMin) / 2
center = cvcorelib.CCVector3d(*center)

vp = pycc.ecvViewportParameters()
vp.setCameraCenter(center, False)
vp.setFocalDistance(150)

glWindow = pc.getDisplay()
glWindow.setViewportParameters(vp)
glWindow.redraw()
