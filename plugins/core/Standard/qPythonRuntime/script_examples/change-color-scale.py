import pycc

CC = pycc.GetInstance()

entities = CC.getSelectedEntities()
if not entities:
    raise RuntimeError("No entity selected. Select a point cloud first.")

pc = entities[0]
if pc.getNumberOfScalarFields() == 0:
    raise RuntimeError("The selected entity has no scalar field to change the color scale of.")

sf = pc.getScalarField(0)

# Get histogram values
hist = sf.getHistogram()
print(hist)

# Change the color scale displayed
scale = pycc.ccColorScalesManager.GetDefaultScale(pycc.ccColorScalesManager.YELLOW_BROWN)
sf.setColorScale(scale)
pycc.GetInstance().updateUI()
pycc.GetInstance().redrawAll()