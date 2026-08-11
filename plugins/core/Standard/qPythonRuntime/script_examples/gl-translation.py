"""
This scripts applies a translation of the first selected entity.
It's an OpenGL translation, so coordinates are not changed, its just visual
"""
import pycc
import cccorelib


CC = pycc.GetInstance()

entities = CC.getSelectedEntities()
if not entities:
    raise RuntimeError("No entity selected. Select an entity first.")

entity = entities[0]

# Translating the entity

glMat = entity.getGLTransformation()
translation = glMat.getTranslationAsVec3D()
translation.x += 10.0
glMat.setTranslation(translation)

entity.setGLTransformation(glMat)
entity.applyGLTransformation_recursive()
pycc.GetInstance().redrawAll()
