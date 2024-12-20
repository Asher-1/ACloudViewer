import numpy as np
import pycc
import cccorelib

CC = pycc.GetInstance()


def main():
    print("hello world")
    
    params = pycc.FileIOFilter.LoadParameters()
    params.parentWidget = CC.getMainWindow()

    hierObj = CC.loadFile(r"/home/asher/develop/data/cloud/cloud/las/LPine1_demo.laz", params)
    print(hierObj, "is hierarchy obj:", hierObj.isHierarchy(), "is enabled ?:", hierObj.isEnabled())

    obj = hierObj.getChild(0)
    obj.setEnabled(False)
    assert obj.isEnabled() == False
    obj.setEnabled(True)
    assert obj.isEnabled() == True

    if not obj.nameShownIn3D():
        obj.showNameIn3D(True)

    CC.setSelectedInDB(obj, True)

    entities = CC.getSelectedEntities()
    print(entities)

    if not entities:
        raise RuntimeError("No entities selected")
        return

    pc = entities[0]
    print(f"The point cloud has {pc.size()} points")
    print(pc.getName())
    pc.setName("Renamed from python")

    sf = pc.getScalarField(0).asArray()
    print(pc.getScalarField(0).getName())
    print(f"{sf}")
    sf[:] = 42.0
    print(f"{sf}")


def main2():
    point = cccorelib.CCVector3(1.0, 2.0, 3.0)
    print(f"First Point: {point}")

    point.x = 17.256
    point.y = 12.5
    point.z = 42.65
    assert np.isclose(point.x, 17.256)
    assert np.isclose(point.y, 12.5)
    assert np.isclose(point.z, 42.65)

    point1 = cccorelib.CCVector3(25.0, 15.0, 30.0)
    print(f"Point1: {point1}")
    point2 = cccorelib.CCVector3(5.0, 5.0, 5.0)
    print(f"Point2: {point2}")

    p = point1 - point2
    print(f"point1-point2: {p}")


if __name__ == '__main__':
    main()
    main2()
