"""
An example of how to create a "machine" mesh by composing the geometric
primitives provided by the ccGenericPrimitive factory
(ccBox, ccSphere, ccCylinder, ccCone, ccTorus, ccPlane, ccDisc, ...).

Every primitive is an independent ccMesh entity (ccGenericPrimitive derives
from ccMesh). Each one is positioned by passing its transformation directly
to the primitive constructor (transMat), exactly like the C++ primitive
factory dialog does (e.g. `new ccSphere(radius, &transMat)`), so the transform
is baked into the real vertex coordinates -- not a rendering-only GL transform.

All the primitives are then merged into a SINGLE ccMesh entity (a real
mesh-typed object in the DB tree, like the cube produced by creating_a_mesh.py)
before being added to the DB.

Run it inside the ACloudViewer Python runtime (qPythonRuntime).
"""
import numpy as np

import cccorelib
import pycc


def make_transform(x, y, z):
    """Return a ccGLMatrix that only translates by (x, y, z)."""
    mat = pycc.ccGLMatrix()
    mat.setTranslation(cccorelib.CCVector3(x, y, z))
    return mat


def merge_primitives(primitives, name):
    """Merge a list of ccMesh primitives into a single ccMesh.

    The vertices of every primitive are gathered into one ccPointCloud and
    the triangles are re-indexed with the corresponding offset, producing a
    single real MESH entity.
    """
    xs, ys, zs = [], [], []
    triangles = []
    offset = 0

    for prim in primitives:
        cloud = prim.getAssociatedCloud()
        # Collect the (already transformed) vertices of this primitive
        cloud.placeIteratorAtBeginning()
        while True:
            p = cloud.getNextPoint()
            if p is None:
                break
            xs.append(p.x)
            ys.append(p.y)
            zs.append(p.z)

        # Collect its triangles, re-indexed by the current vertex offset
        for t in range(prim.size()):
            idx = prim.getTriangleVertIndexes(t)
            triangles.append((idx.i1 + offset, idx.i2 + offset, idx.i3 + offset))

        offset += cloud.size()

    vertices = pycc.ccPointCloud(np.array(xs), np.array(ys), np.array(zs))
    mesh = pycc.ccMesh(vertices)
    for (i1, i2, i3) in triangles:
        mesh.addTriangle(i1, i2, i3)
    mesh.setName(name)
    return mesh


def main():
    primitives = []

    def add(prim):
        primitives.append(prim)

    # --- base plate -------------------------------------------------------
    add(pycc.ccPlane(4.0, 6.0, make_transform(0.0, 0.0, 0.0), "BasePlate"))

    # --- body -------------------------------------------------------------
    add(pycc.ccBox(cccorelib.CCVector3(2.0, 3.0, 1.5),
                   make_transform(0.0, 0.0, 1.5), "Body"))

    # --- head (cone) ------------------------------------------------------
    add(pycc.ccCone(0.8, 0.8, 0.8, 0.0, 0.0,
                    make_transform(0.0, 0.0, 3.5), "Head"))

    # --- two spherical shoulder joints -----------------------------------
    for side in (-1.0, 1.0):
        add(pycc.ccSphere(0.4, make_transform(side * 1.2, 0.0, 2.6),
                          "Shoulder"))

    # --- two cylindrical arms --------------------------------------------
    for side in (-1.0, 1.0):
        add(pycc.ccCylinder(0.25, 2.0, make_transform(side * 1.6, 0.0, 1.6),
                            "Arm"))

    # --- four torus wheels -----------------------------------------------
    for front in (-1.5, 1.5):
        for side in (-1.7, 1.7):
            add(pycc.ccTorus(0.3, 0.45, 2.0 * 3.141592653589793, False, 0.0,
                             make_transform(side, front, 0.2), "Wheel"))

    # --- antenna (thin cylinder + sphere) --------------------------------
    add(pycc.ccCylinder(0.05, 1.2, make_transform(0.0, 0.0, 4.3), "Antenna"))
    add(pycc.ccSphere(0.15, make_transform(0.0, 0.0, 5.0), "AntennaTip"))

    machine = merge_primitives(primitives, "Machine")

    CC = pycc.GetInstance()
    # updateZoom=True -> camera frames the freshly created geometry (same as
    # the C++ primitive factory dialog: addToDB(primitive, true, true, true)).
    CC.addToDB(machine, updateZoom=True, autoExpandDBTree=True)
    CC.updateUI()


if __name__ == "__main__":
    main()
