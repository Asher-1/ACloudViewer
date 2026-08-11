"""
An example of how to create a "machine" by composing the geometric
primitives provided by the ccGenericPrimitive factory
(ccBox, ccSphere, ccCylinder, ccCone, ccTorus, ccPlane, ccDisc, ...).

Every primitive is an independent ccMesh entity (ccGenericPrimitive derives
from ccMesh), so each component shows up in the DB tree as a real MESH-typed
object. Each one is positioned by passing its transformation directly to the
primitive constructor (transMat), exactly like the C++ primitive factory dialog
does (e.g. `new ccSphere(radius, &transMat)`), so the transform is baked into
the real vertex coordinates.

IMPORTANT (double-transform pitfall)
------------------------------------
`ccGenericPrimitive::applyTransformationToVertices()` bakes transMat into the
vertices AND keeps it as a GL transformation (`setGLTransformation`). The
renderer then applies that GL transform AGAIN on top of the already-moved
vertices, which shifts every entity (each component drifts from its intended
pose). To avoid this, after construction we call `resetGLTransformation()` so
the baked vertex positions are used exactly once.

Colouring
---------
Each component gets its own pastel colour via `setColor(pycc.Rgb(...))`, so the
robot reads as a coherent, cute palette (warm body / cool accents) instead of a
monochrome blob. `showColors(True)` makes sure the per-component colours are
actually rendered.

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


# --- Cute palette ---------------------------------------------------------
# A warm gradient from the ground up, with cool accents on the moving parts.
COL_BASE   = pycc.Rgb(176, 192, 208)   # soft blue-grey ground plate
COL_BODY   = pycc.Rgb(255, 160, 122)   # warm coral torso
COL_NECK   = pycc.Rgb(255, 196, 138)   # lighter coral neck
COL_HEAD   = pycc.Rgb(255, 230, 170)   # creamy yellow head
COL_SHOUL  = pycc.Rgb(188, 176, 232)   # soft lavender shoulders
COL_ARM    = pycc.Rgb(142, 184, 235)   # sky-blue arms
COL_WHEEL  = pycc.Rgb(96, 100, 116)    # dark slate wheels
COL_ANTEN  = pycc.Rgb(240, 106, 104)   # coral-red antenna
COL_TIP    = pycc.Rgb(255, 218, 100)   # golden antenna tip

_KEEP = {}  # module-level refs so pycc doesn't GC the group / children


def main():
    CC = pycc.GetInstance()

    # One top-level group that will contain every component.
    # Children stay real MESH entities; the group is just a DB-tree container.
    machine = pycc.ccHObject("Machine")
    parts = []

    def add(prim, color):
        # transMat was baked into the vertices by the primitive constructor.
        # Clear the leftover GL transformation so the renderer does NOT apply
        # it a second time (otherwise every component drifts off position).
        prim.resetGLTransformation()
        # Give each component its own colour and make sure it is displayed.
        prim.setColor(color)
        prim.showColors(True)
        machine.addChild(prim)
        # keep a Python reference so pycc doesn't garbage-collect the child
        parts.append(prim)

    # --- base plate -------------------------------------------------------
    add(pycc.ccPlane(4.0, 6.0, make_transform(0.0, 0.0, 0.0), "BasePlate"),
        COL_BASE)

    # --- body -------------------------------------------------------------
    # box(2,3,1.5) centred at z=1.5  ->  top face at z=2.25
    add(pycc.ccBox(cccorelib.CCVector3(2.0, 3.0, 1.5),
                   make_transform(0.0, 0.0, 1.5), "Body"), COL_BODY)

    # --- neck (connects body top z=2.25 to head bottom z=3.5) ------------
    # The gap is 1.25 tall; a slim cylinder fills it so head and body connect.
    add(pycc.ccCylinder(0.45, 1.25, make_transform(0.0, 0.0, 2.875), "Neck"),
        COL_NECK)

    # --- head (cone) ------------------------------------------------------
    # cone bottom at z=3.5, top at z=4.3
    add(pycc.ccCone(0.8, 0.8, 0.8, 0.0, 0.0,
                    make_transform(0.0, 0.0, 3.5), "Head"), COL_HEAD)

    # --- two spherical shoulder joints -----------------------------------
    for side in (-1.0, 1.0):
        add(pycc.ccSphere(0.4, make_transform(side * 1.2, 0.0, 2.6),
                          "Shoulder"), COL_SHOUL)

    # --- two cylindrical arms --------------------------------------------
    for side in (-1.0, 1.0):
        add(pycc.ccCylinder(0.25, 2.0, make_transform(side * 1.6, 0.0, 1.6),
                            "Arm"), COL_ARM)

    # --- four torus wheels -----------------------------------------------
    for front in (-1.5, 1.5):
        for side in (-1.7, 1.7):
            add(pycc.ccTorus(0.3, 0.45, 2.0 * 3.141592653589793, False, 0.0,
                             make_transform(side, front, 0.2), "Wheel"),
                COL_WHEEL)

    # --- antenna (thin cylinder + sphere) --------------------------------
    # antenna base sits on top of the head (z=4.3)
    add(pycc.ccCylinder(0.05, 1.2, make_transform(0.0, 0.0, 4.3), "Antenna"),
        COL_ANTEN)
    add(pycc.ccSphere(0.15, make_transform(0.0, 0.0, 5.0), "AntennaTip"),
        COL_TIP)

    # Add the whole group to the DB tree (updateZoom frames the geometry)
    CC.addToDB(machine, updateZoom=True, autoExpandDBTree=True)
    # keep references alive for the whole Python session
    _KEEP["machine"] = machine
    _KEEP["parts"] = parts
    CC.updateUI()


if __name__ == "__main__":
    main()
