# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import cloudViewer as cv3d

if __name__ == "__main__":
    knot_data = cv3d.data.KnotMesh()
    print(f"Reading mesh from file: knot.ply stored at {knot_data.path}")
    mesh = cv3d.io.read_triangle_mesh(knot_data.path)
    print(mesh)
    print("Saving mesh to file: copy_of_knot.ply")
    cv3d.io.write_triangle_mesh("copy_of_knot.ply", mesh)
