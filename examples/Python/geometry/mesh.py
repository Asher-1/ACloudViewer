# cloudViewer: www.cloudViewer.org
# The MIT License (MIT)
# See license file or visit www.cloudViewer.org for details

# examples/Python/Basic/mesh.py

import copy
import numpy as np
import cloudViewer as cv3d

if __name__ == "__main__":

    print("Testing mesh in cloudViewer ...")
    mesh = cv3d.io.read_triangle_mesh("../../test_data/knot.ply")
    print(mesh)
    print(np.asarray(mesh.get_vertices()))
    print(np.asarray(mesh.get_triangles()))
    print("")

    # triangulation
    # DELAUNAY_2D_AXIS_ALIGNED or DELAUNAY_2D_BEST_LS_PLANE.
    triangulation_type = cv3d.geometry.DELAUNAY_2D_BEST_LS_PLANE
    new_mesh = cv3d.geometry.ccMesh.triangulate(cloud=mesh.get_associated_cloud(), type=triangulation_type)
    new_mesh.compute_vertex_normals()
    cv3d.visualization.draw_geometries([new_mesh])

    # triangulation between with two polylines
    entity = cv3d.io.read_entity("../../test_data/polylines/polylines.bin")
    polylines = entity.filter_children(recursive=False, filter=cv3d.geometry.ccHObject.POLY_LINE)
    print(polylines)
    assert len(polylines) > 1
    mesh_polys = cv3d.geometry.ccMesh.triangulate_two_polylines(poly1=polylines[0], poly2=polylines[1])
    mesh_polys.compute_vertex_normals()
    cv3d.visualization.draw_geometries(polylines + [mesh_polys])

    print("Try to render a mesh with normals (exist: " +
          str(mesh.has_vertex_normals()) + ") and colors (exist: " +
          str(mesh.has_vertex_colors()) + ")")
    mesh.set_temp_color([0, 0, 1])
    mesh.set_opacity(0.5)
    mesh.show_colors(True)
    cv3d.visualization.draw_geometries([mesh])
    print("A mesh with no normals and no colors does not seem good.")

    print("Computing normal and rendering it.")
    mesh.compute_vertex_normals()
    print(np.asarray(mesh.get_triangle_normals()))
    cv3d.visualization.draw_geometries([mesh])

    print("We make a partial mesh of only the first half triangles.")
    mesh1 = copy.deepcopy(mesh)
    triangles = np.asarray(mesh1.get_triangles())
    triangle_normals = np.asarray(mesh1.get_triangle_normals())
    mesh1.set_triangles(cv3d.utility.Vector3iVector(triangles[:len(triangles) // 2, :]))
    mesh1.set_triangle_normals(cv3d.utility.Vector3dVector(triangle_normals[:len(triangle_normals) // 2, :]))
    print(mesh1.get_triangles())
    cv3d.visualization.draw_geometries([mesh1])

    print("Painting the mesh")
    mesh1.paint_uniform_color([1, 0.706, 0])
    cv3d.visualization.draw_geometries([mesh1])
