# cloudViewer: www.cloudViewer.org
# The MIT License (MIT)
# See license file or visit www.cloudViewer.org for details

# examples/Python/Basic/file_io.py

import cloudViewer as cv3d
if __name__ == "__main__":

    print("Testing IO for point cloud ...")
    pcd = cv3d.io.read_point_cloud("../../test_data/fragment.pcd")
    print(pcd)
    cv3d.io.write_point_cloud("copy_of_fragment.pcd", pcd)

    print("Testing IO for meshes ...")
    mesh = cv3d.io.read_triangle_mesh("../../test_data/knot.ply")
    print(mesh)
    cv3d.io.write_triangle_mesh("copy_of_knot.ply", mesh)

    print("Testing IO for textured meshes ...")
    textured_mesh = cv3d.io.read_triangle_mesh("../../test_data/crate/crate.obj")
    print(textured_mesh)
    cv3d.io.write_triangle_mesh("copy_of_crate.obj",
                               textured_mesh,
                               write_triangle_uvs=True)
    copy_textured_mesh = cv3d.io.read_triangle_mesh('copy_of_crate.obj')
    print(copy_textured_mesh)
    cv3d.visualization.draw_geometries([copy_textured_mesh])

    print("Testing IO for images ...")
    img = cv3d.io.read_image("../../test_data/lena_color.jpg")
    print(img)
    cv3d.visualization.draw_geometries([img])
    cv3d.io.write_image("copy_of_lena_color.jpg", img)
