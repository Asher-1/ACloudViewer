# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import cloudViewer as cv3d


def generate_from_point_cloud():
    print("Load a ply point cloud, print it, and render it")
    ply_data = cv3d.data.PLYPointCloud()
    pc = cv3d.io.read_point_cloud(ply_data.path)
    facet = cv3d.geometry.ccFacet.Create(cloud=pc, max_edge_length=0)
    print(facet)
    facet.get_polygon().set_temp_color([0, 0, 0.5])
    facet.get_polygon().set_opacity(0.5)
    facet.get_polygon().clear_triangle_normals()
    facet.get_polygon().compute_vertex_normals()
    facet.get_contour().set_color([1, 0, 1])
    facet.get_contour().set_width(9)
    facet.show_normal_vector(True)
    return [facet]


def generate_from_file():
    print("Load a facets data, print it, and render it")
    cv3d.data.set_custom_downloads_prefix(
        "https://github.com/Asher-1/cloudViewer_downloads/releases/download/")
    facets_data = cv3d.data.FacetsModel()
    entity = cv3d.io.read_entity(facets_data.path)
    facets = entity.filter_children(recursive=False,
                                    filter=cv3d.geometry.ccHObject.FACET)
    print(facets)
    for facet in facets:
        facet.get_polygon().set_temp_color([0.5, 0, 0])
        facet.get_polygon().set_opacity(0.5)
        facet.get_polygon().clear_triangle_normals()
        facet.get_polygon().compute_vertex_normals()
        facet.get_contour().set_color([0, 1, 0])
        facet.get_contour().set_width(9)
        facet.show_normal_vector(True)
    return facets


def facets_generator():
    yield "facet1", generate_from_file()
    yield "facet2", generate_from_point_cloud()


if __name__ == "__main__":
    cv3d.utility.set_verbosity_level(cv3d.utility.Debug)
    for name, facets in facets_generator():
        print("{}: RMS {}, Area {}, normal {}".format(name, facets[0].get_rms(),
                                                      facets[0].get_area(),
                                                      facets[0].get_normal()))
        cv3d.visualization.draw_geometries(facets, mesh_show_back_face=True)
