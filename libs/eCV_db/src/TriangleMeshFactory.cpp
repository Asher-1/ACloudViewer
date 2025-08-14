// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <IntersectionTest.h>
#include <Logging.h>
#include <Parallel.h>

#include <Eigen/Dense>
#include <numeric>
#include <queue>
#include <random>
#include <tuple>

#include "ecvHObjectCaster.h"
#include "ecvKDTreeFlann.h"
#include "ecvMesh.h"
#include "ecvPointCloud.h"
#include "ecvQhull.h"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace cloudViewer;

ccMesh &ccMesh::removeDuplicatedVertices() {
    typedef std::tuple<double, double, double> Coordinate3;
    std::unordered_map<Coordinate3, size_t,
                       utility::hash_tuple::hash<Coordinate3>>
            point_to_old_index;
    std::vector<int> index_old_to_new(getVerticeSize());
    bool has_vert_normal = hasNormals();
    bool has_vert_color = hasColors();
    size_t old_vertex_num = getVerticeSize();
    size_t k = 0;                                  // new index
    for (size_t i = 0; i < old_vertex_num; i++) {  // old index
        Coordinate3 coord = std::make_tuple(getVertice(i)(0), getVertice(i)(1),
                                            getVertice(i)(2));
        if (point_to_old_index.find(coord) == point_to_old_index.end()) {
            point_to_old_index[coord] = i;
            setVertice(k, getVertice(i));
            if (has_vert_normal) setVertexNormal(k, getVertexNormal(i));
            if (has_vert_color) setVertexColor(k, getVertexColor(i));
            index_old_to_new[i] = (int)k;
            k++;
        } else {
            index_old_to_new[i] = index_old_to_new[point_to_old_index[coord]];
        }
    }

    // do some cleaning
    {
        ccPointCloud *cloud = ccHObjectCaster::ToPointCloud(m_associatedCloud);
        if (cloud) {
            cloud->resize(static_cast<unsigned>(k));
        } else {
            utility::LogDebug(
                    "[removeUnreferencedVertices] ccMesh has not associated "
                    "cloud.");
        }
    }

    if (k < old_vertex_num) {
        for (auto &triangle : *getTrianglesPtr()) {
            triangle.i1 = static_cast<unsigned>(index_old_to_new[triangle.i1]);
            triangle.i2 = static_cast<unsigned>(index_old_to_new[triangle.i2]);
            triangle.i3 = static_cast<unsigned>(index_old_to_new[triangle.i3]);
        }
        if (hasAdjacencyList()) {
            computeAdjacencyList();
        }
    }
    utility::LogDebug(
            "[removeDuplicatedVertices] {:d} vertices have been removed.",
            (int)(old_vertex_num - k));

    return *this;
}

ccMesh &ccMesh::removeDuplicatedTriangles() {
    if (hasTriangleUvs()) {
        utility::LogWarning(
                "[removeDuplicatedTriangles] This mesh contains triangle uvs "
                "that are not handled in this function");
    }
    typedef std::tuple<int, int, int> Index3;
    std::unordered_map<Index3, size_t, utility::hash_tuple::hash<Index3>>
            triangle_to_old_index;
    bool has_tri_normal = hasTriNormals();
    size_t old_triangle_num = size();
    size_t k = 0;
    for (size_t i = 0; i < old_triangle_num; i++) {
        Index3 index;
        // We first need to find the minimum index. Because triangle (0-1-2)
        // and triangle (2-0-1) are the same.
        Eigen::Vector3i triangleInd = getTriangle(i);
        if (triangleInd(0) <= triangleInd(1)) {
            if (triangleInd(0) <= triangleInd(2)) {
                index = std::make_tuple(triangleInd(0), triangleInd(1),
                                        triangleInd(2));
            } else {
                index = std::make_tuple(triangleInd(2), triangleInd(0),
                                        triangleInd(1));
            }
        } else {
            if (triangleInd(1) <= triangleInd(2)) {
                index = std::make_tuple(triangleInd(1), triangleInd(2),
                                        triangleInd(0));
            } else {
                index = std::make_tuple(triangleInd(2), triangleInd(0),
                                        triangleInd(1));
            }
        }
        if (triangle_to_old_index.find(index) == triangle_to_old_index.end()) {
            triangle_to_old_index[index] = i;
            setTriangle(k, triangleInd);
            if (has_tri_normal) setTriangleNorm(k, getTriangleNorm(i));
            k++;
        }
    }

    // do some cleaning
    resize(k);
    if (has_tri_normal) m_triNormals->resize(k);
    if (k < old_triangle_num && hasAdjacencyList()) {
        computeAdjacencyList();
    }
    utility::LogDebug(
            "[removeDuplicatedTriangles] {:d} triangles have been removed.",
            (int)(old_triangle_num - k));

    return *this;
}

ccMesh &ccMesh::removeUnreferencedVertices() {
    std::vector<bool> vertex_has_reference(getVerticeSize(), false);
    for (const auto &triangle : *getTrianglesPtr()) {
        vertex_has_reference[triangle.i1] = true;
        vertex_has_reference[triangle.i2] = true;
        vertex_has_reference[triangle.i3] = true;
    }

    std::vector<int> index_old_to_new(getVerticeSize());
    bool has_vert_normal = hasNormals();
    bool has_vert_color = hasColors();
    size_t old_vertex_num = getVerticeSize();
    size_t k = 0;                                  // new index
    for (size_t i = 0; i < old_vertex_num; i++) {  // old index
        if (vertex_has_reference[i]) {
            setVertice(k, getVertice(i));
            if (has_vert_normal) setVertexNormal(k, getVertexNormal(i));
            if (has_vert_color) setVertexColor(k, getVertexColor(i));
            index_old_to_new[i] = (int)k;
            k++;
        } else {
            index_old_to_new[i] = -1;
        }
    }

    // do some cleaning
    {
        ccPointCloud *cloud = ccHObjectCaster::ToPointCloud(m_associatedCloud);
        if (cloud) {
            cloud->resize(static_cast<unsigned>(k));
        } else {
            utility::LogDebug(
                    "[removeUnreferencedVertices] ccMesh has not associated "
                    "cloud.");
        }
    }

    if (k < old_vertex_num) {
        for (auto &triangle : *getTrianglesPtr()) {
            triangle.i1 = static_cast<unsigned>(index_old_to_new[triangle.i1]);
            triangle.i2 = static_cast<unsigned>(index_old_to_new[triangle.i2]);
            triangle.i3 = static_cast<unsigned>(index_old_to_new[triangle.i3]);
        }
        if (hasAdjacencyList()) {
            computeAdjacencyList();
        }
    }
    utility::LogDebug(
            "[removeUnreferencedVertices] {:d} vertices have been removed.",
            (int)(old_vertex_num - k));

    return *this;
}

ccMesh &ccMesh::removeDegenerateTriangles() {
    if (hasTriangleUvs()) {
        utility::LogWarning(
                "[removeDegenerateTriangles] This mesh contains triangle uvs "
                "that are not handled in this function");
    }
    bool has_tri_normal = hasTriNormals();
    size_t old_triangle_num = size();
    size_t k = 0;
    for (size_t i = 0; i < old_triangle_num; i++) {
        auto triangle = getTriangle(i);
        if (triangle(0) != triangle(1) && triangle(1) != triangle(2) &&
            triangle(2) != triangle(0)) {
            setTriangle(k, getTriangle(i));
            if (has_tri_normal) setTriangleNorm(k, getTriangleNorm(i));
            k++;
        }
    }

    // do some cleaning
    resize(k);
    if (has_tri_normal) m_triNormals->resize(k);
    if (k < old_triangle_num && hasAdjacencyList()) {
        computeAdjacencyList();
    }
    utility::LogDebug(
            "[RemoveDegenerateTriangles] {:d} triangles have been "
            "removed.",
            (int)(old_triangle_num - k));
    return *this;
}

ccMesh &ccMesh::removeNonManifoldEdges() {
    if (hasTriangleUvs()) {
        utility::LogWarning(
                "[RemoveNonManifoldEdges] This mesh contains triangle uvs that "
                "are not handled in this function");
    }
    std::vector<double> triangle_areas;
    getSurfaceArea(triangle_areas);

    bool mesh_is_edge_manifold = false;
    while (!mesh_is_edge_manifold) {
        mesh_is_edge_manifold = true;
        auto edges_to_triangles = getEdgeToTrianglesMap();

        for (auto &kv : edges_to_triangles) {
            size_t n_edge_triangle_refs = kv.second.size();
            // check if the given edge is manifold
            // (has exactly 1, or 2 adjacent triangles)
            if (n_edge_triangle_refs == 1u || n_edge_triangle_refs == 2u) {
                continue;
            }

            // There is at least one edge that is non-manifold
            mesh_is_edge_manifold = false;

            // if the edge is non-manifold, then check if a referenced
            // triangle has already been removed
            // (triangle area has been set to < 0), otherwise remove triangle
            // with smallest surface area until number of adjacent triangles
            // is <= 2.
            // 1) count triangles that are not marked deleted
            int n_triangles = 0;
            for (int tidx : kv.second) {
                if (triangle_areas[tidx] > 0) {
                    n_triangles++;
                }
            }
            // 2) mark smallest triangles as deleted by setting
            // surface area to -1
            int n_triangles_to_delete = n_triangles - 2;
            while (n_triangles_to_delete > 0) {
                // find triangle with smallest area
                int min_tidx = -1;
                double min_area = std::numeric_limits<double>::max();
                for (int tidx : kv.second) {
                    double area = triangle_areas[tidx];
                    if (area > 0 && area < min_area) {
                        min_tidx = tidx;
                        min_area = area;
                    }
                }

                // mark triangle as deleted by setting area to -1
                triangle_areas[min_tidx] = -1;
                n_triangles_to_delete--;
            }
        }

        // delete marked triangles
        bool has_tri_normal = hasTriNormals();
        size_t to_tidx = 0;
        for (size_t from_tidx = 0; from_tidx < size(); ++from_tidx) {
            if (triangle_areas[from_tidx] > 0) {
                setTriangle(to_tidx, getTriangle(from_tidx));
                triangle_areas[to_tidx] = triangle_areas[from_tidx];
                if (has_tri_normal) {
                    setTriangleNorm(to_tidx, getTriangleNorm(from_tidx));
                }
                to_tidx++;
            }
        }

        // do some cleaning
        {
            resize(to_tidx);
            triangle_areas.resize(to_tidx);
            if (has_tri_normal) {
                m_triNormals->resize(to_tidx);
            }
        }
    }
    return *this;
}

ccMesh &ccMesh::mergeCloseVertices(double eps) {
    cloudViewer::geometry::KDTreeFlann kdtree(*this);
    // precompute all neighbours
    utility::LogDebug("Precompute Neighbours");
    std::vector<std::vector<int>> nbs(getVerticeSize());
#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
        num_threads(utility::EstimateMaxThreads())
#endif
    for (int idx = 0; idx < int(getVerticeSize()); ++idx) {
        std::vector<double> dists2;
        kdtree.SearchRadius(getVertice(static_cast<size_t>(idx)), eps, nbs[idx],
                            dists2);
    }
    utility::LogDebug("Done Precompute Neighbours");

    ccPointCloud *cloud = ccHObjectCaster::ToPointCloud(m_associatedCloud);
    assert(cloud);

    bool has_vertex_normals = hasNormals();
    bool has_vertex_colors = hasColors();
    std::vector<CCVector3> new_vertices;
    std::vector<CCVector3> new_vertex_normals;
    ColorsTableType new_vertex_colors = *cloud->rgbColors();
    std::unordered_map<int, int> new_vert_mapping;
    for (int vidx = 0; vidx < int(getVerticeSize()); ++vidx) {
        if (new_vert_mapping.count(vidx) > 0) {
            continue;
        }

        int new_vidx = int(new_vertices.size());
        new_vert_mapping[vidx] = new_vidx;

        CCVector3 vertex = getVertice(static_cast<size_t>(vidx));
        CCVector3 normal;
        if (has_vertex_normals) {
            normal = getVertexNormal(static_cast<size_t>(vidx));
        }
        Eigen::Vector3d color;
        if (has_vertex_colors) {
            color = getVertexColor(static_cast<size_t>(vidx));
        }
        int n = 1;
        for (int nb : nbs[vidx]) {
            if (vidx == nb || new_vert_mapping.count(nb) > 0) {
                continue;
            }
            vertex += getVertice(static_cast<size_t>(nb));
            if (has_vertex_normals) {
                normal += getVertexNormal(static_cast<size_t>(nb));
            }
            if (has_vertex_colors) {
                color += getVertexColor(static_cast<size_t>(nb));
            }
            new_vert_mapping[nb] = new_vidx;
            n += 1;
        }
        new_vertices.push_back(vertex / n);
        if (has_vertex_normals) {
            new_vertex_normals.push_back(normal / n);
        }
        if (has_vertex_colors) {
            new_vertex_colors.push_back(ecvColor::Rgb::FromEigen(color / n));
        }
    }
    utility::LogDebug("Merged {} vertices",
                      getVerticeSize() - new_vertices.size());

    std::swap(cloud->getPoints(), new_vertices);
    if (has_vertex_normals) {
        cloud->setPointNormals(new_vertex_normals);
    }
    if (has_vertex_colors) {
        std::swap(*cloud->rgbColors(), new_vertex_colors);
    }

    for (auto &triangle : *getTrianglesPtr()) {
        triangle.i1 = static_cast<unsigned>(new_vert_mapping[triangle.i1]);
        triangle.i2 = static_cast<unsigned>(new_vert_mapping[triangle.i2]);
        triangle.i3 = static_cast<unsigned>(new_vert_mapping[triangle.i3]);
    }

    if (hasTriNormals()) {
        computePerTriangleNormals();
    }

    return *this;
}

ccMesh &ccMesh::paintUniformColor(const Eigen::Vector3d &color) {
    if (getAssociatedCloud() &&
        getAssociatedCloud()->isKindOf(CV_TYPES::POINT_CLOUD)) {
        ccPointCloud *cloud =
                ccHObjectCaster::ToPointCloud(getAssociatedCloud());
        cloud->paintUniformColor(color);
    }
    return *this;
}

std::tuple<std::shared_ptr<ccMesh>, std::vector<size_t>>
ccMesh::computeConvexHull() const {
    return cloudViewer::utility::Qhull::ComputeConvexHull(getVertices());
}

std::shared_ptr<ccMesh> ccMesh::filterSharpen(int number_of_iterations,
                                              double strength,
                                              FilterScope scope) const {
    ccPointCloud *cloud = ccHObjectCaster::ToPointCloud(m_associatedCloud);
    assert(cloud);

    bool filter_vertex =
            scope == FilterScope::All || scope == FilterScope::Vertex;
    bool filter_normal =
            (scope == FilterScope::All || scope == FilterScope::Normal) &&
            cloud->hasNormals();
    bool filter_color =
            (scope == FilterScope::All || scope == FilterScope::Color) &&
            cloud->hasColors();

    std::vector<CCVector3> prev_vertices = cloud->getPoints();

    std::vector<CCVector3> prev_vertex_normals;
    if (filter_normal) {
        prev_vertex_normals = cloud->getPointNormals();
    }
    ColorsTableType prev_vertex_colors;
    if (filter_color) {
        prev_vertex_colors = *cloud->rgbColors();
    }

    ccPointCloud *baseVertices = new ccPointCloud("vertices");
    assert(baseVertices);
    baseVertices->setEnabled(false);
    // DGM: no need to lock it as it is only used by one mesh!
    baseVertices->setLocked(false);
    std::shared_ptr<ccMesh> mesh =
            cloudViewer::make_shared<ccMesh>(baseVertices);
    mesh->addChild(baseVertices);

    baseVertices->resize(cloud->size());
    if (cloud->hasNormals()) {
        baseVertices->resizeTheNormsTable();
    }
    if (cloud->hasColors()) {
        baseVertices->resizeTheRGBTable();
    }

    mesh->setTriangles(getTriangles());
    mesh->adjacency_list_ = adjacency_list_;
    if (!mesh->hasAdjacencyList()) {
        mesh->computeAdjacencyList();
    }

    for (int iter = 0; iter < number_of_iterations; ++iter) {
        for (size_t vidx = 0; vidx < baseVertices->size(); ++vidx) {
            CCVector3 vertex_sum(0, 0, 0);
            CCVector3 normal_sum(0, 0, 0);
            Eigen::Vector3d color_sum(0, 0, 0);
            for (int nbidx : mesh->adjacency_list_[vidx]) {
                if (filter_vertex) {
                    vertex_sum += prev_vertices[nbidx];
                }
                if (filter_normal) {
                    normal_sum += prev_vertex_normals[nbidx];
                }
                if (filter_color) {
                    color_sum +=
                            ecvColor::Rgb::ToEigen(prev_vertex_colors.getValue(
                                    static_cast<size_t>(nbidx)));
                }
            }

            size_t nb_size = mesh->adjacency_list_[vidx].size();
            if (filter_vertex) {
                CCVector3 p = prev_vertices[vidx] +
                              static_cast<float>(strength) *
                                      (prev_vertices[vidx] *
                                               static_cast<float>(nb_size) -
                                       vertex_sum);
                baseVertices->setPoint(vidx, p);
            }

            if (filter_normal) {
                CCVector3 p = prev_vertex_normals[vidx] +
                              static_cast<float>(strength) *
                                      (prev_vertex_normals[vidx] *
                                               static_cast<float>(nb_size) -
                                       normal_sum);
                baseVertices->setPointNormal(vidx, p);
            }
            if (filter_color) {
                baseVertices->setPointColor(
                        vidx,
                        ecvColor::Rgb::ToEigen(prev_vertex_colors[vidx]) +
                                strength * (ecvColor::Rgb::ToEigen(
                                                    prev_vertex_colors[vidx]) *
                                                    nb_size -
                                            color_sum));
            }
        }
        if (iter < number_of_iterations - 1) {
            std::swap(baseVertices->getPoints(), prev_vertices);
            if (filter_normal) {
                prev_vertex_normals = baseVertices->getPointNormals();
            }
            if (filter_color) {
                std::swap(*baseVertices->rgbColors(), prev_vertex_colors);
            }
        }
    }

    if (hasTriNormals()) {
        mesh->computeTriangleNormals();
    }

    // do some cleaning
    {
        baseVertices->shrinkToFit();
        mesh->shrinkToFit();
        NormsIndexesTableType *normals = mesh->getTriNormsTable();
        if (normals) {
            normals->shrink_to_fit();
        }
    }
    return mesh;
}

std::shared_ptr<ccMesh> ccMesh::filterSmoothSimple(int number_of_iterations,
                                                   FilterScope scope) const {
    ccPointCloud *cloud = ccHObjectCaster::ToPointCloud(m_associatedCloud);
    assert(cloud);

    bool filter_vertex =
            scope == FilterScope::All || scope == FilterScope::Vertex;
    bool filter_normal =
            (scope == FilterScope::All || scope == FilterScope::Normal) &&
            cloud->hasNormals();
    bool filter_color =
            (scope == FilterScope::All || scope == FilterScope::Color) &&
            cloud->hasColors();

    std::vector<CCVector3> prev_vertices = cloud->getPoints();
    std::vector<CCVector3> prev_vertex_normals;
    if (filter_normal) {
        prev_vertex_normals = cloud->getPointNormals();
    }
    ColorsTableType prev_vertex_colors;
    if (filter_color) {
        prev_vertex_colors = *cloud->rgbColors();
    }

    ccPointCloud *baseVertices = new ccPointCloud("vertices");
    assert(baseVertices);
    baseVertices->setEnabled(false);
    // DGM: no need to lock it as it is only used by one mesh!
    baseVertices->setLocked(false);
    std::shared_ptr<ccMesh> mesh =
            cloudViewer::make_shared<ccMesh>(baseVertices);
    mesh->addChild(baseVertices);

    baseVertices->resize(cloud->size());
    if (cloud->hasNormals()) {
        baseVertices->resizeTheNormsTable();
    }
    if (cloud->hasColors()) {
        baseVertices->resizeTheRGBTable();
    }

    mesh->setTriangles(getTriangles());
    mesh->adjacency_list_ = adjacency_list_;
    if (!mesh->hasAdjacencyList()) {
        mesh->computeAdjacencyList();
    }

    for (int iter = 0; iter < number_of_iterations; ++iter) {
        for (size_t vidx = 0; vidx < baseVertices->size(); ++vidx) {
            CCVector3 vertex_sum(0, 0, 0);
            CCVector3 normal_sum(0, 0, 0);
            Eigen::Vector3d color_sum(0, 0, 0);
            for (int nbidx : mesh->adjacency_list_[vidx]) {
                if (filter_vertex) {
                    vertex_sum += prev_vertices[nbidx];
                }
                if (filter_normal) {
                    normal_sum += prev_vertex_normals[nbidx];
                }
                if (filter_color) {
                    color_sum += ecvColor::Rgb::ToEigen(
                            prev_vertex_colors.at(static_cast<size_t>(nbidx)));
                }
            }

            size_t nb_size = mesh->adjacency_list_[vidx].size();
            if (filter_vertex) {
                baseVertices->setPoint(
                        vidx,
                        (prev_vertices[vidx] + vertex_sum) / (1 + nb_size));
            }
            if (filter_normal) {
                baseVertices->setPointNormal(
                        vidx, (prev_vertex_normals[vidx] + normal_sum) /
                                      (1 + nb_size));
            }
            if (filter_color) {
                baseVertices->setPointColor(
                        vidx,
                        (ecvColor::Rgb::ToEigen(prev_vertex_colors[vidx]) +
                         color_sum) /
                                (1 + nb_size));
            }
        }
        if (iter < number_of_iterations - 1) {
            std::swap(baseVertices->getPoints(), prev_vertices);
            if (filter_normal) {
                prev_vertex_normals = baseVertices->getPointNormals();
            }
            if (filter_color) {
                std::swap(*baseVertices->rgbColors(), prev_vertex_colors);
            }
        }
    }

    if (hasTriNormals()) {
        mesh->computeTriangleNormals();
    }

    // do some cleaning
    {
        baseVertices->shrinkToFit();
        mesh->shrinkToFit();
        NormsIndexesTableType *normals = mesh->getTriNormsTable();
        if (normals) {
            normals->shrink_to_fit();
        }
    }
    return mesh;
}

void ccMesh::filterSmoothLaplacianHelper(
        std::shared_ptr<ccMesh> &mesh,
        const std::vector<CCVector3> &prev_vertices,
        const std::vector<CCVector3> &prev_vertex_normals,
        const ColorsTableType &prev_vertex_colors,
        const std::vector<std::unordered_set<int>> &adjacency_list,
        double lambda,
        bool filter_vertex,
        bool filter_normal,
        bool filter_color) const {
    ccPointCloud *cloud =
            ccHObjectCaster::ToPointCloud(mesh->getAssociatedCloud());
    assert(cloud);

    for (size_t vidx = 0; vidx < cloud->size(); ++vidx) {
        CCVector3 vertex_sum(0, 0, 0);
        CCVector3 normal_sum(0, 0, 0);
        Eigen::Vector3d color_sum(0, 0, 0);
        double total_weight = 0;
        for (int nbidx : mesh->adjacency_list_[vidx]) {
            auto diff = prev_vertices[vidx] - prev_vertices[nbidx];
            double dist = diff.norm();
            double weight = 1. / (dist + 1e-12);
            total_weight += weight;

            if (filter_vertex) {
                vertex_sum += static_cast<float>(weight) * prev_vertices[nbidx];
            }
            if (filter_normal) {
                normal_sum +=
                        static_cast<float>(weight) * prev_vertex_normals[nbidx];
            }
            if (filter_color) {
                color_sum += weight *
                             ecvColor::Rgb::ToEigen(prev_vertex_colors[nbidx]);
            }
        }

        if (filter_vertex) {
            cloud->setPoint(vidx, prev_vertices[vidx] +
                                          static_cast<float>(lambda) *
                                                  (vertex_sum / total_weight -
                                                   prev_vertices[vidx]));
        }
        if (filter_normal) {
            cloud->setPointNormal(vidx,
                                  prev_vertex_normals[vidx] +
                                          static_cast<float>(lambda) *
                                                  (normal_sum / total_weight -
                                                   prev_vertex_normals[vidx]));
        }
        if (filter_color) {
            Eigen::Vector3d C =
                    ecvColor::Rgb::ToEigen(prev_vertex_colors[vidx]) +
                    lambda * (color_sum / total_weight -
                              ecvColor::Rgb::ToEigen(prev_vertex_colors[vidx]));
            cloud->setPointColor(vidx, C);
        }
    }
}

std::shared_ptr<ccMesh> ccMesh::filterSmoothLaplacian(int number_of_iterations,
                                                      double lambda,
                                                      FilterScope scope) const {
    ccPointCloud *cloud = ccHObjectCaster::ToPointCloud(m_associatedCloud);
    assert(cloud);

    bool filter_vertex =
            scope == FilterScope::All || scope == FilterScope::Vertex;
    bool filter_normal =
            (scope == FilterScope::All || scope == FilterScope::Normal) &&
            cloud->hasNormals();
    bool filter_color =
            (scope == FilterScope::All || scope == FilterScope::Color) &&
            cloud->hasColors();

    std::vector<CCVector3> prev_vertices = cloud->getPoints();
    std::vector<CCVector3> prev_vertex_normals;
    if (filter_normal) {
        prev_vertex_normals = cloud->getPointNormals();
    }
    ColorsTableType prev_vertex_colors;
    if (filter_color) {
        prev_vertex_colors = *cloud->rgbColors();
    }

    ccPointCloud *baseVertices = new ccPointCloud("vertices");
    assert(baseVertices);
    baseVertices->setEnabled(false);
    // DGM: no need to lock it as it is only used by one mesh!
    baseVertices->setLocked(false);
    std::shared_ptr<ccMesh> mesh =
            cloudViewer::make_shared<ccMesh>(baseVertices);
    mesh->addChild(baseVertices);

    baseVertices->resize(cloud->size());
    if (cloud->hasNormals()) {
        baseVertices->resizeTheNormsTable();
    }
    if (cloud->hasColors()) {
        baseVertices->resizeTheRGBTable();
    }

    mesh->setTriangles(getTriangles());
    mesh->adjacency_list_ = adjacency_list_;
    if (!mesh->hasAdjacencyList()) {
        mesh->computeAdjacencyList();
    }

    for (int iter = 0; iter < number_of_iterations; ++iter) {
        filterSmoothLaplacianHelper(mesh, prev_vertices, prev_vertex_normals,
                                    prev_vertex_colors, mesh->adjacency_list_,
                                    lambda, filter_vertex, filter_normal,
                                    filter_color);
        if (iter < number_of_iterations - 1) {
            std::swap(baseVertices->getPoints(), prev_vertices);
            if (filter_normal) {
                prev_vertex_normals = baseVertices->getPointNormals();
            }
            if (filter_color) {
                std::swap(*baseVertices->rgbColors(), prev_vertex_colors);
            }
        }
    }

    if (hasTriNormals()) {
        mesh->computeTriangleNormals();
    }

    // do some cleaning
    {
        baseVertices->shrinkToFit();
        mesh->shrinkToFit();
        NormsIndexesTableType *normals = mesh->getTriNormsTable();
        if (normals) {
            normals->shrink_to_fit();
        }
    }
    return mesh;
}

std::shared_ptr<ccMesh> ccMesh::filterSmoothTaubin(int number_of_iterations,
                                                   double lambda,
                                                   double mu,
                                                   FilterScope scope) const {
    ccPointCloud *cloud = ccHObjectCaster::ToPointCloud(m_associatedCloud);
    assert(cloud);

    bool filter_vertex =
            scope == FilterScope::All || scope == FilterScope::Vertex;
    bool filter_normal =
            (scope == FilterScope::All || scope == FilterScope::Normal) &&
            cloud->hasNormals();
    bool filter_color =
            (scope == FilterScope::All || scope == FilterScope::Color) &&
            cloud->hasColors();

    std::vector<CCVector3> prev_vertices = cloud->getPoints();
    std::vector<CCVector3> prev_vertex_normals;
    if (filter_normal) {
        prev_vertex_normals = cloud->getPointNormals();
    }
    ColorsTableType prev_vertex_colors;
    if (filter_color) {
        prev_vertex_colors = *cloud->rgbColors();
    }

    ccPointCloud *baseVertices = new ccPointCloud("vertices");
    assert(baseVertices);
    baseVertices->setEnabled(false);
    // DGM: no need to lock it as it is only used by one mesh!
    baseVertices->setLocked(false);
    std::shared_ptr<ccMesh> mesh =
            cloudViewer::make_shared<ccMesh>(baseVertices);
    mesh->addChild(baseVertices);

    baseVertices->resize(cloud->size());
    if (cloud->hasNormals()) {
        baseVertices->resizeTheNormsTable();
    }
    if (cloud->hasColors()) {
        baseVertices->resizeTheRGBTable();
    }

    mesh->setTriangles(getTriangles());
    mesh->adjacency_list_ = adjacency_list_;
    if (!mesh->hasAdjacencyList()) {
        mesh->computeAdjacencyList();
    }

    for (int iter = 0; iter < number_of_iterations; ++iter) {
        filterSmoothLaplacianHelper(mesh, prev_vertices, prev_vertex_normals,
                                    prev_vertex_colors, mesh->adjacency_list_,
                                    lambda, filter_vertex, filter_normal,
                                    filter_color);

        std::swap(baseVertices->getPoints(), prev_vertices);
        if (filter_normal) {
            prev_vertex_normals = baseVertices->getPointNormals();
        }
        if (filter_color) {
            std::swap(*baseVertices->rgbColors(), prev_vertex_colors);
        }

        filterSmoothLaplacianHelper(mesh, prev_vertices, prev_vertex_normals,
                                    prev_vertex_colors, mesh->adjacency_list_,
                                    mu, filter_vertex, filter_normal,
                                    filter_color);

        if (iter < number_of_iterations - 1) {
            std::swap(baseVertices->getPoints(), prev_vertices);
            if (filter_normal) {
                prev_vertex_normals = baseVertices->getPointNormals();
            }
            if (filter_color) {
                std::swap(*baseVertices->rgbColors(), prev_vertex_colors);
            }
        }
    }

    if (hasTriNormals()) {
        mesh->computeTriangleNormals();
    }

    // do some cleaning
    {
        baseVertices->shrinkToFit();
        mesh->shrinkToFit();
        NormsIndexesTableType *normals = mesh->getTriNormsTable();
        if (normals) {
            normals->shrink_to_fit();
        }
    }
    return mesh;
}

int ccMesh::eulerPoincareCharacteristic() const {
    std::unordered_set<Eigen::Vector2i,
                       utility::hash_eigen<Eigen::Vector2i>>
            edges;

    for (auto triangle : getTriangles()) {
        edges.emplace(GetOrderedEdge(triangle(0), triangle(1)));
        edges.emplace(GetOrderedEdge(triangle(0), triangle(2)));
        edges.emplace(GetOrderedEdge(triangle(1), triangle(2)));
    }

    int E = int(edges.size());
    int V = int(getVerticeSize());
    int F = int(size());
    return V + F - E;
}

std::vector<Eigen::Vector2i> ccMesh::getNonManifoldEdges(
        bool allow_boundary_edges /* = true */) const {
    auto edges = getEdgeToTrianglesMap();
    std::vector<Eigen::Vector2i> non_manifold_edges;
    for (auto &kv : edges) {
        if ((allow_boundary_edges &&
             (kv.second.size() < 1 || kv.second.size() > 2)) ||
            (!allow_boundary_edges && kv.second.size() != 2)) {
            non_manifold_edges.push_back(kv.first);
        }
    }
    return non_manifold_edges;
}

bool ccMesh::isEdgeManifold(bool allow_boundary_edges /* = true */) const {
    auto edges = getEdgeToTrianglesMap();
    for (auto &kv : edges) {
        if ((allow_boundary_edges &&
             (kv.second.size() < 1 || kv.second.size() > 2)) ||
            (!allow_boundary_edges && kv.second.size() != 2)) {
            return false;
        }
    }
    return true;
}

std::vector<int> ccMesh::getNonManifoldVertices() const {
    std::vector<std::unordered_set<int>> vert_to_triangles(getVerticeSize());
    for (size_t tidx = 0; tidx < size(); ++tidx) {
        const auto &tria = getTriangle(tidx);
        vert_to_triangles[tria(0)].emplace(int(tidx));
        vert_to_triangles[tria(1)].emplace(int(tidx));
        vert_to_triangles[tria(2)].emplace(int(tidx));
    }

    std::vector<int> non_manifold_verts;
    for (int vidx = 0; vidx < int(getVerticeSize()); ++vidx) {
        const auto &triangles = vert_to_triangles[vidx];
        if (triangles.size() == 0) {
            continue;
        }

        // collect edges and vertices
        std::unordered_map<int, std::unordered_set<int>> edges;
        for (int tidx : triangles) {
            const auto &triangle = getTriangle(static_cast<size_t>(tidx));
            if (triangle(0) != vidx && triangle(1) != vidx) {
                edges[triangle(0)].emplace(triangle(1));
                edges[triangle(1)].emplace(triangle(0));
            } else if (triangle(0) != vidx && triangle(2) != vidx) {
                edges[triangle(0)].emplace(triangle(2));
                edges[triangle(2)].emplace(triangle(0));
            } else if (triangle(1) != vidx && triangle(2) != vidx) {
                edges[triangle(1)].emplace(triangle(2));
                edges[triangle(2)].emplace(triangle(1));
            }
        }

        // test if vertices are connected
        std::queue<int> next;
        std::unordered_set<int> visited;
        next.push(edges.begin()->first);
        visited.emplace(edges.begin()->first);
        while (!next.empty()) {
            int vert = next.front();
            next.pop();

            for (auto nb : edges[vert]) {
                if (visited.count(nb) == 0) {
                    visited.emplace(nb);
                    next.emplace(nb);
                }
            }
        }
        if (visited.size() != edges.size()) {
            non_manifold_verts.push_back(vidx);
        }
    }

    return non_manifold_verts;
}

bool ccMesh::isVertexManifold() const {
    return getNonManifoldVertices().empty();
}

std::vector<Eigen::Vector2i> ccMesh::getSelfIntersectingTriangles() const {
    std::vector<Eigen::Vector2i> self_intersecting_triangles;
    for (size_t tidx0 = 0; tidx0 < size() - 1; ++tidx0) {
        const Eigen::Vector3i &tria_p = getTriangle(tidx0);
        const Eigen::Vector3d &p0 = getVertice(static_cast<size_t>(tria_p(0)));
        const Eigen::Vector3d &p1 = getVertice(static_cast<size_t>(tria_p(1)));
        const Eigen::Vector3d &p2 = getVertice(static_cast<size_t>(tria_p(2)));
        for (size_t tidx1 = tidx0 + 1; tidx1 < size(); ++tidx1) {
            const Eigen::Vector3i &tria_q = getTriangle(tidx1);
            // check if neighbour triangle
            if (tria_p(0) == tria_q(0) || tria_p(0) == tria_q(1) ||
                tria_p(0) == tria_q(2) || tria_p(1) == tria_q(0) ||
                tria_p(1) == tria_q(1) || tria_p(1) == tria_q(2) ||
                tria_p(2) == tria_q(0) || tria_p(2) == tria_q(1) ||
                tria_p(2) == tria_q(2)) {
                continue;
            }

            // check for intersection
            const Eigen::Vector3d &q0 =
                    getVertice(static_cast<size_t>(tria_q(0)));
            const Eigen::Vector3d &q1 =
                    getVertice(static_cast<size_t>(tria_q(1)));
            const Eigen::Vector3d &q2 =
                    getVertice(static_cast<size_t>(tria_q(2)));
            if (utility::IntersectionTest::TriangleTriangle3d(p0, p1, p2, q0,
                                                              q1, q2)) {
                self_intersecting_triangles.push_back(
                        Eigen::Vector2i(tidx0, tidx1));
            }
        }
    }
    return self_intersecting_triangles;
}

bool ccMesh::isSelfIntersecting() const {
    return !getSelfIntersectingTriangles().empty();
}

bool ccMesh::isBoundingBoxIntersecting(const ccMesh &other) const {
    return utility::IntersectionTest::AABBAABB(getMinBound(), getMaxBound(),
                                               other.getMinBound(),
                                               other.getMaxBound());
}

bool ccMesh::isIntersecting(const ccMesh &other) const {
    if (!isBoundingBoxIntersecting(other)) {
        return false;
    }
    for (size_t tidx0 = 0; tidx0 < size(); ++tidx0) {
        const Eigen::Vector3i &tria_p = getTriangle(tidx0);
        const Eigen::Vector3d &p0 = getVertice(static_cast<size_t>(tria_p(0)));
        const Eigen::Vector3d &p1 = getVertice(static_cast<size_t>(tria_p(1)));
        const Eigen::Vector3d &p2 = getVertice(static_cast<size_t>(tria_p(2)));
        for (size_t tidx1 = 0; tidx1 < other.size(); ++tidx1) {
            const Eigen::Vector3i &tria_q = other.getTriangle(tidx1);
            const Eigen::Vector3d &q0 =
                    other.getVertice(static_cast<size_t>(tria_q(0)));
            const Eigen::Vector3d &q1 =
                    other.getVertice(static_cast<size_t>(tria_q(1)));
            const Eigen::Vector3d &q2 =
                    other.getVertice(static_cast<size_t>(tria_q(2)));
            if (utility::IntersectionTest::TriangleTriangle3d(p0, p1, p2, q0,
                                                              q1, q2)) {
                return true;
            }
        }
    }
    return false;
}

template <typename F>
bool OrientTriangleHelper(const std::vector<Eigen::Vector3i> &triangles,
                          F &swap) {
    std::unordered_map<Eigen::Vector2i, Eigen::Vector2i,
                       utility::hash_eigen<Eigen::Vector2i>>
            edge_to_orientation;
    std::unordered_set<int> unvisited_triangles;
    std::unordered_map<Eigen::Vector2i, std::unordered_set<int>,
                       utility::hash_eigen<Eigen::Vector2i>>
            adjacent_triangles;
    std::queue<int> triangle_queue;

    auto VerifyAndAdd = [&](int vidx0, int vidx1) {
        Eigen::Vector2i key = ccMesh::GetOrderedEdge(vidx0, vidx1);
        if (edge_to_orientation.count(key) > 0) {
            if (edge_to_orientation.at(key)(0) == vidx0) {
                return false;
            }
        } else {
            edge_to_orientation[key] = Eigen::Vector2i(vidx0, vidx1);
        }
        return true;
    };
    auto AddTriangleNbsToQueue = [&](const Eigen::Vector2i &edge) {
        for (int nb_tidx : adjacent_triangles[edge]) {
            triangle_queue.push(nb_tidx);
        }
    };

    for (size_t tidx = 0; tidx < triangles.size(); ++tidx) {
        unvisited_triangles.insert(int(tidx));
        const auto &triangle = triangles[tidx];
        int vidx0 = triangle(0);
        int vidx1 = triangle(1);
        int vidx2 = triangle(2);
        adjacent_triangles[ccMesh::GetOrderedEdge(vidx0, vidx1)].insert(
                int(tidx));
        adjacent_triangles[ccMesh::GetOrderedEdge(vidx1, vidx2)].insert(
                int(tidx));
        adjacent_triangles[ccMesh::GetOrderedEdge(vidx2, vidx0)].insert(
                int(tidx));
    }

    while (!unvisited_triangles.empty()) {
        int tidx;
        if (triangle_queue.empty()) {
            tidx = *unvisited_triangles.begin();
        } else {
            tidx = triangle_queue.front();
            triangle_queue.pop();
        }
        if (unvisited_triangles.count(tidx) > 0) {
            unvisited_triangles.erase(tidx);
        } else {
            continue;
        }

        const auto &triangle = triangles[tidx];
        int vidx0 = triangle(0);
        int vidx1 = triangle(1);
        int vidx2 = triangle(2);
        Eigen::Vector2i key01 = ccMesh::GetOrderedEdge(vidx0, vidx1);
        Eigen::Vector2i key12 = ccMesh::GetOrderedEdge(vidx1, vidx2);
        Eigen::Vector2i key20 = ccMesh::GetOrderedEdge(vidx2, vidx0);
        bool exist01 = edge_to_orientation.count(key01) > 0;
        bool exist12 = edge_to_orientation.count(key12) > 0;
        bool exist20 = edge_to_orientation.count(key20) > 0;

        if (!(exist01 || exist12 || exist20)) {
            edge_to_orientation[key01] = Eigen::Vector2i(vidx0, vidx1);
            edge_to_orientation[key12] = Eigen::Vector2i(vidx1, vidx2);
            edge_to_orientation[key20] = Eigen::Vector2i(vidx2, vidx0);
        } else {
            // one flip is allowed
            if (exist01 && edge_to_orientation.at(key01)(0) == vidx0) {
                std::swap(vidx0, vidx1);
                swap(tidx, 0, 1);
            } else if (exist12 && edge_to_orientation.at(key12)(0) == vidx1) {
                std::swap(vidx1, vidx2);
                swap(tidx, 1, 2);
            } else if (exist20 && edge_to_orientation.at(key20)(0) == vidx2) {
                std::swap(vidx2, vidx0);
                swap(tidx, 2, 0);
            }

            // check if each edge looks in different direction compared to
            // existing ones if not existend, add the edge to map
            if (!VerifyAndAdd(vidx0, vidx1)) {
                return false;
            }
            if (!VerifyAndAdd(vidx1, vidx2)) {
                return false;
            }
            if (!VerifyAndAdd(vidx2, vidx0)) {
                return false;
            }
        }

        AddTriangleNbsToQueue(key01);
        AddTriangleNbsToQueue(key12);
        AddTriangleNbsToQueue(key20);
    }
    return true;
}

bool ccMesh::isOrientable() const {
    auto NoOp = [](int, int, int) {};
    return OrientTriangleHelper(getTriangles(), NoOp);
}

bool ccMesh::isWatertight() const {
    return isEdgeManifold(false) && isVertexManifold() && !isSelfIntersecting();
}

bool ccMesh::orientTriangles() {
    auto SwapTriangleOrder = [&](int tidx, int idx0, int idx1) {
        std::swap(getTriangleVertIndexes(static_cast<unsigned>(tidx))->i[idx0],
                  getTriangleVertIndexes(static_cast<unsigned>(tidx))->i[idx1]);
    };
    return OrientTriangleHelper(getTriangles(), SwapTriangleOrder);
}

std::tuple<std::vector<int>, std::vector<size_t>, std::vector<double>>
ccMesh::clusterConnectedTriangles() const {
    std::vector<int> triangle_clusters(this->size(), -1);
    std::vector<size_t> num_triangles;
    std::vector<double> areas;

    utility::LogDebug("[ClusterConnectedTriangles] Compute triangle adjacency");
    auto edges_to_triangles = getEdgeToTrianglesMap();
    std::vector<std::unordered_set<int>> adjacency_list(this->size());
#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
        num_threads(utility::EstimateMaxThreads())
#endif
    for (int tidx = 0; tidx < int(size()); ++tidx) {
        const auto &triangle = getTriangle(static_cast<size_t>(tidx));
        for (auto tnb :
             edges_to_triangles[GetOrderedEdge(triangle(0), triangle(1))]) {
            adjacency_list[tidx].insert(tnb);
        }
        for (auto tnb :
             edges_to_triangles[GetOrderedEdge(triangle(0), triangle(2))]) {
            adjacency_list[tidx].insert(tnb);
        }
        for (auto tnb :
             edges_to_triangles[GetOrderedEdge(triangle(1), triangle(2))]) {
            adjacency_list[tidx].insert(tnb);
        }
    }
    utility::LogDebug(
            "[ClusterConnectedTriangles] Done computing triangle adjacency");

    int cluster_idx = 0;
    for (int tidx = 0; tidx < int(this->size()); ++tidx) {
        if (triangle_clusters[tidx] != -1) {
            continue;
        }

        std::queue<int> triangle_queue;
        int cluster_n_triangles = 0;
        double cluster_area = 0;

        triangle_queue.push(tidx);
        triangle_clusters[tidx] = cluster_idx;
        while (!triangle_queue.empty()) {
            int cluster_tidx = triangle_queue.front();
            triangle_queue.pop();

            cluster_n_triangles++;
            cluster_area += getTriangleArea(cluster_tidx);

            for (auto tnb : adjacency_list[cluster_tidx]) {
                if (triangle_clusters[tnb] == -1) {
                    triangle_queue.push(tnb);
                    triangle_clusters[tnb] = cluster_idx;
                }
            }
        }

        num_triangles.push_back(cluster_n_triangles);
        areas.push_back(cluster_area);
        cluster_idx++;
    }

    utility::LogDebug(
            "[ClusterConnectedTriangles] Done clustering, #clusters={}",
            cluster_idx);
    return std::make_tuple(triangle_clusters, num_triangles, areas);
}

void ccMesh::removeTrianglesByIndex(
        const std::vector<size_t> &triangle_indices) {
    std::vector<bool> triangle_mask(size(), false);
    for (auto tidx : triangle_indices) {
        if (tidx >= 0 && tidx < size()) {
            triangle_mask[tidx] = true;
        } else {
            utility::LogWarning(
                    "[RemoveTriangles] contains triangle index {} that is not "
                    "within the bounds",
                    tidx);
        }
    }

    removeTrianglesByMask(triangle_mask);
}

void ccMesh::removeTrianglesByMask(const std::vector<bool> &triangle_mask) {
    if (triangle_mask.size() != this->size()) {
        utility::LogError("triangle_mask has a different size than triangles_");
    }

    bool has_tri_normal = hasTriNormals();
    int to_tidx = 0;
    for (size_t from_tidx = 0; from_tidx < this->size(); ++from_tidx) {
        if (!triangle_mask[from_tidx]) {
            setTriangle(to_tidx, getTriangle(from_tidx));
            if (has_tri_normal) {
                setTriangleNorm(to_tidx, getTriangleNorm(from_tidx));
            }
            to_tidx++;
        }
    }

    this->resize(to_tidx);
    if (has_tri_normal) {
        getTriNormsTable()->resize(to_tidx);
    }
}

void ccMesh::removeVerticesByIndex(const std::vector<size_t> &vertex_indices) {
    std::vector<bool> vertex_mask(getVerticeSize(), false);
    for (auto vidx : vertex_indices) {
        if (vidx >= 0 && vidx < getVerticeSize()) {
            vertex_mask[vidx] = true;
        } else {
            utility::LogWarning(
                    "[RemoveVerticessByIndex] contains vertex index {} that is "
                    "not within the bounds",
                    vidx);
        }
    }

    removeVerticesByMask(vertex_mask);
}

void ccMesh::removeVerticesByMask(const std::vector<bool> &vertex_mask) {
    ccPointCloud *cloud = ccHObjectCaster::ToPointCloud(m_associatedCloud);
    assert(cloud);

    if (vertex_mask.size() != cloud->size()) {
        utility::LogError("vertex_mask has a different size than vertices_");
    }

    bool has_normal = cloud->hasNormals();
    bool has_color = cloud->hasColors();
    int to_vidx = 0;
    std::unordered_map<int, int> vertex_map;
    for (unsigned from_vidx = 0; from_vidx < cloud->size(); ++from_vidx) {
        if (!vertex_mask[from_vidx]) {
            vertex_map[static_cast<int>(from_vidx)] = static_cast<int>(to_vidx);
            cloud->setPoint(to_vidx, *cloud->getPoint(from_vidx));
            if (has_normal) {
                cloud->setPointNormal(to_vidx,
                                      cloud->getPointNormal(from_vidx));
            }
            if (has_color) {
                cloud->setPointColor(to_vidx, cloud->getPointColor(from_vidx));
            }
            to_vidx++;
        }
    }

    cloud->resize(to_vidx);

    std::vector<bool> triangle_mask(this->size());
    for (unsigned tidx = 0; tidx < this->size(); ++tidx) {
        cloudViewer::VerticesIndexes *tria = getTriangleVertIndexes(tidx);
        triangle_mask[tidx] = vertex_mask[tria->i[0]] ||
                              vertex_mask[tria->i[1]] ||
                              vertex_mask[tria->i[2]];
        if (!triangle_mask[tidx]) {
            tria->i[0] = vertex_map[tria->i[0]];
            tria->i[1] = vertex_map[tria->i[1]];
            tria->i[2] = vertex_map[tria->i[2]];
        }
    }
    removeTrianglesByMask(triangle_mask);
}

std::unordered_map<Eigen::Vector2i,
                   double,
                   utility::hash_eigen<Eigen::Vector2i>>
ccMesh::computeEdgeWeightsCot(
        const std::unordered_map<Eigen::Vector2i,
                                 std::vector<int>,
                                 utility::hash_eigen<Eigen::Vector2i>>
                &edges_to_vertices,
        double min_weight) const {
    std::unordered_map<Eigen::Vector2i, double,
                       utility::hash_eigen<Eigen::Vector2i>>
            weights;

    for (const auto &edge_v2s : edges_to_vertices) {
        Eigen::Vector2i edge = edge_v2s.first;
        double weight_sum = 0;
        int N = 0;
        for (int v2 : edge_v2s.second) {
            Eigen::Vector3d a = getVertice(static_cast<size_t>(edge(0))) -
                                getVertice(static_cast<size_t>(v2));
            Eigen::Vector3d b = getVertice(static_cast<size_t>(edge(1))) -
                                getVertice(static_cast<size_t>(v2));

            double weight = a.dot(b) / (a.cross(b)).norm();
            weight_sum += weight;
            N++;
        }
        double weight = N > 0 ? weight_sum / N : 0;
        if (weight < min_weight) {
            weights[edge] = min_weight;
        } else {
            weights[edge] = weight;
        }
    }
    return weights;
}

ccMesh &ccMesh::computeTriangleNormals(bool normalized /* = true*/) {
    computePerTriangleNormals();
    if (normalized) {
        normalizeNormals();
    }
    return *this;
}

ccMesh &ccMesh::computeVertexNormals(bool normalized /* = true*/) {
    if (!hasTriNormals()) {
        computeTriangleNormals(false);
    }
    computePerVertexNormals();

    if (normalized) {
        normalizeNormals();
    }
    return *this;
}

ccMesh &ccMesh::normalizeNormals() {
    ccPointCloud *cloud = ccHObjectCaster::ToPointCloud(m_associatedCloud);
    if (cloud && cloud->hasNormals()) {
        cloud->normalizeNormals();
    }

    if (hasTriNormals()) {
        for (size_t i = 0; i < m_triNormals->size(); ++i) {
            ccNormalVectors::GetNormalPtr(m_triNormals->getValue(i))
                    .normalize();
        }
    }

    return *this;
}

ccMesh &ccMesh::computeAdjacencyList() {
    adjacency_list_.clear();
    adjacency_list_.resize(getVerticeSize());
    for (size_t i = 0; i < size(); ++i) {
        Eigen::Vector3i triangle = getTriangle(i);
        adjacency_list_[triangle(0)].insert(triangle(1));
        adjacency_list_[triangle(0)].insert(triangle(2));
        adjacency_list_[triangle(1)].insert(triangle(0));
        adjacency_list_[triangle(1)].insert(triangle(2));
        adjacency_list_[triangle(2)].insert(triangle(0));
        adjacency_list_[triangle(2)].insert(triangle(1));
    }
    return *this;
}

std::shared_ptr<ccPointCloud> ccMesh::samplePointsPoissonDisk(
        size_t number_of_points,
        double init_factor /* = 5 */,
        const std::shared_ptr<ccPointCloud> pcl_init /* = nullptr */,
        bool use_triangle_normal /* = false */,
        int seed /* = -1 */) {
    if (number_of_points <= 0) {
        utility::LogError("[SamplePointsPoissonDisk] number_of_points <= 0");
    }
    if (size() == 0) {
        utility::LogError(
                "[SamplePointsPoissonDisk] input mesh has no triangles");
    }
    if (pcl_init == nullptr && init_factor < 1) {
        utility::LogError(
                "[SamplePointsPoissonDisk] either pass pcl_init with #points "
                "> number_of_points or init_factor > 1");
    }
    if (pcl_init != nullptr && pcl_init->size() < number_of_points) {
        utility::LogError(
                "[SamplePointsPoissonDisk] either pass pcl_init with #points "
                "> number_of_points, or init_factor > 1");
    }

    // Compute area of each triangle and sum surface area
    std::vector<double> triangle_areas;
    double surface_area = getSurfaceArea(triangle_areas);

    // Compute init points using uniform sampling
    std::shared_ptr<ccPointCloud> pcl;
    if (pcl_init == nullptr) {
        pcl = samplePointsUniformlyImpl(size_t(init_factor * number_of_points),
                                        triangle_areas, surface_area,
                                        use_triangle_normal, seed);
    } else {
        pcl = cloudViewer::make_shared<ccPointCloud>();
        *pcl = *pcl_init;
    }

    // Set-up sample elimination
    double alpha = 8;    // constant defined in paper
    double beta = 0.5;   // constant defined in paper
    double gamma = 1.5;  // constant defined in paper
    double ratio = double(number_of_points) / double(pcl->size());
    double r_max = 2 * std::sqrt((surface_area / number_of_points) /
                                 (2 * std::sqrt(3.)));
    double r_min = r_max * beta * (1 - std::pow(ratio, gamma));

    std::vector<double> weights(pcl->size());
    std::vector<bool> deleted(pcl->size(), false);
    cloudViewer::geometry::KDTreeFlann kdtree(*pcl);

    auto WeightFcn = [&](double d2) {
        double d = std::sqrt(d2);
        if (d < r_min) {
            d = r_min;
        }
        return std::pow(1 - d / r_max, alpha);
    };

    auto ComputePointWeight = [&](int pidx0) {
        std::vector<int> nbs;
        std::vector<double> dists2;
        kdtree.SearchRadius(pcl->getEigenPoint(static_cast<size_t>(pidx0)),
                            r_max, nbs, dists2);
        double weight = 0;
        for (size_t nbidx = 0; nbidx < nbs.size(); ++nbidx) {
            int pidx1 = nbs[nbidx];
            // only count weights if not the same point if not deleted
            if (pidx0 == pidx1 || deleted[pidx1]) {
                continue;
            }
            weight += WeightFcn(dists2[nbidx]);
        }

        weights[pidx0] = weight;
    };

    // init weights and priority queue
    typedef std::tuple<int, double> QueueEntry;
    auto WeightCmp = [](const QueueEntry &a, const QueueEntry &b) {
        return std::get<1>(a) < std::get<1>(b);
    };
    std::priority_queue<QueueEntry, std::vector<QueueEntry>,
                        decltype(WeightCmp)>
            queue(WeightCmp);
    for (size_t pidx0 = 0; pidx0 < pcl->size(); ++pidx0) {
        ComputePointWeight(int(pidx0));
        queue.push(QueueEntry(int(pidx0), weights[pidx0]));
    };

    // sample elimination
    size_t current_number_of_points = pcl->size();
    while (current_number_of_points > number_of_points) {
        int pidx;
        double weight;
        std::tie(pidx, weight) = queue.top();
        queue.pop();

        // test if the entry is up to date (because of reinsert)
        if (deleted[pidx] || weight != weights[pidx]) {
            continue;
        }

        // delete current sample
        deleted[pidx] = true;
        current_number_of_points--;

        // update weights
        std::vector<int> nbs;
        std::vector<double> dists2;
        kdtree.SearchRadius(pcl->getEigenPoint(static_cast<size_t>(pidx)),
                            r_max, nbs, dists2);
        for (int nb : nbs) {
            ComputePointWeight(nb);
            queue.push(QueueEntry(nb, weights[nb]));
        }
    }

    // update pcl
    bool has_vert_normal = pcl->hasNormals();
    bool has_vert_color = pcl->hasColors();
    size_t next_free = 0;
    for (size_t idx = 0; idx < pcl->size(); ++idx) {
        if (!deleted[idx]) {
            pcl->setPoint(next_free,
                          *pcl->getPoint(static_cast<unsigned>(idx)));
            if (has_vert_normal) {
                pcl->setPointNormal(
                        next_free,
                        pcl->getPointNormal(static_cast<unsigned>(idx)));
            }
            if (has_vert_color) {
                pcl->setPointColor(
                        static_cast<unsigned>(next_free),
                        pcl->getPointColor(static_cast<unsigned>(idx)));
            }
            next_free++;
        }
    }

    // pcl->shrinkToFit();
    pcl->resize(static_cast<unsigned>(next_free));
    if (has_vert_normal) {
        pcl->resizeTheNormsTable();
    }
    if (has_vert_color) {
        pcl->resizeTheRGBTable();
    }

    return pcl;
}

std::unordered_map<Eigen::Vector2i,
                   std::vector<int>,
                   utility::hash_eigen<Eigen::Vector2i>>
ccMesh::getEdgeToTrianglesMap() const {
    std::unordered_map<Eigen::Vector2i, std::vector<int>,
                       utility::hash_eigen<Eigen::Vector2i>>
            trias_per_edge;
    auto AddEdge = [&](int vidx0, int vidx1, int tidx) {
        trias_per_edge[GetOrderedEdge(vidx0, vidx1)].push_back(tidx);
    };

    Eigen::Vector3i triangle;
    for (size_t tidx = 0; tidx < size(); ++tidx) {
        getTriangleVertIndexes(tidx, triangle);
        AddEdge(triangle(0), triangle(1), int(tidx));
        AddEdge(triangle(1), triangle(2), int(tidx));
        AddEdge(triangle(2), triangle(0), int(tidx));
    }
    return trias_per_edge;
}

std::unordered_map<Eigen::Vector2i,
                   std::vector<int>,
                   utility::hash_eigen<Eigen::Vector2i>>
ccMesh::getEdgeToVerticesMap() const {
    std::unordered_map<Eigen::Vector2i, std::vector<int>,
                       utility::hash_eigen<Eigen::Vector2i>>
            trias_per_edge;
    auto AddEdge = [&](int vidx0, int vidx1, int vidx2) {
        trias_per_edge[GetOrderedEdge(vidx0, vidx1)].push_back(vidx2);
    };

    Eigen::Vector3i triangle;
    for (size_t tidx = 0; tidx < size(); ++tidx) {
        getTriangleVertIndexes(tidx, triangle);
        AddEdge(triangle(0), triangle(1), triangle(2));
        AddEdge(triangle(1), triangle(2), triangle(0));
        AddEdge(triangle(2), triangle(0), triangle(1));
    }
    return trias_per_edge;
}

double ccMesh::ComputeTriangleArea(const Eigen::Vector3d &p0,
                                   const Eigen::Vector3d &p1,
                                   const Eigen::Vector3d &p2) {
    const Eigen::Vector3d x = p0 - p1;
    const Eigen::Vector3d y = p0 - p2;
    double area = 0.5 * x.cross(y).norm();
    return area;
}

double ccMesh::getTriangleArea(size_t triangle_idx) const {
    Eigen::Vector3d vertex0, vertex1, vertex2;
    getTriangleVertices(static_cast<unsigned int>(triangle_idx), vertex0.data(),
                        vertex1.data(), vertex2.data());
    return ComputeTriangleArea(vertex0, vertex1, vertex2);
}

double ccMesh::getSurfaceArea() const {
    double surface_area = 0;
    for (size_t tidx = 0; tidx < size(); ++tidx) {
        double triangle_area = getTriangleArea(tidx);
        surface_area += triangle_area;
    }
    return surface_area;
}

double ccMesh::getSurfaceArea(std::vector<double> &triangle_areas) const {
    double surface_area = 0;
    triangle_areas.resize(size());
    for (size_t tidx = 0; tidx < size(); ++tidx) {
        double triangle_area = getTriangleArea(tidx);
        triangle_areas[tidx] = triangle_area;
        surface_area += triangle_area;
    }
    return surface_area;
}

double ccMesh::getVolume() const {
    // Computes the signed volume of the tetrahedron defined by
    // the three triangle vertices and the origin. The sign is determined by
    // checking if the origin is at the same side as the normal with respect to
    // the triangle.
    auto GetSignedVolumeOfTriangle = [&](size_t tidx) {
        const Eigen::Vector3i &triangle = getTriangle(tidx);
        const Eigen::Vector3d &vertex0 = getVertice(triangle(0));
        const Eigen::Vector3d &vertex1 = getVertice(triangle(1));
        const Eigen::Vector3d &vertex2 = getVertice(triangle(2));
        return vertex0.dot(vertex1.cross(vertex2)) / 6.0;
    };

    if (!isWatertight()) {
        utility::LogError(
                "The mesh is not watertight, and the volume cannot be "
                "computed.");
    }
    if (!isOrientable()) {
        utility::LogError(
                "The mesh is not orientable, and the volume cannot be "
                "computed.");
    }

    double volume = 0;
    std::int64_t num_triangles = static_cast<std::int64_t>(this->size());
#pragma omp parallel for reduction(+ : volume)
    for (std::int64_t tidx = 0; tidx < num_triangles; ++tidx) {
        volume += GetSignedVolumeOfTriangle(tidx);
    }
    return std::abs(volume);
}

Eigen::Vector4d ccMesh::ComputeTrianglePlane(const Eigen::Vector3d &p0,
                                             const Eigen::Vector3d &p1,
                                             const Eigen::Vector3d &p2) {
    const Eigen::Vector3d e0 = p1 - p0;
    const Eigen::Vector3d e1 = p2 - p0;
    Eigen::Vector3d abc = e0.cross(e1);
    double norm = abc.norm();
    // if the three points are co-linear, return invalid plane
    if (norm == 0) {
        return Eigen::Vector4d(0, 0, 0, 0);
    }
    abc /= norm;
    double d = -abc.dot(p0);
    return Eigen::Vector4d(abc(0), abc(1), abc(2), d);
}

Eigen::Vector4d ccMesh::getTrianglePlane(size_t triangle_idx) const {
    Eigen::Vector3d vertex0, vertex1, vertex2;
    getTriangleVertices(static_cast<unsigned int>(triangle_idx), vertex0.data(),
                        vertex1.data(), vertex2.data());
    return ComputeTrianglePlane(vertex0, vertex1, vertex2);
}

std::shared_ptr<ccPointCloud> ccMesh::samplePointsUniformlyImpl(
        size_t number_of_points,
        std::vector<double> &triangle_areas,
        double surface_area,
        bool use_triangle_normal,
        int seed) {
    // triangle areas to cdf
    triangle_areas[0] /= surface_area;
    for (size_t tidx = 1; tidx < size(); ++tidx) {
        triangle_areas[tidx] =
                triangle_areas[tidx] / surface_area + triangle_areas[tidx - 1];
    }

    // sample point cloud
    bool has_vert_normal = m_associatedCloud->hasNormals();
    bool has_vert_color = m_associatedCloud->hasColors();
    if (seed == -1) {
        std::random_device rd;
        seed = rd();
    }
    std::mt19937 mt(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    auto pcd = cloudViewer::make_shared<ccPointCloud>();
    pcd->resize(static_cast<unsigned>(number_of_points));
    if (has_vert_normal || use_triangle_normal) {
        pcd->resizeTheNormsTable();
    }
    if (use_triangle_normal && !hasTriNormals()) {
        computeNormals(true);
    }
    if (has_vert_color) {
        pcd->resizeTheRGBTable();
    }
    size_t point_idx = 0;
    for (size_t tidx = 0; tidx < size(); ++tidx) {
        size_t n = size_t(std::round(triangle_areas[tidx] * number_of_points));
        while (point_idx < n) {
            double r1 = dist(mt);
            double r2 = dist(mt);
            double a = (1 - std::sqrt(r1));
            double b = std::sqrt(r1) * (1 - r2);
            double c = std::sqrt(r1) * r2;

            Eigen::Vector3d vert0, vert1, vert2;
            getTriangleVertices(static_cast<unsigned int>(tidx), vert0.data(),
                                vert1.data(), vert2.data());
            Eigen::Vector3d temp = a * vert0 + b * vert1 + c * vert2;
            pcd->setPoint(point_idx, temp);

            assert(m_associatedCloud);
            ccPointCloud *cloud = (ccPointCloud *)m_associatedCloud;
            const cloudViewer::VerticesIndexes *tri =
                    getTriangleVertIndexes(static_cast<unsigned int>(tidx));

            if (has_vert_normal && !use_triangle_normal) {
                Eigen::Vector3d N = a * cloud->getEigenNormal(tri->i1) +
                                    b * cloud->getEigenNormal(tri->i2) +
                                    c * cloud->getEigenNormal(tri->i3);
                pcd->setPointNormal(point_idx, N);
            }
            if (use_triangle_normal) {
                pcd->setPointNormal(point_idx, getTriangleNorm(tidx));
            }
            if (has_vert_color) {
                Eigen::Vector3d C = a * cloud->getEigenColor(tri->i1) +
                                    b * cloud->getEigenColor(tri->i2) +
                                    c * cloud->getEigenColor(tri->i3);
                pcd->setPointColor(point_idx, C);
            }

            point_idx++;
        }
    }

    return pcd;
}

std::shared_ptr<ccPointCloud> ccMesh::samplePointsUniformly(
        size_t number_of_points,
        bool use_triangle_normal /* = false */,
        int seed /* = -1 */) {
    if (number_of_points <= 0) {
        utility::LogError("[SamplePointsUniformly] number_of_points <= 0");
    }
    if (size() == 0) {
        utility::LogError(
                "[SamplePointsUniformly] input mesh has no triangles");
    }

    // Compute area of each triangle and sum surface area
    std::vector<double> triangle_areas;
    double surface_area = getSurfaceArea(triangle_areas);

    return samplePointsUniformlyImpl(number_of_points, triangle_areas,
                                     surface_area, use_triangle_normal, seed);
}

std::shared_ptr<ccMesh> ccMesh::CreateTetrahedron(
        double radius /* = 1.0*/, bool create_uv_map /* = false*/) {
    ccPointCloud *baseVertices = new ccPointCloud("vertices");
    assert(baseVertices);
    auto mesh = cloudViewer::make_shared<ccMesh>(baseVertices);
    if (radius <= 0) {
        utility::LogError("[CreateTetrahedron] radius <= 0");
    }

    // Vertices.
    baseVertices->addEigenPoint(
            radius * Eigen::Vector3d(std::sqrt(8. / 9.), 0, -1. / 3.));
    baseVertices->addEigenPoint(radius * Eigen::Vector3d(-std::sqrt(2. / 9.),
                                                         std::sqrt(2. / 3.),
                                                         -1. / 3.));
    baseVertices->addEigenPoint(radius * Eigen::Vector3d(-std::sqrt(2. / 9.),
                                                         -std::sqrt(2. / 3.),
                                                         -1. / 3.));
    baseVertices->addEigenPoint(radius * Eigen::Vector3d(0., 0., 1.));

    // Triangles.
    mesh->addTriangle(Eigen::Vector3i(0, 2, 1));
    mesh->addTriangle(Eigen::Vector3i(0, 3, 2));
    mesh->addTriangle(Eigen::Vector3i(0, 1, 3));
    mesh->addTriangle(Eigen::Vector3i(1, 2, 3));

    // UV Map.
    if (create_uv_map) {
        mesh->triangle_uvs_ = {{0.866, 0.5},  {0.433, 0.75}, {0.433, 0.25},
                               {0.866, 0.5},  {0.866, 1.0},  {0.433, 0.75},
                               {0.866, 0.5},  {0.433, 0.25}, {0.866, 0.0},
                               {0.433, 0.25}, {0.433, 0.75}, {0.0, 0.5}};
    }

    // do some cleaning
    {
        baseVertices->shrinkToFit();
        mesh->shrinkToFit();
        NormsIndexesTableType *normals = mesh->getTriNormsTable();
        if (normals) {
            normals->shrink_to_fit();
        }
    }

    baseVertices->setEnabled(false);
    // DGM: no need to lock it as it is only used by one mesh!
    baseVertices->setLocked(false);
    mesh->addChild(baseVertices);
    return mesh;
}

std::shared_ptr<ccMesh> ccMesh::CreateOctahedron(
        double radius /* = 1.0*/, bool create_uv_map /* = false*/) {
    ccPointCloud *baseVertices = new ccPointCloud("vertices");
    assert(baseVertices);
    auto mesh = cloudViewer::make_shared<ccMesh>(baseVertices);

    if (radius <= 0) {
        utility::LogError("[CreateOctahedron] radius <= 0");
    }

    // Vertices.
    baseVertices->addEigenPoint(radius * Eigen::Vector3d(1, 0, 0));
    baseVertices->addEigenPoint(radius * Eigen::Vector3d(0, 1, 0));
    baseVertices->addEigenPoint(radius * Eigen::Vector3d(0, 0, 1));
    baseVertices->addEigenPoint(radius * Eigen::Vector3d(-1, 0, 0));
    baseVertices->addEigenPoint(radius * Eigen::Vector3d(0, -1, 0));
    baseVertices->addEigenPoint(radius * Eigen::Vector3d(0, 0, -1));

    // Triangles.
    mesh->addTriangle(Eigen::Vector3i(0, 1, 2));
    mesh->addTriangle(Eigen::Vector3i(1, 3, 2));
    mesh->addTriangle(Eigen::Vector3i(3, 4, 2));
    mesh->addTriangle(Eigen::Vector3i(4, 0, 2));
    mesh->addTriangle(Eigen::Vector3i(0, 5, 1));
    mesh->addTriangle(Eigen::Vector3i(1, 5, 3));
    mesh->addTriangle(Eigen::Vector3i(3, 5, 4));
    mesh->addTriangle(Eigen::Vector3i(4, 5, 0));

    // UV Map.
    if (create_uv_map) {
        mesh->triangle_uvs_ = {
                {0.0, 0.75},    {0.1444, 0.5},  {0.2887, 0.75}, {0.1444, 0.5},
                {0.433, 0.5},   {0.2887, 0.75}, {0.433, 0.5},   {0.5773, 0.75},
                {0.2887, 0.75}, {0.5773, 0.75}, {0.433, 1.0},   {0.2887, 0.75},
                {0.0, 0.25},    {0.2887, 0.25}, {0.1444, 0.5},  {0.1444, 0.5},
                {0.2887, 0.25}, {0.433, 0.5},   {0.433, 0.5},   {0.2887, 0.25},
                {0.5773, 0.25}, {0.5773, 0.25}, {0.2887, 0.25}, {0.433, 0.0}};
    }

    // do some cleaning
    {
        baseVertices->shrinkToFit();
        mesh->shrinkToFit();
        NormsIndexesTableType *normals = mesh->getTriNormsTable();
        if (normals) {
            normals->shrink_to_fit();
        }
    }

    baseVertices->setEnabled(false);
    // DGM: no need to lock it as it is only used by one mesh!
    baseVertices->setLocked(false);
    mesh->addChild(baseVertices);

    return mesh;
}

std::shared_ptr<ccMesh> ccMesh::CreateIcosahedron(
        double radius /* = 1.0*/, bool create_uv_map /* = false*/) {
    ccPointCloud *baseVertices = new ccPointCloud("vertices");
    assert(baseVertices);
    auto mesh = cloudViewer::make_shared<ccMesh>(baseVertices);
    if (radius <= 0) {
        utility::LogError("[CreateIcosahedron] radius <= 0");
    }
    const double p = (1. + std::sqrt(5.)) / 2.;

    // Vertices.
    baseVertices->addEigenPoint(radius * Eigen::Vector3d(-1, 0, p));
    baseVertices->addEigenPoint(radius * Eigen::Vector3d(1, 0, p));
    baseVertices->addEigenPoint(radius * Eigen::Vector3d(1, 0, -p));
    baseVertices->addEigenPoint(radius * Eigen::Vector3d(-1, 0, -p));
    baseVertices->addEigenPoint(radius * Eigen::Vector3d(0, -p, 1));
    baseVertices->addEigenPoint(radius * Eigen::Vector3d(0, p, 1));
    baseVertices->addEigenPoint(radius * Eigen::Vector3d(0, p, -1));
    baseVertices->addEigenPoint(radius * Eigen::Vector3d(0, -p, -1));
    baseVertices->addEigenPoint(radius * Eigen::Vector3d(-p, -1, 0));
    baseVertices->addEigenPoint(radius * Eigen::Vector3d(p, -1, 0));
    baseVertices->addEigenPoint(radius * Eigen::Vector3d(p, 1, 0));
    baseVertices->addEigenPoint(radius * Eigen::Vector3d(-p, 1, 0));

    // Triangles.
    mesh->addTriangle(Eigen::Vector3i(0, 4, 1));
    mesh->addTriangle(Eigen::Vector3i(0, 1, 5));
    mesh->addTriangle(Eigen::Vector3i(1, 4, 9));
    mesh->addTriangle(Eigen::Vector3i(1, 9, 10));
    mesh->addTriangle(Eigen::Vector3i(1, 10, 5));
    mesh->addTriangle(Eigen::Vector3i(0, 8, 4));
    mesh->addTriangle(Eigen::Vector3i(0, 11, 8));
    mesh->addTriangle(Eigen::Vector3i(0, 5, 11));
    mesh->addTriangle(Eigen::Vector3i(5, 6, 11));
    mesh->addTriangle(Eigen::Vector3i(5, 10, 6));
    mesh->addTriangle(Eigen::Vector3i(4, 8, 7));
    mesh->addTriangle(Eigen::Vector3i(4, 7, 9));
    mesh->addTriangle(Eigen::Vector3i(3, 6, 2));
    mesh->addTriangle(Eigen::Vector3i(3, 2, 7));
    mesh->addTriangle(Eigen::Vector3i(2, 6, 10));
    mesh->addTriangle(Eigen::Vector3i(2, 10, 9));
    mesh->addTriangle(Eigen::Vector3i(2, 9, 7));
    mesh->addTriangle(Eigen::Vector3i(3, 11, 6));
    mesh->addTriangle(Eigen::Vector3i(3, 8, 11));
    mesh->addTriangle(Eigen::Vector3i(3, 7, 8));

    // UV Map.
    if (create_uv_map) {
        mesh->triangle_uvs_ = {
                {0.0001, 0.1819}, {0.1575, 0.091},  {0.1575, 0.2728},
                {0.0001, 0.3637}, {0.1575, 0.2728}, {0.1575, 0.4546},
                {0.1575, 0.2728}, {0.1575, 0.091},  {0.3149, 0.1819},
                {0.1575, 0.2728}, {0.3149, 0.1819}, {0.3149, 0.3637},
                {0.1575, 0.2728}, {0.3149, 0.3637}, {0.1575, 0.4546},
                {0.0001, 0.909},  {0.1575, 0.8181}, {0.1575, 0.9999},
                {0.0001, 0.7272}, {0.1575, 0.6363}, {0.1575, 0.8181},
                {0.0001, 0.5454}, {0.1575, 0.4546}, {0.1575, 0.6363},
                {0.1575, 0.4546}, {0.3149, 0.5454}, {0.1575, 0.6363},
                {0.1575, 0.4546}, {0.3149, 0.3637}, {0.3149, 0.5454},
                {0.1575, 0.9999}, {0.1575, 0.8181}, {0.3149, 0.909},
                {0.1575, 0.091},  {0.3149, 0.0001}, {0.3149, 0.1819},
                {0.3149, 0.7272}, {0.3149, 0.5454}, {0.4724, 0.6363},
                {0.3149, 0.7272}, {0.4724, 0.8181}, {0.3149, 0.909},
                {0.4724, 0.4546}, {0.3149, 0.5454}, {0.3149, 0.3637},
                {0.4724, 0.2728}, {0.3149, 0.3637}, {0.3149, 0.1819},
                {0.4724, 0.091},  {0.3149, 0.1819}, {0.3149, 0.0001},
                {0.3149, 0.7272}, {0.1575, 0.6363}, {0.3149, 0.5454},
                {0.3149, 0.7272}, {0.1575, 0.8181}, {0.1575, 0.6363},
                {0.3149, 0.7272}, {0.3149, 0.909},  {0.1575, 0.8181}};
    }

    // do some cleaning
    {
        baseVertices->shrinkToFit();
        mesh->shrinkToFit();
        NormsIndexesTableType *normals = mesh->getTriNormsTable();
        if (normals) {
            normals->shrink_to_fit();
        }
    }

    baseVertices->setEnabled(false);
    // DGM: no need to lock it as it is only used by one mesh!
    baseVertices->setLocked(false);
    mesh->addChild(baseVertices);
    return mesh;
}

std::shared_ptr<ccMesh> ccMesh::CreatePlane(double width /* = 1.0*/,
                                            double height /* = 1.0*/,
                                            bool create_uv_map /* = false*/) {
    ccPointCloud *baseVertices = new ccPointCloud("vertices");
    assert(baseVertices);
    auto mesh = cloudViewer::make_shared<ccMesh>(baseVertices);
    if (width <= 0) {
        utility::LogError("[CreatePlane] width <= 0");
    }
    if (height <= 0) {
        utility::LogError("[CreatePlane] height <= 0");
    }

    // B ------ C
    // |        |
    // A ------ D
    if (!baseVertices->resize(4)) {
        utility::LogError("not enough memory!");
    }
    *baseVertices->getPointPtr(0) = Eigen::Vector3d(-width / 2, -height / 2, 0);
    *baseVertices->getPointPtr(1) = Eigen::Vector3d(-width / 2, height / 2, 0);
    *baseVertices->getPointPtr(2) = Eigen::Vector3d(width / 2, height / 2, 0);
    *baseVertices->getPointPtr(3) = Eigen::Vector3d(width / 2, -height / 2, 0);
    // Triangles.
    mesh->addTriangle(Eigen::Vector3i(0, 2, 1));  // A C B
    mesh->addTriangle(Eigen::Vector3i(0, 3, 2));  // A D C

    // do some cleaning
    {
        baseVertices->shrinkToFit();
        mesh->shrinkToFit();
        NormsIndexesTableType *normals = mesh->getTriNormsTable();
        if (normals) {
            normals->shrink_to_fit();
        }
    }

    baseVertices->setEnabled(false);
    // DGM: no need to lock it as it is only used by one mesh!
    baseVertices->setLocked(false);
    mesh->addChild(baseVertices);
    return mesh;
}

std::shared_ptr<ccMesh> ccMesh::CreateBox(
        double width /* = 1.0*/,
        double height /* = 1.0*/,
        double depth /* = 1.0*/,
        bool create_uv_map /* = false*/,
        bool map_texture_to_each_face /*= false*/) {
    ccPointCloud *baseVertices = new ccPointCloud("vertices");
    assert(baseVertices);
    auto mesh = cloudViewer::make_shared<ccMesh>(baseVertices);
    if (width <= 0) {
        utility::LogError("[CreateBox] width <= 0");
    }
    if (height <= 0) {
        utility::LogError("[CreateBox] height <= 0");
    }
    if (depth <= 0) {
        utility::LogError("[CreateBox] depth <= 0");
    }

    // Vertices.
    if (!baseVertices->resize(8)) {
        utility::LogError("not enough memory!");
    }
    *baseVertices->getPointPtr(0) = Eigen::Vector3d(0.0, 0.0, 0.0);
    *baseVertices->getPointPtr(1) = Eigen::Vector3d(width, 0.0, 0.0);
    *baseVertices->getPointPtr(2) = Eigen::Vector3d(0.0, 0.0, depth);
    *baseVertices->getPointPtr(3) = Eigen::Vector3d(width, 0.0, depth);
    *baseVertices->getPointPtr(4) = Eigen::Vector3d(0.0, height, 0.0);
    *baseVertices->getPointPtr(5) = Eigen::Vector3d(width, height, 0.0);
    *baseVertices->getPointPtr(6) = Eigen::Vector3d(0.0, height, depth);
    *baseVertices->getPointPtr(7) = Eigen::Vector3d(width, height, depth);

    // Triangles.
    mesh->addTriangle(Eigen::Vector3i(4, 7, 5));
    mesh->addTriangle(Eigen::Vector3i(4, 6, 7));
    mesh->addTriangle(Eigen::Vector3i(0, 2, 4));
    mesh->addTriangle(Eigen::Vector3i(2, 6, 4));
    mesh->addTriangle(Eigen::Vector3i(0, 1, 2));
    mesh->addTriangle(Eigen::Vector3i(1, 3, 2));
    mesh->addTriangle(Eigen::Vector3i(1, 5, 7));
    mesh->addTriangle(Eigen::Vector3i(1, 7, 3));
    mesh->addTriangle(Eigen::Vector3i(2, 3, 7));
    mesh->addTriangle(Eigen::Vector3i(2, 7, 6));
    mesh->addTriangle(Eigen::Vector3i(0, 4, 1));
    mesh->addTriangle(Eigen::Vector3i(1, 4, 5));

    // UV Map.
    if (create_uv_map) {
        if (map_texture_to_each_face) {
            mesh->triangle_uvs_ = {
                    {0.0, 0.0}, {1.0, 1.0}, {1.0, 0.0}, {0.0, 0.0}, {0.0, 1.0},
                    {1.0, 1.0}, {0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {0.0, 1.0},
                    {1.0, 1.0}, {1.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0},
                    {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}, {0.0, 0.0}, {1.0, 0.0},
                    {1.0, 1.0}, {0.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}, {0.0, 0.0},
                    {1.0, 0.0}, {1.0, 1.0}, {0.0, 0.0}, {1.0, 1.0}, {0.0, 1.0},
                    {0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 0.0}, {0.0, 1.0},
                    {1.0, 1.0}};
        } else {
            mesh->triangle_uvs_ = {
                    {0.5, 0.5},   {0.75, 0.75}, {0.5, 0.75},  {0.5, 0.5},
                    {0.75, 0.5},  {0.75, 0.75}, {0.25, 0.5},  {0.25, 0.25},
                    {0.5, 0.5},   {0.25, 0.25}, {0.5, 0.25},  {0.5, 0.5},
                    {0.25, 0.5},  {0.25, 0.75}, {0.0, 0.5},   {0.25, 0.75},
                    {0.0, 0.75},  {0.0, 0.5},   {0.25, 0.75}, {0.5, 0.75},
                    {0.5, 1.0},   {0.25, 0.75}, {0.5, 1.0},   {0.25, 1.0},
                    {0.25, 0.25}, {0.25, 0.0},  {0.5, 0.0},   {0.25, 0.25},
                    {0.5, 0.0},   {0.5, 0.25},  {0.25, 0.5},  {0.5, 0.5},
                    {0.25, 0.75}, {0.25, 0.75}, {0.5, 0.5},   {0.5, 0.75}};
        }
    }

    // do some cleaning
    {
        baseVertices->shrinkToFit();
        mesh->shrinkToFit();
        NormsIndexesTableType *normals = mesh->getTriNormsTable();
        if (normals) {
            normals->shrink_to_fit();
        }
    }

    baseVertices->setEnabled(false);
    // DGM: no need to lock it as it is only used by one mesh!
    baseVertices->setLocked(false);
    mesh->addChild(baseVertices);
    return mesh;
}

std::shared_ptr<ccMesh> ccMesh::CreateSphere(double radius /* = 1.0*/,
                                             int resolution /* = 20*/,
                                             bool create_uv_map /* = false*/) {
    ccPointCloud *baseVertices = new ccPointCloud("vertices");
    assert(baseVertices);
    auto mesh = cloudViewer::make_shared<ccMesh>(baseVertices);

    if (radius <= 0) {
        utility::LogError("[CreateSphere] radius <= 0");
    }
    if (resolution <= 0) {
        utility::LogError("[CreateSphere] resolution <= 0");
    }

    if (!baseVertices->resize(2 * resolution * (resolution - 1) + 2)) {
        utility::LogError("not enough memory!");
    }

    std::unordered_map<int64_t, std::pair<double, double>> map_vertices_to_uv;
    std::unordered_map<int64_t, std::pair<double, double>>
            map_cut_vertices_to_uv;

    *baseVertices->getPointPtr(0) = Eigen::Vector3d(0.0, 0.0, radius);
    *baseVertices->getPointPtr(1) = Eigen::Vector3d(0.0, 0.0, -radius);
    double step = M_PI / (double)resolution;
    for (int i = 1; i < resolution; i++) {
        double alpha = step * i;
        double uv_row = (1.0 / (resolution)) * i;
        int base = 2 + 2 * resolution * (i - 1);
        for (int j = 0; j < 2 * resolution; j++) {
            double theta = step * j;
            double uv_col = (1.0 / (2.0 * resolution)) * j;
            Eigen::Vector3d temp =
                    Eigen::Vector3d(sin(alpha) * cos(theta),
                                    sin(alpha) * sin(theta), cos(alpha)) *
                    radius;
            baseVertices->setEigenPoint(static_cast<std::size_t>(base + j),
                                        temp);
            if (create_uv_map) {
                map_vertices_to_uv[base + j] = std::make_pair(uv_row, uv_col);
            }
        }
        if (create_uv_map) {
            map_cut_vertices_to_uv[base] = std::make_pair(uv_row, 1.0);
        }
    }

    // Triangles for poles.
    for (int j = 0; j < 2 * resolution; j++) {
        int j1 = (j + 1) % (2 * resolution);
        int base = 2;
        mesh->addTriangle(Eigen::Vector3i(0, base + j, base + j1));
        base = 2 + 2 * resolution * (resolution - 2);
        mesh->addTriangle(Eigen::Vector3i(1, base + j1, base + j));
    }

    // UV coordinates mapped to triangles for poles.
    if (create_uv_map) {
        for (int j = 0; j < 2 * resolution - 1; j++) {
            int j1 = (j + 1) % (2 * resolution);
            int base = 2;
            double width = 1.0 / (2.0 * resolution);
            double base_offset = width / 2.0;
            double uv_col = base_offset + width * j;
            mesh->triangle_uvs_.emplace_back(0.0, uv_col);
            mesh->triangle_uvs_.emplace_back(
                    Eigen::Vector2d(map_vertices_to_uv[base + j].first,
                                    map_vertices_to_uv[base + j].second));
            mesh->triangle_uvs_.emplace_back(
                    Eigen::Vector2d(map_vertices_to_uv[base + j1].first,
                                    map_vertices_to_uv[base + j1].second));

            base = 2 + 2 * resolution * (resolution - 2);
            mesh->triangle_uvs_.emplace_back(Eigen::Vector2d(1.0, uv_col));
            mesh->triangle_uvs_.emplace_back(
                    Eigen::Vector2d(map_vertices_to_uv[base + j1].first,
                                    map_vertices_to_uv[base + j1].second));
            mesh->triangle_uvs_.emplace_back(
                    Eigen::Vector2d(map_vertices_to_uv[base + j].first,
                                    map_vertices_to_uv[base + j].second));
        }

        // UV coordinates mapped to triangles for poles, with cut-vertices.
        int j = 2 * resolution - 1;
        int base = 2;
        double width = 1.0 / (2.0 * resolution);
        double base_offset = width / 2.0;
        double uv_col = base_offset + width * j;
        mesh->triangle_uvs_.emplace_back(Eigen::Vector2d(0.0, uv_col));
        mesh->triangle_uvs_.emplace_back(
                Eigen::Vector2d(map_vertices_to_uv[base + j].first,
                                map_vertices_to_uv[base + j].second));
        mesh->triangle_uvs_.emplace_back(
                Eigen::Vector2d(map_cut_vertices_to_uv[base].first,
                                map_cut_vertices_to_uv[base].second));

        base = 2 + 2 * resolution * (resolution - 2);
        mesh->triangle_uvs_.emplace_back(Eigen::Vector2d(1.0, uv_col));
        mesh->triangle_uvs_.emplace_back(
                Eigen::Vector2d(map_cut_vertices_to_uv[base].first,
                                map_cut_vertices_to_uv[base].second));
        mesh->triangle_uvs_.emplace_back(
                Eigen::Vector2d(map_vertices_to_uv[base + j].first,
                                map_vertices_to_uv[base + j].second));
    }

    // Triangles for non-polar region.
    for (int i = 1; i < resolution - 1; i++) {
        int base1 = 2 + 2 * resolution * (i - 1);
        int base2 = base1 + 2 * resolution;
        for (int j = 0; j < 2 * resolution; j++) {
            int j1 = (j + 1) % (2 * resolution);
            mesh->addTriangle(
                    Eigen::Vector3i(base2 + j, base1 + j1, base1 + j));
            mesh->addTriangle(
                    Eigen::Vector3i(base2 + j, base2 + j1, base1 + j1));
        }
    }

    // UV coordinates mapped to triangles for non-polar region.
    if (create_uv_map) {
        for (int i = 1; i < resolution - 1; i++) {
            int base1 = 2 + 2 * resolution * (i - 1);
            int base2 = base1 + 2 * resolution;
            for (int j = 0; j < 2 * resolution - 1; j++) {
                int j1 = (j + 1) % (2 * resolution);
                mesh->triangle_uvs_.emplace_back(
                        map_vertices_to_uv[base2 + j].first,
                        map_vertices_to_uv[base2 + j].second);
                mesh->triangle_uvs_.emplace_back(
                        Eigen::Vector2d(map_vertices_to_uv[base1 + j1].first,
                                        map_vertices_to_uv[base1 + j1].second));
                mesh->triangle_uvs_.emplace_back(
                        Eigen::Vector2d(map_vertices_to_uv[base1 + j].first,
                                        map_vertices_to_uv[base1 + j].second));

                mesh->triangle_uvs_.emplace_back(
                        Eigen::Vector2d(map_vertices_to_uv[base2 + j].first,
                                        map_vertices_to_uv[base2 + j].second));
                mesh->triangle_uvs_.emplace_back(
                        Eigen::Vector2d(map_vertices_to_uv[base2 + j1].first,
                                        map_vertices_to_uv[base2 + j1].second));
                mesh->triangle_uvs_.emplace_back(
                        Eigen::Vector2d(map_vertices_to_uv[base1 + j1].first,
                                        map_vertices_to_uv[base1 + j1].second));
            }

            // UV coordinates mapped to triangles for non-polar region with
            // cut-vertices.
            mesh->triangle_uvs_.emplace_back(Eigen::Vector2d(
                    map_vertices_to_uv[base2 + 2 * resolution - 1].first,
                    map_vertices_to_uv[base2 + 2 * resolution - 1].second));
            mesh->triangle_uvs_.emplace_back(
                    Eigen::Vector2d(map_cut_vertices_to_uv[base1].first,
                                    map_cut_vertices_to_uv[base1].second));
            mesh->triangle_uvs_.emplace_back(Eigen::Vector2d(
                    map_vertices_to_uv[base1 + 2 * resolution - 1].first,
                    map_vertices_to_uv[base1 + 2 * resolution - 1].second));

            mesh->triangle_uvs_.emplace_back(Eigen::Vector2d(
                    map_vertices_to_uv[base2 + 2 * resolution - 1].first,
                    map_vertices_to_uv[base2 + 2 * resolution - 1].second));
            mesh->triangle_uvs_.emplace_back(
                    Eigen::Vector2d(map_cut_vertices_to_uv[base2].first,
                                    map_cut_vertices_to_uv[base2].second));
            mesh->triangle_uvs_.emplace_back(
                    Eigen::Vector2d(map_cut_vertices_to_uv[base1].first,
                                    map_cut_vertices_to_uv[base1].second));
        }
    }

    // do some cleaning
    {
        baseVertices->shrinkToFit();
        mesh->shrinkToFit();
        NormsIndexesTableType *normals = mesh->getTriNormsTable();
        if (normals) {
            normals->shrink_to_fit();
        }
    }

    baseVertices->setEnabled(false);
    // DGM: no need to lock it as it is only used by one mesh!
    baseVertices->setLocked(false);
    mesh->addChild(baseVertices);
    return mesh;
}

std::shared_ptr<ccMesh> ccMesh::CreateCylinder(
        double radius /* = 1.0*/,
        double height /* = 2.0*/,
        int resolution /* = 20*/,
        int split /* = 4*/,
        bool create_uv_map /* = false*/) {
    ccPointCloud *baseVertices = new ccPointCloud("vertices");
    assert(baseVertices);
    auto mesh = cloudViewer::make_shared<ccMesh>(baseVertices);
    if (radius <= 0) {
        utility::LogError("[CreateCylinder] radius <= 0");
    }
    if (height <= 0) {
        utility::LogError("[CreateCylinder] height <= 0");
    }
    if (resolution <= 0) {
        utility::LogError("[CreateCylinder] resolution <= 0");
    }
    if (split <= 0) {
        utility::LogError("[CreateCylinder] split <= 0");
    }

    if (!baseVertices->resize(resolution * (split + 1) + 2)) {
        utility::LogError("not enough memory!");
    }
    *baseVertices->getPointPtr(0) = Eigen::Vector3d(0.0, 0.0, height * 0.5);
    *baseVertices->getPointPtr(1) = Eigen::Vector3d(0.0, 0.0, -height * 0.5);
    double step = M_PI * 2.0 / (double)resolution;
    double h_step = height / (double)split;
    for (int i = 0; i <= split; i++) {
        for (int j = 0; j < resolution; j++) {
            double theta = step * j;
            baseVertices->setEigenPoint(
                    static_cast<std::size_t>(2 + resolution * i + j),
                    Eigen::Vector3d(cos(theta) * radius, sin(theta) * radius,
                                    height * 0.5 - h_step * i));
        }
    }

    std::unordered_map<int64_t, std::pair<double, double>> map_vertices_to_uv;
    std::unordered_map<int64_t, std::pair<double, double>>
            map_cut_vertices_to_uv;

    // Mapping vertices to UV coordinates.
    if (create_uv_map) {
        for (int i = 0; i <= split; i++) {
            double uv_row = (1.0 / (double)split) * i;
            for (int j = 0; j < resolution; j++) {
                // double theta = step * j;
                double uv_col = (1.0 / (double)resolution) * j;
                map_vertices_to_uv[2 + resolution * i + j] =
                        std::make_pair(uv_row, uv_col);
            }
            map_cut_vertices_to_uv[2 + resolution * i] =
                    std::make_pair(uv_row, 1.0);
        }
    }

    // Triangles for top and bottom face.
    for (int j = 0; j < resolution; j++) {
        int j1 = (j + 1) % resolution;
        int base = 2;
        mesh->addTriangle(Eigen::Vector3i(0, base + j, base + j1));
        base = 2 + resolution * split;
        mesh->addTriangle(Eigen::Vector3i(1, base + j1, base + j));
    }

    // UV coordinates mapped to triangles for top and bottom face.
    if (create_uv_map) {
        for (int j = 0; j < resolution; j++) {
            int j1 = (j + 1) % resolution;
            double theta = step * j;
            double theta1 = step * j1;
            double uv_radius = 0.25;

            mesh->triangle_uvs_.push_back(Eigen::Vector2d(0.75, 0.25));
            mesh->triangle_uvs_.push_back(
                    Eigen::Vector2d(0.75 + uv_radius * cos(theta),
                                    0.25 + uv_radius * sin(theta)));
            mesh->triangle_uvs_.push_back(
                    Eigen::Vector2d(0.75 + uv_radius * cos(theta1),
                                    0.25 + uv_radius * sin(theta1)));

            mesh->triangle_uvs_.push_back(Eigen::Vector2d(0.75, 0.75));
            mesh->triangle_uvs_.push_back(
                    Eigen::Vector2d(0.75 + uv_radius * cos(theta1),
                                    0.75 + uv_radius * sin(theta1)));
            mesh->triangle_uvs_.push_back(
                    Eigen::Vector2d(0.75 + uv_radius * cos(theta),
                                    0.75 + uv_radius * sin(theta)));
        }
    }

    // Triangles for cylindrical surface.
    for (int i = 0; i < split; i++) {
        int base1 = 2 + resolution * i;
        int base2 = base1 + resolution;
        for (int j = 0; j < resolution; j++) {
            int j1 = (j + 1) % resolution;
            mesh->addTriangle(
                    Eigen::Vector3i(base2 + j, base1 + j1, base1 + j));
            mesh->addTriangle(
                    Eigen::Vector3i(base2 + j, base2 + j1, base1 + j1));
        }
    }

    // UV coordinates mapped to triangles for cylindrical surface.
    if (create_uv_map) {
        for (int i = 0; i < split; i++) {
            int base1 = 2 + resolution * i;
            int base2 = base1 + resolution;
            for (int j = 0; j < resolution - 1; j++) {
                int j1 = (j + 1) % resolution;

                mesh->triangle_uvs_.emplace_back(
                        map_vertices_to_uv[base2 + j].first,
                        map_vertices_to_uv[base2 + j].second);
                mesh->triangle_uvs_.emplace_back(
                        Eigen::Vector2d(map_vertices_to_uv[base1 + j1].first,
                                        map_vertices_to_uv[base1 + j1].second));
                mesh->triangle_uvs_.emplace_back(
                        Eigen::Vector2d(map_vertices_to_uv[base1 + j].first,
                                        map_vertices_to_uv[base1 + j].second));

                mesh->triangle_uvs_.emplace_back(
                        Eigen::Vector2d(map_vertices_to_uv[base2 + j].first,
                                        map_vertices_to_uv[base2 + j].second));
                mesh->triangle_uvs_.emplace_back(
                        Eigen::Vector2d(map_vertices_to_uv[base2 + j1].first,
                                        map_vertices_to_uv[base2 + j1].second));
                mesh->triangle_uvs_.emplace_back(
                        Eigen::Vector2d(map_vertices_to_uv[base1 + j1].first,
                                        map_vertices_to_uv[base1 + j1].second));
            }

            // UV coordinates mapped to triangles for cylindrical surface with
            // cut-vertices.
            mesh->triangle_uvs_.emplace_back(Eigen::Vector2d(
                    map_vertices_to_uv[base2 + resolution - 1].first,
                    map_vertices_to_uv[base2 + resolution - 1].second));
            mesh->triangle_uvs_.emplace_back(
                    Eigen::Vector2d(map_cut_vertices_to_uv[base1].first,
                                    map_cut_vertices_to_uv[base1].second));
            mesh->triangle_uvs_.emplace_back(Eigen::Vector2d(
                    map_vertices_to_uv[base1 + resolution - 1].first,
                    map_vertices_to_uv[base1 + resolution - 1].second));

            mesh->triangle_uvs_.emplace_back(Eigen::Vector2d(
                    map_vertices_to_uv[base2 + resolution - 1].first,
                    map_vertices_to_uv[base2 + resolution - 1].second));
            mesh->triangle_uvs_.emplace_back(
                    Eigen::Vector2d(map_cut_vertices_to_uv[base2].first,
                                    map_cut_vertices_to_uv[base2].second));
            mesh->triangle_uvs_.emplace_back(
                    Eigen::Vector2d(map_cut_vertices_to_uv[base1].first,
                                    map_cut_vertices_to_uv[base1].second));
        }
    }

    // do some cleaning
    {
        baseVertices->shrinkToFit();
        mesh->shrinkToFit();
        NormsIndexesTableType *normals = mesh->getTriNormsTable();
        if (normals) {
            normals->shrink_to_fit();
        }
    }

    baseVertices->setEnabled(false);
    // DGM: no need to lock it as it is only used by one mesh!
    baseVertices->setLocked(false);
    mesh->addChild(baseVertices);
    return mesh;
}

std::shared_ptr<ccMesh> ccMesh::CreateCone(double radius /* = 1.0*/,
                                           double height /* = 2.0*/,
                                           int resolution /* = 20*/,
                                           int split /* = 4*/,
                                           bool create_uv_map /* = false*/) {
    ccPointCloud *baseVertices = new ccPointCloud("vertices");
    assert(baseVertices);
    auto mesh = cloudViewer::make_shared<ccMesh>(baseVertices);

    if (radius <= 0) {
        utility::LogError("[CreateCone] radius <= 0");
    }
    if (height <= 0) {
        utility::LogError("[CreateCone] height <= 0");
    }
    if (resolution <= 0) {
        utility::LogError("[CreateCone] resolution <= 0");
    }
    if (split <= 0) {
        utility::LogError("[CreateCone] split <= 0");
    }

    if (!baseVertices->resize(resolution * split + 2)) {
        utility::LogError("not enough memory!");
    }
    *baseVertices->getPointPtr(0) = Eigen::Vector3d(0.0, 0.0, 0.0);
    *baseVertices->getPointPtr(1) = Eigen::Vector3d(0.0, 0.0, height);
    double step = M_PI * 2.0 / (double)resolution;
    double h_step = height / (double)split;
    double r_step = radius / (double)split;
    std::unordered_map<int64_t, std::pair<double, double>> map_vertices_to_uv;
    for (int i = 0; i < split; i++) {
        int base = 2 + resolution * i;
        double r = r_step * (split - i);
        for (int j = 0; j < resolution; j++) {
            double theta = step * j;
            baseVertices->setEigenPoint(
                    static_cast<std::size_t>(base + j),
                    Eigen::Vector3d(cos(theta) * r, sin(theta) * r,
                                    h_step * i));

            // Mapping vertices to UV coordinates.
            if (create_uv_map) {
                double factor = 0.25 * r / radius;
                map_vertices_to_uv[base + j] = std::make_pair(
                        factor * cos(theta), factor * sin(theta));
            }
        }
    }

    for (int j = 0; j < resolution; j++) {
        int j1 = (j + 1) % resolution;
        int base = 2;
        mesh->addTriangle(Eigen::Vector3i(0, base + j1, base + j));
        base = 2 + resolution * (split - 1);
        mesh->addTriangle(Eigen::Vector3i(1, base + j, base + j1));
    }

    if (create_uv_map) {
        for (int j = 0; j < resolution; j++) {
            int j1 = (j + 1) % resolution;
            // UV coordinates mapped to triangles for bottom surface.
            int base = 2;
            mesh->triangle_uvs_.push_back(Eigen::Vector2d(0.5, 0.25));
            mesh->triangle_uvs_.push_back(Eigen::Vector2d(
                    0.5 + map_vertices_to_uv[base + j1].first,
                    0.25 + map_vertices_to_uv[base + j1].second));
            mesh->triangle_uvs_.push_back(Eigen::Vector2d(
                    0.5 + map_vertices_to_uv[base + j].first,
                    0.25 + map_vertices_to_uv[base + j].second));

            // UV coordinates mapped to triangles for top segment of conical
            // surface.
            base = 2 + resolution * (split - 1);
            mesh->triangle_uvs_.push_back(Eigen::Vector2d(0.5, 0.75));
            mesh->triangle_uvs_.push_back(Eigen::Vector2d(
                    0.5 + map_vertices_to_uv[base + j].first,
                    0.75 + map_vertices_to_uv[base + j].second));
            mesh->triangle_uvs_.push_back(Eigen::Vector2d(
                    0.5 + map_vertices_to_uv[base + j1].first,
                    0.75 + map_vertices_to_uv[base + j1].second));
        }
    }

    // Triangles for conical surface other than top-segment.
    for (int i = 0; i < split - 1; i++) {
        int base1 = 2 + resolution * i;
        int base2 = base1 + resolution;
        for (int j = 0; j < resolution; j++) {
            int j1 = (j + 1) % resolution;
            mesh->addTriangle(
                    Eigen::Vector3i(base2 + j1, base1 + j, base1 + j1));
            mesh->addTriangle(
                    Eigen::Vector3i(base2 + j1, base2 + j, base1 + j));
        }
    }

    // UV coordinates mapped to triangles for conical surface other than
    // top-segment.
    if (create_uv_map) {
        for (int i = 0; i < split - 1; i++) {
            int base1 = 2 + resolution * i;
            int base2 = base1 + resolution;
            for (int j = 0; j < resolution; j++) {
                int j1 = (j + 1) % resolution;
                mesh->triangle_uvs_.emplace_back(
                        0.5 + map_vertices_to_uv[base2 + j1].first,
                        0.75 + map_vertices_to_uv[base2 + j1].second);
                mesh->triangle_uvs_.emplace_back(Eigen::Vector2d(
                        0.5 + map_vertices_to_uv[base1 + j].first,
                        0.75 + map_vertices_to_uv[base1 + j].second));
                mesh->triangle_uvs_.emplace_back(Eigen::Vector2d(
                        0.5 + map_vertices_to_uv[base1 + j1].first,
                        0.75 + map_vertices_to_uv[base1 + j1].second));

                mesh->triangle_uvs_.emplace_back(Eigen::Vector2d(
                        0.5 + map_vertices_to_uv[base2 + j1].first,
                        0.75 + map_vertices_to_uv[base2 + j1].second));
                mesh->triangle_uvs_.emplace_back(Eigen::Vector2d(
                        0.5 + map_vertices_to_uv[base2 + j].first,
                        0.75 + map_vertices_to_uv[base2 + j].second));
                mesh->triangle_uvs_.emplace_back(Eigen::Vector2d(
                        0.5 + map_vertices_to_uv[base1 + j].first,
                        0.75 + map_vertices_to_uv[base1 + j].second));
            }
        }
    }

    // do some cleaning
    {
        baseVertices->shrinkToFit();
        mesh->shrinkToFit();
        NormsIndexesTableType *normals = mesh->getTriNormsTable();
        if (normals) {
            normals->shrink_to_fit();
        }
    }

    baseVertices->setEnabled(false);
    // DGM: no need to lock it as it is only used by one mesh!
    baseVertices->setLocked(false);
    mesh->addChild(baseVertices);
    return mesh;
}

std::shared_ptr<ccMesh> ccMesh::CreateTorus(double torus_radius /* = 1.0 */,
                                            double tube_radius /* = 0.5 */,
                                            int radial_resolution /* = 20 */,
                                            int tubular_resolution /* = 20 */) {
    ccPointCloud *baseVertices = new ccPointCloud("vertices");
    assert(baseVertices);
    auto mesh = cloudViewer::make_shared<ccMesh>(baseVertices);

    if (torus_radius <= 0) {
        utility::LogError("[CreateTorus] torus_radius <= 0");
    }
    if (tube_radius <= 0) {
        utility::LogError("[CreateTorus] tube_radius <= 0");
    }
    if (radial_resolution <= 0) {
        utility::LogError("[CreateTorus] radial_resolution <= 0");
    }
    if (tubular_resolution <= 0) {
        utility::LogError("[CreateTorus] tubular_resolution <= 0");
    }

    if (!baseVertices->resize(radial_resolution * tubular_resolution) ||
        !mesh->resize(2 * radial_resolution * tubular_resolution)) {
        utility::LogError("not enough memory!");
    }

    auto vert_idx = [&](int uidx, int vidx) {
        return uidx * tubular_resolution + vidx;
    };
    double u_step = 2 * M_PI / double(radial_resolution);
    double v_step = 2 * M_PI / double(tubular_resolution);
    Eigen::Vector3d temp;
    for (int uidx = 0; uidx < radial_resolution; ++uidx) {
        double u = uidx * u_step;
        Eigen::Vector3d w(cos(u), sin(u), 0);
        for (int vidx = 0; vidx < tubular_resolution; ++vidx) {
            double v = vidx * v_step;
            temp = torus_radius * w + tube_radius * cos(v) * w +
                   Eigen::Vector3d(0, 0, tube_radius * sin(v));
            *baseVertices->getPointPtr(
                    static_cast<unsigned int>(vert_idx(uidx, vidx))) = temp;

            int tri_idx = (uidx * tubular_resolution + vidx) * 2;
            mesh->setTriangle(
                    static_cast<unsigned int>(tri_idx + 0),
                    Eigen::Vector3i(
                            vert_idx((uidx + 1) % radial_resolution, vidx),
                            vert_idx((uidx + 1) % radial_resolution,
                                     (vidx + 1) % tubular_resolution),
                            vert_idx(uidx, vidx)));
            mesh->setTriangle(
                    static_cast<unsigned int>(tri_idx + 1),
                    Eigen::Vector3i(
                            vert_idx(uidx, vidx),
                            vert_idx((uidx + 1) % radial_resolution,
                                     (vidx + 1) % tubular_resolution),
                            vert_idx(uidx, (vidx + 1) % tubular_resolution)));
        }
    }

    // do some cleaning
    {
        baseVertices->shrinkToFit();
        mesh->shrinkToFit();
        NormsIndexesTableType *normals = mesh->getTriNormsTable();
        if (normals) {
            normals->shrink_to_fit();
        }
    }

    baseVertices->setEnabled(false);
    // DGM: no need to lock it as it is only used by one mesh!
    baseVertices->setLocked(false);
    mesh->addChild(baseVertices);
    return mesh;
}

std::shared_ptr<ccMesh> ccMesh::CreateArrow(double cylinder_radius /* = 1.0*/,
                                            double cone_radius /* = 1.5*/,
                                            double cylinder_height /* = 5.0*/,
                                            double cone_height /* = 4.0*/,
                                            int resolution /* = 20*/,
                                            int cylinder_split /* = 4*/,
                                            int cone_split /* = 1*/) {
    if (cylinder_radius <= 0) {
        utility::LogError("[CreateArrow] cylinder_radius <= 0");
    }
    if (cone_radius <= 0) {
        utility::LogError("[CreateArrow] cone_radius <= 0");
    }
    if (cylinder_height <= 0) {
        utility::LogError("[CreateArrow] cylinder_height <= 0");
    }
    if (cone_height <= 0) {
        utility::LogError("[CreateArrow] cone_height <= 0");
    }
    if (resolution <= 0) {
        utility::LogError("[CreateArrow] resolution <= 0");
    }
    if (cylinder_split <= 0) {
        utility::LogError("[CreateArrow] cylinder_split <= 0");
    }
    if (cone_split <= 0) {
        utility::LogError("[CreateArrow] cone_split <= 0");
    }
    Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();
    auto mesh_cylinder = CreateCylinder(cylinder_radius, cylinder_height,
                                        resolution, cylinder_split);
    transformation(2, 3) = cylinder_height * 0.5;
    mesh_cylinder->transform(transformation);
    auto mesh_cone =
            CreateCone(cone_radius, cone_height, resolution, cone_split);
    transformation(2, 3) = cylinder_height;
    mesh_cone->transform(transformation);
    auto mesh_arrow = mesh_cylinder;
    *mesh_arrow += *mesh_cone;
    return mesh_arrow;
}

std::shared_ptr<ccMesh> ccMesh::CreateCoordinateFrame(
        double size /* = 1.0*/,
        const Eigen::Vector3d &origin /* = Eigen::Vector3d(0.0, 0.0, 0.0)*/) {
    if (size <= 0) {
        utility::LogError("[CreateCoordinateFrame] size <= 0");
    }
    auto mesh_frame = CreateSphere(0.06 * size);
    mesh_frame->computeVertexNormals();
    mesh_frame->paintUniformColor(Eigen::Vector3d(0.5, 0.5, 0.5));

    std::shared_ptr<ccMesh> mesh_arrow;
    Eigen::Matrix4d transformation;

    mesh_arrow = CreateArrow(0.035 * size, 0.06 * size, 0.8 * size, 0.2 * size);
    mesh_arrow->computeVertexNormals();
    mesh_arrow->paintUniformColor(Eigen::Vector3d(1.0, 0.0, 0.0));
    mesh_arrow->showColors(true);
    transformation << 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1;
    mesh_arrow->transform(transformation);
    *mesh_frame += *mesh_arrow;

    mesh_arrow = CreateArrow(0.035 * size, 0.06 * size, 0.8 * size, 0.2 * size);
    mesh_arrow->computeVertexNormals();
    mesh_arrow->paintUniformColor(Eigen::Vector3d(0.0, 1.0, 0.0));
    transformation << 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1;
    mesh_arrow->transform(transformation);
    *mesh_frame += *mesh_arrow;

    mesh_arrow = CreateArrow(0.035 * size, 0.06 * size, 0.8 * size, 0.2 * size);
    mesh_arrow->computeVertexNormals();
    mesh_arrow->paintUniformColor(Eigen::Vector3d(0.0, 0.0, 1.0));
    transformation << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;
    mesh_arrow->transform(transformation);
    *mesh_frame += *mesh_arrow;

    transformation = Eigen::Matrix4d::Identity();
    transformation.block<3, 1>(0, 3) = origin;
    mesh_frame->transform(transformation);

    return mesh_frame;
}

std::shared_ptr<ccMesh> ccMesh::CreateMoebius(int length_split /* = 70 */,
                                              int width_split /* = 15 */,
                                              int twists /* = 1 */,
                                              double radius /* = 1 */,
                                              double flatness /* = 1 */,
                                              double width /* = 1 */,
                                              double scale /* = 1 */) {
    ccPointCloud *baseVertices = new ccPointCloud("vertices");
    assert(baseVertices);
    auto mesh = cloudViewer::make_shared<ccMesh>(baseVertices);

    if (length_split <= 0) {
        utility::LogError("[CreateMoebius] length_split <= 0");
    }
    if (width_split <= 0) {
        utility::LogError("[CreateMoebius] width_split <= 0");
    }
    if (twists < 0) {
        utility::LogError("[CreateMoebius] twists < 0");
    }
    if (radius <= 0) {
        utility::LogError("[CreateMoebius] radius <= 0");
    }
    if (flatness == 0) {
        utility::LogError("[CreateMoebius] flatness == 0");
    }
    if (width <= 0) {
        utility::LogError("[CreateMoebius] width <= 0");
    }
    if (scale <= 0) {
        utility::LogError("[CreateMoebius] scale <= 0");
    }

    if (!baseVertices->resize(length_split * width_split)) {
        utility::LogError("not enough memory!");
    }

    double u_step = 2 * M_PI / length_split;
    double v_step = width / (width_split - 1);
    for (int uidx = 0; uidx < length_split; ++uidx) {
        double u = uidx * u_step;
        double cos_u = std::cos(u);
        double sin_u = std::sin(u);
        for (int vidx = 0; vidx < width_split; ++vidx) {
            unsigned int idx =
                    static_cast<unsigned int>(uidx * width_split + vidx);
            double v = -width / 2.0 + vidx * v_step;
            double alpha = twists * 0.5 * u;
            double cos_alpha = std::cos(alpha);
            double sin_alpha = std::sin(alpha);
            baseVertices->getPointPtr(idx)->x =
                    static_cast<PointCoordinateType>(
                            scale * ((cos_alpha * cos_u * v) + radius * cos_u));
            baseVertices->getPointPtr(idx)->y =
                    static_cast<PointCoordinateType>(
                            scale * ((cos_alpha * sin_u * v) + radius * sin_u));
            baseVertices->getPointPtr(idx)->z =
                    static_cast<PointCoordinateType>(scale * sin_alpha * v *
                                                     flatness);
        }
    }

    for (int uidx = 0; uidx < length_split - 1; ++uidx) {
        for (int vidx = 0; vidx < width_split - 1; ++vidx) {
            if ((uidx + vidx) % 2 == 0) {
                mesh->addTriangle(
                        Eigen::Vector3i(uidx * width_split + vidx,
                                        (uidx + 1) * width_split + vidx + 1,
                                        uidx * width_split + vidx + 1));
                mesh->addTriangle(
                        Eigen::Vector3i(uidx * width_split + vidx,
                                        (uidx + 1) * width_split + vidx,
                                        (uidx + 1) * width_split + vidx + 1));
            } else {
                mesh->addTriangle(
                        Eigen::Vector3i(uidx * width_split + vidx + 1,
                                        uidx * width_split + vidx,
                                        (uidx + 1) * width_split + vidx));
                mesh->addTriangle(
                        Eigen::Vector3i(uidx * width_split + vidx + 1,
                                        (uidx + 1) * width_split + vidx,
                                        (uidx + 1) * width_split + vidx + 1));
            }
        }
    }

    int uidx = length_split - 1;
    for (int vidx = 0; vidx < width_split - 1; ++vidx) {
        if (twists % 2 == 1) {
            if ((uidx + vidx) % 2 == 0) {
                mesh->addTriangle(
                        Eigen::Vector3i((width_split - 1) - (vidx + 1),
                                        uidx * width_split + vidx,
                                        uidx * width_split + vidx + 1));
                mesh->addTriangle(Eigen::Vector3i(
                        (width_split - 1) - vidx, uidx * width_split + vidx,
                        (width_split - 1) - (vidx + 1)));
            } else {
                mesh->addTriangle(Eigen::Vector3i(uidx * width_split + vidx,
                                                  uidx * width_split + vidx + 1,
                                                  (width_split - 1) - vidx));
                mesh->addTriangle(Eigen::Vector3i(
                        (width_split - 1) - vidx, uidx * width_split + vidx + 1,
                        (width_split - 1) - (vidx + 1)));
            }
        } else {
            if ((uidx + vidx) % 2 == 0) {
                mesh->addTriangle(
                        Eigen::Vector3i(uidx * width_split + vidx, vidx + 1,
                                        uidx * width_split + vidx + 1));
                mesh->addTriangle(Eigen::Vector3i(uidx * width_split + vidx,
                                                  vidx, vidx + 1));
            } else {
                mesh->addTriangle(
                        Eigen::Vector3i(uidx * width_split + vidx, vidx,
                                        uidx * width_split + vidx + 1));
                mesh->addTriangle(Eigen::Vector3i(uidx * width_split + vidx + 1,
                                                  vidx, vidx + 1));
            }
        }
    }

    // do some cleaning
    {
        baseVertices->shrinkToFit();
        mesh->shrinkToFit();
        NormsIndexesTableType *normals = mesh->getTriNormsTable();
        if (normals) {
            normals->shrink_to_fit();
        }
    }

    baseVertices->setEnabled(false);
    // DGM: no need to lock it as it is only used by one mesh!
    baseVertices->setLocked(false);
    mesh->addChild(baseVertices);

    return mesh;
}
