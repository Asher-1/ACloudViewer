// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// CV_CORE_LIB
#include <CVConst.h>
#include <CVLog.h>

// LOCAL
#include "ecvHObjectCaster.h"
#include "ecvMesh.h"
#include "ecvPointCloud.h"
#include "ecvQhull.h"
#include "ecvTetraMesh.h"

// QHULL_LIB
#include "libqhullcpp/PointCoordinates.h"
#include "libqhullcpp/Qhull.h"
#include "libqhullcpp/QhullFacet.h"
#include "libqhullcpp/QhullFacetList.h"
#include "libqhullcpp/QhullVertexSet.h"

// SYSTEM
#include <unordered_map>
#include <unordered_set>

namespace cloudViewer {
namespace geometry {

std::tuple<std::shared_ptr<ccMesh>, std::vector<size_t>>
Qhull::ComputeConvexHull(const std::vector<Eigen::Vector3d>& points) {
    return ComputeConvexHull(CCVector3::fromArrayContainer(points));
}

std::tuple<std::shared_ptr<ccMesh>, std::vector<size_t>>
Qhull::ComputeConvexHull(const std::vector<CCVector3>& points) {
    ccPointCloud* baseVertices = new ccPointCloud("vertices");
    assert(baseVertices);
    baseVertices->setEnabled(false);
    // DGM: no need to lock it as it is only used by one mesh!
    baseVertices->setLocked(false);
    auto convex_hull = std::make_shared<ccMesh>(baseVertices);
    convex_hull->addChild(baseVertices);
    convex_hull->setName("ConvexMesh");
    std::vector<size_t> pt_map;

    std::vector<double> qhull_points_data(points.size() * 3);
    for (size_t pidx = 0; pidx < points.size(); ++pidx) {
        const auto& pt = points[pidx];
        qhull_points_data[pidx * 3 + 0] = pt[0];
        qhull_points_data[pidx * 3 + 1] = pt[1];
        qhull_points_data[pidx * 3 + 2] = pt[2];
    }

    orgQhull::PointCoordinates qhull_points(3, "");
    qhull_points.append(qhull_points_data);

    orgQhull::Qhull qhull;
    qhull.runQhull(qhull_points.comment().c_str(), qhull_points.dimension(),
                   qhull_points.count(), qhull_points.coordinates(), "Qt");

    orgQhull::QhullFacetList facets = qhull.facetList();

    if (!convex_hull->resize(facets.count())) {
        assert(false);
    }
    std::unordered_map<unsigned int, unsigned int> vert_map;
    std::unordered_set<unsigned int> inserted_vertices;
    int tidx = 0;
    for (orgQhull::QhullFacetList::iterator it = facets.begin();
         it != facets.end(); ++it) {
        if (!(*it).isGood()) continue;

        orgQhull::QhullFacet f = *it;
        orgQhull::QhullVertexSet vSet = f.vertices();
        int triangle_subscript = 0;
        for (orgQhull::QhullVertexSet::iterator vIt = vSet.begin();
             vIt != vSet.end(); ++vIt) {
            orgQhull::QhullVertex v = *vIt;
            orgQhull::QhullPoint p = v.point();

            unsigned int vidx = static_cast<unsigned int>(p.id());
            convex_hull->getTriangleVertIndexes(tidx)->i[triangle_subscript] =
                    vidx;
            triangle_subscript++;

            if (inserted_vertices.count(vidx) == 0) {
                inserted_vertices.insert(vidx);
                vert_map[vidx] = baseVertices->size();
                double* coords = p.coordinates();
                baseVertices->addPoint(
                        CCVector3(static_cast<PointCoordinateType>(coords[0]),
                                  static_cast<PointCoordinateType>(coords[1]),
                                  static_cast<PointCoordinateType>(coords[2])));
                pt_map.push_back(vidx);
            }
        }

        tidx++;
    }

    for (unsigned int i = 0; i < convex_hull->size(); ++i) {
        cloudViewer::VerticesIndexes* tsi =
                convex_hull->getTriangleVertIndexes(i);
        tsi->i1 = vert_map[tsi->i1];
        tsi->i2 = vert_map[tsi->i2];
        tsi->i3 = vert_map[tsi->i3];
    }

    // do some cleaning
    {
        baseVertices->shrinkToFit();
        convex_hull->shrinkToFit();
        NormsIndexesTableType* normals = convex_hull->getTriNormsTable();
        if (normals) {
            normals->shrink_to_fit();
        }
    }

    return std::make_tuple(convex_hull, pt_map);
}

std::tuple<std::shared_ptr<TetraMesh>, std::vector<size_t>>
Qhull::ComputeDelaunayTetrahedralization(const std::vector<CCVector3>& points) {
    return ComputeDelaunayTetrahedralization(
            CCVector3::fromArrayContainer(points));
}

std::tuple<std::shared_ptr<TetraMesh>, std::vector<size_t>>
Qhull::ComputeDelaunayTetrahedralization(
        const std::vector<Eigen::Vector3d>& points) {
    typedef decltype(TetraMesh::tetras_)::value_type Vector4i;
    auto delaunay_triangulation = std::make_shared<TetraMesh>();
    std::vector<size_t> pt_map;

    if (points.size() < 4) {
        CVLog::Error(
                "[ComputeDelaunayTriangulation3D] not enough points to create "
                "a tetrahedral mesh.");
    }

    // qhull cannot deal with this case
    if (points.size() == 4) {
        delaunay_triangulation->vertices_ = points;
        delaunay_triangulation->tetras_.push_back(Vector4i(0, 1, 2, 3));
        return std::make_tuple(delaunay_triangulation, pt_map);
    }

    std::vector<double> qhull_points_data(points.size() * 3);
    for (size_t pidx = 0; pidx < points.size(); ++pidx) {
        const auto& pt = points[pidx];
        qhull_points_data[pidx * 3 + 0] = pt[0];
        qhull_points_data[pidx * 3 + 1] = pt[1];
        qhull_points_data[pidx * 3 + 2] = pt[2];
    }

    orgQhull::PointCoordinates qhull_points(3, "");
    qhull_points.append(qhull_points_data);

    orgQhull::Qhull qhull;
    qhull.runQhull(qhull_points.comment().c_str(), qhull_points.dimension(),
                   qhull_points.count(), qhull_points.coordinates(),
                   "d Qbb Qt");

    orgQhull::QhullFacetList facets = qhull.facetList();
    delaunay_triangulation->tetras_.resize(facets.count());
    std::unordered_map<int, int> vert_map;
    std::unordered_set<int> inserted_vertices;
    int tidx = 0;
    for (orgQhull::QhullFacetList::iterator it = facets.begin();
         it != facets.end(); ++it) {
        if (!(*it).isGood()) continue;

        orgQhull::QhullFacet f = *it;
        orgQhull::QhullVertexSet vSet = f.vertices();
        int tetra_subscript = 0;
        for (orgQhull::QhullVertexSet::iterator vIt = vSet.begin();
             vIt != vSet.end(); ++vIt) {
            orgQhull::QhullVertex v = *vIt;
            orgQhull::QhullPoint p = v.point();

            int vidx = p.id();
            delaunay_triangulation->tetras_[tidx](tetra_subscript) = vidx;
            tetra_subscript++;

            if (inserted_vertices.count(vidx) == 0) {
                inserted_vertices.insert(vidx);
                vert_map[vidx] = int(delaunay_triangulation->vertices_.size());
                double* coords = p.coordinates();
                delaunay_triangulation->vertices_.push_back(
                        Eigen::Vector3d(coords[0], coords[1], coords[2]));
                pt_map.push_back(vidx);
            }
        }

        tidx++;
    }

    for (auto& tetra : delaunay_triangulation->tetras_) {
        tetra(0) = vert_map[tetra(0)];
        tetra(1) = vert_map[tetra(1)];
        tetra(2) = vert_map[tetra(2)];
        tetra(3) = vert_map[tetra(3)];
    }

    return std::make_tuple(delaunay_triangulation, pt_map);
}

}  // namespace geometry
}  // namespace cloudViewer
