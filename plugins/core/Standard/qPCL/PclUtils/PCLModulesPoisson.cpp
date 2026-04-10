// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Poisson surface reconstruction is compiled WITHOUT PCL_NO_PRECOMPILE.
// PCL 1.14's bundled poisson4 headers (octree_poisson.hpp, sparse_matrix.hpp)
// contain known bugs that surface only when full template expansion is forced
// by PCL_NO_PRECOMPILE.  By omitting that macro here, we link against the
// pre-instantiated symbols in the PCL shared library and avoid the broken
// template code paths entirely.

#include <PclUtils/PCLModules.h>
#include <pcl/search/kdtree.h>
#include <pcl/surface/poisson.h>

namespace PCLModules {

int GetPoissonReconstruction(const PointCloudNormal::ConstPtr &cloudWithNormals,
                             PCLMesh &outMesh,
                             int degree,
                             int treeDepth,
                             int isoDivideDepth,
                             int solverDivideDepth,
                             float scale,
                             float samplesPerNode,
                             bool useConfidence,
                             bool useManifold,
                             bool outputPolygons) {
    pcl::search::KdTree<PointNT>::Ptr kdtree(new pcl::search::KdTree<PointNT>);
    kdtree->setInputCloud(cloudWithNormals);
    pcl::Poisson<PointNT> pn;
    pn.setConfidence(useConfidence);
    pn.setDegree(degree);
    pn.setDepth(treeDepth);
    pn.setIsoDivide(isoDivideDepth);
    pn.setManifold(useManifold);
    pn.setOutputPolygons(outputPolygons);
    pn.setSamplesPerNode(samplesPerNode);
    pn.setScale(scale);
    pn.setSolverDivide(solverDivideDepth);
    pn.setSearchMethod(kdtree);
    pn.setInputCloud(cloudWithNormals);
    pn.performReconstruction(outMesh);

    return 1;
}

}  // namespace PCLModules
