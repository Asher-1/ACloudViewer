// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/**
 * @file ecvPclTools.h
 * @brief PCL-dependent utility functions for polylines, point clouds,
 *        clusters, and segmentation groups.
 *
 * These functions convert between PCL and CloudViewer (cc) data structures.
 * For non-PCL utilities (vector ops, color conversion), see ecvTools.h.
 */

#include <PclUtils/PCLCloud.h>
#include <PclUtils/PCLConv.h>
#include <PclUtils/sm2cc.h>
#include <ecvColorTypes.h>
#include <ecvPointCloud.h>
#include <ecvPolyline.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/PointIndices.h>
#include <pcl/Vertices.h>

namespace ecvPclTools {

static ccPolyline* GetPolylines(PCLCloud::Ptr& curve_sm,
                                const QString& name = "curve",
                                bool closed = true,
                                const ecvColor::Rgb& color = ecvColor::green) {
    if (curve_sm->width * curve_sm->height == 0) {
        return nullptr;
    }

    ccPointCloud* polyVertices = pcl2cc::Convert(*curve_sm, true, true);
    {
        if (!polyVertices) {
            return nullptr;
        }
        polyVertices->setPointSize(5);
        polyVertices->showColors(true);
        polyVertices->setTempColor(ecvColor::red);
    }

    ccPolyline* curvePoly = new ccPolyline(polyVertices);
    {
        if (!curvePoly) {
            return nullptr;
        }

        int verticesCount = polyVertices->size();
        if (curvePoly->reserve(verticesCount)) {
            curvePoly->addPointIndex(0, verticesCount);
            curvePoly->setColor(color);
            curvePoly->showColors(true);
            curvePoly->setVisible(true);
            curvePoly->setClosed(closed);
            curvePoly->setName(name);
            curvePoly->addChild(polyVertices);
        } else {
            delete curvePoly;
            curvePoly = nullptr;
            return nullptr;
        }
    }

    return curvePoly;
}

static void AddPointCloud(ccHObject* ecvGroup,
                          PCLCloud::Ptr sm_cloud,
                          bool& error,
                          QString label,
                          const ccPointCloud* ccCloud = nullptr,
                          bool randomColors = true) {
    ccPointCloud* out_cloud_cc = pcl2cc::Convert(*sm_cloud);

    if (!out_cloud_cc) {
        // not enough memory!
        error = true;
    } else {
        // shall we colorize it with a random color?
        if (randomColors) {
            ecvColor::Rgb col = ecvColor::Generator::Random();
            out_cloud_cc->setRGBColor(col);
            out_cloud_cc->showColors(true);
            out_cloud_cc->showSF(false);  // just in case
        }

        QString cloudName = QString("%1 %2 (size=%3)")
                                    .arg(label)
                                    .arg(ecvGroup->getChildrenNumber())
                                    .arg(out_cloud_cc->size());
        out_cloud_cc->setName(cloudName);

        // copy global shift & scale
        if (ccCloud) {
            out_cloud_cc->setGlobalScale(ccCloud->getGlobalScale());
            out_cloud_cc->setGlobalShift(ccCloud->getGlobalShift());
        }

        ecvGroup->addChild(out_cloud_cc);
    }
}

/// @param ccCloud Source cloud for metadata.
/// @param xyzCloud XYZ point cloud.
/// @param cluster_indices Cluster index sets.
/// @param minPointsPerComponent Minimum points per cluster.
/// @param randomColors If true, assign random colors per cluster.
/// @param error Set to true on failure.
/// @return Group containing cluster clouds, or nullptr.
static ccHObject* GetClousterGroup(
        const ccPointCloud* ccCloud,
        const PointCloudT::ConstPtr xyzCloud,
        const std::vector<pcl::PointIndices>& cluster_indices,
        unsigned minPointsPerComponent,
        bool randomColors,
        bool& error) {
    if (!xyzCloud) {
        return nullptr;
    }

    // we create a new group to store all input CCs as 'clusters'
    ccHObject* ecvGroup = new ccHObject(ccCloud ? ccCloud->getName()
                                                : "" + QString(" [clusters]-"));
    ecvGroup->setVisible(true);
    error = false;

    // for each cluster
    for (std::vector<pcl::PointIndices>::const_iterator it =
                 cluster_indices.begin();
         it != cluster_indices.end(); ++it) {
        PointCloudT::Ptr cloud_cluster(new PointCloudT);
        for (std::vector<int>::const_iterator pit = it->indices.begin();
             pit != it->indices.end(); ++pit)
            cloud_cluster->points.push_back(xyzCloud->points[*pit]);
        cloud_cluster->width =
                static_cast<uint32_t>(cloud_cluster->points.size());
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        PCLCloud::Ptr sm_cloud(new PCLCloud);
        TO_PCL_CLOUD(*cloud_cluster, *sm_cloud);

        if (sm_cloud->height * sm_cloud->width == 0) {
            error = true;
            continue;
        }

        AddPointCloud(ecvGroup, sm_cloud, error, "cluster", ccCloud,
                      randomColors);
    }

    if (ecvGroup->getChildrenNumber() == 0) {
        delete ecvGroup;
        ecvGroup = nullptr;
    }

    return ecvGroup;
}

/// @param ccCloud Source cloud.
/// @param cluster_indices Cluster index sets (per-point indices).
/// @param minPointsPerComponent Minimum points per cluster.
/// @param randomColors If true, assign random colors per cluster.
/// @param error Set to true on failure.
/// @return Group containing cluster clouds, or nullptr.
static ccHObject* GetClousterGroup(
        const ccPointCloud* ccCloud,
        const std::vector<std::vector<size_t>>& cluster_indices,
        unsigned minPointsPerComponent,
        bool randomColors,
        bool& error) {
    if (!ccCloud) {
        return nullptr;
    }

    // we create a new group to store all input CCs as 'clusters'
    ccHObject* ecvGroup = new ccHObject(ccCloud->getName());
    ecvGroup->setVisible(true);
    error = false;

    // for each cluster
    for (auto& cluster : cluster_indices) {
        ccPointCloud* cloud = ccPointCloud::From(ccCloud, cluster);
        if (cloud) {
            // shall we colorize it with a random color?
            if (randomColors) {
                ecvColor::Rgb col = ecvColor::Generator::Random();
                cloud->setRGBColor(col);
                cloud->showColors(true);
                cloud->showSF(false);  // just in case
            }

            QString cloudName = QString("%1 %2 (size=%3)")
                                        .arg("cluster")
                                        .arg(ecvGroup->getChildrenNumber())
                                        .arg(cloud->size());
            cloud->setName(cloudName);

            ecvGroup->addChild(cloud);
        } else {
            error = true;
            continue;
        }
    }

    if (ecvGroup->getChildrenNumber() == 0) {
        delete ecvGroup;
        ecvGroup = nullptr;
    }

    return ecvGroup;
}

/// @param ccCloud Source cloud for metadata.
/// @param cloudRemained Remaining points after segmentation.
/// @param cloudExtractions Extracted segments.
/// @param randomColors If true, assign random colors.
/// @param error Set to true on failure.
/// @return Group containing remaining and segment clouds, or nullptr.
static ccHObject* GetSegmentationGroup(
        const ccPointCloud* ccCloud,
        const PointCloudT::ConstPtr cloudRemained,
        const std::vector<PointCloudT::Ptr>& cloudExtractions,
        bool randomColors,
        bool& error) {
    // we create a new group to store all input CCs as 'clusters'
    ccHObject* ecvGroup = new ccHObject(
            ccCloud ? ccCloud->getName() : "" + QString(" [segmentation]-"));
    ecvGroup->setVisible(true);
    error = false;

    // save remaining
    if (cloudRemained) {
        PCLCloud::Ptr sm_cloud(new PCLCloud);
        TO_PCL_CLOUD(*cloudRemained, *sm_cloud);
        if (sm_cloud->height * sm_cloud->width != 0) {
            AddPointCloud(ecvGroup, sm_cloud, error, "remaining", ccCloud,
                          randomColors);
        } else {
            error = true;
        }
    }

    // for each extracted segment saving
    for (std::vector<PointCloudT::Ptr>::const_iterator it =
                 cloudExtractions.begin();
         it != cloudExtractions.end(); ++it) {
        PCLCloud::Ptr sm_cloud(new PCLCloud);
        TO_PCL_CLOUD(**it, *sm_cloud);
        if (sm_cloud->height * sm_cloud->width == 0) {
            error = true;
            continue;
        }

        AddPointCloud(ecvGroup, sm_cloud, error, "segment", ccCloud,
                      randomColors);
    }

    if (ecvGroup->getChildrenNumber() == 0) {
        delete ecvGroup;
        ecvGroup = nullptr;
    }

    return ecvGroup;
}

};  // namespace ecvPclTools
