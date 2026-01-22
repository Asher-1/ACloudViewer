// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "RegionGrowingSegmentation.h"

#include <Utils/cc2sm.h>
#include <Utils/sm2cc.h>

#include "PclUtils/PCLModules.h"
#include "dialogs/RegionGrowingSegmentationDlg.h"
#ifdef LP_PCL_PATCH_ENABLED
#include <Utils/copy.h>
#endif

// CV_DB_LIB
#include <ecvPointCloud.h>

// ECV_PLUGINS
#include <ecvMainAppInterface.h>

// QT
#include <QMainWindow>

// SYSTEM
#include <iostream>
#include <sstream>

RegionGrowingSegmentation::RegionGrowingSegmentation()
    : BasePclModule(PclModuleDescription(
              tr("Region Growing Segmentation"),
              tr("Region Growing Segmentation"),
              tr("Region Growing Segmentation from clouds"),
              ":/toolbar/PclAlgorithms/icons/Regiongrowing.png")),
      m_dialog(nullptr),
      m_basedRgb(false),
      m_k_search(50),
      m_min_cluster_size(50),
      m_max_cluster_size(100000),
      m_neighbour_number(30),
      m_smoothness_theta(0.052359878f),
      m_curvature(1.0f),
      m_min_cluster_size_rgb(100),
      m_neighbors_distance(10.0f),
      m_point_color_diff(6.0f),
      m_region_color_diff(5.0f) {}

RegionGrowingSegmentation::~RegionGrowingSegmentation() {
    // we must delete parent-less dialogs ourselves!
    if (m_dialog && m_dialog->parent() == nullptr) delete m_dialog;
}

int RegionGrowingSegmentation::checkSelected() {
    // do we have a selected cloud?
    int have_cloud = isFirstSelectedCcPointCloud();
    if (have_cloud != 1) return -11;

    return 1;
}

int RegionGrowingSegmentation::openInputDialog() {
    // initialize the dialog object
    if (!m_dialog)
        m_dialog = new RegionGrowingSegmentationDlg(
                m_app ? m_app->getActiveWindow() : 0);

    ccPointCloud* cloud = getSelectedEntityAsCCPointCloud();
    if (cloud->hasColors() || cloud->hasScalarFields()) {
        m_dialog->rgRGBTab->setEnabled(true);
        m_basedRgb = true;
    } else {
        m_dialog->rgRGBTab->setEnabled(false);
        m_basedRgb = false;
    }

    if (!m_dialog->exec()) return 0;

    return 1;
}

void RegionGrowingSegmentation::getParametersFromDialog() {
    if (!m_dialog) return;

    switch (m_dialog->rgTab->currentIndex()) {
        // Basic Region Growing Segmentation
        case 0: {
            m_min_cluster_size =
                    m_dialog->min_cluster_size_input->text().toInt();
            m_max_cluster_size =
                    m_dialog->max_cluster_size_input->text().toInt();
            m_neighbour_number = m_dialog->neighbour_num_input->text().toUInt();
            m_smoothness_theta =
                    m_dialog->smoothness_theta_input->text().toFloat();
            m_curvature = m_dialog->curvature_input->text().toFloat();
            m_basedRgb = false;
        } break;
        // Region Growing Segmentation Based on RGB
        case 1: {
            m_min_cluster_size_rgb =
                    m_dialog->min_cluster_size2_input->text().toInt();
            m_neighbors_distance =
                    m_dialog->neighbours_dist_input->text().toFloat();
            m_point_color_diff =
                    m_dialog->point_color_diff_input->text().toFloat();
            m_region_color_diff =
                    m_dialog->region_color_diff_input->text().toFloat();
            m_basedRgb = true;
        } break;
    }
}

int RegionGrowingSegmentation::checkParameters() { return 1; }

int RegionGrowingSegmentation::compute() {
    ccPointCloud* cloud = getSelectedEntityAsCCPointCloud();
    if (!cloud) return -1;

    PCLCloud::Ptr sm_cloud = cc2smReader(cloud).getAsSM();
    if (!sm_cloud) return -1;

    // initialize all possible clouds
    std::vector<pcl::PointIndices> clusters;
    PointCloudT::Ptr xyzCloud(new PointCloudT);
    PointCloudRGB::Ptr rgbCloud(new PointCloudRGB);
    PointCloudRGB::Ptr cloudSegmented(new PointCloudRGB);

    std::stringstream name;
    if (m_basedRgb)  // Region Growing Segmentation Based on RGB
    {
        FROM_PCL_CLOUD(*sm_cloud, *rgbCloud);
        int result = PCLModules::GetRegionGrowingRGB(
                rgbCloud, clusters, cloudSegmented, m_min_cluster_size_rgb,
                m_neighbors_distance, m_point_color_diff, m_region_color_diff);
        if (result < 0) return -1;
        name << "RGB_RegionGrowing_clusters_" << clusters.size();
    } else  // Basic Region Growing Segmentation
    {
        FROM_PCL_CLOUD(*sm_cloud, *xyzCloud);
        int result = PCLModules::GetRegionGrowing(
                xyzCloud, clusters, cloudSegmented, m_k_search,
                m_min_cluster_size, m_max_cluster_size, m_neighbour_number,
                m_smoothness_theta, m_curvature);
        if (result < 0) return -1;
        name << "Basic_RegionGrowing_clusters_" << clusters.size();
    }

    PCLCloud out_cloud_sm;
    TO_PCL_CLOUD(*cloudSegmented, out_cloud_sm);

    if (out_cloud_sm.height * out_cloud_sm.width == 0) {
        // cloud is empty
        return -53;
    }

    ccPointCloud* out_cloud_cc = pcl2cc::Convert(out_cloud_sm);
    if (!out_cloud_cc) {
        // conversion failed (not enough memory?)
        return -1;
    }

    out_cloud_cc->setName(name.str().c_str());
    // copy global shift & scale
    out_cloud_cc->setGlobalScale(cloud->getGlobalScale());
    out_cloud_cc->setGlobalShift(cloud->getGlobalShift());

    cloud->setEnabled(false);
    if (cloud->getParent()) cloud->getParent()->addChild(out_cloud_cc);

    emit newEntity(out_cloud_cc);

    return 1;
}

QString RegionGrowingSegmentation::getErrorMessage(int errorCode) {
    switch (errorCode) {
            // THESE CASES CAN BE USED TO OVERRIDE OR ADD FILTER-SPECIFIC ERRORS
            // CODES ALSO IN DERIVED CLASSES DEFULAT MUST BE ""

        case -51:
            return tr(
                    "Selected entity does not have any suitable scalar field "
                    "or RGB");
        case -52:
            return tr(
                    "Wrong Parameters. One or more parameters cannot be "
                    "accepted");
        case -53:
            return tr(
                    "Region Growing Segmentation does not returned any point. "
                    "Try relaxing your parameters");
    }

    return BasePclModule::getErrorMessage(errorCode);
}
