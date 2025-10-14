// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "EuclideanClusterSegmentation.h"

#include "PclUtils/PCLModules.h"
#include "PclUtils/cc2sm.h"
#include "PclUtils/sm2cc.h"
#include "Tools/ecvTools.h"  // must below above three
#include "dialogs/EuclideanClusterDlg.h"

// ECV_DB_LIB
#include <ecvPointCloud.h>

// ECV_PLUGINS
#include <ecvMainAppInterface.h>

// QT
#include <QMainWindow>

// SYSTEM
#include <iostream>
#include <sstream>

EuclideanClusterSegmentation::EuclideanClusterSegmentation()
    : BasePclModule(PclModuleDescription(
              tr("EuclideanCluster Segmentation"),
              tr("EuclideanCluster Segmentation"),
              tr("EuclideanCluster Segmentation from clouds"),
              ":/toolbar/PclAlgorithms/icons/"
              "EuclideanClusterSegmentation.png")),
      m_dialog(nullptr),
      m_randomClusterColor(false),
      m_clusterTolerance(0.02f),
      m_minClusterSize(100),
      m_maxClusterSize(25000) {}

EuclideanClusterSegmentation::~EuclideanClusterSegmentation() {
    // we must delete parent-less dialogs ourselves!
    if (m_dialog && m_dialog->parent() == nullptr) delete m_dialog;
}

int EuclideanClusterSegmentation::checkSelected() {
    // do we have a selected cloud?
    int have_cloud = isFirstSelectedCcPointCloud();
    if (have_cloud != 1) return -11;

    return 1;
}

int EuclideanClusterSegmentation::openInputDialog() {
    // initialize the dialog object
    if (!m_dialog)
        m_dialog =
                new EuclideanClusterDlg(m_app ? m_app->getActiveWindow() : 0);

    if (!m_dialog->exec()) return 0;

    return 1;
}

void EuclideanClusterSegmentation::getParametersFromDialog() {
    if (!m_dialog) return;

    // get the parameters from the dialog
    m_randomClusterColor = m_dialog->randomClusterColorCheckBox->isChecked();
    m_clusterTolerance =
            static_cast<float>(m_dialog->clusterToleranceSpinbox->value());
    m_minClusterSize = m_dialog->minClusterSizeSpinBox->value();
    m_maxClusterSize = m_dialog->maxClusterSizeSpinBox->value();
}

int EuclideanClusterSegmentation::checkParameters() {
    if (m_minClusterSize > m_maxClusterSize) {
        return -52;
    }
    return 1;
}

int EuclideanClusterSegmentation::compute() {
    ccPointCloud* cloud = getSelectedEntityAsCCPointCloud();
    if (!cloud) return -1;

    // get xyz as pcl point cloud
    PointCloudT::Ptr xyzCloud = cc2smReader(cloud).getXYZ2();
    if (!xyzCloud) return -1;

    // cluster segmentation
    std::vector<pcl::PointIndices> cluster_indices;
    if (!PCLModules::EuclideanCluster<PointT>(
                xyzCloud, cluster_indices, m_clusterTolerance, m_minClusterSize,
                m_maxClusterSize)) {
        return -1;
    }
    if (cluster_indices.size() == 0 || cluster_indices.size() > 300) {
        return -53;
    }

    // for each cluster
    std::vector<std::vector<size_t>> clusterIndices;
    for (auto& cluster : cluster_indices) {
        std::vector<size_t> cs;
        cs.resize(cluster.indices.size());
        for (size_t i = 0; i < cluster.indices.size(); ++i) {
            cs[i] = (static_cast<size_t>(cluster.indices[i]));
        }

        clusterIndices.push_back(cs);
    }

    bool error = false;
    ccHObject* group =
            ecvTools::GetClousterGroup(cloud, clusterIndices, m_minClusterSize,
                                       m_randomClusterColor, error);

    if (group) {
        group->setName(group->getName() +
                       tr("-Tolerance(%1)-ClusterSize(%2-%3)")
                               .arg(m_clusterTolerance)
                               .arg(m_minClusterSize)
                               .arg(m_maxClusterSize));

        unsigned count = group->getChildrenNumber();
        m_app->dispToConsole(tr("[EuclideanClusterSegmentation] %1 cluster(s) "
                                "where created from cloud '%2'")
                                     .arg(cluster_indices.size())
                                     .arg(cloud->getName()));

        if (error) {
            m_app->dispToConsole(tr("Error(s) occurred during the generation "
                                    "of clusters! Result may be incomplete"),
                                 ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
        }

        cloud->setEnabled(false);
        if (cloud->getParent()) cloud->getParent()->addChild(group);

        emit newEntity(group);

    } else if (error) {
        return -54;
    } else {
        return -1;
    }

    return 1;
}

QString EuclideanClusterSegmentation::getErrorMessage(int errorCode) {
    switch (errorCode) {
            // THESE CASES CAN BE USED TO OVERRIDE OR ADD FILTER-SPECIFIC ERRORS
            // CODES ALSO IN DERIVED CLASSES DEFULAT MUST BE ""
        case -51:
            return tr(
                    "Selected entity does not have any suitable scalar field "
                    "or RGB.");
        case -52:
            return tr(
                    "Wrong Parameters. One or more parameters cannot be "
                    "accepted");
        case -53:
            return tr(
                    "EuclideanCluster Segmentation could not get any cluster "
                    "or "
                    "the clusters are more than 300 for the given dataset. Try "
                    "relaxing your parameters");
        case -54:
            return tr("An error occurred during the generation of clusters!");
    }

    return BasePclModule::getErrorMessage(errorCode);
}
