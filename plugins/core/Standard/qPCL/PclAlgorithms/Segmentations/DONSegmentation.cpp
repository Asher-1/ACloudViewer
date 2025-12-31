// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "DONSegmentation.h"

#include <Utils/cc2sm.h>
#include <Utils/sm2cc.h>

#include "PclUtils/PCLModules.h"
#include "Tools/Common/ecvTools.h"  // must below above three
#include "dialogs/DONSegmentationDlg.h"

// ECV_DB_LIB
#include <ecvPointCloud.h>

// ECV_PLUGINS
#include <ecvMainAppInterface.h>

// QT
#include <QMainWindow>

// SYSTEM
#include <iostream>
#include <sstream>

DONSegmentation::DONSegmentation()
    : BasePclModule(
              PclModuleDescription(tr("DoN Segmentation"),
                                   tr("DoN Segmentation"),
                                   tr("DoN Segmentation from clouds"),
                                   ":/toolbar/PclAlgorithms/icons/DoN.png")),
      m_dialog(nullptr),
      m_comparisonField("curvature"),
      m_comparisonTypes("GT"),
      m_smallScale(5.0f),
      m_largeScale(10.0f),
      m_minDonMagnitude(0.3f),
      m_maxDonMagnitude(1.3f),
      m_clusterTolerance(0.02f),
      m_minClusterSize(100),
      m_maxClusterSize(25000),
      m_randomClusterColor(false) {}

DONSegmentation::~DONSegmentation() {
    // we must delete parent-less dialogs ourselves!
    if (m_dialog && m_dialog->parent() == nullptr) delete m_dialog;
}

int DONSegmentation::checkSelected() {
    // do we have a selected cloud?
    int have_cloud = isFirstSelectedCcPointCloud();
    if (have_cloud != 1) return -11;

    return 1;
}

int DONSegmentation::openInputDialog() {
    // initialize the dialog object
    if (!m_dialog)
        m_dialog = new DONSegmentationDlg(m_app ? m_app->getActiveWindow() : 0);

    if (!m_dialog->exec()) return 0;

    return 1;
}

void DONSegmentation::getParametersFromDialog() {
    if (!m_dialog) return;

    // get the parameters from the dialog
    m_smallScale = static_cast<float>(m_dialog->smallScaleSpinBox->value());
    m_largeScale = static_cast<float>(m_dialog->largeScaleSpinBox->value());

    m_minDonMagnitude =
            static_cast<float>(m_dialog->minDonMagnitudeSpinBox->value());
    m_maxDonMagnitude =
            static_cast<float>(m_dialog->maxDonMagnitudeSpinBox->value());
    m_comparisonField = m_dialog->getComparisonField();
    m_dialog->getComparisonTypes(m_comparisonTypes);

    m_randomClusterColor = m_dialog->randomClusterColorCheckBox->isChecked();
    m_clusterTolerance =
            static_cast<float>(m_dialog->clusterToleranceSpinBox->value());
    m_minClusterSize = m_dialog->minClusterSizeSpinBox->value();
    m_maxClusterSize = m_dialog->maxClusterSizeSpinBox->value();
}

int DONSegmentation::checkParameters() {
    if (m_minClusterSize > m_maxClusterSize || m_smallScale > m_largeScale ||
        m_minDonMagnitude > m_maxDonMagnitude ||
        m_comparisonTypes.size() == 0) {
        return -52;
    }
    return 1;
}

int DONSegmentation::compute() {
    ccPointCloud* cloud = getSelectedEntityAsCCPointCloud();
    if (!cloud) return -1;

    // get xyz as pcl point cloud
    PointCloudT::Ptr xyzCloud = cc2smReader(cloud).getXYZ2();
    if (!xyzCloud) return -1;

    std::vector<int> dummy;
    PCLModules::RemoveNaN<PointT>(xyzCloud, xyzCloud, dummy);

    // 1. update parameters
    double cloudReolution =
            PCLModules::ComputeCloudResolution<PointT>(xyzCloud);
    {
        m_smallScale *= cloudReolution;
        m_largeScale *= cloudReolution;
        m_clusterTolerance *= cloudReolution;
    }

    // 2. compute small and large scale normals
    PointCloudNormal::Ptr normals_small_scale(new PointCloudNormal);
    PointCloudNormal::Ptr normals_large_scale(new PointCloudNormal);
    {
        if (!PCLModules::ComputeNormals<PointT, PointNT>(
                    xyzCloud, normals_small_scale, m_smallScale, false, true)) {
            return -1;
        }

        if (!PCLModules::ComputeNormals<PointT, PointNT>(
                    xyzCloud, normals_large_scale, m_largeScale, false, true)) {
            return -1;
        }
    }

    // 3. Create output cloud for DoN results
    PointCloudNormal::Ptr doncloud(new PointCloudNormal);
    pcl::copyPointCloud<PointT, PointNT>(*xyzCloud, *doncloud);
    if (!PCLModules::DONEstimation<PointT, PointNT, PointNT>(
                xyzCloud, normals_small_scale, normals_large_scale, doncloud)) {
        return -1;
    }

    // 4. Filter by field magnitude
    {
        // init condition parameters
        PCLModules::ConditionParameters param;
        {
            if (m_comparisonTypes.size() == 1) {
                param.condition_type_ = PCLModules::ConditionParameters::
                        ConditionType::CONDITION_OR;
            } else {
                param.condition_type_ = PCLModules::ConditionParameters::
                        ConditionType::CONDITION_AND;
            }
            for (const QString& type : m_comparisonTypes) {
                PCLModules::ConditionParameters::ComparisonParam comparison;
                if (type == "GT") {
                    comparison.comparison_type_ =
                            PCLModules::ConditionParameters::ComparisonType::GT;
                } else if (type == "GE") {
                    comparison.comparison_type_ =
                            PCLModules::ConditionParameters::ComparisonType::GE;
                } else if (type == "LT") {
                    comparison.comparison_type_ =
                            PCLModules::ConditionParameters::ComparisonType::LT;
                } else if (type == "LE") {
                    comparison.comparison_type_ =
                            PCLModules::ConditionParameters::ComparisonType::LE;
                } else if (type == "EQ") {
                    comparison.comparison_type_ =
                            PCLModules::ConditionParameters::ComparisonType::EQ;
                }
                comparison.fieldName_ = m_comparisonField.toStdString();
                comparison.min_threshold_ = m_minDonMagnitude;
                comparison.max_threshold_ = m_maxDonMagnitude;
                param.condition_params_.push_back(comparison);
            }
        }
        PointCloudNormal::Ptr doncloud_filtered(new PointCloudNormal);
        if (!PCLModules::ConditionalRemovalFilter<PointNT>(
                    doncloud, param, doncloud_filtered, false)) {
            return -1;
        }
        doncloud = doncloud_filtered;
        if (doncloud->width * doncloud->height == 0) {
            return -1;
        }
    }

    // 5. cluster segmentation
    std::vector<pcl::PointIndices> cluster_indices;
    if (!PCLModules::EuclideanCluster<PointNT>(
                doncloud, cluster_indices, m_clusterTolerance, m_minClusterSize,
                m_maxClusterSize)) {
        return -1;
    }

    if (cluster_indices.size() == 0 || cluster_indices.size() > 300) {
        return -53;
    }

    // 6. processing result
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
        m_app->dispToConsole(tr("[DONSegmentation] %1 cluster(s) where created "
                                "from cloud '%2'")
                                     .arg(count)
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

QString DONSegmentation::getErrorMessage(int errorCode) {
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
                    "Difference of Normals Segmentation could not get any "
                    "cluster or "
                    "the clusters are more than 300 for the given dataset. Try "
                    "relaxing your parameters");
        case -54:
            return tr("An error occurred during the generation of clusters!");
    }

    return BasePclModule::getErrorMessage(errorCode);
}