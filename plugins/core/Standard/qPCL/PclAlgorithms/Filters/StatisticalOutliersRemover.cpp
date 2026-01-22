// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "StatisticalOutliersRemover.h"

// LOCAL
#include <Utils/cc2sm.h>
#include <Utils/sm2cc.h>

#include "PclUtils/PCLModules.h"
#include "dialogs/StatisticalOutliersRemoverDlg.h"

// ECV_PLUGINS
#include <ecvMainAppInterface.h>

// CV_DB_LIB
#include <ecvPointCloud.h>

// QT
#include <QMainWindow>

StatisticalOutliersRemover::StatisticalOutliersRemover()
    : BasePclModule(PclModuleDescription(
              tr("Statistical Outlier Removal"),
              tr("Filter outlier data based on point neighborhood statistics"),
              tr("Filter the points that are farther of their neighbors than "
                 "the average (plus a number of times the standard deviation)"),
              ":/toolbar/PclAlgorithms/icons/sor_outlier_remover.png")),
      m_dialog(nullptr),
      m_k(0),
      m_std(0.0f) {}

StatisticalOutliersRemover::~StatisticalOutliersRemover() {
    // we must delete parent-less dialogs ourselves!
    if (m_dialog && m_dialog->parent() == nullptr) delete m_dialog;
}

int StatisticalOutliersRemover::openInputDialog() {
    if (!m_dialog) {
        m_dialog = new SORDialog(m_app ? m_app->getMainWindow() : nullptr);
    }

    return m_dialog->exec() ? 1 : 0;
}

void StatisticalOutliersRemover::getParametersFromDialog() {
    // get values from dialog
    if (m_dialog) {
        m_k = m_dialog->spinK->value();
        m_std = static_cast<float>(m_dialog->spinStd->value());
    }
}

int StatisticalOutliersRemover::compute() {
    // get selected as pointcloud
    ccPointCloud* cloud = this->getSelectedEntityAsCCPointCloud();
    if (!cloud) return -1;

    // now as sensor message
    PCLCloud::Ptr tmp_cloud = cc2smReader(cloud).getAsSM();
    if (!tmp_cloud) return -1;

    PCLCloud::Ptr outcloud(new PCLCloud);
    int result = PCLModules::RemoveOutliersStatistical<PCLCloud>(
            tmp_cloud, outcloud, m_k, m_std);
    if (result < 0) return -1;

    // get back outcloud as a ccPointCloud
    ccPointCloud* final_cloud = pcl2cc::Convert(*outcloud);
    if (!final_cloud) return -1;

    // create a suitable name for the entity
    final_cloud->setName(
            QString("%1_k%2_std%3").arg(cloud->getName()).arg(m_k).arg(m_std));
    // final_cloud->setDisplay(cloud->getDisplay());
    // copy global shift & scale
    final_cloud->setGlobalScale(cloud->getGlobalScale());
    final_cloud->setGlobalShift(cloud->getGlobalShift());

    // disable original cloud
    cloud->setEnabled(false);
    if (cloud->getParent()) cloud->getParent()->addChild(final_cloud);

    emit newEntity(final_cloud);

    return 1;
}
