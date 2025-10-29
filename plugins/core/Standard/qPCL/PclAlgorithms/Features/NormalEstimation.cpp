// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "NormalEstimation.h"

// LOCAL
#include "PclUtils/PCLModules.h"
#include "PclUtils/cc2sm.h"
#include "PclUtils/sm2cc.h"
#include "dialogs/NormalEstimationDlg.h"

// ECV_PLUGINS
#include <ecvMainAppInterface.h>

// ECV_DB_LIB
#include <ecvPointCloud.h>

// QT
#include <QMainWindow>
#include <QThread>

NormalEstimation::NormalEstimation()
    : BasePclModule(PclModuleDescription(
              tr("Estimate Normals"),
              tr("Estimate Normals and Curvature"),
              tr("Estimate Normals and Curvature for the selected entity"),
              ":/toolbar/PclAlgorithms/icons/normal_curvature.png")),
      m_dialog(nullptr),
      m_knn_radius(10),
      m_radius(0),
      m_useKnn(false),
      m_overwrite_curvature(false) {
    m_overwrite_curvature = true;
}

NormalEstimation::~NormalEstimation() {
    // we must delete parent-less dialogs ourselves!
    if (m_dialog && m_dialog->parent() == nullptr) delete m_dialog;
}

int NormalEstimation::openInputDialog() {
    if (!m_dialog) {
        m_dialog = new NormalEstimationDialog(m_app ? m_app->getActiveWindow()
                                                    : 0);

        // initially these are invisible
        m_dialog->surfaceComboBox->setVisible(false);
        m_dialog->searchSurfaceCheckBox->setVisible(false);
    }

    ccPointCloud* cloud = getSelectedEntityAsCCPointCloud();
    if (cloud) {
        ccBBox bBox = cloud->getOwnBB();
        if (bBox.isValid())
            m_dialog->radiusDoubleSpinBox->setValue(bBox.getDiagNorm() * 0.005);
    }

    return m_dialog->exec() ? 1 : 0;
}

void NormalEstimation::getParametersFromDialog() {
    assert(m_dialog);
    if (!m_dialog) return;

    // fill in parameters from dialog
    m_useKnn = m_dialog->useKnnCheckBox->isChecked();
    m_overwrite_curvature = m_dialog->curvatureCheckBox->isChecked();
    m_knn_radius = m_dialog->knnSpinBox->value();
    m_radius = static_cast<float>(m_dialog->radiusDoubleSpinBox->value());
}

int NormalEstimation::compute() {
    // pointer to selected cloud
    ccPointCloud* cloud = getSelectedEntityAsCCPointCloud();
    if (!cloud) return -1;

    // if we have normals delete them!
    if (cloud->hasNormals()) cloud->unallocateNorms();

    // get xyz as pcl point cloud
    PointCloudT::Ptr pcl_cloud = cc2smReader(cloud).getXYZ2();
    if (!pcl_cloud) return -1;

    // create storage for normals
    PointCloudNormal::Ptr normals(new PointCloudNormal);

    // now compute
    int result = PCLModules::ComputeNormals<PointT, pcl::PointNormal>(
            pcl_cloud, normals, m_useKnn ? m_knn_radius : m_radius, m_useKnn);
    if (result < 0) return -1;

    PCLCloud::Ptr sm_normals(new PCLCloud);
    TO_PCL_CLOUD(*normals, *sm_normals);

    pcl2cc::CopyNormals(*sm_normals, *cloud);
    pcl2cc::CopyScalarField(*sm_normals, "curvature", *cloud,
                            m_overwrite_curvature);

    emit entityHasChanged(cloud);

    return 1;
}
