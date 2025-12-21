// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "MarchingCubeReconstruction.h"

#include <Utils/cc2sm.h>
#include <Utils/sm2cc.h>

#include "PclUtils/PCLModules.h"
#include "dialogs/MarchingCubeDlg.h"

// ECV_DB_LIB
#include <ecvMesh.h>
#include <ecvPointCloud.h>

// ECV_PLUGINS
#include <ecvMainAppInterface.h>

// QT
#include <QMainWindow>

// SYSTEM
#include <iostream>

MarchingCubeReconstruction::MarchingCubeReconstruction()
    : BasePclModule(PclModuleDescription(
              tr("Marching Cube"),
              tr("Marching Cube Reconstruction"),
              tr("Marching Cube Reconstruction from clouds"),
              ":/toolbar/PclAlgorithms/icons/MarchingCubeReconstruction.png")),
      m_dialog(nullptr),
      m_normalSearchRadius(0),
      m_knn_radius(20),
      m_useKnn(true),
      m_marchingMethod(0),
      m_epsilon(0.01f),
      m_isoLevel(0.0f),
      m_gridResolution(50),
      m_percentageExtendGrid(0.0f) {}

MarchingCubeReconstruction::~MarchingCubeReconstruction() {
    // we must delete parent-less dialogs ourselves!
    if (m_dialog && m_dialog->parent() == nullptr) delete m_dialog;
}

int MarchingCubeReconstruction::checkSelected() {
    // do we have a selected cloud?
    int have_cloud = isFirstSelectedCcPointCloud();
    if (have_cloud != 1) return -11;

    return 1;
}

int MarchingCubeReconstruction::openInputDialog() {
    // initialize the dialog object
    if (!m_dialog)
        m_dialog = new MarchingCubeDlg(m_app ? m_app->getActiveWindow() : 0);

    ccPointCloud* cloud = getSelectedEntityAsCCPointCloud();
    if (cloud) {
        ccBBox bBox = cloud->getOwnBB();
        if (bBox.isValid())
            m_dialog->normalSearchRadius->setValue(bBox.getDiagNorm() * 0.005);
    }

    if (!m_dialog->exec()) return 0;

    return 1;
}

void MarchingCubeReconstruction::getParametersFromDialog() {
    if (!m_dialog) return;

    // get the parameters from the dialog
    m_useKnn = m_dialog->useKnnCheckBoxForTriangulation->isChecked();
    m_knn_radius = m_dialog->knnSpinBoxForTriangulation->value();
    m_normalSearchRadius =
            static_cast<float>(m_dialog->normalSearchRadius->value());

    m_marchingMethod = m_dialog->MarchingMethodsCombo->currentIndex();
    m_epsilon = static_cast<float>(m_dialog->epsilonSpinBox->value());
    m_isoLevel = static_cast<float>(m_dialog->isoLevelSpinBox->value());
    m_gridResolution =
            static_cast<float>(m_dialog->gridResolutionSpinBox->value());
    m_percentageExtendGrid =
            static_cast<float>(m_dialog->percentageExtendedSpinBox->value());
}

int MarchingCubeReconstruction::checkParameters() { return 1; }

int MarchingCubeReconstruction::compute() {
    ccPointCloud* cloud = getSelectedEntityAsCCPointCloud();
    if (!cloud) return -1;

    // create storage for normals
    PointCloudNormal::Ptr cloudWithNormals(new PointCloudNormal);

    if (!cloud->hasNormals()) {
        // get xyz as pcl point cloud
        PointCloudT::Ptr xyzCloud = cc2smReader(cloud).getXYZ2();
        if (!xyzCloud) return -1;

        // now compute
        CloudNormal::Ptr normals(new CloudNormal);
        int result = PCLModules::ComputeNormals<PointT, NormalT>(
                xyzCloud, normals,
                m_useKnn ? m_knn_radius : m_normalSearchRadius, m_useKnn);
        if (result < 0) return -1;

        // concat points and normals
        pcl::concatenateFields(*xyzCloud, *normals, *cloudWithNormals);
        CVLog::Print(tr(
                "[MarchingCubeReconstruction::compute] generate new normals"));
    } else {
        PCLCloud::Ptr sm_cloud = cc2smReader(cloud).getAsSM();
        FROM_PCL_CLOUD(*sm_cloud, *cloudWithNormals);
        CVLog::Print(
                tr("[MarchingCubeReconstruction::compute] find normals and use "
                   "the normals"));
    }

    // Marching Cube
    PCLMesh mesh;
    PCLModules::MarchingMethod marchingMethod =
            (PCLModules::MarchingMethod)m_marchingMethod;
    if (!PCLModules::GetMarchingCubes<PointNT>(
                cloudWithNormals, marchingMethod, mesh, m_epsilon, m_isoLevel,
                m_gridResolution, m_percentageExtendGrid)) {
        return -1;
    }

    PCLCloud out_cloud_sm(mesh.cloud);
    if (out_cloud_sm.height * out_cloud_sm.width == 0) {
        // cloud is empty
        return -53;
    }

    ccMesh* out_mesh = pcl2cc::Convert(out_cloud_sm, mesh.polygons);
    if (!out_mesh) {
        // conversion failed (not enough memory?)
        return -1;
    }

    unsigned vertCount = out_mesh->getAssociatedCloud()->size();
    unsigned faceCount = out_mesh->size();
    CVLog::Print(tr("[MarchingCube-Reconstruction] %1 points, %2 face(s)")
                         .arg(vertCount)
                         .arg(faceCount));

    out_mesh->setName(tr("Marching Cube"));
    // copy global shift & scale
    out_mesh->getAssociatedCloud()->setGlobalScale(cloud->getGlobalScale());
    out_mesh->getAssociatedCloud()->setGlobalShift(cloud->getGlobalShift());

    if (cloud->getParent()) cloud->getParent()->addChild(out_mesh);

    emit newEntity(out_mesh);

    return 1;
}

QString MarchingCubeReconstruction::getErrorMessage(int errorCode) {
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
                    "Marching Cube Reconstruction does not returned any point. "
                    "Try relaxing your parameters");
        default:
            break;
    }

    return BasePclModule::getErrorMessage(errorCode);
}
