// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ConvexConcaveHullReconstruction.h"

#include <Utils/cc2sm.h>
#include <Utils/sm2cc.h>

#include "PclUtils/PCLModules.h"
#include "dialogs/ConvexConcaveHullDlg.h"

// ECV_DB_LIB
#include <ecvMesh.h>
#include <ecvPointCloud.h>

// ECV_PLUGINS
#include <ecvMainAppInterface.h>

// QT
#include <QInputDialog>
#include <QMainWindow>

// SYSTEM
#include <iostream>
#include <sstream>

ConvexConcaveHullReconstruction::ConvexConcaveHullReconstruction()
    : BasePclModule(PclModuleDescription(
              tr("ConvexConcaveHull Reconstruction"),
              tr("ConvexConcaveHull Reconstruction"),
              tr("ConvexConcaveHull Reconstruction from clouds"),
              ":/toolbar/PclAlgorithms/icons/convex_concave_hull.png")),
      m_dialog(nullptr),
      m_dimension(3),
      m_alpha(0.5) {}

ConvexConcaveHullReconstruction::~ConvexConcaveHullReconstruction() {
    // we must delete parent-less dialogs ourselves!
    if (m_dialog && m_dialog->parent() == nullptr) delete m_dialog;
}

int ConvexConcaveHullReconstruction::checkSelected() {
    // do we have a selected cloud?
    int have_cloud = isFirstSelectedCcPointCloud();
    if (have_cloud != 1) return -11;

    return 1;
}

int ConvexConcaveHullReconstruction::openInputDialog() {
    // initialize the dialog object
    if (!m_dialog)
        m_dialog =
                new ConvexConcaveHullDlg(m_app ? m_app->getActiveWindow() : 0);

    if (!m_dialog->exec()) return 0;

    return 1;
}

void ConvexConcaveHullReconstruction::getParametersFromDialog() {
    if (!m_dialog) return;

    // get the parameters from the dialog
    m_dimension = static_cast<int>(m_dialog->hullDimension->value());
    m_alpha = static_cast<float>(m_dialog->concaveAlpha->value());
}

int ConvexConcaveHullReconstruction::checkParameters() {
    if (m_dimension < 2) {
        return -52;
    }
    return 1;
}

int ConvexConcaveHullReconstruction::compute() {
    ccPointCloud* cloud = getSelectedEntityAsCCPointCloud();
    if (!cloud) return -1;

    // get xyz as pcl point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr xyzCloud = cc2smReader(cloud).getXYZ2();
    if (!xyzCloud) return -1;

    // reconstruction
    PCLMesh mesh;
    int result = -1;
    if (m_alpha > 0) {
        result = PCLModules::GetConcaveHullReconstruction<PointT>(
                xyzCloud, mesh, m_dimension, m_alpha);
    } else {
        result = PCLModules::GetConvexHullReconstruction<PointT>(xyzCloud, mesh,
                                                                 m_dimension);
    }
    if (result < 0) return -1;

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

    if (m_alpha > 0) {
        CVLog::Print(tr("[Concave-Reconstruction] %1 points, %2 face(s)")
                             .arg(vertCount)
                             .arg(faceCount));
        out_mesh->setName(tr("Concave Reconstruction"));
    } else {
        CVLog::Print(tr("[Convex-Reconstruction] %1 points, %2 face(s)")
                             .arg(vertCount)
                             .arg(faceCount));
        out_mesh->setName(tr("Convex Reconstruction"));
    }

    // copy global shift & scale
    out_mesh->getAssociatedCloud()->setGlobalScale(cloud->getGlobalScale());
    out_mesh->getAssociatedCloud()->setGlobalShift(cloud->getGlobalShift());

    if (cloud->getParent()) cloud->getParent()->addChild(out_mesh);

    emit newEntity(out_mesh);

    return 1;
}

QString ConvexConcaveHullReconstruction::getErrorMessage(int errorCode) {
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
                    "Convex Concave Hull Reconstruction does not returned any "
                    "point. Try relaxing your parameters");
        default:
            break;
    }

    return BasePclModule::getErrorMessage(errorCode);
}
