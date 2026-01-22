// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "NurbsSurfaceReconstruction.h"

#include <Utils/cc2sm.h>
#include <Utils/sm2cc.h>

#include "PclUtils/PCLModules.h"
#include "Tools/Common/ecvTools.h"  // must below above three
#include "dialogs/NurbsSurfaceDlg.h"

// CV_DB_LIB
#include <ecvMesh.h>
#include <ecvPointCloud.h>
#include <ecvPolyline.h>

// ECV_PLUGINS
#include <ecvMainAppInterface.h>

// QT
#include <QMainWindow>

// SYSTEM
#include <iostream>

NurbsSurfaceReconstruction::NurbsSurfaceReconstruction()
    : BasePclModule(PclModuleDescription(
              tr("Nurbs Surface Triangulation"),
              tr("Nurbs Surface Triangulation"),
              tr("Nurbs Surface Triangulation from clouds"),
              ":/toolbar/PclAlgorithms/icons/bspline_surface.png")),
      m_dialog(nullptr),
      m_order(3),
      m_curveResolution(4),
      m_meshResolution(128),
      m_refinements(4),
      m_iterations(10),
      m_twoDim(true),
      m_fitBSplineCurve(true),
      m_useVoxelGrid(false),
      m_interiorSmoothness(0.2f),
      m_interiorWeight(1.0f),
      m_boundarySmoothness(0.2f),
      m_boundaryWeight(0.0f) {}

NurbsSurfaceReconstruction::~NurbsSurfaceReconstruction() {
    // we must delete parent-less dialogs ourselves!
    if (m_dialog && m_dialog->parent() == nullptr) delete m_dialog;
}

int NurbsSurfaceReconstruction::checkSelected() {
    // do we have a selected cloud?
    int have_cloud = isFirstSelectedCcPointCloud();
    if (have_cloud != 1) return -11;

    return 1;
}

int NurbsSurfaceReconstruction::openInputDialog() {
    // initialize the dialog object
    if (!m_dialog)
        m_dialog = new NurbsSurfaceDlg(m_app ? m_app->getActiveWindow() : 0);

    if (!m_dialog->exec()) return 0;

    return 1;
}

void NurbsSurfaceReconstruction::getParametersFromDialog() {
    if (!m_dialog) return;

    // get the parameters from the dialog
    m_order = m_dialog->orderSpinBox->value();
    m_iterations = m_dialog->iterationsSpinBox->value();
    m_refinements = m_dialog->refinementsSpinBox->value();
    m_meshResolution = m_dialog->meshResolutionSpinBox->value();
    m_curveResolution = m_dialog->curveResolutionSpinBox->value();
    m_twoDim = m_dialog->twoDimCheckBox->isChecked();
    m_fitBSplineCurve = m_dialog->fitBsplineCurveCheckBox->isChecked();
    m_useVoxelGrid = m_dialog->useVoxelGridCheckBox->isChecked();

    m_interiorWeight = static_cast<float>(m_dialog->interiorWSpinBox->value());
    m_interiorSmoothness =
            static_cast<float>(m_dialog->interiorSSpinBox->value());
    m_boundaryWeight = static_cast<float>(m_dialog->boundaryWSpinBox->value());
    m_boundarySmoothness =
            static_cast<float>(m_dialog->boundarySSpinBox->value());
}

int NurbsSurfaceReconstruction::checkParameters() { return 1; }

int NurbsSurfaceReconstruction::compute() {
    ccPointCloud* cloud = getSelectedEntityAsCCPointCloud();
    if (!cloud) return -1;

    // get xyz as pcl point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr xyzCloud = cc2smReader(cloud).getXYZ2();
    if (!xyzCloud) return -1;

    // 2. voxel grid filter
    if (m_useVoxelGrid) {
        double leafSize =
                3 * PCLModules::ComputeCloudResolution<PointT>(xyzCloud);
        if (leafSize > 0) {
            PointCloudT::Ptr tempCloud(new PointCloudT);
            if (!PCLModules::VoxelGridFilter<PointT>(
                        xyzCloud, tempCloud, leafSize, leafSize, leafSize)) {
                return -1;
            }
            xyzCloud = tempCloud;
        }
    }

    // reconstruction
    PCLMesh mesh;
    PointCloudRGB::Ptr outCurve(new PointCloudRGB);
#if defined(WITH_PCL_NURBS)
    // init parameters
    PCLModules::NurbsParameters nurbsParams;
    {
        nurbsParams.fittingParams_.interior_smoothness = m_interiorSmoothness;
        nurbsParams.fittingParams_.interior_weight = m_interiorWeight;
        nurbsParams.fittingParams_.boundary_smoothness = m_boundarySmoothness;
        nurbsParams.fittingParams_.boundary_weight = m_boundaryWeight;

        nurbsParams.order_ = m_order;
        nurbsParams.meshResolution_ = m_meshResolution;
        nurbsParams.curveResolution_ = m_curveResolution;
        nurbsParams.refinements_ = m_refinements;
        nurbsParams.iterations_ = m_iterations;
        nurbsParams.twoDim_ = m_twoDim;
    }

    if (m_fitBSplineCurve) {
        if (!PCLModules::NurbsSurfaceFitting<PointT>(xyzCloud, nurbsParams,
                                                     mesh, outCurve)) {
            return -1;
        }
    } else {
        if (!PCLModules::NurbsSurfaceFitting<PointT>(xyzCloud, nurbsParams,
                                                     mesh, nullptr)) {
            return -1;
        }
    }
#else
    CVLog::Print(tr(
            "[NurbsSurfaceReconstruction] PCL not supported with nurbs, please "
            "rebuild pcl with -DBUILD_surface_on_nurbs=ON again"));
    return -1;
#endif

    // convert output curve to polyline
    ccPolyline* curvePoly = nullptr;
    if (m_fitBSplineCurve && outCurve->points.size() > 1) {
        PCLCloud::Ptr curve_sm(new PCLCloud);
        TO_PCL_CLOUD(*outCurve, *curve_sm);
        curvePoly = ecvTools::GetPolylines(curve_sm, "nurbs-curve", true);
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
    CVLog::Print(tr("[NurbsSurfaceReconstruction] %1 points, %2 face(s)")
                         .arg(vertCount)
                         .arg(faceCount));

    out_mesh->setName(tr("nurbs-surface-%1").arg(m_order));
    // copy global shift & scale
    out_mesh->getAssociatedCloud()->setGlobalScale(cloud->getGlobalScale());
    out_mesh->getAssociatedCloud()->setGlobalShift(cloud->getGlobalShift());

    if (cloud->getParent()) cloud->getParent()->addChild(out_mesh);

    if (curvePoly) {
        if (cloud->getParent()) cloud->getParent()->addChild(curvePoly);
        emit newEntity(curvePoly);
    }

    emit newEntity(out_mesh);

    return 1;
}

QString NurbsSurfaceReconstruction::getErrorMessage(int errorCode) {
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
                    "Nurbs Surface Triangulation does not returned any point. "
                    "Try relaxing your parameters");
        default:
            break;
    }

    return BasePclModule::getErrorMessage(errorCode);
}
