// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "NurbsCurveFitting.h"

#include <Utils/cc2sm.h>
#include <Utils/sm2cc.h>

#include "PclUtils/PCLModules.h"
#include "Tools/Common/CurveFitting.h"
#include "Tools/Common/ecvTools.h"  // must below above three
#include "dialogs/NurbsCurveFittingDlg.h"

// CV_DB_LIB
#include <ecvPlane.h>
#include <ecvPointCloud.h>
#include <ecvPolyline.h>

// ECV_PLUGINS
#include <ecvMainAppInterface.h>

// QT
#include <QMainWindow>

// SYSTEM
#include <iostream>

NurbsCurveFitting::NurbsCurveFitting()
    : BasePclModule(PclModuleDescription(
              tr("BSpline Curve Fitting"),
              tr("BSpline Curve Fitting"),
              tr("BSpline Curve Fitting from clouds"),
              ":/toolbar/PclAlgorithms/icons/bspline_curve.png")),
      m_dialog(0),
      m_order(3),
      m_useVoxelGrid(false),
      m_exportProjectedCloud(false),
      m_curveFitting3D(true),
      m_closed(false),
      m_curveResolution(8),
      m_minimizationType(0),
      m_controlPoints(10),
      m_curveRscale(1.0f),
      m_curveSmoothness(0.000001f) {}

NurbsCurveFitting::~NurbsCurveFitting() {
    // we must delete parent-less dialogs ourselves!
    if (m_dialog && m_dialog->parent() == 0) delete m_dialog;
}

int NurbsCurveFitting::checkSelected() {
    // do we have a selected cloud?
    int have_cloud = isFirstSelectedCcPointCloud();
    if (have_cloud != 1) return -11;

    return 1;
}

int NurbsCurveFitting::openInputDialog() {
    // initialize the dialog object
    if (!m_dialog)
        m_dialog =
                new NurbsCurveFittingDlg(m_app ? m_app->getActiveWindow() : 0);

    if (!m_dialog->exec()) return 0;

    return 1;
}

void NurbsCurveFitting::getParametersFromDialog() {
    if (!m_dialog) return;

    // get the parameters from the dialog
    m_minimizationType = m_dialog->minimizationMethodsCombo->currentIndex();
    m_order = m_dialog->orderSpinBox->value();
    m_curveResolution = m_dialog->curveResolutionSpinBox->value();
    m_controlPoints = m_dialog->controlPointsSpinBox->value();
    m_useVoxelGrid = m_dialog->useVoxelGridCheckBox->isChecked();
    m_exportProjectedCloud =
            m_dialog->exportProjectedCloudCheckBox->isChecked();
    m_closed = m_dialog->closedCurveCheckBox->isChecked();
    m_curveFitting3D = m_dialog->curve3DFittingCheckBox->isChecked();
    m_exportProjectedCloud =
            m_closed && m_curveFitting3D ? false : m_exportProjectedCloud;
    m_curveRscale = static_cast<float>(m_dialog->curveRscaleSpinBox->value());
    m_curveSmoothness =
            static_cast<float>(m_dialog->curveSmoothnessSpinBox->value());
}

int NurbsCurveFitting::checkParameters() { return 1; }

int NurbsCurveFitting::compute() {
    ccPointCloud* cloud = getSelectedEntityAsCCPointCloud();
    if (!cloud) return -1;

    PointCloudRGB::Ptr outCurve(new PointCloudRGB);
    PointCloudT::Ptr xyzCloud = cc2smReader(cloud).getXYZ2();
    if (!xyzCloud) return -1;

    // step 1. voxel grid filter
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

#if 0
	PointCloudT::Ptr outCurve(new PointCloudT);
	CurveFitting fit;
	fit.setInputcloud(xyzCloud);
	PointCloudT::Ptr point_mean(new PointCloudT);
	std::vector<double> x_mean, y_mean, z_mean;
	double a, b, c;
	// 0.5 represent step length, -1
	fit.grid_mean_xyz(0.5, -1, x_mean, y_mean, z_mean, point_mean);
	fit.polynomial3D_fitting(x_mean, y_mean, z_mean, a, b, c);
	fit.getPolynomial3D(outCurve, 0.01);

	if (outCurve->points.size() > 1)
	{
		PointT sp = outCurve->points[0];
		CVLog::Print(tr("Start Point: X1= %1, Y1= %2, Z1= %3").arg(sp.x).arg(sp.y).arg(sp.z));
		PointT ep = outCurve->points[outCurve->points.size() -1];
		CVLog::Print(tr("End Point: X2= %1, Y2= %2, Z2= %3").arg(ep.x).arg(ep.y).arg(ep.z));
		CVLog::Print(tr("space polynomial curve equation: z = %1 * (x^2+y^2) + %2 * sqrt(x^2+y^2) + %3").arg(a).arg(b).arg(c));
	}
	else
	{
		return -1;
	}
#endif

    // step 2. get cloud plane matrix with x-y plane
    ccGLMatrix makeZPosMatrix;
    if (!m_closed || !m_curveFitting3D) {
        double rms = 0.0;
        CCVector3 C, N;
        ccPlane* pPlane = ccPlane::Fit(cloud, &rms);
        if (!pPlane) {
            return -1;
        }
        CVLog::Print(tr("\t- plane fitting RMS: %1").arg(rms));
        N = pPlane->getNormal();
        delete pPlane;
        pPlane = nullptr;
        // C = *cloudViewer::Neighbourhood(cloud).getGravityCenter();
        makeZPosMatrix = ccGLMatrix::FromToRotation(N, CCVector3(0, 0, PC_ONE));
        // CCVector3 Gt = C;
        // makeZPosMatrix.applyRotation(Gt);
        // makeZPosMatrix.setTranslation(C - Gt);
        Eigen::Matrix4f xoyTransformation(makeZPosMatrix.data());
        pcl::transformPointCloud(*xyzCloud, *xyzCloud, xoyTransformation);
    }

#if defined(WITH_PCL_NURBS)
    // step 3. transform cloud to x-y plane according makeZPosMatrix
    PointCloudT::Ptr zoyCloud(new PointCloudT);
    if (m_curveFitting3D) {
        if (!PCLModules::BSplineCurveFitting3D<PointT>(
                    xyzCloud, outCurve, m_order, m_controlPoints,
                    m_curveResolution, m_curveSmoothness, m_curveRscale,
                    m_closed)) {
            return -1;
        }
        if (outCurve->points.size() == 0) {
            return -53;
        }
    } else {
        // step 4. init parameters and reconstruction on x-y plane
        PCLModules::CurveFittingMethod fittingMethod =
                (PCLModules::CurveFittingMethod)m_minimizationType;
        if (!PCLModules::BSplineCurveFitting2D<PointT>(
                    xyzCloud, fittingMethod, outCurve, m_order, m_controlPoints,
                    m_curveResolution, m_curveSmoothness, m_curveRscale,
                    m_closed)) {
            return -1;
        }
        if (outCurve->points.size() == 0) {
            return -53;
        }
    }
#else
    CVLog::Print(
            tr("[NurbsCurveFitting] PCL not supported with nurbs, please "
               "rebuild pcl with -DBUILD_surface_on_nurbs=ON again"));
    return -1;
#endif

    if (!m_closed || !m_curveFitting3D) {
        // step 5. transform cloud from x-y plane to original plane
        Eigen::Matrix4f inverseXoyTransformation(
                makeZPosMatrix.inverse().data());
        pcl::transformPointCloud(*outCurve, *outCurve,
                                 inverseXoyTransformation);
    }

    // convert output fitting curve to polyline
    ccPolyline* curvePoly = nullptr;
    {
        PCLCloud::Ptr curve_sm(new PCLCloud);
        TO_PCL_CLOUD(*outCurve, *curve_sm);
        curvePoly = ecvTools::GetPolylines(curve_sm, "fitting-curve", m_closed);
        if (!curvePoly) {
            return -53;
        }
        CVLog::Print(tr("[NurbsCurveFitting] curve : %1 point(s)")
                             .arg(curvePoly->size()));
    }

    ccHObject* outObject = nullptr;
    if (m_exportProjectedCloud) {
        outObject = new ccHObject(cloud ? cloud->getName()
                                        : "" + QString(" [curve-fitting]-"));
        if (outObject) {
            outObject->setVisible(true);
            PCLCloud curve_sm;
            TO_PCL_CLOUD(*xyzCloud, curve_sm);
            ccPointCloud* projectedCloud = pcl2cc::Convert(curve_sm);
            if (projectedCloud) {
                projectedCloud->setName("projected-xoy");
                if (cloud) {
                    // copy global shift & scale
                    projectedCloud->setGlobalScale(cloud->getGlobalScale());
                    projectedCloud->setGlobalShift(cloud->getGlobalShift());
                }
                outObject->addChild(projectedCloud);
            }

            outObject->addChild(curvePoly);
        }

        if (outObject->getChildrenNumber() == 0) {
            delete outObject;
            outObject = nullptr;
            return -1;
        }
    } else {
        outObject = curvePoly;
    }

    if (outObject) {
        if (cloud->getParent()) cloud->getParent()->addChild(outObject);
        emit newEntity(outObject);
    } else {
        return -1;
    }

    return 1;
}

QString NurbsCurveFitting::getErrorMessage(int errorCode) {
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
                    "Nurbs Curve Fitting does not returned any point. Try "
                    "relaxing your parameters");
    }

    return BasePclModule::getErrorMessage(errorCode);
}
