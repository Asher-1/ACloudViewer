// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "DistanceComputationTools.h"

// Local dependencies
#include "ccMPlaneErrors.h"
#include "ccMeasurementDevice.h"

constexpr char const *MPLANE_SCALARFIELD_NAME = "MPlane Distance";

ccPlane *ccMeasurementDevice::fitPlaneToPoints(
        const std::vector<ccMPlanePoint> &fittingPoints,
        const QString &planeName) const {
    ccPointCloud planeCloud;
    planeCloud.reserveThePointsTable(
            static_cast<unsigned int>(fittingPoints.size()));
    std::for_each(fittingPoints.cbegin(), fittingPoints.cend(),
                  [&planeCloud](ccMPlanePoint point) mutable {
                      planeCloud.addPoint(point.getCoordinates());
                  });

    ccPlane *plane = ccPlane::Fit(&planeCloud);

    if (plane) {
        setupPlaneUiDisplay(plane, planeName);
        return plane;
    } else {
        throw MplaneFittingError("Could not fit plane");
    }
}

void ccMeasurementDevice::createScalarFieldForCloud(
        const ccPlane *plane,
        ccPointCloud *cloud,
        bool signedMeasurement) const {
    CCVector3 N = plane->getNormal();
    PointCoordinateType equation[4] = {N.x, N.y, N.z, 0};
    plane->getEquation(N, equation[3]);

    ccScalarField *scalarFieldPlane;
    int scalarIndex;
    std::tie(scalarFieldPlane, scalarIndex) =
            findOrCreateScalarfieldForCloud(cloud);
    scalarFieldPlane->clear();

    addPointDistancesToScalarfield(scalarFieldPlane, cloud, equation,
                                   signedMeasurement);

    scalarFieldPlane->computeMinAndMax();
    cloud->setCurrentDisplayedScalarField(scalarIndex);
    cloud->showSFColorsScale(true);
    cloud->showSF(true);
    cloud->redrawDisplay();
}

void ccMeasurementDevice::deleteScalarFieldFromCloud(
        ccPointCloud *cloud) const {
    int scalarIndex = cloud->getScalarFieldIndexByName(MPLANE_SCALARFIELD_NAME);
    if (scalarIndex != -1) {
        cloud->deleteScalarField(scalarIndex);
        cloud->redrawDisplay();
    }
}

float ccMeasurementDevice::measurePointToPlaneDistance(
        const ccPlane *plane,
        const CCVector3 &point,
        bool signedMeasurement) const {
    CCVector3 N = plane->getNormal();
    PointCoordinateType equation[4] = {N.x, N.y, N.z, 0};
    plane->getEquation(N, equation[3]);
    if (signedMeasurement) {
        return cloudViewer::DistanceComputationTools::
                computePoint2PlaneDistance(&point, equation);
    } else {
        return abs(cloudViewer::DistanceComputationTools::
                           computePoint2PlaneDistance(&point, equation));
    }
}

void ccMeasurementDevice::setupPlaneUiDisplay(ccPlane *plane,
                                              const QString planeName) const {
    plane->setColor(ecvColor::Rgb(255, 255, 255));
    plane->setName(planeName);
    plane->enableStippling(true);
    plane->showColors(true);
    plane->setOpacity(0.5);
    plane->applyGLTransformation_recursive();
    plane->setVisible(true);
    plane->setSelectionBehavior(ccHObject::SELECTION_IGNORED);
}

std::tuple<ccScalarField *, int>
ccMeasurementDevice::findOrCreateScalarfieldForCloud(
        ccPointCloud *cloud) const {
    ccScalarField *scalarFieldPlane = nullptr;

    int scalarIndex = cloud->getScalarFieldIndexByName(MPLANE_SCALARFIELD_NAME);
    if (scalarIndex == -1) {
        scalarFieldPlane = new ccScalarField(MPLANE_SCALARFIELD_NAME);
        scalarFieldPlane->reserve(cloud->size());
        scalarIndex = cloud->addScalarField(scalarFieldPlane);
    } else {
        scalarFieldPlane = static_cast<ccScalarField *>(
                cloud->getScalarField(scalarIndex));
    }
    return std::make_tuple(scalarFieldPlane, scalarIndex);
}

void ccMeasurementDevice::addPointDistancesToScalarfield(
        ccScalarField *scalarField,
        const ccPointCloud *cloud,
        const PointCoordinateType *equation,
        bool signedMeasurement) const {
    for (unsigned int i = 0; i < cloud->size(); ++i) {
        const CCVector3 *P = cloud->getPoint(i);
        if (signedMeasurement) {
            scalarField->addElement(
                    cloudViewer::DistanceComputationTools::
                            computePoint2PlaneDistance(P, equation));
        } else {
            scalarField->addElement(
                    abs(cloudViewer::DistanceComputationTools::
                                computePoint2PlaneDistance(P, equation)));
        }
    }
}
