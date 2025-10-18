// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// CC
#include "ecvMainAppInterface.h"
#include "ecvPlane.h"
#include "ecvPointCloud.h"
#include "ecvScalarField.h"

// Local dependecies
#include "ccMPlanePoint.h"

class ccMeasurementDevice {
public:
    explicit ccMeasurementDevice(ecvMainAppInterface *app) : m_app(app) {}

    ccPlane *fitPlaneToPoints(const std::vector<ccMPlanePoint> &fittingPoints,
                              const QString &planeName) const;
    void createScalarFieldForCloud(const ccPlane *plane,
                                   ccPointCloud *cloud,
                                   bool signedMeasurement) const;
    void deleteScalarFieldFromCloud(ccPointCloud *cloud) const;
    float measurePointToPlaneDistance(const ccPlane *plane,
                                      const CCVector3 &point,
                                      bool signedMeasurement) const;

private:
    ecvMainAppInterface *m_app;
    void setupPlaneUiDisplay(ccPlane *plane, const QString planeName) const;
    std::tuple<ccScalarField *, int> findOrCreateScalarfieldForCloud(
            ccPointCloud *cloud) const;
    void addPointDistancesToScalarfield(ccScalarField *scalarField,
                                        const ccPointCloud *cloud,
                                        const PointCoordinateType *equation,
                                        bool signedMeasurement) const;
};
