// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <GeometricalAnalysisTools.h>

#include "ecvHObject.h"

class QWidget;

class ccGenericPointCloud;
class ecvProgressDialog;

namespace ccLibAlgorithms {
//! Returns a default first guess for algorithms kernel size (one cloud)
PointCoordinateType GetDefaultCloudKernelSize(ccGenericPointCloud* cloud,
                                              unsigned knn = 12);

//! Returns a default first guess for algorithms kernel size (several clouds)
PointCoordinateType GetDefaultCloudKernelSize(
        const ccHObject::Container& entities, unsigned knn = 12);

/*** cloudViewer "standalone" algorithms ***/

//! Geometric characteristic (with sub option)
struct GeomCharacteristic {
    GeomCharacteristic(
            cloudViewer::GeometricalAnalysisTools::GeomCharacteristic c,
            int option = 0)
        : charac(c), subOption(option) {}

    cloudViewer::GeometricalAnalysisTools::GeomCharacteristic charac;
    int subOption = 0;
};

//! Set of GeomCharacteristic instances
typedef std::vector<GeomCharacteristic> GeomCharacteristicSet;

//! Computes geometrical characteristics (see
//! GeometricalAnalysisTools::GeomCharacteristic) on a set of entities
bool ComputeGeomCharacteristics(const GeomCharacteristicSet& characteristics,
                                PointCoordinateType radius,
                                ccHObject::Container& entities,
                                const CCVector3* roughnessUpDir = nullptr,
                                QWidget* parent = nullptr);

//! Computes a geometrical characteristic (see
//! GeometricalAnalysisTools::GeomCharacteristic) on a set of entities
bool ComputeGeomCharacteristic(
        cloudViewer::GeometricalAnalysisTools::GeomCharacteristic algo,
        int subOption,
        PointCoordinateType radius,
        ccHObject::Container& entities,
        const CCVector3* roughnessUpDir = nullptr,
        QWidget* parent = nullptr,
        ecvProgressDialog* progressDialog = nullptr);

// cloudViewer algorithms handled by the 'ApplyCCLibAlgorithm' method
enum CC_LIB_ALGORITHM {
    CCLIB_ALGO_SF_GRADIENT,
};

//! Applies a standard cloudViewer algorithm (see CC_LIB_ALGORITHM) on a set of
//! entities
bool ApplyCCLibAlgorithm(CC_LIB_ALGORITHM algo,
                         ccHObject::Container& entities,
                         QWidget* parent = 0,
                         void** additionalParameters = 0);

//! Scale matching algorithms
enum ScaleMatchingAlgorithm { BB_MAX_DIM, BB_VOLUME, PCA_MAX_DIM, ICP_SCALE };

//! Applies a standard cloudViewer algorithm (see CC_LIB_ALGORITHM) on a set of
//! entities
bool ApplyScaleMatchingAlgorithm(ScaleMatchingAlgorithm algo,
                                 ccHObject::Container& entities,
                                 double icpRmsDiff,
                                 int icpFinalOverlap,
                                 unsigned refEntityIndex = 0,
                                 QWidget* parent = 0);
}  // namespace ccLibAlgorithms
