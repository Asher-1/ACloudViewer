// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef Q_CANUPO_PROCESS_HEADER
#define Q_CANUPO_PROCESS_HEADER

// Local
#include "ccPointDescriptor.h"

// cloudViewer
#include <GenericIndexedCloudPersist.h>

// Qt
#include <QString>

class ecvMainAppInterface;
class ccPointCloud;
class QWidget;

//! CANUPO process (classify)
class qCanupoProcess {
public:
    //! Classify parameters
    struct ClassifyParams {
        double samplingDist = 0.0;
        int maxThreadCount = 0;
        double confidenceThreshold = 0.0;
        bool useActiveSFForConfidence = true;
        bool generateAdditionalSF = false;
        bool generateRoughnessSF = false;
    };

    //! Classify a point cloud
    static bool Classify(QString classifierFilename,
                         const ClassifyParams& params,
                         ccPointCloud* cloud,
                         cloudViewer::GenericIndexedCloudPersist* corePoints,
                         CorePointDescSet& corePointsDescriptors,
                         ccPointCloud* realCorePoints = nullptr,
                         ecvMainAppInterface* app = nullptr,
                         QWidget* parentWidget = nullptr,
                         bool silent = false);
};

#endif  // Q_CANUPO_PROCESS_HEADER
