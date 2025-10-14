// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef CC_MPLANE_MEASUREMENT
#define CC_MPLANE_MEASUREMENT

// CC
#include "ecv2DLabel.h"
#include "ecvHObject.h"
#include "ecvMainAppInterface.h"
#include "ecvPickingListener.h"
#include "ecvPlane.h"
#include "ecvPointCloud.h"

// Local dependencies
#include "ccMPlanePoint.h"

class ccMeasurementRecorder {
public:
    ccMeasurementRecorder(ccPointCloud *rootCloud, ecvMainAppInterface *app)
        : m_rootCloud(rootCloud), m_app(app) {}

    void loadDataFromSelectedCloud();

    void addFittingPoint(const ccPickingListener::PickedItem &item);
    void deleteFittingPoint(unsigned int index);
    void renameFittingPoint(const QString &newName,
                            unsigned int fittingPointIndex);
    const std::vector<ccMPlanePoint> &getFittingPoints() const;
    unsigned int getActualFittingPointIndex() const;
    unsigned int getFittingPointAmount() const;

    void addMeasurementPoint(const ccPickingListener::PickedItem &item,
                             float distance);
    const std::vector<ccMPlanePoint> &getMeasurementPoints() const;
    bool renameMeasurement(const QString &newName,
                           unsigned int measurementIndex);
    void updateMeasurement(float distance, unsigned int measurementIndex);

    ccPlane *getPlane() const;
    void setPlane(const ccPlane *plane);
    void deletePlane();

private:
    ecvMainAppInterface *m_app = nullptr;
    ccPointCloud *m_rootCloud = nullptr;
    ccPlane *m_plane = nullptr;

    ccHObject *m_rootFolder = nullptr;
    ccHObject *m_fittingPointFolder = nullptr;
    ccHObject *m_measurementFolder = nullptr;

    std::vector<ccMPlanePoint> m_fittingPoints;
    std::vector<ccMPlanePoint> m_measurementPoints;

private:
    void loadFolders();
    void loadFittingPoints();
    void loadMeasurementPoints();
    void loadAndDeleteFittingPlane();
};

#endif
