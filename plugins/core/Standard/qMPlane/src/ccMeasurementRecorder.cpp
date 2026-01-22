// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ccMeasurementRecorder.h"

#include "ccItemTree.h"

// CV_DB_LIB
#include "ecvScalarField.h"

constexpr char const* ROOT_FOLDER_NAME = "MPlane";
constexpr char const* FITTING_POINT_FOLDER_NAME = "Fitting-points";
constexpr char const* MEASUREMENT_FOLDER_NAME = "Measurements";
constexpr char const* MPLANE_PLANE_NAME = "Fitting-plane";
constexpr char const* MPLANE_SCALARFIELD_NAME = "MPlane Distance";

const std::vector<ccMPlanePoint>& ccMeasurementRecorder::getFittingPoints()
        const {
    return m_fittingPoints;
}

const std::vector<ccMPlanePoint>& ccMeasurementRecorder::getMeasurementPoints()
        const {
    return m_measurementPoints;
}

unsigned int ccMeasurementRecorder::getActualFittingPointIndex() const {
    return static_cast<unsigned int>(m_fittingPoints.size());
}

unsigned int ccMeasurementRecorder::getFittingPointAmount() const {
    return static_cast<unsigned int>(m_fittingPoints.size());
}

void ccMeasurementRecorder::loadDataFromSelectedCloud() {
    loadFolders();
    loadFittingPoints();
    loadMeasurementPoints();
    loadAndDeleteFittingPlane();
}

void ccMeasurementRecorder::addFittingPoint(
        const ccPickingListener::PickedItem& item) {
    cc2DLabel* newLabel = CC_ITEM_TREE::createPointLabel2D(
            QString("Point %1").arg(getActualFittingPointIndex()), m_rootCloud,
            item.clickPoint, item.itemIndex);
    m_fittingPointFolder->addChild(newLabel);
    m_app->addToDB(newLabel, false, true, false, false);
    m_fittingPoints.push_back(ccMPlanePoint(newLabel));
}

void ccMeasurementRecorder::addMeasurementPoint(
        const ccPickingListener::PickedItem& item, float distance) {
    QString measurementName = QString("#%1").arg(m_measurementPoints.size());
    cc2DLabel* measureLabel = CC_ITEM_TREE::createPointLabel2D(
            measurementName, m_rootCloud, item.clickPoint, item.itemIndex);
    m_measurementFolder->addChild(measureLabel);
    m_app->addToDB(measureLabel, false, true, false, false);

    m_measurementPoints.push_back(ccMPlanePoint(measureLabel));
    m_measurementPoints.back().setDistance(distance);
}

void ccMeasurementRecorder::loadFolders() {
    m_rootFolder = CC_ITEM_TREE::findOrCreateContainer(m_rootCloud,
                                                       ROOT_FOLDER_NAME, m_app);
    m_fittingPointFolder = CC_ITEM_TREE::findOrCreateContainer(
            m_rootFolder, FITTING_POINT_FOLDER_NAME, m_app);
    m_measurementFolder = CC_ITEM_TREE::findOrCreateContainer(
            m_rootFolder, MEASUREMENT_FOLDER_NAME, m_app);
}

void ccMeasurementRecorder::loadFittingPoints() {
    std::vector<ccHObject*> fittingPointLabels;
    m_fittingPointFolder->filterChildren(fittingPointLabels, false,
                                         CV_TYPES::LABEL_2D);
    std::transform(std::begin(fittingPointLabels), std::end(fittingPointLabels),
                   std::back_inserter(m_fittingPoints), [](ccHObject* item) {
                       return ccMPlanePoint(static_cast<cc2DLabel*>(item));
                   });
}

void ccMeasurementRecorder::loadMeasurementPoints() {
    std::vector<ccHObject*> measurementLabels;
    m_measurementFolder->filterChildren(measurementLabels, false,
                                        CV_TYPES::LABEL_2D);
    std::transform(std::begin(measurementLabels), std::end(measurementLabels),
                   std::back_inserter(m_measurementPoints),
                   [](ccHObject* item) {
                       return ccMPlanePoint(static_cast<cc2DLabel*>(item));
                   });
}

void ccMeasurementRecorder::loadAndDeleteFittingPlane() {
    m_plane = static_cast<ccPlane*>(CC_ITEM_TREE::findItemInHierarchy(
            m_rootFolder, CC_ITEM_TREE::CC_TYPES_MAP::PLANE,
            MPLANE_PLANE_NAME));
    deletePlane();
}

bool ccMeasurementRecorder::renameMeasurement(const QString& newName,
                                              unsigned int measurementIndex) {
    auto element = std::find_if(std::begin(m_measurementPoints),
                                std::end(m_measurementPoints),
                                [&newName](const ccMPlanePoint& point) {
                                    return point.getName() == newName;
                                });

    if (element != std::end(m_measurementPoints)) {
        return false;
    }

    m_measurementPoints[measurementIndex].setName(newName);
    m_measurementPoints[measurementIndex].getLabel()->redrawDisplay();
    return true;
}

void ccMeasurementRecorder::updateMeasurement(float distance,
                                              unsigned int measurementIndex) {
    m_measurementPoints[measurementIndex].setDistance(distance);
}

ccPlane* ccMeasurementRecorder::getPlane() const { return m_plane; }

void ccMeasurementRecorder::setPlane(const ccPlane* plane) {
    deletePlane();
    m_plane = const_cast<ccPlane*>(plane);

    CCVector3 bbMin, bbMax;
    m_rootCloud->getBoundingBox(bbMin, bbMax);
    float length =
            std::max(abs(bbMax.x - bbMin.x),
                     std::max(abs(bbMax.y - bbMin.y), abs(bbMax.z - bbMin.z)));
    m_plane->setXWidth(PointCoordinateType(length));
    m_plane->setYWidth(PointCoordinateType(length));

    m_measurementFolder->addChild(m_plane);
    // m_plane->setDisplay(m_rootCloud->getDisplay());
    // m_plane->prepareDisplayForRefresh_recursive();
    m_app->addToDB(m_plane);
}

void ccMeasurementRecorder::deletePlane() {
    if (m_plane) {
        m_app->removeFromDB(m_plane);
        m_plane = nullptr;
    }
}

void ccMeasurementRecorder::deleteFittingPoint(unsigned int index) {
    m_app->removeFromDB(m_fittingPoints[index].getLabel());
    m_fittingPoints.erase(m_fittingPoints.begin() + index);
}

void ccMeasurementRecorder::renameFittingPoint(const QString& newName,
                                               unsigned int fittingPointIndex) {
    m_fittingPoints[fittingPointIndex].getLabel()->setName(
            QString("Point %1").arg(fittingPointIndex));
}
