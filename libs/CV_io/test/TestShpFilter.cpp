// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "TestShpFilter.h"

#include <ecvPointCloud.h>

#include <QApplication>
#include <QTemporaryDir>
#include <array>
#include <cmath>

#include "FileIOFilter.h"
#include "ShpFilter.h"
#include "ecvHObject.h"
#include "ecvMesh.h"
#include "ecvPolyline.h"

void TestShpFilter::readPolylineFile(const QString &filePath) const {
    ccHObject container;
    FileIOFilter::LoadParameters params;
    params.alwaysDisplayLoadDialog = false;
    ShpFilter filter;
    CC_FILE_ERROR error = filter.loadFile(filePath, container, params);
    ASSERT_TRUE(error == CC_FERR_NO_ERROR);

    ASSERT_TRUE(container.getChildrenNumber() == 2);

    for (unsigned i(0); i < container.getChildrenNumber(); ++i) {
        ccHObject *child = container.getChild(i);
        ASSERT_TRUE(child->getClassID() == CV_TYPES::POLY_LINE);
    }

    auto *firstPolyline = static_cast<ccPolyline *>(container.getChild(0));
    ASSERT_TRUE(!firstPolyline->is2DMode());
    ASSERT_TRUE(!firstPolyline->isClosed());
    auto *vertices = firstPolyline->getAssociatedCloud();
    ASSERT_TRUE(!vertices->isScalarFieldEnabled());
    ASSERT_TRUE(vertices->size() == 5);

    ScalarType expectedXs[5] = {1.0, 5.0, 5.0, 3.0, 1.0};
    ScalarType expectedYs[5] = {5.0, 5.0, 1.0, 3.0, 1.0};
    for (unsigned i = 0; i < 5; ++i) {
        const CCVector3 *point = vertices->getPoint(i);
        EXPECT_DOUBLE_EQ(point->x, expectedXs[i]);
        EXPECT_DOUBLE_EQ(point->y, expectedYs[i]);
        EXPECT_DOUBLE_EQ(point->z, 0.0);
    }

    auto *secondPolyline = static_cast<ccPolyline *>(container.getChild(1));
    ASSERT_TRUE(!secondPolyline->is2DMode());
    ASSERT_TRUE(!firstPolyline->isClosed());
    vertices = secondPolyline->getAssociatedCloud();
    ASSERT_TRUE(!vertices->isScalarFieldEnabled());
    ASSERT_TRUE(vertices->size() == 2);

    ScalarType expectedXs2[2] = {3.0, 2.0};
    ScalarType expectedYs2[2] = {2.0, 6.0};
    for (unsigned i = 0; i < 2; ++i) {
        const CCVector3 *point = vertices->getPoint(i);
        EXPECT_DOUBLE_EQ(point->x, expectedXs2[i]);
        EXPECT_DOUBLE_EQ(point->y, expectedYs2[i]);
        EXPECT_DOUBLE_EQ(point->z, 0.0);
    }
}

void TestShpFilter::readPolylineMFile(const QString &filePath) const {
    ccHObject container;
    FileIOFilter::LoadParameters params;
    params.alwaysDisplayLoadDialog = false;
    ShpFilter filter;
    CC_FILE_ERROR error = filter.loadFile(filePath, container, params);
    ASSERT_TRUE(error == CC_FERR_NO_ERROR);

    ASSERT_TRUE(container.getChildrenNumber() == 2);

    for (unsigned i(0); i < container.getChildrenNumber(); ++i) {
        ccHObject *child = container.getChild(i);
        ASSERT_TRUE(child->getClassID() == CV_TYPES::POLY_LINE);
    }

    auto *firstPolyline = static_cast<ccPolyline *>(container.getChild(0));
    ASSERT_TRUE(!firstPolyline->isClosed());
    ASSERT_TRUE(!firstPolyline->is2DMode());
    auto *vertices = firstPolyline->getAssociatedCloud();
    ASSERT_TRUE(vertices->size() == 5);

    ScalarType expectedXs[5] = {1.0, 5.0, 5.0, 3.0, 1.0};
    ScalarType expectedYs[5] = {5.0, 5.0, 1.0, 3.0, 1.0};
    for (unsigned i = 0; i < 5; ++i) {
        const CCVector3 *point = vertices->getPoint(i);
        EXPECT_DOUBLE_EQ(point->x, expectedXs[i]);
        EXPECT_DOUBLE_EQ(point->y, expectedYs[i]);
        EXPECT_DOUBLE_EQ(point->z, 0.0);
    }

    if (vertices->isScalarFieldEnabled()) {
        EXPECT_DOUBLE_EQ(vertices->getPointScalarValue(0), 0.0);
        ASSERT_TRUE(std::isnan(vertices->getPointScalarValue(1)));
        EXPECT_DOUBLE_EQ(vertices->getPointScalarValue(2), 3.0);
        ASSERT_TRUE(std::isnan(vertices->getPointScalarValue(3)));
        EXPECT_DOUBLE_EQ(vertices->getPointScalarValue(4), 0.0);
    }

    auto *secondPolyline = static_cast<ccPolyline *>(container.getChild(1));
    ASSERT_TRUE(!secondPolyline->isClosed());
    ASSERT_TRUE(!secondPolyline->is2DMode());
    vertices = secondPolyline->getAssociatedCloud();
    ASSERT_TRUE(vertices->size() == 2);
    ASSERT_TRUE(!vertices->isScalarFieldEnabled());  // All values are nan, so
                                                     // no scalarfield created

    ScalarType expectedXs2[2] = {3.0, 2.0};
    ScalarType expectedYs2[2] = {2.0, 6.0};
    for (unsigned i = 0; i < 2; ++i) {
        const CCVector3 *point = vertices->getPoint(i);
        EXPECT_DOUBLE_EQ(point->x, expectedXs2[i]);
        EXPECT_DOUBLE_EQ(point->y, expectedYs2[i]);
        EXPECT_DOUBLE_EQ(point->z, 0.0);
    }
}

void TestShpFilter::readPolylineZFile(const QString &filePath) const {
    ccHObject container;
    FileIOFilter::LoadParameters params;
    params.alwaysDisplayLoadDialog = false;
    ShpFilter filter;
    CC_FILE_ERROR error = filter.loadFile(filePath, container, params);
    ASSERT_TRUE(error == CC_FERR_NO_ERROR);

    ASSERT_TRUE(container.getChildrenNumber() == 3);
    for (unsigned i(0); i < container.getChildrenNumber(); ++i) {
        ccHObject *child = container.getChild(i);
        ASSERT_TRUE(child->getClassID() == CV_TYPES::POLY_LINE);
    }

    // 1st part of the polyline
    auto *firstPolyline = static_cast<ccPolyline *>(container.getChild(0));
    ASSERT_TRUE(!firstPolyline->isClosed());
    ASSERT_TRUE(!firstPolyline->is2DMode());
    auto *vertices = firstPolyline->getAssociatedCloud();
    ASSERT_TRUE(vertices->size() == 5);
    ASSERT_TRUE(!vertices->isScalarFieldEnabled());  // All values are nan, so
                                                     // no scalarfield created

    // 2nd part of the polyline
    auto *secondPolyline = static_cast<ccPolyline *>(container.getChild(1));
    ASSERT_TRUE(!secondPolyline->isClosed());
    ASSERT_TRUE(!secondPolyline->is2DMode());
    vertices = secondPolyline->getAssociatedCloud();
    ASSERT_TRUE(vertices->size() == 2);
    ASSERT_TRUE(!vertices->isScalarFieldEnabled());  // All values are nan, so
                                                     // no scalarfield created

    // 3rd part of the polyline
    auto *thirdPolyline = static_cast<ccPolyline *>(container.getChild(2));
    ASSERT_TRUE(!thirdPolyline->isClosed());
    ASSERT_TRUE(!thirdPolyline->is2DMode());
    vertices = thirdPolyline->getAssociatedCloud();
    ASSERT_TRUE(vertices->size() == 3);

    if (vertices->isScalarFieldEnabled()) {
        EXPECT_DOUBLE_EQ(vertices->getPointScalarValue(0), 0.0);
        EXPECT_DOUBLE_EQ(vertices->getPointScalarValue(1), 3.0);
        EXPECT_DOUBLE_EQ(vertices->getPointScalarValue(2), 2.0);
    }
}

void TestShpFilter::readMultiPointFile(const QString &filePath) const {
    ccHObject container;
    FileIOFilter::LoadParameters params;
    params.alwaysDisplayLoadDialog = false;
    ShpFilter filter;

    CC_FILE_ERROR error = filter.loadFile(filePath, container, params);
    ASSERT_TRUE(error == CC_FERR_NO_ERROR);

    ASSERT_TRUE(container.getChildrenNumber() == 1);
    ASSERT_TRUE(container.getChild(0)->getClassID() == CV_TYPES::POINT_CLOUD);

    auto *cloud = static_cast<ccPointCloud *>(container.getChild(0));
    ASSERT_TRUE(cloud->size() == 2);
    ASSERT_TRUE(!cloud->isScalarFieldEnabled());

    const CCVector3 *point = cloud->getPoint(0);
    EXPECT_DOUBLE_EQ(point->x, 122.0);
    EXPECT_DOUBLE_EQ(point->y, 37.0);
    EXPECT_DOUBLE_EQ(point->z, 0.0);

    point = cloud->getPoint(1);
    EXPECT_DOUBLE_EQ(point->x, 124.0);
    EXPECT_DOUBLE_EQ(point->y, 32.0);
    EXPECT_DOUBLE_EQ(point->z, 0.0);
}

void TestShpFilter::readMultiPointZFile(const QString &filePath) const {
    CCVector3d bbMin(1422671.7232666016, 4188903.4295959473, 71.99445343017578);
    CCVector3d shift = ecvGlobalShiftManager::BestShift(bbMin);
    bool shiftEnabled = true;

    ccHObject container;
    FileIOFilter::LoadParameters params;
    params.alwaysDisplayLoadDialog = false;
    params.shiftHandlingMode = ecvGlobalShiftManager::Mode::NO_DIALOG;
    params.coordinatesShiftEnabled = &shiftEnabled;
    params.coordinatesShift = &shift;
    params.preserveShiftOnSave = true;
    ShpFilter filter;

    CC_FILE_ERROR error = filter.loadFile(filePath, container, params);
    ASSERT_TRUE(error == CC_FERR_NO_ERROR);

    ASSERT_TRUE(container.getChildrenNumber() == 1);
    ccHObject *item = container.getFirstChild();
    ASSERT_TRUE(item->getClassID() == CV_TYPES::POINT_CLOUD);

    auto *pointCloud = static_cast<ccPointCloud *>(item);
    ASSERT_TRUE(pointCloud->size() == 4);
    ASSERT_TRUE(!pointCloud->isScalarFieldEnabled());

    const CCVector3 *point;
    CCVector3d pg;

    point = pointCloud->getPoint(0);
    pg = pointCloud->toGlobal3d(*point);
    EXPECT_DOUBLE_EQ(pg.x, 1422671.7232666016);
    EXPECT_DOUBLE_EQ(pg.y, 4188903.4295959473);
    EXPECT_DOUBLE_EQ(pg.z, 72.00995635986328);

    point = pointCloud->getPoint(1);
    pg = pointCloud->toGlobal3d(*point);
    EXPECT_DOUBLE_EQ(pg.x, 1422672.1022949219);
    EXPECT_DOUBLE_EQ(pg.y, 4188903.4295959473);
    EXPECT_DOUBLE_EQ(pg.z, 72.0060806274414);

    point = pointCloud->getPoint(2);
    pg = pointCloud->toGlobal3d(*point);
    EXPECT_DOUBLE_EQ(pg.x, 1422671.9127807617);
    EXPECT_DOUBLE_EQ(pg.y, 4188903.7578430176);
    EXPECT_DOUBLE_EQ(pg.z, 72.00220489501953);

    point = pointCloud->getPoint(3);
    pg = pointCloud->toGlobal3d(*point);
    EXPECT_DOUBLE_EQ(pg.x, 1422671.9127807617);
    EXPECT_DOUBLE_EQ(pg.y, 4188903.539001465);
    EXPECT_DOUBLE_EQ(pg.z, 71.99445343017578);
}

void TestShpFilter::readMultipatchFile(const QString &filePath) const {
    ccHObject container;
    FileIOFilter::LoadParameters params;
    params.alwaysDisplayLoadDialog = false;
    ShpFilter filter;

    CC_FILE_ERROR error = filter.loadFile(filePath, container, params);
    ASSERT_TRUE(error == CC_FERR_NO_ERROR);

    ASSERT_TRUE(container.getChildrenNumber() == 2);

    for (unsigned i(0); i < container.getChildrenNumber(); ++i) {
        ASSERT_TRUE(container.getChild(i)->getClassID() == CV_TYPES::MESH);
    }

    constexpr unsigned expectedNumPoints = 10;
    auto *firstMesh = static_cast<ccMesh *>(container.getChild(0));
    auto *vertices = firstMesh->getAssociatedCloud();
    ASSERT_TRUE(vertices->size() == expectedNumPoints);
    ScalarType expectedXs[expectedNumPoints] = {0.0, 0.0, 5.0, 5.0, 5.0,
                                                5.0, 0.0, 0.0, 0.0, 0.0};
    ScalarType expectedYs[expectedNumPoints] = {0.0, 0.0, 0.0, 0.0, 5.0,
                                                5.0, 5.0, 5.0, 0.0, 0.0};
    ScalarType expectedZs[expectedNumPoints] = {0.0, 3.0, 0.0, 3.0, 0.0,
                                                3.0, 0.0, 3.0, 0.0, 3.0};
    for (unsigned i = 0; i < expectedNumPoints; ++i) {
        const CCVector3 *point = vertices->getPoint(i);
        EXPECT_DOUBLE_EQ(point->x, expectedXs[i]);
        EXPECT_DOUBLE_EQ(point->y, expectedYs[i]);
        EXPECT_DOUBLE_EQ(point->z, expectedZs[i]);
    }

    constexpr unsigned expectedNumPoints2 = 6;
    auto *secondMesh = static_cast<ccMesh *>(container.getChild(1));
    vertices = secondMesh->getAssociatedCloud();
    ASSERT_TRUE(vertices->size() == 6);
    ScalarType expectedXs2[expectedNumPoints2] = {2.5, 0.0, 5.0, 5.0, 0.0, 0.0};
    ScalarType expectedYs2[expectedNumPoints2] = {2.5, 0.0, 0.0, 5.0, 5.0, 0.0};
    ScalarType expectedZs2[expectedNumPoints2] = {5.0, 3.0, 3.0, 3.0, 3.0, 3.0};
    for (unsigned i = 0; i < expectedNumPoints2; ++i) {
        const CCVector3 *point = vertices->getPoint(i);
        EXPECT_DOUBLE_EQ(point->x, expectedXs2[i]);
        EXPECT_DOUBLE_EQ(point->y, expectedYs2[i]);
        EXPECT_DOUBLE_EQ(point->z, expectedZs2[i]);
    }
}

void TestShpFilter::readPolygonFile(const QString &filePath) const {
    const unsigned expectedNumPoints =
            14;  // File has 15 points but as its a polygon, CC will keep the 14
                 // first pts
    CCVector3d bbMin(-626146.0444521683, 5219675.646154184, 0);
    CCVector3d shift = ecvGlobalShiftManager::BestShift(bbMin);
    bool shiftEnabled = true;

    ccHObject container;
    FileIOFilter::LoadParameters params;
    params.alwaysDisplayLoadDialog = false;
    params.shiftHandlingMode = ecvGlobalShiftManager::Mode::NO_DIALOG;
    params.coordinatesShiftEnabled = &shiftEnabled;
    params.coordinatesShift = &shift;
    params.preserveShiftOnSave = true;
    ShpFilter filter;

    CC_FILE_ERROR error = filter.loadFile(filePath, container, params);
    ASSERT_TRUE(error == CC_FERR_NO_ERROR);

    ASSERT_TRUE(container.getChildrenNumber() == 1);
    ccHObject *item = container.getFirstChild();
    ASSERT_TRUE(item->getClassID() == CV_TYPES::POLY_LINE);

    auto *poly = static_cast<ccPolyline *>(item);
    ASSERT_TRUE(poly->size() == expectedNumPoints);
    ASSERT_TRUE(!poly->isScalarFieldEnabled());
    ASSERT_TRUE(!poly->is2DMode());
    ASSERT_TRUE(poly->isClosed());

    cloudViewer::GenericIndexedCloudPersist *vertices =
            poly->getAssociatedCloud();
    ASSERT_TRUE(vertices->size() == expectedNumPoints);

    std::array<double, 14> expectedXs{
            -626146.0444521683, -187004.53123683017, -59884.61951660062,
            169316.43343351,    180872.78904444003,  300288.4636907161,
            914701.3703384919,  752912.3917854726,   880032.303505702,
            749060.273248496,   473633.79785466543,  375404.77516176086,
            -212043.3017271784, -187004.53123683017};
    std::array<double, 14> expectedYs{
            6182705.280398346, 6409980.274079968, 6383015.444321131,
            6488948.704087989, 6606438.319465778, 6650737.682641009,
            6236634.939916019, 5878387.91597719,  5391094.921049644,
            5271679.246403368, 5352573.735679878, 5219675.646154184,
            5348721.617142901, 5789789.189626727};

    // SHP coordinates are shifted in double precision and stored as
    // PointCoordinateType; scalar-field precision is a separate build option.
    for (unsigned i(0); i < expectedNumPoints; ++i) {
        const CCVector3 *p = vertices->getPoint(i);
        EXPECT_EQ(p->x,
                  static_cast<PointCoordinateType>(expectedXs[i] + shift.x));
        EXPECT_EQ(p->y,
                  static_cast<PointCoordinateType>(expectedYs[i] + shift.y));
        EXPECT_EQ(p->z, PointCoordinateType(0));
    }
}

void TestShpFilter::readPolygonZFile(const QString &filePath) const {
    CCVector3d bbMin(1422671.7232666016, 4188903.4295959473, 71.99445343017578);
    CCVector3d shift = ecvGlobalShiftManager::BestShift(bbMin);
    bool shiftEnabled = true;

    ccHObject container;
    FileIOFilter::LoadParameters params;
    params.alwaysDisplayLoadDialog = false;
    params.shiftHandlingMode = ecvGlobalShiftManager::Mode::NO_DIALOG;
    params.coordinatesShiftEnabled = &shiftEnabled;
    params.coordinatesShift = &shift;
    params.preserveShiftOnSave = true;
    ShpFilter filter;

    CC_FILE_ERROR error = filter.loadFile(filePath, container, params);
    ASSERT_TRUE(error == CC_FERR_NO_ERROR);

    ASSERT_TRUE(container.getChildrenNumber() == 1);
    ASSERT_TRUE(container.getFirstChild()->getClassID() == CV_TYPES::POLY_LINE);

    auto *polygon = static_cast<ccPolyline *>(container.getFirstChild());
    ASSERT_TRUE(polygon->size() ==
                72);  // File has 73 points, but cc reads 72 as its a polygon
    ASSERT_TRUE(!polygon->is2DMode());
    ASSERT_TRUE(polygon->isClosed());
}

void TestShpFilter::testWritePolyline() const {
    ccHObject container;
    FileIOFilter::LoadParameters params;
    params.alwaysDisplayLoadDialog = false;
    ShpFilter filter;
    CC_FILE_ERROR error = filter.loadFile(LINE_FILE, container, params);
    ASSERT_TRUE(error == CC_FERR_NO_ERROR);

    QTemporaryDir tmpDir;
    ASSERT_TRUE(tmpDir.isValid());
    QString tmpLineM = tmpDir.filePath("line.shp");
    FileIOFilter::SaveParameters saveParams;
    saveParams.alwaysDisplaySaveDialog = false;
    filter.save3DPolyAs2D(true);
    filter.save3DPolyHeightInDBF(false);

    error = filter.saveToFile(&container, tmpLineM, saveParams);
    ASSERT_TRUE(error == CC_FERR_NO_ERROR);

    readPolylineFile(tmpLineM);
}

void TestShpFilter::testWritePolylineM() const {
    ccHObject container;
    FileIOFilter::LoadParameters params;
    params.alwaysDisplayLoadDialog = false;
    ShpFilter filter;
    CC_FILE_ERROR error = filter.loadFile(LINEM_FILE, container, params);
    ASSERT_TRUE(error == CC_FERR_NO_ERROR);

    QTemporaryDir tmpDir;
    ASSERT_TRUE(tmpDir.isValid());
    QString tmpLineM = tmpDir.filePath("linem.shp");
    FileIOFilter::SaveParameters saveParams;
    saveParams.alwaysDisplaySaveDialog = false;

    error = filter.saveToFile(&container, tmpLineM, saveParams);
    ASSERT_TRUE(error == CC_FERR_NO_ERROR);

    readPolylineMFile(tmpLineM);
}

void TestShpFilter::testWritePolylineZ() const {
    ccHObject container;
    FileIOFilter::LoadParameters params;
    params.alwaysDisplayLoadDialog = false;
    ShpFilter filter;
    CC_FILE_ERROR error = filter.loadFile(LINEZ_FILE, container, params);
    ASSERT_TRUE(error == CC_FERR_NO_ERROR);

    QTemporaryDir tmpDir;
    ASSERT_TRUE(tmpDir.isValid());
    QString tmpLineM = tmpDir.filePath("linez.shp");
    FileIOFilter::SaveParameters saveParams;
    saveParams.alwaysDisplaySaveDialog = false;

    error = filter.saveToFile(&container, tmpLineM, saveParams);
    ASSERT_TRUE(error == CC_FERR_NO_ERROR);

    readPolylineZFile(tmpLineM);
}

void TestShpFilter::testWriteMultpatchFile() const {
    ccHObject container;
    FileIOFilter::LoadParameters params;
    params.alwaysDisplayLoadDialog = false;
    ShpFilter filter;
    CC_FILE_ERROR error = filter.loadFile(MULTIPATCH_FILE, container, params);
    ASSERT_TRUE(error == CC_FERR_NO_ERROR);

    QTemporaryDir tmpDir;
    ASSERT_TRUE(tmpDir.isValid());
    QString tmpMultipatch = tmpDir.filePath("line.shp");
    FileIOFilter::SaveParameters saveParams;
    saveParams.alwaysDisplaySaveDialog = false;
    filter.save3DPolyAs2D(true);

    error = filter.saveToFile(&container, tmpMultipatch, saveParams);
    ASSERT_TRUE(error == CC_FERR_NO_ERROR);
    readMultipatchFile(tmpMultipatch);
}

void TestShpFilter::testWriteMultiPointFile() const {
    ccHObject container;
    FileIOFilter::LoadParameters params;
    params.alwaysDisplayLoadDialog = false;
    ShpFilter filter;

    CC_FILE_ERROR error = filter.loadFile(MULTIPOINT_FILE, container, params);
    ASSERT_TRUE(error == CC_FERR_NO_ERROR);

    QTemporaryDir tmpDir;
    ASSERT_TRUE(tmpDir.isValid());
    QString tmpMultipatch = tmpDir.filePath("multipoint.shp");
    FileIOFilter::SaveParameters saveParams;
    saveParams.alwaysDisplaySaveDialog = false;

    error = filter.saveToFile(&container, tmpMultipatch, saveParams);
    ASSERT_TRUE(error == CC_FERR_NO_ERROR);
    readMultiPointFile(tmpMultipatch);
}

void TestShpFilter::testWriteMultiPointZFile() const {
    CCVector3d bbMin(1422671.7232666016, 4188903.4295959473, 71.99445343017578);
    CCVector3d shift = ecvGlobalShiftManager::BestShift(bbMin);
    bool shiftEnabled = true;

    ccHObject container;
    FileIOFilter::LoadParameters params;
    params.alwaysDisplayLoadDialog = false;
    params.shiftHandlingMode = ecvGlobalShiftManager::Mode::NO_DIALOG;
    params.coordinatesShiftEnabled = &shiftEnabled;
    params.coordinatesShift = &shift;
    params.preserveShiftOnSave = true;
    ShpFilter filter;

    CC_FILE_ERROR error = filter.loadFile(MULTIPOINT_Z, container, params);
    ASSERT_TRUE(error == CC_FERR_NO_ERROR);

    QTemporaryDir tmpDir;
    ASSERT_TRUE(tmpDir.isValid());
    QString tmpMultipatch = tmpDir.filePath("multi_pointz.shp");
    FileIOFilter::SaveParameters saveParams;
    saveParams.alwaysDisplaySaveDialog = false;

    error = filter.saveToFile(&container, tmpMultipatch, saveParams);
    ASSERT_TRUE(error == CC_FERR_NO_ERROR);
    readMultiPointZFile(tmpMultipatch);
}

void TestShpFilter::testWritePolygonFile() const {
    CCVector3d bbMin(-626146.0444521683, 5219675.646154184, 0);
    CCVector3d shift = ecvGlobalShiftManager::BestShift(bbMin);
    bool shiftEnabled = true;

    ccHObject container;
    FileIOFilter::LoadParameters params;
    params.alwaysDisplayLoadDialog = false;
    params.shiftHandlingMode = ecvGlobalShiftManager::Mode::NO_DIALOG;
    params.coordinatesShiftEnabled = &shiftEnabled;
    params.coordinatesShift = &shift;
    params.preserveShiftOnSave = true;
    ShpFilter filter;

    CC_FILE_ERROR error = filter.loadFile(POLYGON_FILE, container, params);
    ASSERT_TRUE(error == CC_FERR_NO_ERROR);

    QTemporaryDir tmpDir;
    ASSERT_TRUE(tmpDir.isValid());
    QString tmpMultipatch = tmpDir.filePath("polygon.shp");
    FileIOFilter::SaveParameters saveParams;
    saveParams.alwaysDisplaySaveDialog = false;

    filter.save3DPolyAs2D(true);
    filter.treatClosedPolylinesAsPolygons(true);
    error = filter.saveToFile(&container, tmpMultipatch, saveParams);

    ASSERT_TRUE(error == CC_FERR_NO_ERROR);
    readPolygonFile(tmpMultipatch);
}

void TestShpFilter::testWritePolygonZFile() const {
    CCVector3d bbMin(1422671.7232666016, 4188903.4295959473, 71.99445343017578);
    CCVector3d shift = ecvGlobalShiftManager::BestShift(bbMin);
    bool shiftEnabled = true;

    ccHObject container;
    FileIOFilter::LoadParameters params;
    params.alwaysDisplayLoadDialog = false;
    params.shiftHandlingMode = ecvGlobalShiftManager::Mode::NO_DIALOG;
    params.coordinatesShiftEnabled = &shiftEnabled;
    params.coordinatesShift = &shift;
    params.preserveShiftOnSave = true;
    ShpFilter filter;

    CC_FILE_ERROR error = filter.loadFile(POLYGONZ_FILE, container, params);
    ASSERT_TRUE(error == CC_FERR_NO_ERROR);

    QTemporaryDir tmpDir;
    ASSERT_TRUE(tmpDir.isValid());
    QString tmpMultipatch = tmpDir.filePath("polygon_z.shp");
    FileIOFilter::SaveParameters saveParams;
    saveParams.alwaysDisplaySaveDialog = false;

    filter.save3DPolyAs2D(false);
    filter.treatClosedPolylinesAsPolygons(true);
    error = filter.saveToFile(&container, tmpMultipatch, saveParams);
    ASSERT_TRUE(error == CC_FERR_NO_ERROR);
    readPolygonZFile(tmpMultipatch);
}

void TestShpFilter::readSinglePointZFile(const QString &filePath) const {
    CCVector3d bbMin(1422459.0908050265, 4188942.211755641, 0.0);
    CCVector3d shift = ecvGlobalShiftManager::BestShift(bbMin);
    bool shiftEnabled = true;

    ccHObject container;
    FileIOFilter::LoadParameters params;
    params.alwaysDisplayLoadDialog = false;
    params.shiftHandlingMode = ecvGlobalShiftManager::Mode::NO_DIALOG;
    params.coordinatesShiftEnabled = &shiftEnabled;
    params.coordinatesShift = &shift;
    params.preserveShiftOnSave = true;
    ShpFilter filter;

    CC_FILE_ERROR error = filter.loadFile(filePath, container, params);
    ASSERT_TRUE(error == CC_FERR_NO_ERROR);

    ASSERT_TRUE(container.getChildrenNumber() == 1);
    ASSERT_TRUE(container.getFirstChild()->getClassID() ==
                CV_TYPES::POINT_CLOUD);

    unsigned expectedNumPoints = 2;
    auto *cloud = static_cast<ccPointCloud *>(container.getFirstChild());
    ASSERT_TRUE(cloud->size() == expectedNumPoints);
    ASSERT_TRUE(!cloud->isScalarFieldEnabled());

    std::array<double, 2> expectedXs{1422464.3681007193, 1422459.0908050265};
    std::array<double, 2> expectedYs{4188962.3364355816, 4188942.211755641};
    std::array<double, 2> expectedZs{72.40956470558095, 72.58286959604922};
    for (unsigned i(0); i < expectedNumPoints; ++i) {
        const CCVector3 *p = cloud->getPoint(i);
        EXPECT_EQ(p->x,
                  static_cast<PointCoordinateType>(expectedXs[i] + shift.x));
        EXPECT_EQ(p->y,
                  static_cast<PointCoordinateType>(expectedYs[i] + shift.y));
        EXPECT_EQ(p->z,
                  static_cast<PointCoordinateType>(expectedZs[i] + shift.z));
    }
}

TEST_F(TestShpFilter, ReadPolyline) { readPolylineFile(); }
TEST_F(TestShpFilter, ReadPolylineM) { readPolylineMFile(); }
TEST_F(TestShpFilter, ReadPolylineZ) { readPolylineZFile(); }
TEST_F(TestShpFilter, ReadMultiPoint) { readMultiPointFile(); }
TEST_F(TestShpFilter, ReadMultiPointZ) { readMultiPointZFile(); }
TEST_F(TestShpFilter, ReadMultipatch) { readMultipatchFile(); }
TEST_F(TestShpFilter, ReadPolygon) { readPolygonFile(); }
TEST_F(TestShpFilter, ReadPolygonZ) { readPolygonZFile(); }
TEST_F(TestShpFilter, ReadSinglePointZ) { readSinglePointZFile(); }
TEST_F(TestShpFilter, WritePolyline) { testWritePolyline(); }
TEST_F(TestShpFilter, WritePolylineM) { testWritePolylineM(); }
TEST_F(TestShpFilter, WritePolylineZ) { testWritePolylineZ(); }
TEST_F(TestShpFilter, WriteMultiPoint) { testWriteMultiPointFile(); }
TEST_F(TestShpFilter, WriteMultiPointZ) { testWriteMultiPointZFile(); }
TEST_F(TestShpFilter, WriteMultipatch) { testWriteMultpatchFile(); }
TEST_F(TestShpFilter, WritePolygon) { testWritePolygonFile(); }
TEST_F(TestShpFilter, WritePolygonZ) { testWritePolygonZFile(); }

int main(int argc, char **argv) {
    QApplication application(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
