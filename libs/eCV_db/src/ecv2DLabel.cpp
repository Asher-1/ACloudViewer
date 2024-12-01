// ##########################################################################
// #                                                                        #
// #                              CLOUDVIEWER                               #
// #                                                                        #
// #  This program is free software; you can redistribute it and/or modify  #
// #  it under the terms of the GNU General Public License as published by  #
// #  the Free Software Foundation; version 2 or later of the License.      #
// #                                                                        #
// #  This program is distributed in the hope that it will be useful,       #
// #  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
// #  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          #
// #  GNU General Public License for more details.                          #
// #                                                                        #
// #          COPYRIGHT: EDF R&D / DAHAI LU                                 #
// #                                                                        #
// ##########################################################################

// Local
#include "ecv2DLabel.h"

#include "ecvBasicTypes.h"
#include "ecvDisplayTools.h"
#include "ecvFacet.h"
#include "ecvGenericPointCloud.h"
#include "ecvHObjectCaster.h"
#include "ecvPointCloud.h"
#include "ecvPolyline.h"
#include "ecvScalarField.h"
#include "ecvSphere.h"

// Qt
#include <QSharedPointer>

// System
#include <assert.h>
#include <string.h>

//'Delta' character
// static const QChar MathSymbolDelta(0x0394);
static const QString MathSymbolDelta = "D";
static const QString SEPARATOR = "-";

// unit point marker
static QSharedPointer<ccSphere> c_unitPointMarker(nullptr);
static QSharedPointer<ccFacet> c_unitTriMarker(nullptr);

static const QString CENTER_STRING = QObject::tr("Center");
static const char POINT_INDEX_0[] = "pi0";
static const char POINT_INDEX_1[] = "pi1";
static const char POINT_INDEX_2[] = "pi2";
static const char ENTITY_INDEX_0[] = "ei0";
static const char ENTITY_INDEX_1[] = "ei1";
static const char ENTITY_INDEX_2[] = "ei2";

QString cc2DLabel::PickedPoint::itemTitle() const {
    if (entityCenterPoint) {
        QString title = CENTER_STRING;
        if (entity()) title += QString("@%1").arg(entity()->getUniqueID());
        return title;
    } else {
        return QString::number(index, 10);
    }
}

QString cc2DLabel::PickedPoint::prefix(const char* pointTag) const {
    if (entityCenterPoint) {
        return CENTER_STRING;
    } else if (cloud) {
        return QString("Point #") + pointTag;
    } else if (mesh) {
        return QString("Point@Tri#") + pointTag;
    }

    assert(false);
    return QString();
}

CCVector3 cc2DLabel::PickedPoint::getPointPosition() const {
    CCVector3 P;

    if (cloud) {
        if (entityCenterPoint) {
            return cloud->getOwnBB().getCenter();
        } else {
            P = *cloud->getPointPersistentPtr(index);
        }
    } else if (mesh) {
        if (entityCenterPoint) {
            return mesh->getOwnBB().getCenter();
        } else {
            mesh->computePointPosition(index, uv, P);
        }
    } else {
        assert(false);
    }

    return P;
}

unsigned cc2DLabel::PickedPoint::getUniqueID() const {
    if (cloud) return cloud->getUniqueID();
    if (mesh) return mesh->getUniqueID();

    assert(false);
    return 0;
}

ccGenericPointCloud* cc2DLabel::PickedPoint::cloudOrVertices() const {
    if (cloud) return cloud;
    if (mesh) return mesh->getAssociatedCloud();

    assert(false);
    return nullptr;
}

ccHObject* cc2DLabel::PickedPoint::entity() const {
    if (cloud) return cloud;
    if (mesh) return mesh;

    assert(false);
    return nullptr;
}

cc2DLabel::cc2DLabel(QString name /*=QString()*/)
    : ccHObject(name.isEmpty() ? "label" : name),
      m_showFullBody(true),
      m_dispPointsLegend(false),
      m_dispIn2D(true),
      m_relMarkerScale(1.0f),
      m_historyMessage(QStringList()) {
    m_screenPos[0] = m_screenPos[1] = 0.05f;

    clear(false);

    m_lineID = "labelLine-" + this->getViewId();
    if (c_unitPointMarker) {
        m_sphereIdfix = SEPARATOR + this->getViewId() + SEPARATOR +
                        c_unitPointMarker->getViewId();
    }

    if (c_unitTriMarker) {
        m_surfaceIdfix = this->getViewId() + SEPARATOR +
                         c_unitTriMarker->getPolygon()->getViewId();
        m_contourIdfix = this->getViewId() + SEPARATOR +
                         c_unitTriMarker->getContour()->getViewId();
    }

    lockVisibility(false);
    setEnabled(true);
}

QString cc2DLabel::GetSFValueAsString(const LabelInfo1& info, int precision) {
    if (info.hasSF) {
        if (!ccScalarField::ValidValue(info.sfValue)) {
            return "NaN";
        } else {
            QString sfVal = QString::number(info.sfValue, 'f', precision);
            if (info.sfValueIsShifted) {
                sfVal = QString::number(info.sfShiftedValue, 'f', precision) +
                        QString(" (shifted: %1)").arg(sfVal);
            }
            return sfVal;
        }
    } else {
        return QString();
    }
}

QString cc2DLabel::getTitle(int precision) const {
    QString title;
    size_t count = m_pickedPoints.size();
    if (count == 1) {
        title = m_name;
        title.replace(POINT_INDEX_0, QString::number(m_pickedPoints[0].index));

        // if available, we display the point SF value
        LabelInfo1 info;
        getLabelInfo1(info);
        if (info.hasSF) {
            QString sfVal = GetSFValueAsString(info, precision);
            title = QString("%1 = %2").arg(info.sfName, sfVal);
        }
    } else if (count == 2) {
        LabelInfo2 info;
        getLabelInfo2(info);
        // display distance by default
        double dist = info.diff.normd();
        title = QString("Distance: %1").arg(dist, 0, 'f', precision);
    } else if (count == 3) {
        LabelInfo3 info;
        getLabelInfo3(info);
        // display area by default
        title = QString("Area: %1").arg(info.area, 0, 'f', precision);
    }

    return title;
}

QString cc2DLabel::getName() const {
    QString processedName = m_name;

    size_t count = m_pickedPoints.size();
    if (count > 0) {
        processedName.replace(POINT_INDEX_0,
                              QString::number(m_pickedPoints[0].index));
        if (count > 1) {
            processedName.replace(POINT_INDEX_1,
                                  QString::number(m_pickedPoints[1].index));
            if (m_pickedPoints[0].cloud)
                processedName.replace(ENTITY_INDEX_0,
                                      m_pickedPoints[0].cloud->getViewId());
            if (m_pickedPoints[1].cloud)
                processedName.replace(ENTITY_INDEX_1,
                                      m_pickedPoints[1].cloud->getViewId());
            if (count > 2) {
                processedName.replace(POINT_INDEX_2,
                                      QString::number(m_pickedPoints[2].index));
                if (m_pickedPoints[2].cloud)
                    processedName.replace(ENTITY_INDEX_2,
                                          m_pickedPoints[2].cloud->getViewId());
            }
        }
    }

    return processedName;
}

void cc2DLabel::setPosition(float x, float y) {
    m_screenPos[0] = x;
    m_screenPos[1] = y;
}

bool cc2DLabel::move2D(
        int x, int y, int dx, int dy, int screenWidth, int screenHeight) {
    assert(screenHeight > 0 && screenWidth > 0);

    m_screenPos[0] += static_cast<float>(dx) / screenWidth;
    m_screenPos[1] += static_cast<float>(dy) / screenHeight;

    return true;
}

void cc2DLabel::clear(bool ignoreDependencies, bool ignoreCaption) {
    // clear history
    clearLabel(ignoreCaption);

    if (ignoreDependencies) {
        m_pickedPoints.resize(0);
    } else {
        // remove all dependencies first!
        while (!m_pickedPoints.empty()) {
            m_pickedPoints.back().cloud->removeDependencyWith(this);
            m_pickedPoints.pop_back();
        }
    }

    m_lastScreenPos[0] = m_lastScreenPos[1] = -1;
    m_labelROI = QRect(0, 0, 0, 0);
    setVisible(false);
    setName("Label");

    ecvDisplayTools::UpdateScreen();
}

void cc2DLabel::clear3Dviews() {
    ecvDisplayTools::RemoveWidgets(
            WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_LINE_3D, m_lineID));

    if (c_unitPointMarker) {
        for (int i = 0; i < 3; ++i) {
            // ecvDisplayTools::RemoveWidgets(
            //	WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_POLYGONMESH,
            //		QString::number(i) + m_sphereIdfix));
            ecvDisplayTools::RemoveWidgets(
                    WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_SPHERE,
                                      QString::number(i) + m_sphereIdfix));
        }
    }

    if (c_unitTriMarker) {
        ecvDisplayTools::RemoveWidgets(WIDGETS_PARAMETER(
                WIDGETS_TYPE::WIDGET_POLYLINE, m_contourIdfix));
        ecvDisplayTools::RemoveWidgets(WIDGETS_PARAMETER(
                WIDGETS_TYPE::WIDGET_POLYGONMESH, m_surfaceIdfix));
    }
}

void cc2DLabel::clear2Dviews() {
    if (!m_historyMessage.isEmpty()) {
        for (const QString& text : m_historyMessage) {
            // no more valid? we delete the message
            ecvDisplayTools::RemoveWidgets(
                    WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_T2D, text));
            ecvDisplayTools::RemoveWidgets(
                    WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_RECTANGLE_2D, text));
        }
        m_historyMessage.clear();
    }

    ecvDisplayTools::RemoveWidgets(
            WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_T2D, this->getViewId()));
    ecvDisplayTools::RemoveWidgets(WIDGETS_PARAMETER(
            WIDGETS_TYPE::WIDGET_RECTANGLE_2D, this->getViewId()));
}

void cc2DLabel::clearLabel(bool ignoreCaption) {
    clear3Dviews();
    clear2Dviews();
    if (!ignoreCaption) {
        ecvDisplayTools::RemoveWidgets(WIDGETS_PARAMETER(
                WIDGETS_TYPE::WIDGET_CAPTION, this->getViewId()));
    }
}

void cc2DLabel::updateLabel() {
    CC_DRAW_CONTEXT context;
    ecvDisplayTools::GetContext(context);
    update3DLabelView(context, false);
    update2DLabelView(context, false);
    ecvDisplayTools::UpdateScreen();
}

void cc2DLabel::update3DLabelView(CC_DRAW_CONTEXT& context,
                                  bool updateScreen /* = true */) {
    context.drawingFlags = CC_DRAW_3D | CC_DRAW_FOREGROUND;
    drawMeOnly3D(context);
    if (updateScreen) {
        ecvDisplayTools::UpdateScreen();
    }
}

void cc2DLabel::update2DLabelView(CC_DRAW_CONTEXT& context,
                                  bool updateScreen /* = true */) {
    context.drawingFlags = CC_DRAW_2D | CC_DRAW_FOREGROUND;
    drawMeOnly2D(context);
    if (updateScreen) {
        ecvDisplayTools::UpdateScreen();
    }
}

void cc2DLabel::onDeletionOf(const ccHObject* obj) {
    ccHObject::onDeletionOf(obj);  // remove dependencies, etc.

    // check that associated clouds are not about to be deleted!
    size_t pointsToRemove = 0;
    {
        for (size_t i = 0; i < m_pickedPoints.size(); ++i)
            if (m_pickedPoints[i].cloud == obj) ++pointsToRemove;
    }

    if (pointsToRemove == 0) return;

    if (pointsToRemove == m_pickedPoints.size()) {
        clear(true);  // don't call clear as we don't want/need to update input
                      // object's dependencies!
    } else {
        // remove only the necessary points
        size_t j = 0;
        for (size_t i = 0; i < m_pickedPoints.size(); ++i) {
            if (m_pickedPoints[i].cloud != obj) {
                if (i != j) std::swap(m_pickedPoints[i], m_pickedPoints[j]);
                j++;
            }
        }
        assert(j != 0);
        m_pickedPoints.resize(j);
    }

    updateName();
}

void cc2DLabel::updateName() {
    switch (m_pickedPoints.size()) {
        case 0:
            setName("Label");
            break;
        case 1:
            setName(QString("Point #") + POINT_INDEX_0);
            break;
        case 2:
            if (m_pickedPoints[0].cloud == m_pickedPoints[1].cloud)
                setName(QString("Vector #") + POINT_INDEX_0 + QString(" - #") +
                        POINT_INDEX_1);
            else
                setName(QString("Vector #") + POINT_INDEX_0 + QString("@") +
                        ENTITY_INDEX_0 + QString(" - #") + POINT_INDEX_1 +
                        QString("@") + ENTITY_INDEX_1);
            break;
        case 3:
            if (m_pickedPoints[0].cloud == m_pickedPoints[2].cloud &&
                m_pickedPoints[1].cloud == m_pickedPoints[2].cloud)
                setName(QString("Triplet #") + POINT_INDEX_0 + QString(" - #") +
                        POINT_INDEX_1 + QString(" - #") + POINT_INDEX_2);
            else
                setName(QString("Triplet #") + POINT_INDEX_0 + QString("@") +
                        ENTITY_INDEX_0 + QString(" - #") + POINT_INDEX_1 +
                        QString("@") + ENTITY_INDEX_1 + QString(" - #") +
                        POINT_INDEX_2 + QString("@") + ENTITY_INDEX_2);
            break;
    }
}

bool cc2DLabel::addPickedPoint(ccGenericPointCloud* cloud,
                               unsigned pointIndex,
                               bool entityCenter /*=false*/) {
    if (!cloud || pointIndex >= cloud->size()) return false;

    PickedPoint pp;
    pp.cloud = cloud;
    pp.index = pointIndex;
    pp.entityCenterPoint = entityCenter;

    return addPickedPoint(pp);
}

bool cc2DLabel::addPickedPoint(ccGenericMesh* mesh,
                               unsigned triangleIndex,
                               const CCVector2d& uv,
                               bool entityCenter) {
    if (!mesh || triangleIndex >= mesh->size()) return false;

    PickedPoint pp;
    pp.mesh = mesh;
    pp.index = triangleIndex;
    pp.uv = uv;
    pp.entityCenterPoint = entityCenter;

    return addPickedPoint(pp);
}

bool cc2DLabel::addPickedPoint(const PickedPoint& pp) {
    if (m_pickedPoints.size() == 3) {
        return false;
    }

    try {
        m_pickedPoints.resize(m_pickedPoints.size() + 1);
    } catch (const std::bad_alloc&) {
        // not enough memory
        return false;
    }

    m_pickedPoints.back() = pp;

    // we want to be notified whenever an associated mesh is deleted (in which
    // case we'll automatically clear the label)
    if (pp.entity())
        pp.entity()->addDependency(this, DP_NOTIFY_OTHER_ON_DELETE);
    // we must also warn the cloud or mesh whenever we delete this label
    //--> DGM: automatically done by the previous call to addDependency!

    updateName();

    return true;
}

bool cc2DLabel::toFile_MeOnly(QFile& out) const {
    if (!ccHObject::toFile_MeOnly(out)) return false;

    // points count (dataVersion >= 20)
    uint32_t count = (uint32_t)m_pickedPoints.size();
    if (out.write((const char*)&count, 4) < 0) return WriteError();

    // points & associated cloud ID (dataVersion >= 20)
    for (std::vector<PickedPoint>::const_iterator it = m_pickedPoints.begin();
         it != m_pickedPoints.end(); ++it) {
        // point index
        uint32_t index = (uint32_t)it->index;
        if (out.write((const char*)&index, 4) < 0) return WriteError();
        // cloud ID (will be retrieved later --> make sure that the cloud is
        // saved alongside!)
        uint32_t cloudID = (uint32_t)it->cloud->getUniqueID();
        if (out.write((const char*)&cloudID, 4) < 0) return WriteError();
    }

    // Relative screen position (dataVersion >= 20)
    if (out.write((const char*)m_screenPos, sizeof(float) * 2) < 0)
        return WriteError();

    // Collapsed state (dataVersion >= 20)
    if (out.write((const char*)&m_showFullBody, sizeof(bool)) < 0)
        return WriteError();

    // Show in 2D boolean (dataVersion >= 21)
    if (out.write((const char*)&m_dispIn2D, sizeof(bool)) < 0)
        return WriteError();

    // Show point(s) legend boolean (dataVersion >= 21)
    if (out.write((const char*)&m_dispPointsLegend, sizeof(bool)) < 0)
        return WriteError();

    return true;
}

bool cc2DLabel::fromFile_MeOnly(QFile& in,
                                short dataVersion,
                                int flags,
                                LoadedIDMap& oldToNewIDMap) {
    if (!ccHObject::fromFile_MeOnly(in, dataVersion, flags, oldToNewIDMap))
        return false;

    // points count (dataVersion >= 20)
    uint32_t count = 0;
    if (in.read((char*)&count, 4) < 0) return ReadError();

    // points & associated cloud ID (dataVersion >= 20)
    assert(m_pickedPoints.empty());
    for (uint32_t i = 0; i < count; ++i) {
        // point index
        uint32_t index = 0;
        if (in.read((char*)&index, 4) < 0) return ReadError();
        // cloud ID (will be retrieved later --> make sure that the cloud is
        // saved alongside!)
        uint32_t cloudID = 0;
        if (in.read((char*)&cloudID, 4) < 0) return ReadError();

        //[DIRTY] WARNING: temporarily, we set the cloud unique ID in the
        //'PickedPoint::cloud' pointer!!!
        PickedPoint pp;
        pp.index = (unsigned)index;
        *(uint32_t*)(&pp.cloud) = cloudID;
        m_pickedPoints.push_back(pp);
        if (m_pickedPoints.size() != i + 1) return MemoryError();
    }

    // Relative screen position (dataVersion >= 20)
    if (in.read((char*)m_screenPos, sizeof(float) * 2) < 0) return ReadError();

    // Collapsed state (dataVersion >= 20)
    if (in.read((char*)&m_showFullBody, sizeof(bool)) < 0) return ReadError();

    if (dataVersion > 20) {
        // Show in 2D boolean (dataVersion >= 21)
        if (in.read((char*)&m_dispIn2D, sizeof(bool)) < 0) return ReadError();

        // Show point(s) legend boolean (dataVersion >= 21)
        if (in.read((char*)&m_dispPointsLegend, sizeof(bool)) < 0)
            return ReadError();
    }

    return true;
}

void AddPointCoordinates(QStringList& body,
                         unsigned pointIndex,
                         ccGenericPointCloud* cloud,
                         int precision,
                         QString pointName = QString()) {
    assert(cloud);
    const CCVector3* P = cloud->getPointPersistentPtr(pointIndex);
    bool isShifted = cloud->isShifted();

    QString coordStr = QString("P#%0:").arg(pointIndex);
    if (!pointName.isEmpty())
        coordStr = QString("%1 (%2)").arg(pointName, coordStr);
    if (isShifted) {
        body << coordStr;
        coordStr = QString("  [shifted]");
    }

    coordStr += QString(" (%1;%2;%3)")
                        .arg(P->x, 0, 'f', precision)
                        .arg(P->y, 0, 'f', precision)
                        .arg(P->z, 0, 'f', precision);
    body << coordStr;

    if (isShifted) {
        CCVector3d Pg = cloud->toGlobal3d(*P);
        QString globCoordStr = QString("  [original] (%1;%2;%3)")
                                       .arg(Pg.x, 0, 'f', precision)
                                       .arg(Pg.y, 0, 'f', precision)
                                       .arg(Pg.z, 0, 'f', precision);
        body << globCoordStr;
    }
}

void cc2DLabel::getLabelInfo1(LabelInfo1& info) const {
    info.cloud = 0;
    if (m_pickedPoints.size() != 1) return;

    // cloud and point index
    info.cloud = m_pickedPoints[0].cloud;
    if (!info.cloud) {
        assert(false);
        return;
    }
    info.pointIndex = m_pickedPoints[0].index;
    // normal
    info.hasNormal = info.cloud->hasNormals();
    if (info.hasNormal) {
        info.normal = info.cloud->getPointNormal(info.pointIndex);
    }
    // color
    info.hasRGB = info.cloud->hasColors();
    if (info.hasRGB) {
        info.rgb = info.cloud->getPointColor(info.pointIndex);
    }
    // scalar field
    info.hasSF = info.cloud->hasDisplayedScalarField();
    if (info.hasSF) {
        info.sfValue = info.cloud->getPointScalarValue(info.pointIndex);

        info.sfName = "Scalar";
        // fetch the real scalar field name if possible
        if (info.cloud->isA(CV_TYPES::POINT_CLOUD)) {
            ccPointCloud* pc = static_cast<ccPointCloud*>(info.cloud);
            if (pc->getCurrentDisplayedScalarField()) {
                ccScalarField* sf = pc->getCurrentDisplayedScalarField();
                info.sfName = QString(sf->getName());
                if (ccScalarField::ValidValue(info.sfValue) &&
                    sf->getGlobalShift() != 0) {
                    info.sfShiftedValue = sf->getGlobalShift() + info.sfValue;
                    info.sfValueIsShifted = true;
                }
            }
        }
    }
}

void cc2DLabel::getLabelInfo2(LabelInfo2& info) const {
    info.cloud1 = info.cloud2 = 0;
    if (m_pickedPoints.size() != 2) return;

    // 1st point
    info.cloud1 = m_pickedPoints[0].cloud;
    info.point1Index = m_pickedPoints[0].index;
    const CCVector3* P1 = info.cloud1->getPointPersistentPtr(info.point1Index);
    // 2nd point
    info.cloud2 = m_pickedPoints[1].cloud;
    info.point2Index = m_pickedPoints[1].index;
    const CCVector3* P2 = info.cloud2->getPointPersistentPtr(info.point2Index);

    info.diff = *P2 - *P1;
}

void cc2DLabel::getLabelInfo3(LabelInfo3& info) const {
    info.cloud1 = info.cloud2 = info.cloud3 = 0;
    if (m_pickedPoints.size() != 3) return;
    // 1st point
    info.cloud1 = m_pickedPoints[0].cloud;
    info.point1Index = m_pickedPoints[0].index;
    const CCVector3* P1 = info.cloud1->getPointPersistentPtr(info.point1Index);
    // 2nd point
    info.cloud2 = m_pickedPoints[1].cloud;
    info.point2Index = m_pickedPoints[1].index;
    const CCVector3* P2 = info.cloud2->getPointPersistentPtr(info.point2Index);
    // 3rd point
    info.cloud3 = m_pickedPoints[2].cloud;
    info.point3Index = m_pickedPoints[2].index;
    const CCVector3* P3 = info.cloud3->getPointPersistentPtr(info.point3Index);

    // area
    CCVector3 P1P2 = *P2 - *P1;
    CCVector3 P1P3 = *P3 - *P1;
    CCVector3 P2P3 = *P3 - *P2;
    CCVector3 N = P1P2.cross(P1P3);  // N = ABxAC
    info.area = N.norm() / 2;

    // normal
    N.normalize();
    info.normal = N;

    // edges length
    info.edges.u[0] = P1P2.normd();  // edge 1-2
    info.edges.u[1] = P2P3.normd();  // edge 2-3
    info.edges.u[2] = P1P3.normd();  // edge 3-1

    // angle
    info.angles.u[0] =
            cloudViewer::RadiansToDegrees(P1P2.angle_rad(P1P3));  // angleAtP1
    info.angles.u[1] =
            cloudViewer::RadiansToDegrees(P2P3.angle_rad(-P1P2));  // angleAtP2
    info.angles.u[2] = cloudViewer::RadiansToDegrees(
            P1P3.angle_rad(P2P3));  // angleAtP3 (should be equal to 180-a1-a2!)
}

QStringList cc2DLabel::getLabelContent(int precision) const {
    QStringList body;

    switch (m_pickedPoints.size()) {
        case 0:
            // can happen if the associated cloud(s) has(ve) been deleted!
            body << "Deprecated";
            break;

        case 1:  // point
        {
            LabelInfo1 info;
            getLabelInfo1(info);
            if (!info.cloud) break;

            // coordinates
            AddPointCoordinates(body, info.pointIndex, info.cloud, precision);

            // normal
            if (info.hasNormal) {
                QString normStr =
                        QString("Normal: (%1;%2;%3)")
                                .arg(info.normal.x, 0, 'f', precision)
                                .arg(info.normal.y, 0, 'f', precision)
                                .arg(info.normal.z, 0, 'f', precision);
                body << normStr;
            }
            // color
            if (info.hasRGB) {
                QString colorStr = QString("Color: (%1;%2;%3)")
                                           .arg(info.rgb.r)
                                           .arg(info.rgb.g)
                                           .arg(info.rgb.b);
                body << colorStr;
            }
            // scalar field
            if (info.hasSF) {
                QString sfVal = GetSFValueAsString(info, precision);
                QString sfStr = QString("%1 = %2").arg(info.sfName, sfVal);
                body << sfStr;
            }
        } break;

        case 2:  // vector
        {
            LabelInfo2 info;
            getLabelInfo2(info);
            if (!info.cloud1 || !info.cloud2) break;

            // distance is now the default label title
            // PointCoordinateType dist = info.diff.norm();
            // QString distStr = QString("Distance =
            // %1").arg(dist,0,'f',precision); body << distStr;

            QString vecStr =
                    MathSymbolDelta +
                    QString("X: %1\t").arg(info.diff.x, 0, 'f', precision) +
                    MathSymbolDelta +
                    QString("Y: %1\t").arg(info.diff.y, 0, 'f', precision) +
                    MathSymbolDelta +
                    QString("Z: %1").arg(info.diff.z, 0, 'f', precision);

            body << vecStr;

            PointCoordinateType dXY =
                    sqrt(info.diff.x * info.diff.x + info.diff.y * info.diff.y);
            PointCoordinateType dXZ =
                    sqrt(info.diff.x * info.diff.x + info.diff.z * info.diff.z);
            PointCoordinateType dZY =
                    sqrt(info.diff.z * info.diff.z + info.diff.y * info.diff.y);

            vecStr = MathSymbolDelta +
                     QString("XY: %1\t").arg(dXY, 0, 'f', precision) +
                     MathSymbolDelta +
                     QString("XZ: %1\t").arg(dXZ, 0, 'f', precision) +
                     MathSymbolDelta +
                     QString("ZY: %1").arg(dZY, 0, 'f', precision);
            body << vecStr;

            AddPointCoordinates(body, info.point1Index, info.cloud1, precision);
            AddPointCoordinates(body, info.point2Index, info.cloud2, precision);
        } break;

        case 3:  // triangle/plane
        {
            LabelInfo3 info;
            getLabelInfo3(info);

            // area
            QString areaStr =
                    QString("Area = %1").arg(info.area, 0, 'f', precision);
            body << areaStr;

            // coordinates
            AddPointCoordinates(body, info.point1Index, info.cloud1, precision,
                                "A");
            AddPointCoordinates(body, info.point2Index, info.cloud2, precision,
                                "B");
            AddPointCoordinates(body, info.point3Index, info.cloud3, precision,
                                "C");

            // normal
            QString normStr = QString("Normal: (%1;%2;%3)")
                                      .arg(info.normal.x, 0, 'f', precision)
                                      .arg(info.normal.y, 0, 'f', precision)
                                      .arg(info.normal.z, 0, 'f', precision);
            body << normStr;

            // angles
            QString angleStr =
                    QString("Angles: A=%1 - B=%2 - C=%3 deg.")
                            .arg(info.angles.u[0], 0, 'f', precision)
                            .arg(info.angles.u[1], 0, 'f', precision)
                            .arg(info.angles.u[2], 0, 'f', precision);
            body << angleStr;

            // edges
            QString edgesStr = QString("Edges: AB=%1 - BC=%2 - CA=%3")
                                       .arg(info.edges.u[0], 0, 'f', precision)
                                       .arg(info.edges.u[1], 0, 'f', precision)
                                       .arg(info.edges.u[2], 0, 'f', precision);
            body << edgesStr;
        } break;

        default:
            assert(false);
            break;
    }

    return body;
}

bool cc2DLabel::acceptClick(int x, int y, Qt::MouseButton button) {
    if (button == Qt::MidButton) {
        QRect rect = QRect(0, 0, m_labelROI.width(), m_labelROI.height());
        if (rect.contains(x - m_lastScreenPos[0], y - m_lastScreenPos[1])) {
            // toggle collapse state
            m_showFullBody = !m_showFullBody;
            CC_DRAW_CONTEXT context;
            ecvDisplayTools::GetContext(context);
            update2DLabelView(context, true);
            return true;
        }
    }

    return false;
}

void cc2DLabel::drawMeOnly(CC_DRAW_CONTEXT& context) {
    if (m_pickedPoints.empty()) return;

    // 2D foreground only
    if (!MACRO_Foreground(context)) return;

    // Not compatible with virtual transformation (see
    // ccDrawableObject::enableGLTransformation)
    if (MACRO_VirtualTransEnabled(context)) return;

    if (!isRedraw()) {
        return;
    }

    if (MACRO_Draw3D(context))
        drawMeOnly3D(context);
    else if (MACRO_Draw2D(context))
        drawMeOnly2D(context);
}

void cc2DLabel::drawMeOnly3D(CC_DRAW_CONTEXT& context) {
    // clear history
    clear3Dviews();
    if (!isVisible() || !isEnabled()) {
        return;
    }

    size_t count = m_pickedPoints.size();
    if (count == 0) {
        return;
    }

    if (ecvDisplayTools::GetCurrentScreen() == nullptr) {
        assert(false);
        return;
    }

    // standard case: list names pushing
    bool pushName = MACRO_DrawEntityNames(context);
    if (pushName) {
        // not particularly fast
        if (MACRO_DrawFastNamesOnly(context)) return;
    }

    // bool loop = false;
    switch (count) {
        case 3: {
            // display point marker as spheres
            {
                if (!c_unitTriMarker) {
                    cloudViewer::GenericIndexedCloudPersist* cloud = nullptr;
                    {
                        ccPointCloud* m_polyVertices =
                                new ccPointCloud("vertices");
                        m_polyVertices->resize(3);
                        CCVector3* A = const_cast<CCVector3*>(
                                m_polyVertices->getPointPersistentPtr(0));
                        CCVector3* B = const_cast<CCVector3*>(
                                m_polyVertices->getPointPersistentPtr(1));
                        CCVector3* C = const_cast<CCVector3*>(
                                m_polyVertices->getPointPersistentPtr(2));

                        *A = *(m_pickedPoints[0].cloud->getPoint(
                                m_pickedPoints[0].index));
                        *B = *(m_pickedPoints[1].cloud->getPoint(
                                m_pickedPoints[1].index));
                        *C = *(m_pickedPoints[2].cloud->getPoint(
                                m_pickedPoints[2].index));

                        ccGenericPointCloud* gencloud =
                                ccHObjectCaster::ToGenericPointCloud(
                                        m_polyVertices);
                        if (gencloud) {
                            cloud = static_cast<
                                    cloudViewer::GenericIndexedCloudPersist*>(
                                    gencloud);
                        }
                    }

                    c_unitTriMarker =
                            QSharedPointer<ccFacet>(ccFacet::Create(cloud));

                    if (c_unitTriMarker) {
                        c_unitTriMarker->getPolygon()->setOpacity(0.5);
                        c_unitTriMarker->getPolygon()->setTempColor(
                                ecvColor::yellow);
                        c_unitTriMarker->getPolygon()->setVisible(true);
                        c_unitTriMarker->getContour()->setColor(ecvColor::red);
                        c_unitTriMarker->getContour()->showColors(true);
                        c_unitTriMarker->getContour()->setVisible(true);
                        c_unitTriMarker->setTempColor(ecvColor::darkGrey);
                        c_unitTriMarker->showColors(true);
                        c_unitTriMarker->setVisible(true);
                        c_unitTriMarker->setEnabled(true);
                        m_surfaceIdfix =
                                this->getViewId() + SEPARATOR +
                                c_unitTriMarker->getPolygon()->getViewId();
                        m_contourIdfix =
                                this->getViewId() + SEPARATOR +
                                c_unitTriMarker->getContour()->getViewId();
                        c_unitTriMarker->setFixedId(true);
                        c_unitTriMarker->getContour()->setFixedId(true);
                        c_unitTriMarker->getPolygon()->setFixedId(true);
                    }

                    if (m_surfaceIdfix == "") {
                        m_surfaceIdfix =
                                this->getViewId() + SEPARATOR +
                                c_unitTriMarker->getPolygon()->getViewId();
                    }
                    if (m_contourIdfix == "") {
                        m_contourIdfix =
                                this->getViewId() + SEPARATOR +
                                c_unitTriMarker->getContour()->getViewId();
                    }
                }

                CCVector3* A = const_cast<CCVector3*>(
                        c_unitTriMarker->getContourVertices()
                                ->getPointPersistentPtr(0));
                CCVector3* B = const_cast<CCVector3*>(
                        c_unitTriMarker->getContourVertices()
                                ->getPointPersistentPtr(1));
                CCVector3* C = const_cast<CCVector3*>(
                        c_unitTriMarker->getContourVertices()
                                ->getPointPersistentPtr(2));

                *A = *(m_pickedPoints[0].cloud->getPoint(
                        m_pickedPoints[0].index));
                *B = *(m_pickedPoints[1].cloud->getPoint(
                        m_pickedPoints[1].index));
                *C = *(m_pickedPoints[2].cloud->getPoint(
                        m_pickedPoints[2].index));

                // build-up point maker own 'context'
                CC_DRAW_CONTEXT markerContext = context;
                // we must remove the 'push name flag' so that the sphere
                // doesn't push its own!
                markerContext.drawingFlags &= (~CC_DRAW_ENTITY_NAMES);

                // draw triangle contour
                markerContext.viewID = m_contourIdfix;
                c_unitTriMarker->getContour()->setRedraw(true);
                c_unitTriMarker->getContour()->draw(markerContext);
                // draw triangle mesh surface
                markerContext.viewID = m_surfaceIdfix;
                c_unitTriMarker->getPolygon()->setRedraw(true);
                c_unitTriMarker->getPolygon()->draw(markerContext);
            }
        }

        case 2: {
            if (count == 2) {
                // it's may not displayed on top of the entities
                // line width
                const float c_sizeFactor = 4.0f;
                // contour segments (before the labels!)
                ecvColor::Rgb lineColor =
                        isSelected()
                                ? ecvColor::red
                                : context.labelDefaultMarkerCol /*ecvColor::green.rgba*/
                        ;
                float lineWidth = c_sizeFactor * context.renderZoom;

                const CCVector3* lineSt = m_pickedPoints[0].cloud->getPoint(
                        m_pickedPoints[0].index);
                const CCVector3* lineEd = m_pickedPoints[1].cloud->getPoint(
                        m_pickedPoints[1].index);

                // we draw the line
                WIDGETS_PARAMETER param(WIDGETS_TYPE::WIDGET_LINE_3D, m_lineID);
                param.setLineWidget(
                        LineWidget(*lineSt, *lineEd, lineWidth, lineColor));
                ecvDisplayTools::DrawWidgets(param);
            }
        }

        case 1: {
            // display point marker as spheres
            {
                if (!c_unitPointMarker) {
                    c_unitPointMarker = QSharedPointer<ccSphere>(
                            new ccSphere(1.0f, 0, "PointMarker", 12));
                    c_unitPointMarker->showColors(true);
                    c_unitPointMarker->setVisible(true);
                    c_unitPointMarker->setEnabled(true);
                    c_unitPointMarker->setFixedId(true);
                    m_sphereIdfix = SEPARATOR + this->getViewId() + SEPARATOR +
                                    c_unitPointMarker->getViewId();
                }

                if (m_sphereIdfix == "") {
                    m_sphereIdfix = SEPARATOR + this->getViewId() + SEPARATOR +
                                    c_unitPointMarker->getViewId();
                }

                // build-up point maker own 'context'
                CC_DRAW_CONTEXT markerContext = context;
                // we must remove the 'push name flag' so that the sphere
                // doesn't push its own!
                markerContext.drawingFlags &= (~CC_DRAW_ENTITY_NAMES);

                if (isSelected() && !pushName)
                    c_unitPointMarker->setTempColor(ecvColor::red);
                else
                    c_unitPointMarker->setTempColor(
                            context.labelDefaultMarkerCol);

                const ecvViewportParameters& viewportParams =
                        ecvDisplayTools::GetViewportParameters();
                for (size_t i = 0; i < count; i++) {
                    const CCVector3* P = m_pickedPoints[i].cloud->getPoint(
                            m_pickedPoints[i].index);
                    float scale = context.labelMarkerSize * m_relMarkerScale;
                    if (viewportParams.perspectiveView &&
                        viewportParams.zFar > 0) {
                        // we always project the points in 2D (maybe useful
                        // later, even when displaying the label during the 2D
                        // pass!)
                        ccGLCameraParameters camera;
                        // we can't use the context 'ccGLCameraParameters'
                        // (viewport, modelView matrix, etc. ) because it
                        // doesn't take the temporary 'GL transformation' into
                        // account!
                        ecvDisplayTools::GetGLCameraParameters(camera);

                        // in perspective view, the actual scale depends on the
                        // distance to the camera!
                        double d = (camera.modelViewMat *
                                    CCVector3d::fromArray(P->u))
                                           .norm();
                        // we consider that the 'standard' scale is at half the
                        // depth sqrt = empirical (probably because the marker
                        // size is
                        // already partly compensated by
                        // ecvDisplayTools::computeActualPixelSize())
                        double unitD = viewportParams.zFar / 2;
                        scale = static_cast<float>(scale * sqrt(d / unitD));
                    }

                    WIDGETS_PARAMETER param(WIDGETS_TYPE::WIDGET_SPHERE,
                                            QString::number(i) + m_sphereIdfix);
                    param.radius = scale / 2;
                    m_pickedPoints[i].markerScale = scale / 2;
                    param.center = CCVector3(P->x, P->y, P->z);
                    param.color = ecvColor::FromRgba(ecvColor::ored);
                    ecvDisplayTools::DrawWidgets(param, false);
                    // markerContext.transformInfo.setScale(CCVector3(scale,
                    // scale, scale)); markerContext.viewID = QString::number(i)
                    // + m_sphereIdfix; c_unitPointMarker->setRedraw(true);
                    // c_unitPointMarker->draw(markerContext);
                }
            }
        }
    }
}

// display parameters
static const int c_margin = 5;
static const int c_tabMarginX = 5;  // default: 5
static const int c_tabMarginY = 2;
static const int c_arrowBaseSize = 3;
// static const int c_buttonSize = 10;

static const ecvColor::Rgb c_darkGreen(0, 200, 0);

//! Data table
struct Tab {
    //! Default constructor
    Tab(int _maxBlockPerRow = 2)
        : maxBlockPerRow(_maxBlockPerRow),
          blockCount(0),
          rowCount(0),
          colCount(0) {}

    //! Sets the maximum number of blocks per row
    /** \warning Must be called before adding data!
     **/
    void setMaxBlockPerRow(int maxBlock) { maxBlockPerRow = maxBlock; }

    //! Adds a 2x3 block (must be filled!)
    int add2x3Block() {
        // add columns (if necessary)
        if (colCount < maxBlockPerRow * 2) {
            colCount += 2;
            colContent.resize(colCount);
            colWidth.resize(colCount, 0);
        }
        int blockCol = (blockCount % maxBlockPerRow);
        // add new row
        if (blockCol == 0) rowCount += 3;
        ++blockCount;

        // return the first column index of the block
        return blockCol * 2;
    }

    //! Updates columns width table
    /** \return the total width
     **/
    int updateColumnsWidthTable(const QFontMetrics& fm) {
        // compute min width of each column
        int totalWidth = 0;
        for (int i = 0; i < colCount; ++i) {
            int maxWidth = 0;
            for (int j = 0; j < colContent[i].size(); ++j) {
#if (QT_VERSION <= QT_VERSION_CHECK(5, 0, 0))
                maxWidth = std::max(maxWidth, fm.width(colContent[i][j]));
#else
                maxWidth = std::max(maxWidth,
                                    fm.horizontalAdvance(colContent[i][j]));
#endif
            }
            colWidth[i] = maxWidth;
            totalWidth += maxWidth;
        }
        return totalWidth;
    }

    //! Maximum number of blocks per row
    int maxBlockPerRow;
    //! Number of 2x3 blocks
    int blockCount;
    //! Number of rows
    int rowCount;
    //! Number of columns
    int colCount;
    //! Columns width
    std::vector<int> colWidth;
    //! Columns content
    std::vector<QStringList> colContent;
};

void cc2DLabel::drawMeOnly2D(CC_DRAW_CONTEXT& context) {
    if (!ecvDisplayTools::GetCurrentScreen()) {
        assert(false);
        return;
    }

    // clear history
    clear2Dviews();
    if (!isVisible() || !isEnabled()) {
        clearLabel(false);
        return;
    }

    if (m_pickedPoints.empty()) {
        return;
    }

    // standard case: list names pushing
    bool pushName = MACRO_DrawEntityNames(context);

    size_t count = m_pickedPoints.size();
    assert(count != 0);

    // hack: we display the label connecting 'segments' and the point(s) legend
    // in 2D so that they always appear above the entities
    {
        // don't do this in picking mode!
        if (!pushName) {
            // we always project the points in 2D (maybe useful later, even when
            // displaying the label during the 2D pass!)
            ccGLCameraParameters camera;
            // we can't use the context 'ccGLCameraParameters' (viewport,
            // modelView matrix, etc. ) because it doesn't take the temporary
            // 'GL transformation' into account!
            ecvDisplayTools::GetGLCameraParameters(camera);
            for (size_t i = 0; i < count; i++) {
                // project the point in 2D
                const CCVector3* P3D = m_pickedPoints[i].cloud->getPoint(
                        m_pickedPoints[i].index);
                camera.project(*P3D, m_pickedPoints[i].pos2D);
            }
        }

        // test if the label points are visible
        size_t visibleCount = 0;
        for (unsigned j = 0; j < count; ++j) {
            if (m_pickedPoints[j].pos2D.z >= 0.0 &&
                m_pickedPoints[j].pos2D.z <= 1.0) {
                ++visibleCount;
            }
        }

        if (visibleCount) {
            // no need to display the point(s) legend in picking mode
            if (m_dispPointsLegend && !pushName) {
                QFont font(ecvDisplayTools::
                                   GetTextDisplayFont());  // takes rendering
                                                           // zoom into account!
                // font.setPointSize(font.pointSize() + 2);
                font.setBold(true);
                static const QChar ABC[3] = {'A', 'B', 'C'};

                // draw the label 'legend(s)'
                for (size_t j = 0; j < count; j++) {
                    QString title;
                    if (count == 1)
                        title = getName();  // for single-point labels we prefer
                                            // the name
                    else if (count == 3)
                        title = ABC[j];  // for triangle-labels, we only display
                                         // "A","B","C"
                    else
                        title = QString("P#%0").arg(m_pickedPoints[j].index);

                    ecvDisplayTools::DisplayText(
                            title,
                            static_cast<int>(m_pickedPoints[j].pos2D.x) +
                                    context.labelMarkerTextShift_pix,
                            static_cast<int>(m_pickedPoints[j].pos2D.y) +
                                    context.labelMarkerTextShift_pix,
                            ecvDisplayTools::ALIGN_DEFAULT,
                            context.labelOpacity / 100.0f, ecvColor::white.rgb,
                            &font, this->getViewId());
                }
            }
        } else {
            return;
        }
    }

    // only display lengend other than 2D display
    if (!m_dispIn2D) {
        ecvDisplayTools::RemoveWidgets(WIDGETS_PARAMETER(
                WIDGETS_TYPE::WIDGET_CAPTION, this->getViewId()));
        return;
    }

    // label title
    const int precision = context.dispNumberPrecision;
    QString title = getTitle(precision);

#define DRAW_CONTENT_AS_TAB
#ifdef DRAW_CONTENT_AS_TAB
    // draw contents as an array
    Tab tab(4);
    int rowHeight = 0;
#else
    // simply display the content as text
    QStringList body;
#endif

    // render zoom
    int margin = static_cast<int>(c_margin * context.renderZoom);
    int tabMarginX = static_cast<int>(c_tabMarginX * context.renderZoom);
    int tabMarginY = static_cast<int>(c_tabMarginY * context.renderZoom);
    int arrowBaseSize = static_cast<int>(c_arrowBaseSize * context.renderZoom);

    int titleHeight = 0;
    QFont bodyFont, titleFont;
    if (!pushName) {
        /*** label border ***/
        bodyFont =
                ecvDisplayTools::GetLabelDisplayFont();  // takes rendering zoom
                                                         // into account!
        titleFont = bodyFont;  // takes rendering zoom into account!
        // titleFont.setBold(true);

        QFontMetrics titleFontMetrics(titleFont);
        titleHeight = titleFontMetrics.height();

        QFontMetrics bodyFontMetrics(bodyFont);
        rowHeight = bodyFontMetrics.height();

        // get label box dimension
        int dx = 100;
        int dy = 0;
        {
            // base box dimension
#if (QT_VERSION <= QT_VERSION_CHECK(5, 0, 0))
            dx = std::max(dx, titleFontMetrics.width(title));
#else
            dx = std::max(dx, titleFontMetrics.horizontalAdvance(title));
#endif

            dy += margin;       // top vertical margin
            dy += titleHeight;  // title

            if (m_showFullBody) {
#ifdef DRAW_CONTENT_AS_TAB
                try {
                    if (count == 1) {
                        LabelInfo1 info;
                        getLabelInfo1(info);

                        bool isShifted = info.cloud->isShifted();
                        // 1st block: X, Y, Z (local)
                        {
                            int c = tab.add2x3Block();
                            QChar suffix = ' ';
                            if (isShifted) {
                                suffix = 'l';  //'l' for local
                            }
                            const CCVector3* P =
                                    info.cloud->getPoint(info.pointIndex);
                            tab.colContent[c] << QString("X") + suffix;
                            tab.colContent[c + 1]
                                    << QString::number(P->x, 'f', precision);
                            tab.colContent[c] << QString("Y") + suffix;
                            tab.colContent[c + 1]
                                    << QString::number(P->y, 'f', precision);
                            tab.colContent[c] << QString("Z") + suffix;
                            tab.colContent[c + 1]
                                    << QString::number(P->z, 'f', precision);
                        }
                        // next block:  X, Y, Z (global)
                        if (isShifted) {
                            int c = tab.add2x3Block();
                            CCVector3d P = info.cloud->toGlobal3d(
                                    *info.cloud->getPoint(info.pointIndex));
                            tab.colContent[c] << "Xg ";
                            tab.colContent[c + 1]
                                    << QString::number(P.x, 'f', precision);
                            tab.colContent[c] << "Yg ";
                            tab.colContent[c + 1]
                                    << QString::number(P.y, 'f', precision);
                            tab.colContent[c] << "Zg ";
                            tab.colContent[c + 1]
                                    << QString::number(P.z, 'f', precision);
                        }
                        // next block: normal
                        if (info.hasNormal) {
                            int c = tab.add2x3Block();
                            tab.colContent[c] << "Nx ";
                            tab.colContent[c + 1] << QString::number(
                                    info.normal.x, 'f', precision);
                            tab.colContent[c] << "Ny ";
                            tab.colContent[c + 1] << QString::number(
                                    info.normal.y, 'f', precision);
                            tab.colContent[c] << "Nz ";
                            tab.colContent[c + 1] << QString::number(
                                    info.normal.z, 'f', precision);
                        }

                        // next block: RGB color
                        if (info.hasRGB) {
                            int c = tab.add2x3Block();
                            tab.colContent[c] << " R ";
                            tab.colContent[c + 1]
                                    << QString::number(info.rgb.r);
                            tab.colContent[c] << " G ";
                            tab.colContent[c + 1]
                                    << QString::number(info.rgb.g);
                            tab.colContent[c] << " B ";
                            tab.colContent[c + 1]
                                    << QString::number(info.rgb.b);
                        }
                    } else if (count == 2) {
                        LabelInfo2 info;
                        getLabelInfo2(info);

                        // 1st block: dX, dY, dZ
                        {
                            int c = tab.add2x3Block();
                            tab.colContent[c]
                                    << MathSymbolDelta + QString("X ");
                            tab.colContent[c + 1] << QString::number(
                                    info.diff.x, 'f', precision);
                            tab.colContent[c]
                                    << MathSymbolDelta + QString("Y ");
                            tab.colContent[c + 1] << QString::number(
                                    info.diff.y, 'f', precision);
                            tab.colContent[c]
                                    << MathSymbolDelta + QString("Z ");
                            tab.colContent[c + 1] << QString::number(
                                    info.diff.z, 'f', precision);
                        }
                        // 2nd block: dXY, dXZ, dZY
                        {
                            int c = tab.add2x3Block();
                            PointCoordinateType dXY =
                                    sqrt(info.diff.x * info.diff.x +
                                         info.diff.y * info.diff.y);
                            PointCoordinateType dXZ =
                                    sqrt(info.diff.x * info.diff.x +
                                         info.diff.z * info.diff.z);
                            PointCoordinateType dZY =
                                    sqrt(info.diff.z * info.diff.z +
                                         info.diff.y * info.diff.y);
                            tab.colContent[c]
                                    << " " + MathSymbolDelta + QString("XY ");
                            tab.colContent[c + 1]
                                    << QString::number(dXY, 'f', precision);
                            tab.colContent[c]
                                    << " " + MathSymbolDelta + QString("XZ ");
                            tab.colContent[c + 1]
                                    << QString::number(dXZ, 'f', precision);
                            tab.colContent[c]
                                    << " " + MathSymbolDelta + QString("ZY ");
                            tab.colContent[c + 1]
                                    << QString::number(dZY, 'f', precision);
                        }
                    } else if (count == 3) {
                        LabelInfo3 info;
                        getLabelInfo3(info);
                        tab.setMaxBlockPerRow(2);  // square tab (2x2 blocks)

                        // next block: indexes
                        {
                            int c = tab.add2x3Block();
                            tab.colContent[c] << "index.A  ";
                            tab.colContent[c + 1]
                                    << QString::number(info.point1Index);
                            tab.colContent[c] << "index.B  ";
                            tab.colContent[c + 1]
                                    << QString::number(info.point2Index);
                            tab.colContent[c] << "index.C  ";
                            tab.colContent[c + 1]
                                    << QString::number(info.point3Index);
                        }
                        // next block: edges length
                        {
                            int c = tab.add2x3Block();
                            tab.colContent[c] << "  AB  ";
                            tab.colContent[c + 1] << QString::number(
                                    info.edges.u[0], 'f', precision);
                            tab.colContent[c] << "  BC  ";
                            tab.colContent[c + 1] << QString::number(
                                    info.edges.u[1], 'f', precision);
                            tab.colContent[c] << "  CA  ";
                            tab.colContent[c + 1] << QString::number(
                                    info.edges.u[2], 'f', precision);
                        }
                        // next block: angles
                        {
                            int c = tab.add2x3Block();
                            tab.colContent[c] << "angle.A  ";
                            tab.colContent[c + 1] << QString::number(
                                    info.angles.u[0], 'f', precision);
                            tab.colContent[c] << "angle.B  ";
                            tab.colContent[c + 1] << QString::number(
                                    info.angles.u[1], 'f', precision);
                            tab.colContent[c] << "angle.C  ";
                            tab.colContent[c + 1] << QString::number(
                                    info.angles.u[2], 'f', precision);
                        }
                        // next block: normal
                        {
                            int c = tab.add2x3Block();
                            tab.colContent[c] << "  Nx  ";
                            tab.colContent[c + 1] << QString::number(
                                    info.normal.x, 'f', precision);
                            tab.colContent[c] << "  Ny ";
                            tab.colContent[c + 1] << QString::number(
                                    info.normal.y, 'f', precision);
                            tab.colContent[c] << "  Nz ";
                            tab.colContent[c + 1] << QString::number(
                                    info.normal.z, 'f', precision);
                        }
                    }
                } catch (const std::bad_alloc&) {
                    // not enough memory
                    assert(!pushName);
                    return;
                }

                // compute min width of each column
                int totalWidth = tab.updateColumnsWidthTable(bodyFontMetrics);

                int tabWidth =
                        totalWidth +
                        tab.colCount * (2 * tabMarginX);  // add inner margins
                dx = std::max(dx, tabWidth);
                dy += tab.rowCount *
                      (rowHeight + 2 * tabMarginY);  // add inner margins
                // we also add a margin every 3 rows
                dy += std::max(0, (tab.rowCount / 3) - 1) * margin;
                dy += margin;  // bottom vertical margin
#else
                body = getLabelContent(precision);
                if (!body.empty()) {
                    dy += margin;  // vertical margin above separator
                    for (int j = 0; j < body.size(); ++j) {
                        dx = std::max(dx, bodyFontMetrics.width(body[j]));
                        dy += rowHeight;  // body line height
                    }
                    dy += margin;  // vertical margin below text
                }
#endif  // DRAW_CONTENT_AS_TAB
            }

            dx += margin * 2;  // horizontal margins
        }

        // main rectangle
        m_labelROI = QRect(0, 0, dx, dy);
    }

    // draw label rectangle
    const int xStart = static_cast<int>(context.glW * m_screenPos[0]);
    const int yStart = static_cast<int>(context.glH * (1.0f - m_screenPos[1]));

    m_lastScreenPos[0] = xStart;
    m_lastScreenPos[1] = yStart - m_labelROI.height();

    // colors
    bool highlighted = (!pushName && isSelected());
    // default background color
    unsigned char alpha =
            static_cast<unsigned char>((context.labelOpacity / 100.0) * 255);
    ecvColor::Rgbaub defaultBkgColor(context.labelDefaultBkgCol, alpha);
    // default border color (mustn't be totally transparent!)
    ecvColor::Rgbaub defaultBorderColor(ecvColor::red, 255);
    if (!highlighted) {
        // apply only half of the transparency
        unsigned char halfAlpha = static_cast<unsigned char>(
                (50.0 + context.labelOpacity / 200.0) * 255);
        defaultBorderColor =
                ecvColor::Rgbaub(context.labelDefaultBkgCol, halfAlpha);
    }

    m_labelROI = QRect(xStart, yStart - m_labelROI.height(), m_labelROI.width(),
                       m_labelROI.height());

    ecvColor::Rgbub defaultTextColor;
    if (context.labelOpacity < 40) {
        // under a given opacity level, we use the default text color instead!
        defaultTextColor = context.textDefaultCol;
    } else {
        defaultTextColor = ecvColor::Rgbub(255 - context.labelDefaultBkgCol.r,
                                           255 - context.labelDefaultBkgCol.g,
                                           255 - context.labelDefaultBkgCol.b);
    }

    // display text
    if (!pushName) {
        // label title
        m_historyMessage << title;

        if (m_showFullBody) {
            for (int r = 0; r < tab.rowCount; ++r) {
                QString str;
                for (int c = 0; c < tab.colCount; ++c) {
                    str += tab.colContent[c][r];
                }
                m_historyMessage << str;
            }
        }
    }

    if (!pushName && count > 0 && !m_historyMessage.empty()) {
        // compute arrow head position
        CCVector3 position(0, 0, 0);
        for (size_t i = 0; i < count; ++i) {
            const CCVector3* p =
                    m_pickedPoints[i].cloud->getPoint(m_pickedPoints[i].index);
            position += *p;
        }
        position /= static_cast<PointCoordinateType>(count);

        WIDGETS_PARAMETER param(WIDGETS_TYPE::WIDGET_CAPTION,
                                this->getViewId());
        param.center = position;
        param.pos = CCVector2(m_labelROI.x(), m_labelROI.y());
        param.color = ecvColor::FromRgbub(defaultTextColor);
        param.color.a = defaultBkgColor.a / 255.0f;
        param.text = m_historyMessage.join("\n");
        param.text = param.text.trimmed();
        param.fontSize = bodyFont.pointSize();
        ecvDisplayTools::DrawWidgets(param, false);
    }
}

bool cc2DLabel::pointPicking(const CCVector2d& clickPos,
                             const ccGLCameraParameters& camera,
                             int& nearestPointIndex,
                             double& nearestSquareDist) const {
    nearestPointIndex = -1;
    nearestSquareDist = -1.0;
    {
        // back project the clicked point in 3D
        CCVector3d clickPosd(clickPos.x, clickPos.y, 0.0);
        CCVector3d X(0, 0, 0);
        if (!camera.unproject(clickPosd, X)) {
            return false;
        }

        clickPosd.z = 1.0;
        CCVector3d Y(0, 0, 0);
        if (!camera.unproject(clickPosd, Y)) {
            return false;
        }

        CCVector3d xy = (Y - X);
        xy.normalize();

        for (unsigned i = 0; i < size(); ++i) {
            const PickedPoint& pp = getPickedPoint(i);
            if (pp.markerScale == 0) {
                // never displayed
                continue;
            }

            const CCVector3 P = pp.getPointPosition();

            // warning: we have to handle the relative GL transformation!
            ccGLMatrix trans;
            bool noGLTrans =
                    pp.entity()
                            ? !pp.entity()->getAbsoluteGLTransformation(trans)
                            : true;

            CCVector3d Q2D;
            bool insideFrustum = false;
            if (noGLTrans) {
                camera.project(P, Q2D, &insideFrustum);
            } else {
                CCVector3 P3D = P;
                trans.apply(P3D);
                camera.project(P3D, Q2D, &insideFrustum);
            }

            if (!insideFrustum) {
                continue;
            }

            // closest distance to XY
            CCVector3d XP = (P.toDouble() - X);
            double squareDist = (XP - XP.dot(xy) * xy).norm2();

            if (squareDist <=
                static_cast<double>(pp.markerScale) * pp.markerScale) {
                if (nearestPointIndex < 0 || squareDist < nearestSquareDist) {
                    nearestSquareDist = squareDist;
                    nearestPointIndex = i;
                }
            }
        }
    }

    return (nearestPointIndex >= 0);
}

//! deprecated
void cc2DLabel::drawMeOnly2D_(CC_DRAW_CONTEXT& context) {
    if (ecvDisplayTools::GetCurrentScreen() == nullptr) {
        assert(false);
        return;
    }

    // clear history
    clear2Dviews();
    if (!isVisible() || !isEnabled()) {
        return;
    }

    if (m_pickedPoints.empty()) {
        return;
    }

    // standard case: list names pushing
    bool pushName = MACRO_DrawEntityNames(context);

    float halfW = context.glW / 2.0f;
    float halfH = context.glH / 2.0f;

    size_t count = m_pickedPoints.size();
    assert(count != 0);

    // hack: we display the label connecting 'segments' and the point(s) legend
    // in 2D so that they always appear above the entities
    {
        // don't do this in picking mode!
        if (!pushName) {
            // we always project the points in 2D (maybe useful later, even when
            // displaying the label during the 2D pass!)
            ccGLCameraParameters camera;
            // we can't use the context 'ccGLCameraParameters' (viewport,
            // modelView matrix, etc. ) because it doesn't take the temporary
            // 'GL transformation' into account!
            ecvDisplayTools::GetGLCameraParameters(camera);
            for (size_t i = 0; i < count; i++) {
                // project the point in 2D
                const CCVector3* P3D = m_pickedPoints[i].cloud->getPoint(
                        m_pickedPoints[i].index);
                camera.project(*P3D, m_pickedPoints[i].pos2D);
            }
        }

        // test if the label points are visible
        size_t visibleCount = 0;
        for (unsigned j = 0; j < count; ++j) {
            if (m_pickedPoints[j].pos2D.z >= 0.0 &&
                m_pickedPoints[j].pos2D.z <= 1.0) {
                ++visibleCount;
            }
        }

        if (visibleCount) {
            // no need to display the point(s) legend in picking mode
            if (m_dispPointsLegend && !pushName) {
                QFont font(ecvDisplayTools::
                                   GetTextDisplayFont());  // takes rendering
                                                           // zoom into account!
                // font.setPointSize(font.pointSize() + 2);
                font.setBold(true);
                static const QChar ABC[3] = {'A', 'B', 'C'};

                // draw the label 'legend(s)'
                for (size_t j = 0; j < count; j++) {
                    QString title;
                    if (count == 1)
                        title = getName();  // for single-point labels we prefer
                                            // the name
                    else if (count == 3)
                        title = ABC[j];  // for triangle-labels, we only display
                                         // "A","B","C"
                    else
                        title = QString("P#%0").arg(m_pickedPoints[j].index);

                    m_historyMessage << title;
                    ecvDisplayTools::DisplayText(
                            title,
                            static_cast<int>(m_pickedPoints[j].pos2D.x) +
                                    context.labelMarkerTextShift_pix,
                            static_cast<int>(m_pickedPoints[j].pos2D.y) +
                                    context.labelMarkerTextShift_pix,
                            ecvDisplayTools::ALIGN_DEFAULT,
                            context.labelOpacity / 100.0f, ecvColor::red.rgb,
                            &font, this->getViewId());
                }
            }
        } else {
            // no need to draw anything (might be confusing)
            if (pushName) {
                // glFunc->glPopName();
            }
            return;
        }
    }

    if (!m_dispIn2D) {
        // nothing to do
        if (pushName) {
            // glFunc->glPopName();
        }
        return;
    }

    // label title
    const int precision = context.dispNumberPrecision;
    QString title = getTitle(precision);

#define DRAW_CONTENT_AS_TAB
#ifdef DRAW_CONTENT_AS_TAB
    // draw contents as an array
    Tab tab(4);
    int rowHeight = 0;
#else
    // simply display the content as text
    QStringList body;
#endif

    // render zoom
    int margin = static_cast<int>(c_margin * context.renderZoom);
    int tabMarginX = static_cast<int>(c_tabMarginX * context.renderZoom);
    int tabMarginY = static_cast<int>(c_tabMarginY * context.renderZoom);
    int arrowBaseSize = static_cast<int>(c_arrowBaseSize * context.renderZoom);

    int titleHeight = 0;
    QFont bodyFont, titleFont;
    if (!pushName) {
        /*** label border ***/
        bodyFont =
                ecvDisplayTools::GetLabelDisplayFont();  // takes rendering zoom
                                                         // into account!
        titleFont = bodyFont;  // takes rendering zoom into account!
        // titleFont.setBold(true);

        QFontMetrics titleFontMetrics(titleFont);
        titleHeight = titleFontMetrics.height();

        QFontMetrics bodyFontMetrics(bodyFont);
        rowHeight = bodyFontMetrics.height();

        // get label box dimension
        int dx = 100;
        int dy = 0;
        // int buttonSize    = static_cast<int>(c_buttonSize *
        // context.renderZoom);
        {
            // base box dimension
#if (QT_VERSION <= QT_VERSION_CHECK(5, 0, 0))
            dx = std::max(dx, titleFontMetrics.width(title));
#else
            dx = std::max(dx, titleFontMetrics.horizontalAdvance(title));
#endif

            dy += margin;       // top vertical margin
            dy += titleHeight;  // title

            if (m_showFullBody) {
#ifdef DRAW_CONTENT_AS_TAB
                try {
                    if (count == 1) {
                        LabelInfo1 info;
                        getLabelInfo1(info);

                        bool isShifted = info.cloud->isShifted();
                        // 1st block: X, Y, Z (local)
                        {
                            int c = tab.add2x3Block();
                            QChar suffix;
                            if (isShifted) {
                                suffix = 'l';  //'l' for local
                            }
                            const CCVector3* P =
                                    info.cloud->getPoint(info.pointIndex);
                            tab.colContent[c] << QString("X") + suffix;
                            tab.colContent[c + 1]
                                    << QString::number(P->x, 'f', precision);
                            tab.colContent[c] << QString("Y") + suffix;
                            tab.colContent[c + 1]
                                    << QString::number(P->y, 'f', precision);
                            tab.colContent[c] << QString("Z") + suffix;
                            tab.colContent[c + 1]
                                    << QString::number(P->z, 'f', precision);
                        }
                        // next block:  X, Y, Z (global)
                        if (isShifted) {
                            int c = tab.add2x3Block();
                            CCVector3d P = info.cloud->toGlobal3d(
                                    *info.cloud->getPoint(info.pointIndex));
                            tab.colContent[c] << "Xg";
                            tab.colContent[c + 1]
                                    << QString::number(P.x, 'f', precision);
                            tab.colContent[c] << "Yg";
                            tab.colContent[c + 1]
                                    << QString::number(P.y, 'f', precision);
                            tab.colContent[c] << "Zg";
                            tab.colContent[c + 1]
                                    << QString::number(P.z, 'f', precision);
                        }
                        // next block: normal
                        if (info.hasNormal) {
                            int c = tab.add2x3Block();
                            tab.colContent[c] << "Nx";
                            tab.colContent[c + 1] << QString::number(
                                    info.normal.x, 'f', precision);
                            tab.colContent[c] << "Ny";
                            tab.colContent[c + 1] << QString::number(
                                    info.normal.y, 'f', precision);
                            tab.colContent[c] << "Nz";
                            tab.colContent[c + 1] << QString::number(
                                    info.normal.z, 'f', precision);
                        }

                        // next block: RGB color
                        if (info.hasRGB) {
                            int c = tab.add2x3Block();
                            tab.colContent[c] << "R";
                            tab.colContent[c + 1]
                                    << QString::number(info.rgb.r);
                            tab.colContent[c] << "G";
                            tab.colContent[c + 1]
                                    << QString::number(info.rgb.g);
                            tab.colContent[c] << "B";
                            tab.colContent[c + 1]
                                    << QString::number(info.rgb.b);
                        }
                    } else if (count == 2) {
                        LabelInfo2 info;
                        getLabelInfo2(info);

                        // 1st block: dX, dY, dZ
                        {
                            int c = tab.add2x3Block();
                            tab.colContent[c] << MathSymbolDelta + QString("X");
                            tab.colContent[c + 1] << QString::number(
                                    info.diff.x, 'f', precision);
                            tab.colContent[c] << MathSymbolDelta + QString("Y");
                            tab.colContent[c + 1] << QString::number(
                                    info.diff.y, 'f', precision);
                            tab.colContent[c] << MathSymbolDelta + QString("Z");
                            tab.colContent[c + 1] << QString::number(
                                    info.diff.z, 'f', precision);
                        }
                        // 2nd block: dXY, dXZ, dZY
                        {
                            int c = tab.add2x3Block();
                            PointCoordinateType dXY =
                                    sqrt(info.diff.x * info.diff.x +
                                         info.diff.y * info.diff.y);
                            PointCoordinateType dXZ =
                                    sqrt(info.diff.x * info.diff.x +
                                         info.diff.z * info.diff.z);
                            PointCoordinateType dZY =
                                    sqrt(info.diff.z * info.diff.z +
                                         info.diff.y * info.diff.y);
                            tab.colContent[c]
                                    << MathSymbolDelta + QString("XY");
                            tab.colContent[c + 1]
                                    << QString::number(dXY, 'f', precision);
                            tab.colContent[c]
                                    << MathSymbolDelta + QString("XZ");
                            tab.colContent[c + 1]
                                    << QString::number(dXZ, 'f', precision);
                            tab.colContent[c]
                                    << MathSymbolDelta + QString("ZY");
                            tab.colContent[c + 1]
                                    << QString::number(dZY, 'f', precision);
                        }
                    } else if (count == 3) {
                        LabelInfo3 info;
                        getLabelInfo3(info);
                        tab.setMaxBlockPerRow(2);  // square tab (2x2 blocks)

                        // next block: indexes
                        {
                            int c = tab.add2x3Block();
                            tab.colContent[c] << "index.A";
                            tab.colContent[c + 1]
                                    << QString::number(info.point1Index);
                            tab.colContent[c] << "index.B";
                            tab.colContent[c + 1]
                                    << QString::number(info.point2Index);
                            tab.colContent[c] << "index.C";
                            tab.colContent[c + 1]
                                    << QString::number(info.point3Index);
                        }
                        // next block: edges length
                        {
                            int c = tab.add2x3Block();
                            tab.colContent[c] << "AB";
                            tab.colContent[c + 1] << QString::number(
                                    info.edges.u[0], 'f', precision);
                            tab.colContent[c] << "BC";
                            tab.colContent[c + 1] << QString::number(
                                    info.edges.u[1], 'f', precision);
                            tab.colContent[c] << "CA";
                            tab.colContent[c + 1] << QString::number(
                                    info.edges.u[2], 'f', precision);
                        }
                        // next block: angles
                        {
                            int c = tab.add2x3Block();
                            tab.colContent[c] << "angle.A";
                            tab.colContent[c + 1] << QString::number(
                                    info.angles.u[0], 'f', precision);
                            tab.colContent[c] << "angle.B";
                            tab.colContent[c + 1] << QString::number(
                                    info.angles.u[1], 'f', precision);
                            tab.colContent[c] << "angle.C";
                            tab.colContent[c + 1] << QString::number(
                                    info.angles.u[2], 'f', precision);
                        }
                        // next block: normal
                        {
                            int c = tab.add2x3Block();
                            tab.colContent[c] << "Nx";
                            tab.colContent[c + 1] << QString::number(
                                    info.normal.x, 'f', precision);
                            tab.colContent[c] << "Ny";
                            tab.colContent[c + 1] << QString::number(
                                    info.normal.y, 'f', precision);
                            tab.colContent[c] << "Nz";
                            tab.colContent[c + 1] << QString::number(
                                    info.normal.z, 'f', precision);
                        }
                    }
                } catch (const std::bad_alloc&) {
                    // not enough memory
                    assert(!pushName);
                    return;
                }

                // compute min width of each column
                int totalWidth = tab.updateColumnsWidthTable(bodyFontMetrics);

                int tabWidth =
                        totalWidth +
                        tab.colCount * (2 * tabMarginX);  // add inner margins
                dx = std::max(dx, tabWidth);
                dy += tab.rowCount *
                      (rowHeight + 2 * tabMarginY);  // add inner margins
                // we also add a margin every 3 rows
                dy += std::max(0, (tab.rowCount / 3) - 1) * margin;
                dy += margin;  // bottom vertical margin
#else
                body = getLabelContent(precision);
                if (!body.empty()) {
                    dy += margin;  // vertical margin above separator
                    for (int j = 0; j < body.size(); ++j) {
                        dx = std::max(dx, bodyFontMetrics.width(body[j]));
                        dy += rowHeight;  // body line height
                    }
                    dy += margin;  // vertical margin below text
                }
#endif  // DRAW_CONTENT_AS_TAB
            }

            dx += margin * 2;  // horizontal margins
        }

        // main rectangle
        m_labelROI = QRect(0, 0, dx, dy);
    }

    // draw label rectangle
    const int xStart = static_cast<int>(context.glW * m_screenPos[0]);
    const int yStart = static_cast<int>(context.glH * (1.0f - m_screenPos[1]));

    m_lastScreenPos[0] = xStart;
    m_lastScreenPos[1] = yStart - m_labelROI.height();

    // colors
    bool highlighted = (!pushName && isSelected());
    // default background color
    unsigned char alpha =
            static_cast<unsigned char>((context.labelOpacity / 100.0) * 255);
    ecvColor::Rgbaub defaultBkgColor(context.labelDefaultBkgCol, alpha);
    // default border color (mustn't be totally transparent!)
    ecvColor::Rgbaub defaultBorderColor(ecvColor::red, 255);
    if (!highlighted) {
        // apply only half of the transparency
        unsigned char halfAlpha = static_cast<unsigned char>(
                (50.0 + context.labelOpacity / 200.0) * 255);
        defaultBorderColor =
                ecvColor::Rgbaub(context.labelDefaultBkgCol, halfAlpha);
    }

    m_labelROI = QRect(xStart, yStart - m_labelROI.height(), m_labelROI.width(),
                       m_labelROI.height());

    WIDGETS_PARAMETER param(WIDGETS_TYPE::WIDGET_RECTANGLE_2D,
                            this->getViewId());

    if (!pushName) {
        // compute arrow base position relatively to the label rectangle (for 0
        // to 8)
        int arrowBaseConfig = 0;

        // compute arrow head position
        CCVector3d arrowDest2D(0, 0, 0);
        for (size_t i = 0; i < count; ++i) {
            arrowDest2D += m_pickedPoints[i].pos2D;
        }
        arrowDest2D /= static_cast<PointCoordinateType>(count);

        int iArrowDestX = static_cast<int>(arrowDest2D.x - xStart);
        int iArrowDestY = static_cast<int>(arrowDest2D.y - yStart);
        {
            if (iArrowDestX < 0 /*m_labelROI.left()*/)  // left
                arrowBaseConfig += 0;
            else if (iArrowDestX >
                     m_labelROI.width() /*m_labelROI.right()*/)  // Right
                arrowBaseConfig += 2;
            else  // Middle
                arrowBaseConfig += 1;

            if (iArrowDestY > 0 /*-m_labelROI.top()*/)  // Top
                arrowBaseConfig += 0;
            else if (iArrowDestY <
                     -m_labelROI.height() /*-m_labelROI.bottom()*/)  // Bottom
                arrowBaseConfig += 6;
            else  // Middle
                arrowBaseConfig += 3;
        }

        // we make the arrow base start from the nearest corner
        if (arrowBaseConfig != 4)  // 4 = label above point!
        {
            // glFunc->glColor4ubv(defaultBorderColor.rgba);
            // glFunc->glBegin(GL_TRIANGLE_FAN);
            // glFunc->glVertex2i(iArrowDestX, iArrowDestY);

            WIDGETS_PARAMETER triangleParam(WIDGETS_TYPE::WIDGET_LINE_2D,
                                            this->getViewId());
            triangleParam.color = ecvColor::FromRgba(defaultBorderColor);
            triangleParam.p1 = QPoint(arrowDest2D.x, arrowDest2D.y);
            int newTop = m_labelROI.bottom();
            int newBottom = m_labelROI.top();
            switch (arrowBaseConfig) {
                case 0:  // top-left corner
                {
                    triangleParam.p2 = QPoint(m_labelROI.left(), newTop);
                }
                // triangleParam.p2 = QPoint(m_labelROI.left(), newTop - 2 *
                // arrowBaseSize); triangleParam.p3 = QPoint(m_labelROI.left(),
                // newTop); triangleParam.p4 = QPoint(m_labelROI.left() + 2 *
                // arrowBaseSize, newTop);
                break;
                case 1:  // top-middle edge
                {
                    triangleParam.p2 = QPoint(m_labelROI.center().x(), newTop);
                }
                // triangleParam.p2 = QPoint(std::max(m_labelROI.left(),
                // iArrowDestX - arrowBaseSize), newTop); triangleParam.p3 =
                // QPoint(std::min(m_labelROI.right(), iArrowDestX +
                // arrowBaseSize), newTop);
                break;
                case 2:  // top-right corner
                {
                    triangleParam.p2 = QPoint(m_labelROI.right(), newTop);
                }
                // triangleParam.p2 = QPoint(m_labelROI.right(), newTop - 2 *
                // arrowBaseSize); triangleParam.p3 = QPoint(m_labelROI.right(),
                // newTop); triangleParam.p4 = QPoint(m_labelROI.right() - 2 *
                // arrowBaseSize, newTop);
                break;
                case 3:  // middle-left edge
                {
                    triangleParam.p2 =
                            QPoint(m_labelROI.left(), m_labelROI.center().y());
                }
                // triangleParam.p2 = QPoint(m_labelROI.left(), std::min(newTop,
                // iArrowDestY + arrowBaseSize)); triangleParam.p3 =
                // QPoint(m_labelROI.left(), std::max(newBottom, iArrowDestY -
                // arrowBaseSize));
                break;
                case 4:  // middle of rectangle!
                    break;
                case 5:  // middle-right edge
                {
                    triangleParam.p2 =
                            QPoint(m_labelROI.right(), m_labelROI.center().y());
                }
                // triangleParam.p2 = QPoint(m_labelROI.right(),
                // std::min(newTop, iArrowDestY + arrowBaseSize));
                // triangleParam.p3 = QPoint(m_labelROI.right(),
                // std::max(newBottom, iArrowDestY - arrowBaseSize));
                break;
                case 6:  // bottom-left corner
                {
                    triangleParam.p2 = QPoint(m_labelROI.left(), newBottom);
                }
                // triangleParam.p2 = QPoint(m_labelROI.left(), newBottom + 2 *
                // arrowBaseSize); triangleParam.p3 = QPoint(m_labelROI.left(),
                // newBottom); triangleParam.p4 = QPoint(m_labelROI.left() + 2 *
                // arrowBaseSize, newBottom);
                break;
                case 7:  // bottom-middle edge
                {
                    triangleParam.p2 =
                            QPoint(m_labelROI.center().x(), newBottom);
                }
                /*triangleParam.p2 = QPoint(std::max(m_labelROI.left(),
                iArrowDestX - arrowBaseSize), newBottom); triangleParam.p3 =
                QPoint(std::min(m_labelROI.right(), iArrowDestX +
                arrowBaseSize), newBottom);*/
                break;
                case 8:  // bottom-right corner
                {
                    triangleParam.p2 = QPoint(m_labelROI.right(), newBottom);
                }
                // triangleParam.p2 = QPoint(m_labelROI.right(), newBottom + 2 *
                // arrowBaseSize); triangleParam.p3 = QPoint(m_labelROI.right(),
                // newBottom); triangleParam.p4 = QPoint(m_labelROI.right() - 2
                // * arrowBaseSize, newBottom);
                break;
            }

            ecvDisplayTools::DrawWidgets(triangleParam, false);
        }
    }

    // main rectangle
    param.color.r = defaultBkgColor.r / 255.0f;
    param.color.g = defaultBkgColor.g / 255.0f;
    param.color.b = defaultBkgColor.b / 255.0f;
    param.color.a = defaultBkgColor.a / 255.0f;
    param.filled = true;
    param.rect = m_labelROI;
    ecvDisplayTools::DrawWidgets(param, false);

    if (highlighted) {
        param.color.r = defaultBorderColor.r / 255.0f;
        param.color.g = defaultBorderColor.g / 255.0f;
        param.color.b = defaultBorderColor.b / 255.0f;
        param.color.a = defaultBorderColor.a / 255.0f;
        param.filled = false;
        ecvDisplayTools::DrawWidgets(param, false);
    }

    // display text
    if (!pushName) {
        int xStartRel = margin;
        int yStartRel = 0;
        yStartRel -= titleHeight;

        ecvColor::Rgbub defaultTextColor;
        if (context.labelOpacity < 40) {
            // under a given opacity level, we use the default text color
            // instead!
            defaultTextColor = context.textDefaultCol;
        } else {
            defaultTextColor =
                    ecvColor::Rgbub(255 - context.labelDefaultBkgCol.r,
                                    255 - context.labelDefaultBkgCol.g,
                                    255 - context.labelDefaultBkgCol.b);
        }

        // label title
        m_historyMessage << title;
        ecvDisplayTools::DisplayText(
                title, xStart + xStartRel, yStart + yStartRel,
                ecvDisplayTools::ALIGN_DEFAULT, 0, defaultTextColor.rgb,
                &titleFont, this->getViewId());
        yStartRel -= margin;

        if (m_showFullBody) {
#ifdef DRAW_CONTENT_AS_TAB
            int xCol = xStartRel;
            for (int c = 0; c < tab.colCount; ++c) {
                int width = tab.colWidth[c] + 2 * tabMarginX;
                int height = rowHeight + 2 * tabMarginY;

                int yRow = yStartRel;
                int actualRowCount =
                        std::min(tab.rowCount, tab.colContent[c].size());

                bool labelCol = ((c & 1) == 0);
                const unsigned char* textColor =
                        labelCol ? ecvColor::white.rgb : defaultTextColor.rgb;

                for (int r = 0; r < actualRowCount; ++r) {
                    if (r && (r % 3) == 0) yRow -= margin;

                    if (labelCol) {
                        // draw background
                        int rgbIndex = (r % 3);
                        ecvColor::Rgb tempColor;
                        if (rgbIndex == 0)
                            tempColor = ecvColor::red;
                        else if (rgbIndex == 1)
                            tempColor = c_darkGreen;
                        else if (rgbIndex == 2)
                            tempColor = ecvColor::blue;

                        param.color.r = tempColor.r / 255.0f;
                        param.color.g = tempColor.g / 255.0f;
                        param.color.b = tempColor.b / 255.0f;
                        param.color.a = 1.0f;
                        param.filled = true;
                        param.rect =
                                QRect(m_labelROI.x() + xCol,
                                      m_labelROI.y() + m_labelROI.height() +
                                              yRow - height,
                                      width, height);
                        ecvDisplayTools::DrawWidgets(param, false);
                    }

                    const QString& str = tab.colContent[c][r];

                    int xShift = 0;
                    if (labelCol) {
                        // align characters in the middle
#if (QT_VERSION <= QT_VERSION_CHECK(5, 0, 0))
                        xShift = (tab.colWidth[c] -
                                  QFontMetrics(bodyFont).width(str)) /
                                 2;
#else
                        xShift = (tab.colWidth[c] -
                                  QFontMetrics(bodyFont).horizontalAdvance(
                                          str)) /
                                 2;
#endif
                    } else {
                        // align digits on the right
#if (QT_VERSION <= QT_VERSION_CHECK(5, 0, 0))
                        xShift = tab.colWidth[c] -
                                 QFontMetrics(bodyFont).width(str);
#else
                        xShift = tab.colWidth[c] -
                                 QFontMetrics(bodyFont).horizontalAdvance(str);
#endif
                    }

                    m_historyMessage << str;
                    ecvDisplayTools::DisplayText(
                            str, xStart + xCol + tabMarginX + xShift,
                            yStart + yRow - rowHeight,
                            ecvDisplayTools::ALIGN_DEFAULT, 0, textColor,
                            &bodyFont, this->getViewId());

                    yRow -= height;
                }

                xCol += width;
            }
#else
            if (!body.empty()) {
                // display body
                yStartRel -= margin;
                for (int i = 0; i < body.size(); ++i) {
                    yStartRel -= rowHeight;
                    context.display->displayText(
                            body[i], xStart + xStartRel, yStart + yStartRel,
                            ecvGenericDisplayTools::ALIGN_DEFAULT, 0,
                            defaultTextColor.rgb, &bodyFont);
                }
            }
#endif  // DRAW_CONTENT_AS_TAB
        }
    }

    if (pushName) {
        // glFunc->glPopName();
    }
}
