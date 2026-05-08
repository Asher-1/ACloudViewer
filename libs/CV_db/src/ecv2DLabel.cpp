// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecv2DLabel.h"

#include <ecvGenericGLDisplay.h>
#include <ecvRedrawScope.h>
#include <ecvRepresentationManager.h>
#include <ecvViewManager.h>
#include <ecvViewRepresentation.h>

#include <QCoreApplication>
#include <QThread>

#include "ecvBasicTypes.h"
#include "ecvDisplayTools.h"
#include "ecvFacet.h"
#include "ecvGenericDisplayTools.h"
#include "ecvGenericPointCloud.h"
#include "ecvGuiParameters.h"
#include "ecvHObjectCaster.h"
#include "ecvPointCloud.h"
#include "ecvPolyline.h"
#include "ecvScalarField.h"
#include "ecvSphere.h"

// Qt
#include <QApplication>
#include <QFont>
#include <QLineF>
#include <QPainter>
#include <QScreen>
#include <QSharedPointer>

// Qt5/Qt6 Compatibility
#include <QtCompat.h>

// System
#include <assert.h>
#include <string.h>

#include <algorithm>  // For std::max, std::min

namespace {

static void mirrorUpdateScreenLikeDrawWidgetsSuffix() {
    if (QWidget* w = ecvViewManager::instance().activeWidget()) {
        w->update();
    }
    if (ecvGenericGLDisplay* v =
                ecvViewManager::instance().getEffectiveView()) {
        v->updateScene();
    }
    if (ecvViewManager::instance().viewCount() > 1) {
        ecvViewManager::instance().refreshAll();
    }
}

//! Route widget drawing to the active display when available.
inline void drawWidgetsDispatch(ecvGenericGLDisplay* display,
                                const WIDGETS_PARAMETER& param,
                                bool update = false) {
    if (display) {
        display->drawWidgets(param);
    } else if (ecvGenericGLDisplay* ev =
                       ecvViewManager::instance().getEffectiveView()) {
        ev->drawWidgets(param);
    }
    if (update) {
        mirrorUpdateScreenLikeDrawWidgetsSuffix();
    }
}

}  // namespace

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
      m_relMarkerScale(0.15f),  // Reduced from 1.0f for better visualization -
                                // prevents sphere from obscuring points
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
            if (m_pickedPoints[0].entity())
                processedName.replace(ENTITY_INDEX_0,
                                      m_pickedPoints[0].entity()->getViewId());
            if (m_pickedPoints[1].entity())
                processedName.replace(ENTITY_INDEX_1,
                                      m_pickedPoints[1].entity()->getViewId());
            if (count > 2) {
                processedName.replace(POINT_INDEX_2,
                                      QString::number(m_pickedPoints[2].index));
                if (m_pickedPoints[2].entity())
                    processedName.replace(
                            ENTITY_INDEX_2,
                            m_pickedPoints[2].entity()->getViewId());
            }
        }
    }

    return processedName;
}

void cc2DLabel::setPosition(float x, float y) {
    m_screenPos[0] = std::clamp(x, -0.05f, 0.95f);
    m_screenPos[1] = std::clamp(y, -0.05f, 0.95f);
}

bool cc2DLabel::move2D(
        int x, int y, int dx, int dy, int screenWidth, int screenHeight) {
    assert(screenHeight > 0 && screenWidth > 0);

    float oldX = m_screenPos[0], oldY = m_screenPos[1];
    setPosition(m_screenPos[0] + static_cast<float>(dx) / screenWidth,
                m_screenPos[1] - static_cast<float>(dy) / screenHeight);

    CVLog::PrintDebug(
            "[Label] move2D: '%s' dx=%d dy=%d screenWH=(%d,%d) "
            "pos (%.4f,%.4f)->(%.4f,%.4f)",
            qPrintable(getName()), dx, dy, screenWidth, screenHeight, oldX,
            oldY, m_screenPos[0], m_screenPos[1]);
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
            PickedPoint& pp = m_pickedPoints.back();
            if (pp.entity()) pp.entity()->removeDependencyWith(this);
            m_pickedPoints.pop_back();
        }
    }

    m_lastScreenPos[0] = m_lastScreenPos[1] = -1;
    m_labelROI = QRect(0, 0, 0, 0);
    setVisible(false);
    setName("Label");

    // THREAD-SAFETY: Do NOT remove the thread guard below.
    // BinFilter::LoadFileV2 deserializes entities in a QtConcurrent worker
    // thread.  When a cc2DLabel is created (ccHObject::New → constructor →
    // clear()), ecvRedrawScope triggers vtkRenderWindow::Render() which
    // requires the main/GUI thread for OpenGL context.  Calling Render()
    // from a worker thread causes SIGABRT (Qt assertion).
    // See: GDB backtrace — cc2DLabel::clear → ecvRedrawScope::~ecvRedrawScope
    //      → ecvGLView::redraw → vtkGenericOpenGLRenderWindow::Render.
    if (getDisplay() &&
        QThread::currentThread() == QCoreApplication::instance()->thread()) {
        ecvRedrawScope scope({this});
    }
}

void cc2DLabel::clear3Dviews() {
    auto doRemoveOn = [](ecvGenericGLDisplay* d, WIDGETS_TYPE t,
                         const QString& id) {
        WIDGETS_PARAMETER p(t, id);
        if (d) {
            p.context.display = d;
            d->removeWidgets(p);
        } else {
            ecvDisplayTools::RemoveWidgets(p);
        }
    };

    auto removeAllWidgetsOn = [&](ecvGenericGLDisplay* d) {
        doRemoveOn(d, WIDGETS_TYPE::WIDGET_LINE_3D, m_lineID);
        if (c_unitPointMarker) {
            for (int i = 0; i < 3; ++i) {
                doRemoveOn(d, WIDGETS_TYPE::WIDGET_POINT,
                           QString::number(i) + m_sphereIdfix);
                doRemoveOn(d, WIDGETS_TYPE::WIDGET_SPHERE,
                           QString::number(i) + m_sphereIdfix);
            }
        }
        if (c_unitTriMarker) {
            doRemoveOn(d, WIDGETS_TYPE::WIDGET_POLYLINE, m_contourIdfix);
            doRemoveOn(d, WIDGETS_TYPE::WIDGET_POLYGONMESH, m_surfaceIdfix);
        }
    };

    // Remove from ALL registered views to guarantee actors are cleaned up
    // regardless of which view originally rendered them.
    const auto& views = ecvViewManager::instance().getAllViews();
    for (auto* view : views) {
        if (view) removeAllWidgetsOn(view);
    }

    // Fallback: also remove via the static path (covers the primary VtkVis
    // when no views are registered or the entity was drawn before any
    // view was created).
    if (views.isEmpty()) {
        removeAllWidgetsOn(nullptr);
    }
}

void cc2DLabel::clear2Dviews() {
    auto doRemoveOn = [](ecvGenericGLDisplay* d, WIDGETS_TYPE t,
                         const QString& id) {
        WIDGETS_PARAMETER p(t, id);
        if (d) {
            p.context.display = d;
            d->removeWidgets(p);
        } else {
            ecvDisplayTools::RemoveWidgets(p);
        }
    };

    auto remove2DWidgetsOn = [&](ecvGenericGLDisplay* d) {
        if (!m_historyMessage.isEmpty()) {
            for (const QString& text : m_historyMessage) {
                doRemoveOn(d, WIDGETS_TYPE::WIDGET_T2D, text);
                doRemoveOn(d, WIDGETS_TYPE::WIDGET_RECTANGLE_2D, text);
            }
        }

        doRemoveOn(d, WIDGETS_TYPE::WIDGET_T2D, this->getViewId());
        doRemoveOn(d, WIDGETS_TYPE::WIDGET_RECTANGLE_2D, this->getViewId());

        size_t count = m_pickedPoints.size();
        for (size_t j = 0; j < count; ++j) {
            QString legendId =
                    QString("%1_legend_%2").arg(this->getViewId()).arg(j);
            doRemoveOn(d, WIDGETS_TYPE::WIDGET_T2D, legendId);
            doRemoveOn(d, WIDGETS_TYPE::WIDGET_RECTANGLE_2D, legendId);
        }

        doRemoveOn(d, WIDGETS_TYPE::WIDGET_CAPTION, this->getViewId());
    };

    const auto& views = ecvViewManager::instance().getAllViews();
    for (auto* view : views) {
        if (view) remove2DWidgetsOn(view);
    }
    if (views.isEmpty()) {
        remove2DWidgetsOn(nullptr);
    }

    m_historyMessage.clear();
}

void cc2DLabel::clearLabel(bool ignoreCaption) {
    clear3Dviews();
    clear2Dviews();
    if (!ignoreCaption) {
        const auto& views = ecvViewManager::instance().getAllViews();
        for (auto* view : views) {
            if (!view) continue;
            WIDGETS_PARAMETER p(WIDGETS_TYPE::WIDGET_CAPTION,
                                this->getViewId());
            p.context.display = view;
            view->removeWidgets(p);
        }
        if (views.isEmpty()) {
            WIDGETS_PARAMETER p(WIDGETS_TYPE::WIDGET_CAPTION,
                                this->getViewId());
            ecvDisplayTools::RemoveWidgets(p);
        }
    }
}

void cc2DLabel::updateLabel() {
    CC_DRAW_CONTEXT context;
    ecvGenericGLDisplay* disp = getDisplay();
    if (disp) {
        disp->getContext(context);
    } else if (auto* ev = ecvViewManager::instance().getEffectiveView()) {
        ev->getContext(context);
    } else {
        return;
    }
    update3DLabelView(context, false);
    update2DLabelView(context, false);
    { ecvRedrawScope scope({this}); }
}

void cc2DLabel::update3DLabelView(CC_DRAW_CONTEXT& context,
                                  bool updateScreen /* = true */) {
    context.drawingFlags = CC_DRAW_3D | CC_DRAW_FOREGROUND;
    drawMeOnly3D(context);
    if (updateScreen) {
        {
            ecvRedrawScope scope({this});
        }
    }
}

void cc2DLabel::update2DLabelView(CC_DRAW_CONTEXT& context,
                                  bool updateScreen /* = true */) {
    context.drawingFlags = CC_DRAW_2D | CC_DRAW_FOREGROUND;
    drawMeOnly2D(context);
    if (updateScreen) {
        {
            ecvRedrawScope scope({this});
        }
    }
}

void cc2DLabel::onDeletionOf(const ccHObject* obj) {
    ccHObject::onDeletionOf(obj);  // remove dependencies, etc.

    // check that associated entities (clouds or meshes) are not about to be
    // deleted
    size_t pointsToRemove = 0;
    {
        for (size_t i = 0; i < m_pickedPoints.size(); ++i)
            if (m_pickedPoints[i].entity() == obj) ++pointsToRemove;
    }

    if (pointsToRemove == 0) return;

    if (pointsToRemove == m_pickedPoints.size()) {
        clear(true);  // don't call clear as we don't want/need to update input
                      // object's dependencies!
    } else {
        // remove only the necessary points
        size_t j = 0;
        for (size_t i = 0; i < m_pickedPoints.size(); ++i) {
            if (m_pickedPoints[i].entity() != obj) {
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
            if (m_pickedPoints[0].entity() == m_pickedPoints[1].entity())
                setName(QString("Vector #") + POINT_INDEX_0 + QString(" - #") +
                        POINT_INDEX_1);
            else
                setName(QString("Vector #") + POINT_INDEX_0 + QString("@") +
                        ENTITY_INDEX_0 + QString(" - #") + POINT_INDEX_1 +
                        QString("@") + ENTITY_INDEX_1);
            break;
        case 3:
            if (m_pickedPoints[0].entity() == m_pickedPoints[2].entity() &&
                m_pickedPoints[1].entity() == m_pickedPoints[2].entity())
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

bool cc2DLabel::toFile_MeOnly(QFile& out, short dataVersion) const {
    assert(out.isOpen() && (out.openMode() & QIODevice::WriteOnly));
    if (dataVersion < 50) {
        assert(false);
        return false;
    }

    if (!ccHObject::toFile_MeOnly(out, dataVersion)) return false;

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

    // Relative marker scale (dataVersion >= 49) - IMPORTANT for sphere size!
    // This is always written when saving, but only read when dataVersion >= 49
    // to maintain backward compatibility with version 48 and earlier
    if (out.write((const char*)&m_relMarkerScale, sizeof(float)) < 0)
        return WriteError();

    return true;
}

short cc2DLabel::minimumFileVersion_MeOnly() const {
    return std::max(static_cast<short>(50),
                    ccHObject::minimumFileVersion_MeOnly());
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

    if (dataVersion > 48) {
        // Relative marker scale (dataVersion >= 49) - IMPORTANT for sphere
        // size! Read the saved value to preserve custom sphere sizes
        if (in.read((char*)&m_relMarkerScale, sizeof(float)) < 0)
            return ReadError();
    }
    // else: use constructor default value (0.15f) for old files (version <= 48)
    // This automatically fixes sphere size for old labels

    return true;
}

void AddPointCoordinates(QStringList& body,
                         const cc2DLabel::PickedPoint& pp,
                         int precision,
                         QString pointName = QString()) {
    ccGenericPointCloud* cloud = pp.cloudOrVertices();
    if (!cloud) return;

    CCVector3 P = pp.getPointPosition();
    bool isShifted = cloud->isShifted();

    QString coordStr;
    if (pp.mesh)
        coordStr = QString("P@Tri#%0:").arg(pp.index);
    else
        coordStr = QString("P#%0:").arg(pp.index);
    if (!pointName.isEmpty())
        coordStr = QString("%1 (%2)").arg(pointName, coordStr);
    if (isShifted) {
        body << coordStr;
        coordStr = QString("  [shifted]");
    }

    coordStr += QString(" (%1;%2;%3)")
                        .arg(P.x, 0, 'f', precision)
                        .arg(P.y, 0, 'f', precision)
                        .arg(P.z, 0, 'f', precision);
    body << coordStr;

    if (isShifted) {
        CCVector3d Pg = cloud->toGlobal3d(P);
        QString globCoordStr = QString("  [original] (%1;%2;%3)")
                                       .arg(Pg.x, 0, 'f', precision)
                                       .arg(Pg.y, 0, 'f', precision)
                                       .arg(Pg.z, 0, 'f', precision);
        body << globCoordStr;
    }
}

void cc2DLabel::getLabelInfo1(LabelInfo1& info) const {
    info.cloud = nullptr;
    if (m_pickedPoints.size() != 1) return;

    const PickedPoint& pp = m_pickedPoints[0];
    info.cloud = pp.cloudOrVertices();
    if (!info.cloud) {
        assert(false);
        return;
    }
    info.pointIndex = pp.index;

    // point-level attributes (normal, color, SF) are only meaningful
    // for direct cloud picks where index is a point index;
    // for mesh picks, index is a triangle index
    if (!pp.cloud) return;

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
    info.cloud1 = info.cloud2 = nullptr;
    if (m_pickedPoints.size() != 2) return;

    info.cloud1 = m_pickedPoints[0].cloudOrVertices();
    info.point1Index = m_pickedPoints[0].index;
    info.cloud2 = m_pickedPoints[1].cloudOrVertices();
    info.point2Index = m_pickedPoints[1].index;

    CCVector3 P1 = m_pickedPoints[0].getPointPosition();
    CCVector3 P2 = m_pickedPoints[1].getPointPosition();
    info.diff = P2 - P1;
}

void cc2DLabel::getLabelInfo3(LabelInfo3& info) const {
    info.cloud1 = info.cloud2 = info.cloud3 = nullptr;
    if (m_pickedPoints.size() != 3) return;

    info.cloud1 = m_pickedPoints[0].cloudOrVertices();
    info.point1Index = m_pickedPoints[0].index;
    info.cloud2 = m_pickedPoints[1].cloudOrVertices();
    info.point2Index = m_pickedPoints[1].index;
    info.cloud3 = m_pickedPoints[2].cloudOrVertices();
    info.point3Index = m_pickedPoints[2].index;

    CCVector3 P1 = m_pickedPoints[0].getPointPosition();
    CCVector3 P2 = m_pickedPoints[1].getPointPosition();
    CCVector3 P3 = m_pickedPoints[2].getPointPosition();

    // area
    CCVector3 P1P2 = P2 - P1;
    CCVector3 P1P3 = P3 - P1;
    CCVector3 P2P3 = P3 - P2;
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
            AddPointCoordinates(body, m_pickedPoints[0], precision);

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

            AddPointCoordinates(body, m_pickedPoints[0], precision);
            AddPointCoordinates(body, m_pickedPoints[1], precision);
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
            AddPointCoordinates(body, m_pickedPoints[0], precision, "A");
            AddPointCoordinates(body, m_pickedPoints[1], precision, "B");
            AddPointCoordinates(body, m_pickedPoints[2], precision, "C");

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
    if (button == Qt::MiddleButton || button == Qt::RightButton) {
        QRect rect = QRect(0, 0, m_labelROI.width(), m_labelROI.height());
        if (rect.contains(x - m_lastScreenPos[0], y - m_lastScreenPos[1])) {
            m_showFullBody = !m_showFullBody;
            CC_DRAW_CONTEXT context;
            ecvGenericGLDisplay* disp = getDisplay();
            if (disp) {
                disp->getContext(context);
            } else if (auto* ev =
                               ecvViewManager::instance().getEffectiveView()) {
                ev->getContext(context);
            } else {
                return false;
            }
            update2DLabelView(context, true);
            return true;
        }
    }

    return false;
}

void cc2DLabel::drawMeOnly(CC_DRAW_CONTEXT& context) {
    if (m_pickedPoints.empty()) return;
    if (!MACRO_Foreground(context)) return;
    if (MACRO_VirtualTransEnabled(context)) return;

    if (!isRedraw() && !context.forceRedraw) {
        return;
    }

    // Labels are bound to exactly ONE display window via getDisplay().
    // In multi-view mode, an unbound label (getDisplay() == nullptr) is
    // auto-bound to the active view on first draw, then stays there
    // permanently.  This is simpler than per-view representation logic
    // and matches the expected UX: labels belong to the window where
    // they were created, regardless of which window is currently active.
    if (context.display) {
        ecvGenericGLDisplay* myDisp = getDisplay();
        if (!myDisp) {
            // Unbound label: bind to current active view once.
            auto* activeView = ecvViewManager::instance().getActiveView();
            if (activeView && activeView == context.display) {
                const_cast<cc2DLabel*>(this)->setDisplay(context.display);
            } else {
                return;
            }
        } else if (myDisp != context.display) {
            return;
        }
    }

    if (MACRO_Draw3D(context)) {
        drawMeOnly3D(context);
    } else if (MACRO_Draw2D(context)) {
        drawMeOnly2D(context);
    }
}

void cc2DLabel::drawMeOnly3D(CC_DRAW_CONTEXT& context) {
    clear3Dviews();
    if (!isVisible() || !isEnabled()) {
        return;
    }

    size_t count = m_pickedPoints.size();
    if (count == 0) {
        return;
    }

    if (!context.display || !context.display->asWidget()) {
        assert(false);
        return;
    }

    // standard case: list names pushing
    bool entityPickingMode = MACRO_EntityPicking(context);
    if (entityPickingMode) {
        // not particularly fast
        if (MACRO_FastEntityPicking(context)) return;
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

                        *A = m_pickedPoints[0].getPointPosition();
                        *B = m_pickedPoints[1].getPointPosition();
                        *C = m_pickedPoints[2].getPointPosition();

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
                        if (c_unitTriMarker->getPolygon()) {
                            c_unitTriMarker->getPolygon()->setOpacity(0.5);
                            c_unitTriMarker->getPolygon()->setTempColor(
                                    ecvColor::yellow);
                            c_unitTriMarker->getPolygon()->setVisible(true);
                        }
                        if (c_unitTriMarker->getContour()) {
                            c_unitTriMarker->getContour()->setColor(
                                    ecvColor::red);
                            c_unitTriMarker->getContour()->showColors(true);
                            c_unitTriMarker->getContour()->setVisible(true);
                        }
                        c_unitTriMarker->setTempColor(ecvColor::darkGrey);
                        c_unitTriMarker->showColors(true);
                        c_unitTriMarker->setVisible(true);
                        c_unitTriMarker->setEnabled(true);
                        if (c_unitTriMarker->getPolygon()) {
                            m_surfaceIdfix =
                                    this->getViewId() + SEPARATOR +
                                    c_unitTriMarker->getPolygon()->getViewId();
                            c_unitTriMarker->getPolygon()->setFixedId(true);
                        }
                        if (c_unitTriMarker->getContour()) {
                            m_contourIdfix =
                                    this->getViewId() + SEPARATOR +
                                    c_unitTriMarker->getContour()->getViewId();
                            c_unitTriMarker->getContour()->setFixedId(true);
                        }
                        c_unitTriMarker->setFixedId(true);
                    }

                    if (m_surfaceIdfix == "" && c_unitTriMarker &&
                        c_unitTriMarker->getPolygon()) {
                        m_surfaceIdfix =
                                this->getViewId() + SEPARATOR +
                                c_unitTriMarker->getPolygon()->getViewId();
                    }
                    if (m_contourIdfix == "" && c_unitTriMarker &&
                        c_unitTriMarker->getContour()) {
                        m_contourIdfix =
                                this->getViewId() + SEPARATOR +
                                c_unitTriMarker->getContour()->getViewId();
                    }
                }

                if (!c_unitTriMarker ||
                    !c_unitTriMarker->getContourVertices()) {
                    break;
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

                *A = m_pickedPoints[0].getPointPosition();
                *B = m_pickedPoints[1].getPointPosition();
                *C = m_pickedPoints[2].getPointPosition();

                // build-up point maker own 'context'
                CC_DRAW_CONTEXT markerContext = context;
                // we must remove the 'push name flag' so that the sphere
                // doesn't push its own!
                markerContext.drawingFlags &= (~CC_ENTITY_PICKING);

                // draw triangle contour
                if (c_unitTriMarker->getContour()) {
                    if (context.display)
                        c_unitTriMarker->getContour()->setDisplay(
                                context.display);
                    markerContext.viewID = m_contourIdfix;
                    c_unitTriMarker->getContour()->setRedraw(true);
                    c_unitTriMarker->getContour()->draw(markerContext);
                }
                // draw triangle mesh surface
                if (c_unitTriMarker->getPolygon()) {
                    if (context.display)
                        c_unitTriMarker->getPolygon()->setDisplay(
                                context.display);
                    markerContext.viewID = m_surfaceIdfix;
                    c_unitTriMarker->getPolygon()->setRedraw(true);
                    c_unitTriMarker->getPolygon()->draw(markerContext);
                }
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

                CCVector3 lineSt = m_pickedPoints[0].getPointPosition();
                CCVector3 lineEd = m_pickedPoints[1].getPointPosition();

                // we draw the line
                WIDGETS_PARAMETER param(WIDGETS_TYPE::WIDGET_LINE_3D, m_lineID);
                param.context.display = context.display;
                param.setLineWidget(
                        LineWidget(lineSt, lineEd, lineWidth, lineColor));
                drawWidgetsDispatch(context.display, param, false);
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
                markerContext.drawingFlags &= (~CC_ENTITY_PICKING);

                ecvColor::Rgb markerColor =
                        (isSelected() && !entityPickingMode)
                                ? ecvColor::red
                                : context.labelDefaultMarkerCol;
                c_unitPointMarker->setTempColor(markerColor);

                static constexpr float LABEL_MARKER_PIXEL_SIZE = 10.0f;
                for (size_t i = 0; i < count; i++) {
                    CCVector3 P = m_pickedPoints[i].getPointPosition();

                    WIDGETS_PARAMETER param(WIDGETS_TYPE::WIDGET_POINT,
                                            QString::number(i) + m_sphereIdfix);
                    param.pointSize = LABEL_MARKER_PIXEL_SIZE;
                    param.context.display = context.display;
                    m_pickedPoints[i].markerScale = LABEL_MARKER_PIXEL_SIZE;
                    param.center = P;
                    param.color = ecvColor::FromRgbub(markerColor);
                    drawWidgetsDispatch(context.display, param, false);
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
    if (!context.display || !context.display->asWidget()) {
        assert(false);
        return;
    }

    m_overlayData.clear();
    m_historyMessage.clear();

    clear2Dviews();
    if (!isVisible() || !isEnabled()) {
        clearLabel(false);
        return;
    }

    if (m_pickedPoints.empty()) {
        return;
    }

    // standard case: list names pushing
    bool entityPickingMode = MACRO_EntityPicking(context);

    size_t count = m_pickedPoints.size();
    assert(count != 0);

    // hack: we display the label connecting 'segments' and the point(s) legend
    // in 2D so that they always appear above the entities
    {
        // don't do this in picking mode!
        if (!entityPickingMode) {
            // we always project the points in 2D (maybe useful later, even when
            // displaying the label during the 2D pass!)
            ccGLCameraParameters camera;
            if (context.display) {
                context.display->getGLCameraParameters(camera);
            } else if (auto* ev =
                               ecvViewManager::instance().getEffectiveView()) {
                ev->getGLCameraParameters(camera);
            }
            for (size_t i = 0; i < count; i++) {
                CCVector3 P3D = m_pickedPoints[i].getPointPosition();
                camera.projectSafe(P3D, m_pickedPoints[i].pos2D);
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
            if (!entityPickingMode) {
                const float dpr = context.devicePixelRatio;
                const float lH = context.glH / dpr;

                // Connecting segments between picked points
                // (drawn before m_dispIn2D check, always visible when
                // points are visible — consistent with CloudCompare)
                if (count > 1) {
                    m_overlayData.segmentColor =
                            isSelected()
                                    ? QColor(Qt::red)
                                    : QColor(context.labelDefaultMarkerCol.r,
                                             context.labelDefaultMarkerCol.g,
                                             context.labelDefaultMarkerCol.b);
                    for (size_t i = 0; i < count; ++i) {
                        size_t j = (i + 1) % count;
                        if (count == 2 && i == 1) break;
                        float px0 =
                                static_cast<float>(m_pickedPoints[i].pos2D.x) /
                                dpr;
                        float py0 =
                                lH - 1.0f -
                                static_cast<float>(m_pickedPoints[i].pos2D.y) /
                                        dpr;
                        float px1 =
                                static_cast<float>(m_pickedPoints[j].pos2D.x) /
                                dpr;
                        float py1 =
                                lH - 1.0f -
                                static_cast<float>(m_pickedPoints[j].pos2D.y) /
                                        dpr;
                        LabelOverlayData::Segment2D seg;
                        seg.from = QPointF(px0, py0);
                        seg.to = QPointF(px1, py1);
                        m_overlayData.segments.push_back(seg);
                    }
                }

                // Point legends (controlled by m_dispPointsLegend)
                if (m_dispPointsLegend) {
                    QFont font = QApplication::font();
                    if (context.display) {
                        font.setPointSize(static_cast<int>(
                                context.display->getDisplayParameters()
                                        .defaultFontSize));
                    } else if (auto* ev = ecvViewManager::instance()
                                                  .getEffectiveView()) {
                        font.setPointSize(static_cast<int>(
                                ev->getDisplayParameters().defaultFontSize));
                    } else {
                        font.setPointSize(static_cast<int>(
                                ecvGui::Parameters().defaultFontSize));
                    }
                    {
                        int basePt = font.pointSize();
                        if (basePt < 6) basePt = 6;
                        font.setPointSize(std::min(basePt, 9));
                    }
                    font.setBold(false);
                    static const QChar ABC[3] = {'A', 'B', 'C'};

                    for (size_t j = 0; j < count; j++) {
                        QString legendText;
                        if (count == 1)
                            legendText = getName();
                        else if (count == 3)
                            legendText = ABC[j];

                        if (legendText.isEmpty()) continue;

                        LabelOverlayData::Legend leg;
                        leg.text = legendText;
                        leg.font = font;
                        leg.color = Qt::white;
                        float vtkX =
                                static_cast<float>(m_pickedPoints[j].pos2D.x) +
                                context.labelMarkerTextShift_pix;
                        float vtkY =
                                static_cast<float>(m_pickedPoints[j].pos2D.y) +
                                context.labelMarkerTextShift_pix;
                        leg.pos = QPointF(vtkX / dpr, lH - 1.0f - vtkY / dpr);
                        m_overlayData.legends.push_back(leg);
                    }
                }
            }
        } else {
            return;
        }
    }

    // Legend-only mode: mark overlay valid so legends are painted
    if (!m_overlayData.legends.isEmpty() || !m_overlayData.segments.isEmpty()) {
        m_overlayData.valid = true;
    }

    // Only display full panel when dispIn2D is set
    if (!m_dispIn2D) {
        WIDGETS_PARAMETER capP(WIDGETS_TYPE::WIDGET_CAPTION, this->getViewId());
        capP.context.display = context.display;
        if (context.display) {
            context.display->removeWidgets(capP);
        } else {
            ecvDisplayTools::RemoveWidgets(capP);
        }
        m_labelROI = QRect(0, 0, 0, 0);
        m_lastScreenPos[0] = m_lastScreenPos[1] = -1;
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
    if (!entityPickingMode) {
        /*** label border ***/
        bodyFont = QApplication::font();
        ecvGenericGLDisplay* labelDisp =
                context.display ? context.display
                                : ecvViewManager::instance().getEffectiveView();
        const ecvGui::ParamStruct& lp =
                labelDisp ? labelDisp->getDisplayParameters()
                          : ecvGui::Parameters();
        bodyFont.setPointSize(ecvGenericDisplayTools::FontSizeModifier(
                static_cast<int>(lp.labelFontSize), context.renderZoom));
        titleFont = bodyFont;  // label body + title use same base zoom

        {
            int bodyPt = bodyFont.pointSize();
            int titlePt = titleFont.pointSize();
            if (bodyPt < 8) bodyPt = 8;
            if (titlePt < 9) titlePt = 9;
            bodyFont.setPointSize(std::min(bodyPt, 11));
            titleFont.setPointSize(std::min(titlePt, 12));
        }
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
            dx = std::max(dx,
                          QTCOMPAT_FONTMETRICS_WIDTH(titleFontMetrics, title));

            dy += margin;       // top vertical margin
            dy += titleHeight;  // title

            if (m_showFullBody) {
#ifdef DRAW_CONTENT_AS_TAB
                try {
                    if (count == 1) {
                        LabelInfo1 info;
                        getLabelInfo1(info);

                        ccGenericPointCloud* cloud =
                                m_pickedPoints[0].cloudOrVertices();
                        if (cloud) {
                            bool isShifted = cloud->isShifted();
                            CCVector3 Pt = m_pickedPoints[0].getPointPosition();
                            // 1st block: X, Y, Z (local)
                            {
                                int c = tab.add2x3Block();
                                QChar suffix = ' ';
                                if (isShifted) {
                                    suffix = 'l';  //'l' for local
                                }
                                tab.colContent[c] << QString("X") + suffix;
                                tab.colContent[c + 1] << QString::number(
                                        Pt.x, 'f', precision);
                                tab.colContent[c] << QString("Y") + suffix;
                                tab.colContent[c + 1] << QString::number(
                                        Pt.y, 'f', precision);
                                tab.colContent[c] << QString("Z") + suffix;
                                tab.colContent[c + 1] << QString::number(
                                        Pt.z, 'f', precision);
                            }
                            // next block:  X, Y, Z (global)
                            if (isShifted) {
                                int c = tab.add2x3Block();
                                CCVector3d P = cloud->toGlobal3d(Pt);
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
                    assert(!entityPickingMode);
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
                        dx = std::max(dx, QTCOMPAT_FONTMETRICS_WIDTH(
                                                  bodyFontMetrics, body[j]));
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
    // CRITICAL FIX for macOS: m_screenPos is relative coordinates (0.0 to 1.0)
    // context.glW and context.glH are physical pixels (already scaled by
    // devicePixelRatio) m_screenPos is stored relative to LOGICAL pixels (as
    // seen in move2D implementation) We need to use logical pixels for position
    // calculation to match the coordinate system used by move2D and DrawWidgets
    // const int xStart = static_cast<int>(context.glW * m_screenPos[0]);
    // const int yStart = static_cast<int>(context.glH * (1.0f -
    // m_screenPos[1]));
    const float logicalW = context.glW / context.devicePixelRatio;
    const float logicalH = context.glH / context.devicePixelRatio;
    const int xStart = static_cast<int>(logicalW * m_screenPos[0]);
    const int yStart = static_cast<int>(logicalH * (1.0f - m_screenPos[1]));

    m_lastScreenPos[0] = xStart;
    m_lastScreenPos[1] = yStart - m_labelROI.height();

    // colors
    bool highlighted = (!entityPickingMode && isSelected());
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
    if (!entityPickingMode) {
        // label title
        m_historyMessage << title;

        if (m_showFullBody) {
            // Create QFontMetrics for text alignment calculations
            QFontMetrics bodyFontMetrics(bodyFont);

            for (int r = 0; r < tab.rowCount; ++r) {
                QString str;
                for (int c = 0; c < tab.colCount; ++c) {
                    QString cellContent = tab.colContent[c][r];
                    // Calculate actual text width
#if (QT_VERSION <= QT_VERSION_CHECK(5, 0, 0))
                    int textWidth = bodyFontMetrics.width(cellContent);
#else
                    int textWidth =
                            bodyFontMetrics.horizontalAdvance(cellContent);
#endif
                    // Calculate target width (column width + margin for
                    // spacing)
                    int targetWidth = tab.colWidth[c];
                    if (c < tab.colCount - 1) {
                        // Add margin after each column except the last
                        targetWidth += tabMarginX;
                    }
                    // Add spaces to align text
                    int spaceWidth = textWidth < targetWidth
                                             ? targetWidth - textWidth
                                             : 0;
                    if (spaceWidth > 0) {
                        // Calculate number of spaces needed (approximate)
                        // Use average character width for spacing
#if (QT_VERSION <= QT_VERSION_CHECK(5, 0, 0))
                        int spaceCharWidth = bodyFontMetrics.width(' ');
#else
                        int spaceCharWidth =
                                bodyFontMetrics.horizontalAdvance(' ');
#endif
                        int numSpaces = spaceCharWidth > 0
                                                ? (spaceWidth + spaceCharWidth -
                                                   1) / spaceCharWidth
                                                : 0;
                        cellContent += QString(numSpaces, ' ');
                    }
                    str += cellContent;
                }
                m_historyMessage << str;
            }
        }
    }

    if (!entityPickingMode && count > 0 && !m_historyMessage.empty()) {
        const float dpr = context.devicePixelRatio;
        const float lH = context.glH / dpr;

        m_overlayData.panelRect = QRectF(m_labelROI);
        m_overlayData.bkgColor = QColor(defaultBkgColor.r, defaultBkgColor.g,
                                        defaultBkgColor.b, defaultBkgColor.a);
        m_overlayData.borderColor =
                QColor(defaultBorderColor.r, defaultBorderColor.g,
                       defaultBorderColor.b, defaultBorderColor.a);
        m_overlayData.textColor = QColor(defaultTextColor.r, defaultTextColor.g,
                                         defaultTextColor.b);
        m_overlayData.highlighted = highlighted;
        m_overlayData.title = title;
        m_overlayData.titleFont = titleFont;
        m_overlayData.bodyFont = bodyFont;
        m_overlayData.titleHeight = titleHeight;
        m_overlayData.rowHeight = rowHeight;
        m_overlayData.margin = margin;
        m_overlayData.tabMarginX = tabMarginX;
        m_overlayData.tabMarginY = tabMarginY;

        if (m_showFullBody && m_historyMessage.size() > 1) {
            m_overlayData.bodyLines = m_historyMessage.mid(1);
        } else {
            m_overlayData.bodyLines.clear();
        }

        // Store per-column structure for CloudCompare-style colored label cells
        m_overlayData.columns.clear();
        m_overlayData.tabCells.clear();
        m_overlayData.tabRowCount = 0;
        if (m_showFullBody && tab.colCount > 0) {
            m_overlayData.tabRowCount = tab.rowCount;
            int xCol = 0;
            for (int c = 0; c < tab.colCount; ++c) {
                LabelOverlayData::TabColumn col;
                col.xOffset = xCol;
                col.width = tab.colWidth[c] + 2 * tabMarginX;
                col.isLabel = ((c & 1) == 0);
                m_overlayData.columns.push_back(col);
                xCol += col.width;
            }
            for (int r = 0; r < tab.rowCount; ++r) {
                QVector<QString> rowCells;
                for (int c = 0; c < tab.colCount; ++c) {
                    if (r < static_cast<int>(tab.colContent[c].size()))
                        rowCells.push_back(tab.colContent[c][r]);
                    else
                        rowCells.push_back(QString());
                }
                m_overlayData.tabCells.push_back(rowCells);
            }
        }

        // Arrow wedge: from panel edge to centroid of projected points
        QPointF centroid2D(0, 0);
        for (size_t i = 0; i < count; ++i) {
            float cx = static_cast<float>(m_pickedPoints[i].pos2D.x) / dpr;
            float cy = lH - 1.0f -
                       static_cast<float>(m_pickedPoints[i].pos2D.y) / dpr;
            centroid2D += QPointF(cx, cy);
        }
        centroid2D /= static_cast<qreal>(count);

        QRectF panel = m_overlayData.panelRect;
        QPointF edgePt;
        if (centroid2D.y() > panel.bottom()) {
            edgePt =
                    QPointF(qBound(panel.left(), centroid2D.x(), panel.right()),
                            panel.bottom());
        } else if (centroid2D.y() < panel.top()) {
            edgePt =
                    QPointF(qBound(panel.left(), centroid2D.x(), panel.right()),
                            panel.top());
        } else if (centroid2D.x() < panel.left()) {
            edgePt = QPointF(panel.left(), qBound(panel.top(), centroid2D.y(),
                                                  panel.bottom()));
        } else {
            edgePt = QPointF(panel.right(), qBound(panel.top(), centroid2D.y(),
                                                   panel.bottom()));
        }

        int arrowBase = static_cast<int>(c_arrowBaseSize * context.renderZoom);
        QPointF dir = centroid2D - edgePt;
        qreal len = QLineF(edgePt, centroid2D).length();
        if (len > 3.0) {
            QPointF perp(-dir.y() / len, dir.x() / len);
            QPolygonF wedge;
            wedge << (edgePt + perp * arrowBase) << centroid2D
                  << (edgePt - perp * arrowBase);
            m_overlayData.arrowPolygon = wedge;
        } else {
            m_overlayData.arrowPolygon.clear();
        }

        m_overlayData.valid = true;
    }
}

void cc2DLabel::paintOverlay(QPainter& painter) const {
    if (!m_overlayData.valid) return;

    const auto& od = m_overlayData;
    const QRectF& panel = od.panelRect;
    bool hasPanel = panel.width() > 0 && panel.height() > 0;

    static const QColor s_darkGreen(0, 200, 0);

    // CloudCompare: depth off, draw point-to-point segments BEFORE panel
    if (!od.segments.isEmpty()) {
        QPen segPen(od.segmentColor, 4.0);
        painter.setPen(segPen);
        painter.setBrush(Qt::NoBrush);
        for (const auto& seg : od.segments) {
            painter.drawLine(seg.from, seg.to);
        }
    }

    if (hasPanel) {
        // Arrow wedge (panel edge → centroid, drawn before panel)
        if (!od.arrowPolygon.isEmpty()) {
            painter.setPen(Qt::NoPen);
            painter.setBrush(od.borderColor);
            painter.drawPolygon(od.arrowPolygon);
        }

        // Panel background
        painter.setPen(Qt::NoPen);
        painter.setBrush(od.bkgColor);
        painter.drawRect(panel);

        // Panel border (CloudCompare: line width 3.0 * renderZoom)
        QPen borderPen(od.borderColor, od.highlighted ? 3.0 : 2.0);
        painter.setPen(borderPen);
        painter.setBrush(Qt::NoBrush);
        painter.drawRect(panel);

        // Title
        painter.setPen(od.textColor);
        painter.setFont(od.titleFont);
        QRectF titleRect(panel.left() + od.margin, panel.top() + od.margin,
                         panel.width() - 2 * od.margin, od.titleHeight);
        painter.drawText(titleRect, Qt::AlignLeft | Qt::AlignVCenter, od.title);

        // Body: per-column rendering (CloudCompare style)
        if (!od.columns.isEmpty() && !od.tabCells.isEmpty()) {
            int rowH = od.rowHeight + 2 * od.tabMarginY;
            QFontMetrics bfm(od.bodyFont);
            painter.setFont(od.bodyFont);

            for (int c = 0; c < od.columns.size(); ++c) {
                const auto& col = od.columns[c];
                qreal xCol = panel.left() + od.margin + col.xOffset;
                qreal yRow = panel.top() + od.margin + od.titleHeight +
                             od.tabMarginY + 2;

                int actualRowCount = qMin(od.tabRowCount, od.tabCells.size());
                for (int r = 0; r < actualRowCount; ++r) {
                    if (r > 0 && (r % 3) == 0) yRow += od.margin;

                    if (col.isLabel) {
                        int rgbIdx = r % 3;
                        QColor cellBg;
                        if (rgbIdx == 0)
                            cellBg = QColor(255, 0, 0);
                        else if (rgbIdx == 1)
                            cellBg = s_darkGreen;
                        else
                            cellBg = QColor(0, 0, 255);

                        QRectF cellRect(xCol, yRow, col.width, rowH);
                        painter.setPen(Qt::NoPen);
                        painter.setBrush(cellBg);
                        painter.drawRect(cellRect);
                    }

                    const QString& str = (c < od.tabCells[r].size())
                                                 ? od.tabCells[r][c]
                                                 : QString();
                    int xShift = 0;
                    if (col.isLabel) {
                        xShift = (col.width - 2 * od.tabMarginX -
                                  bfm.horizontalAdvance(str)) /
                                 2;
                        painter.setPen(Qt::white);
                    } else {
                        xShift = col.width - 2 * od.tabMarginX -
                                 bfm.horizontalAdvance(str);
                        painter.setPen(od.textColor);
                    }
                    painter.setBrush(Qt::NoBrush);
                    QRectF textRect(xCol + od.tabMarginX + xShift, yRow,
                                    col.width, rowH);
                    painter.drawText(textRect, Qt::AlignLeft | Qt::AlignVCenter,
                                     str);

                    yRow += rowH;
                }
            }
        } else if (!od.bodyLines.isEmpty()) {
            // Fallback: no column info, draw flattened rows
            qreal yOff = panel.top() + od.margin + od.titleHeight +
                         od.tabMarginY + 2;
            int rowH = od.rowHeight + 2 * od.tabMarginY;
            painter.setFont(od.bodyFont);
            painter.setPen(od.textColor);
            for (int i = 0; i < od.bodyLines.size(); ++i) {
                QRectF rowRect(panel.left() + od.margin, yOff,
                               panel.width() - 2 * od.margin, rowH);
                painter.drawText(rowRect, Qt::AlignLeft | Qt::AlignVCenter,
                                 od.bodyLines[i]);
                yOff += rowH;
                if (((i + 1) % 3) == 0 && i + 1 < od.bodyLines.size())
                    yOff += od.margin;
            }
        }
    }

    // CloudCompare: ABC legends near 3D markers = white text on black
    // semi-transparent background (same as displayText with bkgAlpha)
    for (const auto& leg : od.legends) {
        QFont boldFont = leg.font;
        boldFont.setBold(true);
        painter.setFont(boldFont);
        QFontMetrics fm(boldFont);
        QRect textRect = fm.boundingRect(leg.text);
        int margin = fm.height() / 4;
        QRect bgRect(leg.pos.x() - margin, leg.pos.y() - fm.ascent() - margin,
                     textRect.width() + 2 * margin, fm.height() + 2 * margin);
        painter.setPen(Qt::NoPen);
        painter.setBrush(QColor(0, 0, 0, 178));
        painter.drawRect(bgRect);
        painter.setPen(Qt::white);
        painter.setBrush(Qt::NoBrush);
        painter.drawText(leg.pos, leg.text);
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
                camera.projectSafe(P, Q2D, &insideFrustum);
            } else {
                CCVector3 P3D = P;
                trans.apply(P3D);
                camera.projectSafe(P3D, Q2D, &insideFrustum);
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
