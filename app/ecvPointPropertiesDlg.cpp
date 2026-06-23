// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvPointPropertiesDlg.h"

// Local
#include "ecvCommon.h"
#include "ecvGuiParameters.h"

// CV_DB_LIB
#include <CVLog.h>
#include <ecv2DLabel.h>
#include <ecv2DViewportLabel.h>
#include <ecvGenericGLDisplay.h>
#include <ecvGenericMesh.h>
#include <ecvHObjectCaster.h>
#include <ecvPointCloud.h>
#include <ecvRedrawScope.h>
#include <ecvViewManager.h>

// cloudViewer
#include <ScalarField.h>

// Qt
#include <QInputDialog>

// System
#include <assert.h>

#include <cmath>

ccPointPropertiesDlg::ccPointPropertiesDlg(ccPickingHub* pickingHub,
                                           QWidget* parent)
    : ccPointPickingGenericInterface(pickingHub, parent),
      Ui::PointPropertiesDlg(),
      m_pickingMode(POINT_INFO) {
    setupUi(this);

    connect(closeButton, &QToolButton::clicked, this,
            &ccPointPropertiesDlg::onClose);
    connect(pointPropertiesButton, &QToolButton::clicked, this,
            &ccPointPropertiesDlg::activatePointPropertiesDisplay);
    connect(pointPointDistanceButton, &QToolButton::clicked, this,
            &ccPointPropertiesDlg::activateDistanceDisplay);
    connect(pointsAngleButton, &QToolButton::clicked, this,
            &ccPointPropertiesDlg::activateAngleDisplay);
    connect(rectZoneToolButton, &QToolButton::clicked, this,
            &ccPointPropertiesDlg::activate2DZonePicking);
    connect(saveLabelButton, &QToolButton::clicked, this,
            &ccPointPropertiesDlg::exportCurrentLabel);
    connect(razButton, &QToolButton::clicked, this,
            &ccPointPropertiesDlg::initializeState);

    // for points picking
    m_label = new cc2DLabel();
    m_label->setSelected(true);

    // for 2D zone picking
    m_rect2DLabel = new cc2DViewportLabel();
    m_rect2DLabel->setVisible(false);  //=invalid
    m_rect2DLabel->setSelected(true);  //=closed
}

ccPointPropertiesDlg::~ccPointPropertiesDlg() {
    if (m_label) delete m_label;
    m_label = nullptr;

    if (m_rect2DLabel) delete m_rect2DLabel;
    m_rect2DLabel = nullptr;
}

bool ccPointPropertiesDlg::linkWith(QWidget* win) {
    assert(m_label && m_rect2DLabel);

    if (!ccPointPickingGenericInterface::linkWith(win)) {
        return false;
    }

    if (auto* view = ecvViewManager::instance().getEffectiveView()) {
        view->removeFromOwnDB(m_label);
        view->removeFromOwnDB(m_rect2DLabel);
        view->setInteractionMode(ecvGenericGLDisplay::MODE_TRANSFORM_CAMERA);
    }

    m_rect2DLabel->setVisible(false);  //=invalid
    m_rect2DLabel->setSelected(true);  //=closed
    m_label->clear();

    // new window?
    if (ecvViewManager::instance().activeWidget()) {
        if (auto* view = ecvViewManager::instance().getEffectiveView()) {
            view->addToOwnDB(m_label);
            view->addToOwnDB(m_rect2DLabel);
        }
        ecvViewManager& vm = ecvViewManager::instance();
        connect(&vm, &ecvViewManager::mouseMoved, this,
                &ccPointPropertiesDlg::update2DZone, Qt::UniqueConnection);
        connect(&vm, &ecvViewManager::leftButtonClicked, this,
                &ccPointPropertiesDlg::processClickedPoint,
                Qt::UniqueConnection);
        connect(&vm, &ecvViewManager::buttonReleased, this,
                &ccPointPropertiesDlg::close2DZone, Qt::UniqueConnection);
    }

    return true;
}

bool ccPointPropertiesDlg::start() {
    activatePointPropertiesDisplay();
    return ccPointPickingGenericInterface::start();
}

void ccPointPropertiesDlg::stop(bool state) {
    initializeState();

    if (ecvViewManager::instance().activeWidget()) {
        if (auto* view = ecvViewManager::instance().getEffectiveView()) {
            view->setInteractionMode(
                    ecvGenericGLDisplay::MODE_TRANSFORM_CAMERA);
        }
    }

    ccPointPickingGenericInterface::stop(state);
}

void ccPointPropertiesDlg::onClose() { stop(false); }

void ccPointPropertiesDlg::activatePointPropertiesDisplay() {
    if (ecvViewManager::instance().activeWidget()) {
        if (auto* view = ecvViewManager::instance().getEffectiveView()) {
            view->setInteractionMode(
                    ecvGenericGLDisplay::MODE_TRANSFORM_CAMERA);
        }
    }

    m_pickingMode = POINT_INFO;
    pointPropertiesButton->setDown(true);
    pointPointDistanceButton->setDown(false);
    pointsAngleButton->setDown(false);
    rectZoneToolButton->setDown(false);
    m_label->setVisible(true);
    m_rect2DLabel->setVisible(false);
}

void ccPointPropertiesDlg::activateDistanceDisplay() {
    m_pickingMode = POINT_POINT_DISTANCE;
    pointPropertiesButton->setDown(false);
    pointPointDistanceButton->setDown(true);
    pointsAngleButton->setDown(false);
    rectZoneToolButton->setDown(false);
    m_label->setVisible(true);
    m_rect2DLabel->setVisible(false);

    if (auto* view = ecvViewManager::instance().getEffectiveView()) {
        view->setInteractionMode(ecvGenericGLDisplay::MODE_TRANSFORM_CAMERA);
    }
}

void ccPointPropertiesDlg::activateAngleDisplay() {
    m_pickingMode = POINTS_ANGLE;
    pointPropertiesButton->setDown(false);
    pointPointDistanceButton->setDown(false);
    pointsAngleButton->setDown(true);
    rectZoneToolButton->setDown(false);
    m_label->setVisible(true);
    m_rect2DLabel->setVisible(false);

    if (auto* view = ecvViewManager::instance().getEffectiveView()) {
        view->setInteractionMode(ecvGenericGLDisplay::MODE_TRANSFORM_CAMERA);
    }
}

void ccPointPropertiesDlg::activate2DZonePicking() {
    m_pickingMode = RECT_ZONE;
    pointPropertiesButton->setDown(false);
    pointPointDistanceButton->setDown(false);
    pointsAngleButton->setDown(false);
    rectZoneToolButton->setDown(true);
    m_label->setVisible(false);
    // m_rect2DLabel->setVisible(false);

    if (auto* view = ecvViewManager::instance().getEffectiveView()) {
        view->setInteractionMode(
                ecvGenericGLDisplay::INTERACT_SEND_ALL_SIGNALS);
    }
}

void ccPointPropertiesDlg::initializeState() {
    assert(m_label && m_rect2DLabel);
    m_label->clear(false, false);
    m_rect2DLabel->setVisible(false);  //=invalid
    m_rect2DLabel->setSelected(true);  //=closed
}

void ccPointPropertiesDlg::exportCurrentLabel() {
    ccHObject* labelObject = 0;
    if (m_pickingMode == RECT_ZONE) {
        labelObject = (m_rect2DLabel->isSelected() && m_rect2DLabel->isVisible()
                               ? m_rect2DLabel
                               : 0);
    } else {
        labelObject = (m_label && m_label->size() > 0 ? m_label : 0);
        if (labelObject) {
            m_label->setDisplayedIn2D(true);
            m_label->displayPointLegend(m_label->size() == 3);
        }
    }

    if (!labelObject) {
        return;
    }

    // detach current label from window
    if (ecvViewManager::instance().activeWidget()) {
        if (auto* view = ecvViewManager::instance().getEffectiveView())
            view->removeFromOwnDB(labelObject);
    }
    labelObject->setSelected(false);

    ccHObject* newLabelObject = 0;
    if (m_pickingMode == RECT_ZONE) {
        newLabelObject = m_rect2DLabel = new cc2DViewportLabel();
        m_rect2DLabel->setVisible(false);  //=invalid
        m_rect2DLabel->setSelected(true);  //=closed
    } else {
        ccHObject* parentEntity = m_label->getPickedPoint(0).entity();
        if (parentEntity) {
            parentEntity->addChild(labelObject);
            ecvGenericGLDisplay* activeView =
                    ecvViewManager::instance().getActiveView();
            if (activeView) {
                labelObject->setDisplay(activeView);
            } else if (parentEntity->getDisplay()) {
                labelObject->setDisplay(parentEntity->getDisplay());
            }
        } else {
            CVLog::Warning("Parent entity not found for label!");
        }
        newLabelObject = m_label = new cc2DLabel();
        m_label->setSelected(true);
    }

    emit newLabel(labelObject);

    if (ecvViewManager::instance().activeWidget()) {
        if (auto* view = ecvViewManager::instance().getEffectiveView())
            view->addToOwnDB(newLabelObject);
        { ecvRedrawScope scope({newLabelObject}, true, true); }
    }
}

void ccPointPropertiesDlg::processPickedPoint(const PickedItem& picked) {
    if (!picked.entity) return;
    assert(m_label);

    switch (m_pickingMode) {
        case POINT_INFO:
            m_label->clear();
            break;
        case POINT_POINT_DISTANCE:
            if (m_label->size() >= 2) m_label->clear();
            break;
        case POINTS_ANGLE:
            if (m_label->size() >= 3) m_label->clear();
            break;
        case RECT_ZONE:
            return;  // we don't use this slot for 2D mode
    }

    bool addOk = false;
    if (picked.entity->isKindOf(CV_TYPES::POINT_CLOUD)) {
        addOk = m_label->addPickedPoint(
                ccHObjectCaster::ToGenericPointCloud(picked.entity),
                picked.itemIndex, picked.entityCenter);
    } else if (picked.entity->isKindOf(CV_TYPES::MESH)) {
        ccGenericMesh* mesh = ccHObjectCaster::ToGenericMesh(picked.entity);
        if (mesh && picked.itemIndex < mesh->size()) {
            CCVector3 A, B, C;
            mesh->getTriangleVertices(picked.itemIndex, A, B, C);
            CCVector3 v0 = A - C, v1 = B - C, v2 = picked.P3D - C;
            double d00 = v0.dot(v0), d01 = v0.dot(v1), d11 = v1.dot(v1);
            double d20 = v2.dot(v0), d21 = v2.dot(v1);
            double denom = d00 * d11 - d01 * d01;
            CCVector2d uv(0, 0);
            if (std::abs(denom) > 1.0e-12) {
                uv.x = (d11 * d20 - d01 * d21) / denom;
                uv.y = (d00 * d21 - d01 * d20) / denom;
            }
            addOk = m_label->addPickedPoint(mesh, picked.itemIndex, uv,
                                            picked.entityCenter);
        }
    }
    if (!addOk) {
        return;
    }

    m_label->setVisible(true);
    m_label->setDisplayedIn2D(true);
    m_label->displayPointLegend(
            m_label->size() ==
            3);  // we need to display 'A', 'B' and 'C' for 3-points labels

    {
        ecvGenericGLDisplay* pickView =
                picked.pickView ? picked.pickView
                                : ecvViewManager::instance().getActiveView();
        if (pickView && m_label->getDisplay() != pickView) {
            m_label->setDisplay(pickView);
        }
    }

    if (m_label->size() == 1 && ecvViewManager::instance().activeWidget()) {
        if (auto* view = ecvViewManager::instance().getEffectiveView()) {
            const float gw = static_cast<float>(view->glWidth());
            const float gh = static_cast<float>(view->glHeight());
            if (gw > 0.0f && gh > 0.0f) {
                m_label->setPosition(
                        static_cast<float>(picked.clickPoint.x() + 20) / gw,
                        static_cast<float>(picked.clickPoint.y() + 20) / gh);
            }
        }
    }

    // output info to Console
    QStringList body = m_label->getLabelContent(
            ecvGui::Parameters().displayedNumPrecision);
    CVLog::Print(QString("[Picked] ") + m_label->getName());
    for (QString& row : body) {
        CVLog::Print(QString("[Picked]\t- ") + row);
    }

    if (ecvViewManager::instance().activeWidget()) {
        m_label->setEnabled(true);
        m_label->updateLabel();
    }
}

void ccPointPropertiesDlg::processClickedPoint(int x, int y) {
    if (m_pickingMode != RECT_ZONE) {
        return;
    }

    if (!m_rect2DLabel || !ecvViewManager::instance().activeWidget()) {
        assert(false);
        return;
    }

    CCVector3d pos2D(0, 0, 0);
    if (auto* v = ecvViewManager::instance().getEffectiveView()) {
        pos2D = v->toVtkCoordinates(x, y);
    }

    if (m_rect2DLabel->isSelected())  // already closed? we start a new label
    {
        float roi[4] = {static_cast<float>(pos2D.x),
                        static_cast<float>(pos2D.y), 0, 0};

        if (auto* view = ecvViewManager::instance().getEffectiveView())
            m_rect2DLabel->setParameters(view->getViewportParameters());
        m_rect2DLabel->setVisible(false);   //=invalid
        m_rect2DLabel->setSelected(false);  //=not closed
        m_rect2DLabel->setRoi(roi);
        m_rect2DLabel->setName("");  // reset name before display!
    } else                           // we close the existing one
    {
        float roi[4] = {m_rect2DLabel->roi()[0], m_rect2DLabel->roi()[1],
                        static_cast<float>(pos2D.x),
                        static_cast<float>(pos2D.y)};
        m_rect2DLabel->setRoi(roi);
        m_rect2DLabel->setVisible(true);   //=valid
        m_rect2DLabel->setSelected(true);  //=closed
    }

    if (ecvViewManager::instance().activeWidget()) m_rect2DLabel->updateLabel();
}

void ccPointPropertiesDlg::update2DZone(int x,
                                        int y,
                                        Qt::MouseButtons buttons) {
    if (m_pickingMode != RECT_ZONE) {
        return;
    }

    if (m_rect2DLabel->isSelected()) {
        return;
    }

    if (!ecvViewManager::instance().activeWidget()) {
        assert(false);
        return;
    }

    CCVector3d pos2D(0, 0, 0);
    if (auto* v = ecvViewManager::instance().getEffectiveView()) {
        pos2D = v->toVtkCoordinates(x, y);
    }

    float roi[4] = {m_rect2DLabel->roi()[0], m_rect2DLabel->roi()[1],
                    static_cast<float>(pos2D.x), static_cast<float>(pos2D.y)};
    m_rect2DLabel->setRoi(roi);
    m_rect2DLabel->setVisible(true);

    if (ecvViewManager::instance().activeWidget()) m_rect2DLabel->updateLabel();
}

static QString s_last2DLabelComment("");
void ccPointPropertiesDlg::close2DZone() {
    if (m_pickingMode != RECT_ZONE) return;

    if (m_rect2DLabel->isSelected() || !m_rect2DLabel->isVisible()) return;

    m_rect2DLabel->setSelected(true);

    bool ok;
    QString title = QInputDialog::getText(this, "Set area label title",
                                          "Title:", QLineEdit::Normal,
                                          s_last2DLabelComment, &ok);
    if (!ok) {
        m_rect2DLabel->setVisible(false);
    } else {
        m_rect2DLabel->setName(title);
        s_last2DLabelComment = title;
    }

    if (ecvViewManager::instance().activeWidget()) m_rect2DLabel->updateLabel();
}
