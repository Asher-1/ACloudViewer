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
#include <ecvDisplayTools.h>
#include <ecvPointCloud.h>

// cloudViewer
#include <ScalarField.h>

// Qt
#include <QInputDialog>

// System
#include <assert.h>

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

    ecvDisplayTools::RemoveFromOwnDB(m_label);
    ecvDisplayTools::RemoveFromOwnDB(m_rect2DLabel);
    ecvDisplayTools::SetInteractionMode(ecvDisplayTools::TRANSFORM_CAMERA());

    m_rect2DLabel->setVisible(false);  //=invalid
    m_rect2DLabel->setSelected(true);  //=closed
    m_label->clear();

    // new window?
    if (ecvDisplayTools::GetCurrentScreen()) {
        ecvDisplayTools::AddToOwnDB(m_label);
        ecvDisplayTools::AddToOwnDB(m_rect2DLabel);
        connect(ecvDisplayTools::TheInstance(), &ecvDisplayTools::mouseMoved,
                this, &ccPointPropertiesDlg::update2DZone);
        connect(ecvDisplayTools::TheInstance(),
                &ecvDisplayTools::leftButtonClicked, this,
                &ccPointPropertiesDlg::processClickedPoint);
        connect(ecvDisplayTools::TheInstance(),
                &ecvDisplayTools::buttonReleased, this,
                &ccPointPropertiesDlg::close2DZone);
    }

    return true;
}

bool ccPointPropertiesDlg::start() {
    activatePointPropertiesDisplay();
    return ccPointPickingGenericInterface::start();
}

void ccPointPropertiesDlg::stop(bool state) {
    initializeState();

    if (ecvDisplayTools::GetCurrentScreen())
        ecvDisplayTools::SetInteractionMode(
                ecvDisplayTools::TRANSFORM_CAMERA());

    ccPointPickingGenericInterface::stop(state);
}

void ccPointPropertiesDlg::onClose() { stop(false); }

void ccPointPropertiesDlg::activatePointPropertiesDisplay() {
    if (ecvDisplayTools::GetCurrentScreen())
        ecvDisplayTools::SetInteractionMode(
                ecvDisplayTools::TRANSFORM_CAMERA());

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

    ecvDisplayTools::SetInteractionMode(ecvDisplayTools::TRANSFORM_CAMERA());
}

void ccPointPropertiesDlg::activateAngleDisplay() {
    m_pickingMode = POINTS_ANGLE;
    pointPropertiesButton->setDown(false);
    pointPointDistanceButton->setDown(false);
    pointsAngleButton->setDown(true);
    rectZoneToolButton->setDown(false);
    m_label->setVisible(true);
    m_rect2DLabel->setVisible(false);

    ecvDisplayTools::SetInteractionMode(ecvDisplayTools::TRANSFORM_CAMERA());
}

void ccPointPropertiesDlg::activate2DZonePicking() {
    m_pickingMode = RECT_ZONE;
    pointPropertiesButton->setDown(false);
    pointPointDistanceButton->setDown(false);
    pointsAngleButton->setDown(false);
    rectZoneToolButton->setDown(true);
    m_label->setVisible(false);
    // m_rect2DLabel->setVisible(false);

    ecvDisplayTools::SetInteractionMode(
            ecvDisplayTools::INTERACT_SEND_ALL_SIGNALS);
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
        if (labelObject && !ecvDisplayTools::USE_2D) {
            m_label->setDisplayedIn2D(false);
            m_label->displayPointLegend(true);
        }
    }

    if (!labelObject) {
        return;
    }

    // detach current label from window
    if (ecvDisplayTools::GetCurrentScreen())
        ecvDisplayTools::RemoveFromOwnDB(labelObject);
    labelObject->setSelected(false);

    ccHObject* newLabelObject = 0;
    if (m_pickingMode == RECT_ZONE) {
        newLabelObject = m_rect2DLabel = new cc2DViewportLabel();
        m_rect2DLabel->setVisible(false);  //=invalid
        m_rect2DLabel->setSelected(true);  //=closed
    } else {
        // attach old label to first point cloud by default
        m_label->getPickedPoint(0).cloud->addChild(labelObject);
        newLabelObject = m_label = new cc2DLabel();
        m_label->setSelected(true);
    }

    emit newLabel(labelObject);

    if (ecvDisplayTools::GetCurrentScreen()) {
        ecvDisplayTools::AddToOwnDB(newLabelObject);
        ecvDisplayTools::SetRedrawRecursive(false);
        newLabelObject->setRedraw(true);
        ecvDisplayTools::RedrawDisplay(true);
    }
}

void ccPointPropertiesDlg::processPickedPoint(ccPointCloud* cloud,
                                              unsigned pointIndex,
                                              int x,
                                              int y) {
    assert(cloud);
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

    m_label->addPickedPoint(cloud, pointIndex);
    m_label->setVisible(true);
    m_label->displayPointLegend(
            m_label->size() ==
            3);  // we need to display 'A', 'B' and 'C' for 3-points labels
    if (m_label->size() == 1 && ecvDisplayTools::GetCurrentScreen()) {
        m_label->setPosition(
                static_cast<float>(x + 20) / ecvDisplayTools::GlWidth(),
                static_cast<float>(y + 20) / ecvDisplayTools::GlWidth());
    }

    // output info to Console
    QStringList body = m_label->getLabelContent(
            ecvGui::Parameters().displayedNumPrecision);
    CVLog::Print(QString("[Picked] ") + m_label->getName());
    for (QString& row : body) {
        CVLog::Print(QString("[Picked]\t- ") + row);
    }

    if (ecvDisplayTools::GetCurrentScreen()) {
        m_label->setEnabled(true);
        m_label->updateLabel();
    }
}

void ccPointPropertiesDlg::processClickedPoint(int x, int y) {
    if (m_pickingMode != RECT_ZONE) {
        return;
    }

    if (!m_rect2DLabel || !ecvDisplayTools::GetCurrentScreen()) {
        assert(false);
        return;
    }

    CCVector3d pos2D = ecvDisplayTools::ToVtkCoordinates(x, y);

    if (m_rect2DLabel->isSelected())  // already closed? we start a new label
    {
        float roi[4] = {static_cast<float>(pos2D.x),
                        static_cast<float>(pos2D.y), 0, 0};

        if (ecvDisplayTools::GetCurrentScreen()) {
            m_rect2DLabel->setParameters(
                    ecvDisplayTools::GetViewportParameters());
        }
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

    if (ecvDisplayTools::GetCurrentScreen()) {
        m_rect2DLabel->updateLabel();
    }
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

    if (!ecvDisplayTools::GetCurrentScreen()) {
        assert(false);
        return;
    }

    CCVector3d pos2D = ecvDisplayTools::ToVtkCoordinates(x, y);

    float roi[4] = {m_rect2DLabel->roi()[0], m_rect2DLabel->roi()[1],
                    static_cast<float>(pos2D.x), static_cast<float>(pos2D.y)};
    m_rect2DLabel->setRoi(roi);
    m_rect2DLabel->setVisible(true);

    if (ecvDisplayTools::GetCurrentScreen()) {
        m_rect2DLabel->updateLabel();
    }
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

    if (ecvDisplayTools::GetCurrentScreen()) {
        m_rect2DLabel->updateLabel();
    }
}
