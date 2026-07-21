// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qManualCalib.h"

#include <ecvMainAppInterface.h>

#include <QMainWindow>

#include "ManualAvmAdjustDlg.h"
#include "ManualSensorCalibDlg.h"

qManualCalib::qManualCalib(QObject* parent)
    : QObject(parent),
      ccStdPluginInterface(":/CC/plugin/qManualCalib/info.json") {}

QList<QAction*> qManualCalib::getActions() {
    if (!m_actionSensorCalib) {
        m_actionSensorCalib =
                new QAction(tr("Sensor Extrinsic Calibration"), this);
        m_actionSensorCalib->setToolTip(
                tr("Manually adjust sensor extrinsic parameters "
                   "(roll/pitch/yaw/x/y/z) with real-time visual feedback"));
        m_actionSensorCalib->setIcon(QIcon(QString::fromUtf8(
                ":/CC/plugin/qManualCalib/images/sensorCalibIcon.svg")));
        connect(m_actionSensorCalib, &QAction::triggered, this,
                &qManualCalib::doSensorCalib);
    }

    if (!m_actionAvmAdjust) {
        m_actionAvmAdjust = new QAction(tr("AVM View Adjustment"), this);
        m_actionAvmAdjust->setToolTip(
                tr("Adjust Around View Monitor (AVM) virtual camera parameters "
                   "for panoramic view generation"));
        m_actionAvmAdjust->setIcon(QIcon(QString::fromUtf8(
                ":/CC/plugin/qManualCalib/images/avmAdjustIcon.svg")));
        connect(m_actionAvmAdjust, &QAction::triggered, this,
                &qManualCalib::doAvmAdjust);
    }

    return {m_actionSensorCalib, m_actionAvmAdjust};
}

void qManualCalib::onNewSelection(
        const ccHObject::Container& /*selectedEntities*/) {}

void qManualCalib::doSensorCalib() {
    if (!m_app) return;

    if (!m_sensorCalibDlg) {
        m_sensorCalibDlg =
                new ManualSensorCalibDlg(m_app, m_app->getMainWindow());
        m_app->registerOverlayDialog(m_sensorCalibDlg, Qt::BottomRightCorner);
    }

    m_sensorCalibDlg->linkWith(m_app->getActiveWindow());

    if (m_sensorCalibDlg->start()) {
        m_app->updateOverlayDialogsPlacement();
    }
}

void qManualCalib::doAvmAdjust() {
    if (!m_app) return;

    if (!m_avmAdjustDlg) {
        m_avmAdjustDlg = new ManualAvmAdjustDlg(m_app, m_app->getMainWindow());
        m_app->registerOverlayDialog(m_avmAdjustDlg, Qt::BottomRightCorner);
    }

    m_avmAdjustDlg->linkWith(m_app->getActiveWindow());

    if (m_avmAdjustDlg->start()) {
        m_app->updateOverlayDialogsPlacement();
    }
}
