// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qSAM3.h"

#include <ecvImage.h>
#include <ecvMainAppInterface.h>
#include <ecvPluginDbNaming.h>

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QMainWindow>
#include <QMessageBox>
#include <QTimer>

#include "SAM3Dialog.h"

#ifdef AICore_ENABLED
#include "aicore/backend_capi.h"
#endif

qSAM3::qSAM3(QObject* parent)
    : QObject(parent), ccStdPluginInterface(":/CC/plugin/qSAM3/info.json") {
    m_action = new QAction(tr("SAM3 Image & Video Segmentation"), this);
    m_action->setToolTip(
            tr("Segment Anything 2 / 2.1 / 3: interactive point, box, "
               "text-prompt segmentation (GGML)"));
    m_action->setIcon(QIcon(":/CC/plugin/qSAM3/images/qSAM3.svg"));
    connect(m_action, &QAction::triggered, this, &qSAM3::showDialog);
}

QList<QAction*> qSAM3::getActions() { return {m_action}; }

void qSAM3::onNewSelection(const ccHObject::Container& selectedEntities) {
    if (!m_dialog || !m_dialog->isVisible()) return;
    QStringList names;
    for (ccHObject* obj : selectedEntities) {
        if (!obj) continue;
        if (obj->isA(CV_TYPES::IMAGE)) {
            names.append(obj->getName());
        }
    }
    if (!names.isEmpty()) {
        m_dialog->applyDbTreeSelection(names);
    }
}

void qSAM3::showDialog() {
    if (!m_app) return;
    if (!m_dialog) {
        m_dialog = new SAM3Dialog(m_app->getMainWindow());
        m_dialog->setAppInterface(m_app);
    }
    m_dialog->show();
    m_dialog->raise();
    m_dialog->activateWindow();
}