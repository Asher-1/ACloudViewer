// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "../include/qCloudLayers.h"

#include "../include/ccCloudLayersDlg.h"

// Qt
#include <QMainWindow>
#include <QtGui>

// qCC_db
#include <ecvPointCloud.h>

// system
#include <assert.h>

qCloudLayers::qCloudLayers(QObject* parent)
    : QObject(parent),
      ccStdPluginInterface(":/CC/plugin/qCloudLayers/info.json"),
      m_action(nullptr),
      m_cloudLayersDlg(nullptr) {}

void qCloudLayers::onNewSelection(
        const ccHObject::Container& selectedEntities) {
    if (m_action) {
        // a single point cloud must be selected
        m_action->setEnabled(
                selectedEntities.size() == 1 &&
                selectedEntities.front()->isA(CV_TYPES::POINT_CLOUD));
    }
}

QList<QAction*> qCloudLayers::getActions() {
    // default action
    if (!m_action) {
        m_action = new QAction(getName(), this);
        m_action->setToolTip(getDescription());
        m_action->setIcon(getIcon());

        // connect signal
        connect(m_action, &QAction::triggered, this, &qCloudLayers::doAction);
    }

    return {m_action};
}

void qCloudLayers::doAction() {
    if (!m_app) {
        assert(false);
        return;
    }

    // check selection
    const ccHObject::Container& selectedEntities = m_app->getSelectedEntities();
    if (!m_app->haveOneSelection() ||
        !selectedEntities.front()->isA(CV_TYPES::POINT_CLOUD)) {
        m_app->dispToConsole("Select only one point cloud!",
                             ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
        return;
    }

    // get first selected cloud
    ccPointCloud* cloud = static_cast<ccPointCloud*>(selectedEntities.front());

    if (!cloud->hasScalarFields()) {
        CVLog::Error("Cloud has no scalar field");
        return;
    }

    // set colors schema to RGB
    m_app->updateUI();

    if (!m_cloudLayersDlg) {
        m_cloudLayersDlg = new ccCloudLayersDlg(m_app, m_app->getMainWindow());
        m_app->registerOverlayDialog(m_cloudLayersDlg, Qt::TopRightCorner);
    }

    m_cloudLayersDlg->linkWith(m_app->getActiveWindow());
    m_cloudLayersDlg->setPointCloud(cloud);

    if (m_cloudLayersDlg->start()) {
        m_app->updateOverlayDialogsPlacement();
    }
}
