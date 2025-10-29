// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QtGui>

// CV
#include "DistanceComputationTools.h"
#include "ecvHObject.h"
#include "ecvPickingHub.h"
#include "ecvPlane.h"
#include "ecvScalarField.h"
#include "qtablewidget.h"

// Local dependencies
#include "qMPlane.h"

qMPlane::qMPlane(QObject *parent)
    : QObject(parent),
      ccStdPluginInterface(":/CC/plugin/qMPlane/info.json"),
      m_action(nullptr) {}

QList<QAction *> qMPlane::getActions() {
    if (!m_action) {
        m_action = new QAction(getName(), this);
        m_action->setToolTip(getDescription());
        m_action->setIcon(getIcon());
        connect(m_action, &QAction::triggered, this, &qMPlane::doAction);
    }
    return {m_action};
}

// Called when an item is selected
void qMPlane::onNewSelection(const ccHObject::Container &selectedEntities) {
    if (m_action == nullptr) {
        return;
    }

    m_action->setEnabled(false);
    if (selectedEntities.size() == 1) {
        ccHObject *object = selectedEntities.at(0);
        if (object->isKindOf(CV_TYPES::POINT_CLOUD)) {
            m_selectedCloud = static_cast<ccPointCloud *>(object);
            m_action->setEnabled(true);
        }
    }
}

// Called when plugin icon is clicked
void qMPlane::doAction() {
    if (m_app == nullptr) {
        Q_ASSERT(false);
        return;
    }

    if (!m_controller) {
        m_controller = std::make_unique<ccMPlaneDlgController>(m_app);
    }
    m_controller->openDialog(m_selectedCloud);
}