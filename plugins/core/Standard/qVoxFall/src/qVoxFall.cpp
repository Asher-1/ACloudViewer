// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qVoxFall.h"

// Qt
#include <QMainWindow>

// local
#include "qVoxFallDialog.h"
#include "qVoxFallDisclaimerDialog.h"
#include "qVoxFallProcess.h"

// qCC_db
#include <ecvMesh.h>

qVoxFall::qVoxFall(QObject *parent)
    : QObject(parent),
      ccStdPluginInterface(":/CC/plugin/qVoxFall/info.json"),
      m_action(nullptr) {}

void qVoxFall::onNewSelection(const ccHObject::Container &selectedEntities) {
    if (m_action) {
        m_action->setEnabled(selectedEntities.size() == 2 &&
                             selectedEntities[0]->isA(CV_TYPES::MESH) &&
                             selectedEntities[1]->isA(CV_TYPES::MESH));
    }

    m_selectedEntities = selectedEntities;
}

QList<QAction *> qVoxFall::getActions() {
    if (!m_action) {
        m_action = new QAction(getName(), this);
        m_action->setToolTip(getDescription());
        m_action->setIcon(getIcon());

        // Connect signal
        connect(m_action, &QAction::triggered, this, &qVoxFall::doAction);
    }

    return {m_action};
}

void qVoxFall::doAction() {
    // disclaimer accepted?
    if (!DisclaimerDialog::show(m_app)) return;

    // m_app should have already been initialized by CC when plugin is loaded!
    assert(m_app);
    if (!m_app) return;

    if (m_selectedEntities.size() != 2 ||
        !m_selectedEntities[0]->isA(CV_TYPES::MESH) ||
        !m_selectedEntities[1]->isA(CV_TYPES::MESH)) {
        m_app->dispToConsole("Select two meshes !",
                             ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
        return;
    }

    ccMesh *mesh1 = ccHObjectCaster::ToMesh(m_selectedEntities[0]);
    ccMesh *mesh2 = ccHObjectCaster::ToMesh(m_selectedEntities[1]);

    m_app->dispToConsole("[VoxFall] Meshes loaded successfully",
                         ecvMainAppInterface::STD_CONSOLE_MESSAGE);

    // display dialog
    qVoxFallDialog dlg(mesh1, mesh2, m_app);
    if (!dlg.exec()) {
        // process cancelled by the user
        return;
    }

    ////display the voxel size in console
    m_app->dispToConsole(
            QString("[VoxFall] Voxel size: %1 m").arg(dlg.getVoxelSize()),
            ecvMainAppInterface::STD_CONSOLE_MESSAGE);

    QString errorMessage;
    if (!qVoxFallProcess::Compute(dlg, errorMessage, true,
                                  m_app->getMainWindow(), m_app)) {
        if (!errorMessage.isEmpty()) {
            m_app->dispToConsole(errorMessage,
                                 ecvMainAppInterface::WRN_CONSOLE_MESSAGE);
        } else {
            mesh1->setEnabled(false);
            m_app->dispToConsole("[VoxFall] Completed!",
                                 ecvMainAppInterface::STD_CONSOLE_MESSAGE);
        }
    }

    //'Compute' may change some parameters of the dialog
    dlg.saveParamsToPersistentSettings();
}
