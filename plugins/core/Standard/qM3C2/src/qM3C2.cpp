// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qM3C2.h"

// LOCAL
#include "qM3C2Commands.h"
#include "qM3C2Dialog.h"
#include "qM3C2DisclaimerDialog.h"
#include "qM3C2Process.h"
#include "qM3C2Tools.h"

// CV_DB_LIB
#include <ecvPointCloud.h>

// Qt
#include <QMainWindow>

qM3C2Plugin::qM3C2Plugin(QObject* parent)
    : QObject(parent),
      ccStdPluginInterface(":/CC/plugin/qM3C2Plugin/info.json"),
      m_action(nullptr) {}

void qM3C2Plugin::onNewSelection(const ccHObject::Container& selectedEntities) {
    if (m_action) {
        m_action->setEnabled(selectedEntities.size() == 2 &&
                             selectedEntities[0]->isA(CV_TYPES::POINT_CLOUD) &&
                             selectedEntities[1]->isA(CV_TYPES::POINT_CLOUD));
    }

    m_selectedEntities = selectedEntities;
}

QList<QAction*> qM3C2Plugin::getActions() {
    if (!m_action) {
        m_action = new QAction(getName(), this);
        m_action->setToolTip(getDescription());
        m_action->setIcon(getIcon());
        connect(m_action, &QAction::triggered, this, &qM3C2Plugin::doAction);
    }

    return QList<QAction*>{m_action};
}

void qM3C2Plugin::doAction() {
    // disclaimer accepted?
    if (!DisclaimerDialog::show(m_app)) return;

    // m_app should have already been initialized by CC when plugin is loaded!
    assert(m_app);
    if (!m_app) return;

    if (m_selectedEntities.size() != 2 ||
        !m_selectedEntities[0]->isA(CV_TYPES::POINT_CLOUD) ||
        !m_selectedEntities[1]->isA(CV_TYPES::POINT_CLOUD)) {
        m_app->dispToConsole(tr("Select two point clouds!"),
                             ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
        return;
    }

    ccPointCloud* cloud1 = ccHObjectCaster::ToPointCloud(m_selectedEntities[0]);
    ccPointCloud* cloud2 = ccHObjectCaster::ToPointCloud(m_selectedEntities[1]);

    // display dialog
    qM3C2Dialog dlg(cloud1, cloud2, m_app);
    if (!dlg.exec()) {
        // process cancelled by the user
        return;
    }

    QString errorMessage;
    ccPointCloud* outputCloud =
            nullptr;  // only necessary for the command line version in fact
    if (!qM3C2Process::Compute(dlg, errorMessage, outputCloud, true,
                               m_app->getMainWindow(), m_app)) {
        m_app->dispToConsole(errorMessage,
                             ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
    }

    //'Compute' may change some parameters of the dialog
    dlg.saveParamsToPersistentSettings();
}

void qM3C2Plugin::registerCommands(ccCommandLineInterface* cmd) {
    if (!cmd) {
        assert(false);
        return;
    }
    cmd->registerCommand(
            ccCommandLineInterface::Command::Shared(new CommandM3C2));
}
