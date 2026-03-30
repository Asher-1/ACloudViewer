// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qPCV.h"

#include "PCVCommand.h"
#include "ccPcvDlg.h"

// CV_CORE_LIB
#include <PCV.h>
#include <ScalarField.h>

// CV_DB_LIB
#include <ecvColorScalesManager.h>
#include <ecvGenericMesh.h>
#include <ecvGenericPointCloud.h>
#include <ecvHObjectCaster.h>
#include <ecvPointCloud.h>
#include <ecvProgressDialog.h>
#include <ecvScalarField.h>

// Qt
#include <QElapsedTimer>
#include <QMainWindow>
#include <QProgressBar>

// persistent settings during a single session
static bool s_firstLaunch = true;
static int s_raysSpinBoxValue = 256;
static int s_resSpinBoxValue = 1024;
static bool s_mode180CheckBoxState = true;
static bool s_closedMeshCheckBoxState = false;

qPCV::qPCV(QObject* parent /*=0*/)
    : QObject(parent),
      ccStdPluginInterface(":/CC/plugin/qPCV/info.json"),
      m_action(nullptr) {}

void qPCV::onNewSelection(const ccHObject::Container& selectedEntities) {
    if (m_action) {
        bool elligibleEntitiies = false;
        for (ccHObject* obj : selectedEntities) {
            if (obj && (obj->isKindOf(CV_TYPES::POINT_CLOUD) ||
                        obj->isKindOf(CV_TYPES::MESH))) {
                elligibleEntitiies = true;
                break;
            }
        }
        m_action->setEnabled(elligibleEntitiies);
    }
}

QList<QAction*> qPCV::getActions() {
    // default action
    if (!m_action) {
        m_action = new QAction(getName(), this);
        m_action->setToolTip(getDescription());
        m_action->setIcon(getIcon());
        // connect signal
        connect(m_action, &QAction::triggered, this, &qPCV::doAction);
    }

    return QList<QAction*>{m_action};
}

void qPCV::doAction() {
    assert(m_app);
    if (!m_app) return;

    const ccHObject::Container& selectedEntities = m_app->getSelectedEntities();

    ccHObject::Container candidates;
    bool hasMeshes = false;
    for (ccHObject* obj : selectedEntities) {
        if (!obj) {
            assert(false);
            continue;
        }

        if (obj->isA(CV_TYPES::POINT_CLOUD)) {
            // we need a real point cloud
            candidates.push_back(obj);
        } else if (obj->isKindOf(CV_TYPES::MESH)) {
            ccGenericMesh* mesh = ccHObjectCaster::ToGenericMesh(obj);
            if (mesh->getAssociatedCloud() &&
                mesh->getAssociatedCloud()->isA(CV_TYPES::POINT_CLOUD)) {
                // we need a mesh with a real point cloud
                candidates.push_back(obj);
                hasMeshes = true;
            }
        }
    }

    ccPcvDlg dlg(m_app->getMainWindow());

    // restore previous dialog state
    if (!s_firstLaunch) {
        dlg.raysSpinBox->setValue(s_raysSpinBoxValue);
        dlg.mode180CheckBox->setChecked(s_mode180CheckBoxState);
        dlg.resSpinBox->setValue(s_resSpinBoxValue);
        dlg.closedMeshCheckBox->setChecked(s_closedMeshCheckBoxState);
    }

    dlg.closedMeshCheckBox->setEnabled(hasMeshes);  // for meshes only

    // for using clouds normals as rays
    std::vector<ccGenericPointCloud*> cloudsWithNormals;
    ccHObject* root = m_app->dbRootObject();
    if (root) {
        ccHObject::Container clouds;
        root->filterChildren(clouds, true, CV_TYPES::POINT_CLOUD);
        for (size_t i = 0; i < clouds.size(); ++i) {
            // we keep only clouds with normals
            ccGenericPointCloud* cloud =
                    ccHObjectCaster::ToGenericPointCloud(clouds[i]);
            if (cloud && cloud->hasNormals()) {
                cloudsWithNormals.push_back(cloud);
                QString cloudTitle = QString("%1 - %2 points")
                                             .arg(cloud->getName())
                                             .arg(cloud->size());
                if (cloud->getParent() &&
                    cloud->getParent()->isKindOf(CV_TYPES::MESH)) {
                    cloudTitle.append(QString(" (%1)").arg(
                            cloud->getParent()->getName()));
                }

                dlg.cloudsComboBox->addItem(cloudTitle);
            }
        }
    }
    if (cloudsWithNormals.empty()) {
        dlg.useCloudRadioButton->setEnabled(false);
    }

    if (!dlg.exec()) {
        return;
    }

    // save dialog state
    {
        s_firstLaunch = false;
        s_raysSpinBoxValue = dlg.raysSpinBox->value();
        s_mode180CheckBoxState = dlg.mode180CheckBox->isChecked();
        s_resSpinBoxValue = dlg.resSpinBox->value();
        s_closedMeshCheckBoxState = dlg.closedMeshCheckBox->isChecked();
    }

    unsigned raysNumber = dlg.raysSpinBox->value();
    unsigned resolution = dlg.resSpinBox->value();
    bool meshIsClosed =
            (hasMeshes ? dlg.closedMeshCheckBox->isChecked() : false);
    bool mode360 = !dlg.mode180CheckBox->isChecked();

    // PCV type ShadeVis
    std::vector<CCVector3d> rays;
    if (!cloudsWithNormals.empty() && dlg.useCloudRadioButton->isChecked()) {
        // Version with cloud normals as light rays
        assert(dlg.cloudsComboBox->currentIndex() <
               static_cast<int>(cloudsWithNormals.size()));
        ccGenericPointCloud* pc =
                cloudsWithNormals[dlg.cloudsComboBox->currentIndex()];
        unsigned count = pc->size();
        try {
            rays.resize(count);
        } catch (const std::bad_alloc&) {
            m_app->dispToConsole(
                    "Not enough memory to generate the set of rays",
                    ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
            return;
        }
        for (unsigned i = 0; i < count; ++i) {
            CCVector3 n = pc->getPointNormal(i);
            rays[i] = CCVector3d(n.x, n.y, n.z);
        }
    } else {
        // generates light directions
        if (!PCV::GenerateRays(raysNumber, rays, mode360)) {
            m_app->dispToConsole("Failed to generate the set of rays",
                                 ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
            return;
        }
    }

    if (rays.empty()) {
        assert(false);
        m_app->dispToConsole("No ray was generated?!",
                             ecvMainAppInterface::WRN_CONSOLE_MESSAGE);
        return;
    }

    ecvProgressDialog pcvProgressCb(true, m_app->getMainWindow());
    pcvProgressCb.setAutoClose(false);

    QElapsedTimer timer;
    timer.start();
    PCVCommand::Process(candidates, rays, meshIsClosed, resolution,
                        &pcvProgressCb, m_app);
    m_app->dispToConsole(
            QString("[PCV] Timing: %1 sec").arg(timer.elapsed() / 1000.0));

    pcvProgressCb.close();

    m_app->updateUI();
    m_app->refreshAll();
}

void qPCV::registerCommands(ccCommandLineInterface* cmd) {
    cmd->registerCommand(
            ccCommandLineInterface::Command::Shared(new PCVCommand));
}