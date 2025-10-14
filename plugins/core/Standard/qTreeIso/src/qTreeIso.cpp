// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qTreeIso.h"

// Qt
#include <QApplication>
#include <QComboBox>
#include <QElapsedTimer>
#include <QMainWindow>
#include <QMessageBox>
#include <QProgressDialog>

// Local
#include "ccTreeIsoDlg.h"
#include "qTreeIsoCommands.h"

// System
#include <assert.h>

#include <iostream>
#include <string>
#include <vector>

// qCC_db
#include <ecvGenericPointCloud.h>
#include <ecvHObjectCaster.h>
#include <ecvMesh.h>
#include <ecvOctree.h>
#include <ecvPointCloud.h>

// TreeIso
#include <TreeIso.h>

qTreeIso::qTreeIso(QObject* parent)
    : QObject(parent),
      ccStdPluginInterface(":/CC/plugin/qTreeIso/info.json"),
      m_action(nullptr) {}

void qTreeIso::onNewSelection(const ccHObject::Container& selectedEntities) {
    if (m_action) {
        bool hasCloud = false;
        for (ccHObject* entity : selectedEntities) {
            if (entity && entity->isA(CV_TYPES::POINT_CLOUD)) {
                hasCloud = true;
                break;
            }
        }
        m_action->setEnabled(hasCloud);
    }
}

QList<QAction*> qTreeIso::getActions() {
    if (!m_action) {
        m_action = new QAction(getName(), this);
        m_action->setToolTip(getDescription());
        m_action->setIcon(getIcon());

        // connect appropriate signal
        connect(m_action, &QAction::triggered, this, &qTreeIso::doAction);
    }

    return {m_action};
}

void qTreeIso::doAction() {
    Parameters parameters;
    ccTreeIsoDlg treeisoDlg(m_app ? m_app->getMainWindow() : nullptr);

    connect(treeisoDlg.pushButtonInitSeg, &QPushButton::clicked, [&] {
        parameters.min_nn1 = treeisoDlg.spinBoxK1->value();
        parameters.reg_strength1 = treeisoDlg.doubleSpinBoxLambda1->value();
        ;
        parameters.decimate_res1 = treeisoDlg.doubleSpinBoxDecRes1->value();

        init_segs(parameters, &treeisoDlg);
    });

    connect(treeisoDlg.pushButtonInterSeg, &QPushButton::clicked, [&] {
        parameters.min_nn2 = treeisoDlg.spinBoxK2->value();
        parameters.reg_strength2 = treeisoDlg.doubleSpinBoxLambda2->value();
        parameters.decimate_res2 = treeisoDlg.doubleSpinBoxDecRes2->value();
        parameters.max_gap = treeisoDlg.doubleSpinBoxMaxGap->value();

        intermediate_segs(parameters, &treeisoDlg);
    });

    connect(treeisoDlg.pushButtonReseg, &QPushButton::clicked, [&] {
        parameters.rel_height_length_ratio =
                treeisoDlg.doubleSpinBoxRelHLRatio->value();
        parameters.vertical_weight = treeisoDlg.doubleSpinBoxVWeight->value();

        final_segs(parameters, &treeisoDlg);
    });
    treeisoDlg.pushButtonInitSeg->setEnabled(true);

    if (treeisoDlg.exec()) {
        if (m_app) {
            m_app->refreshAll();
        } else {
            // m_app should have already been initialized by CC when plugin is
            // loaded!
            assert(false);
        }
    }
}

void qTreeIso::init_segs(const Parameters& parameters,
                         QWidget* parent /*=nullptr*/) {
    // display the progress dialog
    QProgressDialog* progressDlg = new QProgressDialog(parent);
    progressDlg->setWindowTitle("TreeIso Step 1. Initial segmention");
    progressDlg->setLabelText(tr("Computing...."));
    progressDlg->setCancelButton(nullptr);
    progressDlg->setRange(0, 0);  // infinite progress bar
    progressDlg->show();

    if (!TreeIso::Init_seg(parameters.min_nn1, parameters.reg_strength1,
                           parameters.decimate_res1, m_app, progressDlg)) {
        m_app->dispToConsole("Not enough memory",
                             ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
        return;
    }

    progressDlg->close();
    QApplication::processEvents();

    m_app->updateUI();
    m_app->refreshAll();
}

void qTreeIso::intermediate_segs(const Parameters& parameters,
                                 QWidget* parent /*=nullptr*/) {
    // display the progress dialog
    QProgressDialog* progressDlg = new QProgressDialog(parent);
    progressDlg->setWindowTitle("TreeIso Step 2. Interim segmention");
    progressDlg->setLabelText(tr("Computing...."));
    progressDlg->setCancelButton(nullptr);
    progressDlg->setRange(0, 0);  // infinite progress bar
    progressDlg->show();

    if (!TreeIso::Intermediate_seg(parameters.min_nn2, parameters.reg_strength2,
                                   parameters.decimate_res2, parameters.max_gap,
                                   m_app, progressDlg)) {
        progressDlg->hide();
        QApplication::processEvents();
        m_app->updateUI();
        m_app->refreshAll();
        return;
    }

    progressDlg->hide();
    QApplication::processEvents();

    m_app->updateUI();
    m_app->refreshAll();
}

void qTreeIso::final_segs(const Parameters& parameters,
                          QWidget* parent /*=nullptr*/) {
    // display the progress dialog
    QProgressDialog* progressDlg = new QProgressDialog(parent);
    progressDlg->setWindowTitle("TreeIso Step 3. Final segmention");
    progressDlg->setLabelText(tr("Computing...."));
    progressDlg->setCancelButton(nullptr);
    progressDlg->setRange(0, 0);  // infinite progress bar
    progressDlg->show();

    if (!TreeIso::Final_seg(parameters.min_nn2,
                            parameters.rel_height_length_ratio,
                            parameters.vertical_weight, m_app, progressDlg)) {
        progressDlg->hide();
        QApplication::processEvents();
        m_app->updateUI();
        m_app->refreshAll();
        return;
    }

    progressDlg->close();
    QApplication::processEvents();

    m_app->updateUI();
    m_app->refreshAll();
}

void qTreeIso::registerCommands(ccCommandLineInterface* cmd) {
    if (!cmd) {
        assert(false);
        return;
    }
    cmd->registerCommand(
            ccCommandLineInterface::Command::Shared(new CommandTreeIso));
}
