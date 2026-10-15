// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qSpirula.h"

// qSpirula: process-boundary integration of Spirula Studio. The plugin owns
// no training logic — it opens a single dialog whose buttons start the
// official upstream GUI binary, drive headless training, and import the file
// outputs into the DB tree, so the official UI evolution rides the
// SPIRULA_TAG upgrade without any source patches.

#include <ecvMainAppInterface.h>
#include <ecvPluginDbNaming.h>

#include <QAction>
#include <QCoreApplication>
#include <QFileDialog>
#include <QFileInfo>
#include <QMessageBox>
#include <QProgressDialog>
#include <QSettings>
#include <QTranslator>

#include "SpirulaDialog.h"
#include "SpirulaLauncher.h"
#include "SpirulaLocator.h"
#include "SpirulaTrainDialog.h"
#include "SplatImporter.h"

qSpirula::qSpirula(QObject* parent)
    : QObject(parent), ccStdPluginInterface(":/CC/plugin/qSpirula/info.json") {
    // Load the embedded translation catalog matching the persisted language
    // setting (mirrors qAnyLabeling); untranslated strings fall back to the
    // English source text.
    const QString language = QSettings()
                                     .value(QStringLiteral("qSpirula/language"),
                                            QStringLiteral("en_US"))
                                     .toString();
    if (language != QStringLiteral("en_US") &&
        m_translator.load(
                QStringLiteral(":/qSpirula/translations/qspirula_%1.qm")
                        .arg(language))) {
        QCoreApplication::installTranslator(&m_translator);
    }

    m_action = new QAction(tr("Spirula Studio"), this);
    m_action->setIcon(
            QIcon(QStringLiteral(":/CC/plugin/qSpirula/images/icon_128.png")));
    m_action->setToolTip(
            tr("Spirula Studio — launch the official 3DGS trainer GUI, run "
               "headless training, and import results into the DB tree"));
    connect(m_action, &QAction::triggered, this, &qSpirula::showDialog);
}

qSpirula::~qSpirula() = default;

QList<QAction*> qSpirula::getActions() { return {m_action}; }

void qSpirula::showDialog() {
    if (!m_dialog) {
        m_dialog = std::make_unique<SpirulaDialog>();
        connect(m_dialog.get(), &SpirulaDialog::launchRequested, this,
                &qSpirula::launchSpirulaStudio);
        connect(m_dialog.get(), &SpirulaDialog::importRequested, this,
                &qSpirula::importResult);
        connect(m_dialog.get(), &SpirulaDialog::trainRequested, this, [this]() {
            if (!m_trainDialog) {
                m_trainDialog = std::make_unique<SpirulaTrainDialog>();
                connect(m_trainDialog.get(), &SpirulaTrainDialog::runFinished,
                        this, &qSpirula::importPath);
            }
            m_trainDialog->refreshExecutable();
            m_trainDialog->show();
            m_trainDialog->raise();
            m_trainDialog->activateWindow();
        });
    }
    m_dialog->refreshStatus();
    m_dialog->show();
    m_dialog->raise();
    m_dialog->activateWindow();
}

void qSpirula::launchSpirulaStudio() {
    const QString exePath = spirula::locateExecutable();
    QString error;
    if (!spirula::launchGui(exePath, &error)) {
        QMessageBox::warning(nullptr, tr("Spirula Studio"), error);
        return;
    }
    dispToConsole(QStringLiteral("[Spirula] Launched Spirula Studio GUI (%1)")
                          .arg(exePath),
                  ecvMainAppInterface::STD_CONSOLE_MESSAGE);
}

void qSpirula::importResult() {
    static const QString kFilter = QStringLiteral(
            "Spirula results (splat.ply mesh.ply *.ply);;All files (*)");
    const QString picked = QFileDialog::getOpenFileName(
            nullptr, tr("Import Spirula result"), QString(), kFilter);
    if (!picked.isEmpty()) {
        importPath(picked);
    }
}

void qSpirula::importPath(const QString& path) {
    if (!m_app || path.isEmpty()) {
        return;
    }

    const bool looksLikeMesh =
            path.endsWith(QStringLiteral("mesh.ply"), Qt::CaseInsensitive);

    QProgressDialog progress(tr("Importing Spirula result…"), QString(), 0, 0,
                             nullptr);
    progress.setWindowModality(Qt::WindowModal);
    progress.show();

    ccHObject* imported = nullptr;
    QString error;
    if (looksLikeMesh) {
        imported = spirula::loadMesh(path, &error);
    } else {
        const QString splatPath = spirula::resolveSplatPath(path);
        if (splatPath.isEmpty()) {
            error = tr("No splat.ply found in '%1' (run directories and "
                       "step-*.ckpt are resolved by their splat.ply).")
                            .arg(path);
        } else {
            imported = spirula::loadSplatPointCloud(splatPath, &error);
        }
    }
    progress.hide();

    if (!imported) {
        QMessageBox::warning(nullptr, tr("Spirula Studio"),
                             error.isEmpty()
                                     ? tr("Import failed for '%1'.").arg(path)
                                     : error);
        return;
    }

    const QString baseName = QFileInfo(path).baseName();
    const QString uniqueName = ecvPluginDbNaming::makeUnique(
            QStringLiteral("Spirula_") + baseName, m_app);
    imported->setName(uniqueName);
    m_app->addToDB(imported, true, true, false, true);
    m_app->setSelectedInDB(imported, true);
    m_app->zoomOnEntities(imported);
    m_app->refreshAll();

    dispToConsole(QStringLiteral("[Spirula] Added '%1' to the DB tree")
                          .arg(uniqueName),
                  ecvMainAppInterface::STD_CONSOLE_MESSAGE);
}
