// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qVLXSeek.h"

#include <aicore/inference_log.h>
#include <ecvImage.h>
#include <ecvMainAppInterface.h>

#include <QDir>
#include <QFileInfo>

#include "VLXSeekBatchCommand.h"
#include "ecvAICoreRuntimeHelpers.h"
#include "ecvPluginDbNaming.h"

qVLXSeek::qVLXSeek(QObject* parent)
    : QObject(parent), ccStdPluginInterface(":/CC/plugin/qVLXSeek/info.json") {}

void qVLXSeek::onNewSelection(const ccHObject::Container& selectedEntities) {
    if (!m_dialog) return;
    for (ccHObject* entity : selectedEntities) {
        auto* img = ccHObjectCaster::ToImage(entity);
        if (!img) continue;
        // Result images carry their source path as metadata; prefer that,
        // otherwise fall back to the image's file path when one was set.
        QVariant source = img->getMetaData(QStringLiteral("Source"));
        if (source.isValid() && !source.toString().isEmpty() &&
            QFileInfo::exists(source.toString())) {
            m_dialog->setInputPath(source.toString());
            return;
        }
    }
}

QList<QAction*> qVLXSeek::getActions() {
    if (!m_action) {
        m_action =
                new QAction(QIcon(":/CC/plugin/qVLXSeek/images/qVLXSeek.svg"),
                            tr("VLX-Seek grounding"), this);
        m_action->setToolTip(
                tr("VLX-Seek fine-grained open-vocabulary grounding: "
                   "detection, counting, region caption and region OCR via "
                   "region-reference retrieval"));
        connect(m_action, &QAction::triggered, this, &qVLXSeek::showDialog);
    }
    return {m_action};
}

void qVLXSeek::registerCommands(ccCommandLineInterface* cmd) {
    if (!cmd) return;
    cmd->registerCommand(
            ccCommandLineInterface::Command::Shared(new CommandVlxSeekRun()));
}

void qVLXSeek::showDialog() {
    if (!m_dialog) {
        m_dialog = new VLXSeekDialog(nullptr);
        connect(m_dialog, &VLXSeekDialog::runRequested, this,
                [this](const VLXSeekWorker::Settings& settings) {
                    m_lastSettings = settings;
                });
        connect(m_dialog, &VLXSeekDialog::resultReady, this,
                [this](const VLXSeekRunResult& result) {
                    onResultReady(result);
                });
    }
    m_dialog->show();
    m_dialog->raise();
    m_dialog->activateWindow();
}

void qVLXSeek::onResultReady(const VLXSeekRunResult& result) {
    // The dialog's "Export result to DB tree" checkbox (default checked)
    // gates the DB write; the console summary always lands in the shared
    // AICore log route (no plugin-private log widget).
    if (m_dialog && m_dialog->exportToDbRequested()) {
        addResultToDb(result, m_lastSettings);
    }

    aicore_inference_log::log(
            QStringLiteral("[qVLXSeek] %1: %2 box(es), "
                           "answer: %3")
                    .arg(QFileInfo(result.imagePath).fileName())
                    .arg(result.boxes.size())
                    .arg(result.answer.left(160)));
}

void qVLXSeek::addResultToDb(const VLXSeekRunResult& result,
                             const VLXSeekWorker::Settings& settings) {
    if (!m_app || result.annotatedImage.isNull()) return;

    const QString deviceTag = ecvPluginDbNaming::deviceTagFromName(
            result.resolvedDevice.isEmpty() ? settings.device
                                            : result.resolvedDevice);
    const QString modelTag =
            ecvPluginDbNaming::modelTagFromFilename(settings.visionPath);
    const QString name = ecvPluginDbNaming::makeUnique(
            QStringLiteral("VLXSeek_%1_%2")
                    .arg(modelTag, QFileInfo(result.imagePath).baseName()),
            m_app);
    auto* img = new ccImage(result.annotatedImage, name);
    img->setMetaData(QStringLiteral("VLXSeek"), true);
    img->setMetaData(QStringLiteral("VLXSeek/Task"), settings.task);
    img->setMetaData(QStringLiteral("VLXSeek/Text"), settings.text);
    img->setMetaData(QStringLiteral("VLXSeek/Answer"), result.answer);
    img->setMetaData(QStringLiteral("VLXSeek/Boxes"),
                     static_cast<qlonglong>(result.boxes.size()));
    img->setMetaData(QStringLiteral("VLXSeek/Proposals"),
                     static_cast<qlonglong>(result.proposalCount));
    img->setMetaData(QStringLiteral("Runtime (ms)"), result.e2eMs);
    img->setMetaData(QStringLiteral("Source"), result.imagePath);
    if (!result.resolvedDevice.isEmpty())
        img->setMetaData(QStringLiteral("Device"), result.resolvedDevice);
    if (!result.precision.isEmpty())
        img->setMetaData(QStringLiteral("Precision"), result.precision);

    m_app->addToDB(img, true, true, false, true);
    // Select the result so it is immediately visible in the DB tree
    // (the shared AICore plugin finishing touch, qRMBG/qMapAnything
    // pattern).
    m_app->setSelectedInDB(img, true);
}
