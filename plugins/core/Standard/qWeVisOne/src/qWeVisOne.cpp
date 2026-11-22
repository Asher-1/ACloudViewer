// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qWeVisOne.h"

#include <aicore/inference_log.h>
#include <ecvImage.h>
#include <ecvMainAppInterface.h>

#include <QDir>
#include <QFileInfo>
#include <QImageReader>

#include "WeVisOneBatchCommand.h"
#include "WeVisOneModelCatalog.h"
#include "ecvAICoreRuntimeHelpers.h"
#include "ecvPluginDbNaming.h"

qWeVisOne::qWeVisOne(QObject* parent)
    : QObject(parent),
      ccStdPluginInterface(":/CC/plugin/qWeVisOne/info.json") {}

void qWeVisOne::onNewSelection(const ccHObject::Container& selectedEntities) {
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

QList<QAction*> qWeVisOne::getActions() {
    if (!m_action) {
        m_action =
                new QAction(QIcon(":/CC/plugin/qWeVisOne/images/qWeVisOne.svg"),
                            tr("WeVisOne-VL perception"), this);
        m_action->setToolTip(
                tr("WeVisOne-VL unified visual perception: detection, "
                   "referring, pointing, counting, OCR, segmentation, "
                   "matting, visual-prompt grounding and freeform "
                   "instructions"));
        connect(m_action, &QAction::triggered, this, &qWeVisOne::showDialog);
    }
    return {m_action};
}

void qWeVisOne::registerCommands(ccCommandLineInterface* cmd) {
    if (!cmd) return;
    cmd->registerCommand(
            ccCommandLineInterface::Command::Shared(new CommandWeVisOneRun()));
}

void qWeVisOne::showDialog() {
    if (!m_dialog) {
        // The worker thread hands results back through queued connections;
        // without runtime registration Qt silently drops the struct-carrying
        // signal and onResultReady never runs (no DB export, no summary).
        qRegisterMetaType<WeVisOneRunResult>("WeVisOneRunResult");
        qRegisterMetaType<WeVisOneWorker::Settings>("WeVisOneWorker::Settings");
        m_dialog = new WeVisOneDialog(nullptr);
        connect(m_dialog, &WeVisOneDialog::runRequested, this,
                [this](const WeVisOneWorker::Settings& settings) {
                    m_lastSettings = settings;
                });
        connect(m_dialog, &WeVisOneDialog::resultReady, this,
                [this](const WeVisOneRunResult& result) {
                    onResultReady(result);
                });
    }
    m_dialog->show();
    m_dialog->raise();
    m_dialog->activateWindow();
}

void qWeVisOne::onResultReady(const WeVisOneRunResult& result) {
    // A "None" rejection or a text-only freeform answer carries no boxes
    // and no masks — exporting an unmodified image to the DB tree adds
    // noise without information. The full answer is already in the console
    // log below; every visually annotated result is worth keeping.
    const bool visual = !result.boxes.isEmpty() || result.hasMasks;
    if (visual && m_dialog && m_dialog->exportToDbRequested()) {
        addResultToDb(result, m_lastSettings);
    }

    // One line with every user-visible key fact: task mode (first field so
    // scanning the log immediately identifies which of the eleven released
    // tasks produced each row), input, count/shapes, the staged timings
    // (load / inference / e2e), the resolved device and a human-readable
    // multi-line rendering of the structured answer (one aligned line per
    // <ref>/<box> group). For counting tasks the headline uses the model's
    // stated "Total number: N." (upstream demo/app.py semantics) — the box
    // list is localization evidence and may be shorter. The raw token
    // string remains available through the typed result for downstream
    // parsing; the console shows the parsed form, not the token soup.
    const int stated = WeVisOneHelpers::statedTotalCount(result.answer);
    const bool counting = m_lastSettings.task == QStringLiteral("count_box") ||
                          m_lastSettings.task == QStringLiteral("count_point");
    const QString shapes =
            (counting && stated >= 0
                     ? tr("count %1 (%2 box(es))")
                               .arg(stated)
                               .arg(result.boxes.size())
                     : tr("%1 box(es)").arg(result.boxes.size())) +
            tr(", %1 matte(s)").arg(result.hasMasks ? result.matteCount : 0);
    aicore_inference_log::log(
            QStringLiteral("[qWeVisOne] task=%1 | %2\n"
                           "           %3 | "
                           "load %4 ms, infer %5 ms, e2e %6 ms | "
                           "device %7 (%8 %9)\n"
                           "           result: %10")
                    .arg(m_lastSettings.task, result.imagePath)
                    .arg(shapes)
                    .arg(result.modelLoadMs, 0, 'f', 0)
                    .arg(result.inferenceMs, 0, 'f', 0)
                    .arg(result.e2eMs, 0, 'f', 0)
                    .arg(result.resolvedDevice.isEmpty()
                                 ? m_lastSettings.device
                                 : result.resolvedDevice)
                    .arg(m_lastSettings.size)
                    .arg(result.precision.isEmpty() ? m_lastSettings.precision
                                                    : result.precision)
                    .arg(WeVisOneHelpers::formatAnswerForLog(
                                 result.answer,
                                 QImageReader(result.imagePath).size())
                                 .replace(QStringLiteral("\n"),
                                          QStringLiteral(
                                                  "\n                   "))));
}

void qWeVisOne::addResultToDb(const WeVisOneRunResult& result,
                              const WeVisOneWorker::Settings& settings) {
    if (!m_app || result.annotatedImage.isNull()) return;

    const QString modelTag =
            ecvPluginDbNaming::modelTagFromFilename(settings.vlmPath);
    const QString name = ecvPluginDbNaming::makeUnique(
            QStringLiteral("WeVisOne_%1_%2")
                    .arg(modelTag, QFileInfo(result.imagePath).baseName()),
            m_app);
    auto* img = new ccImage(result.annotatedImage, name);
    img->setMetaData(QStringLiteral("WeVisOne"), true);
    img->setMetaData(QStringLiteral("WeVisOne/Task"), settings.task);
    img->setMetaData(QStringLiteral("WeVisOne/Query"), settings.query);
    if (settings.task == QStringLiteral("visual_prompt"))
        img->setMetaData(QStringLiteral("WeVisOne/RefBoxes"),
                         settings.refBoxes);
    img->setMetaData(QStringLiteral("WeVisOne/Answer"), result.answer);
    img->setMetaData(QStringLiteral("WeVisOne/Boxes"),
                     static_cast<qlonglong>(result.boxes.size()));
    img->setMetaData(
            QStringLiteral("WeVisOne/Mattes"),
            static_cast<qlonglong>(result.hasMasks ? result.matteCount : 0));
    img->setMetaData(QStringLiteral("WeVisOne/Size"), settings.size);
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