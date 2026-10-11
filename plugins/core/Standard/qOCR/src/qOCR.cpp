// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qOCR.h"

#include <QDateTime>
#include <QDir>
#include <QFileInfo>
#include <QMainWindow>
#include <QPainter>
#include <QPolygonF>

#include "OCRBatchCommand.h"
#include "OCRWorker.h"
#include "QtCompat.h"
#include "ecvGenericGLDisplay.h"
#include "ecvImage.h"
#include "ecvMainAppInterface.h"
#include "ecvPersistentSettings.h"
#include "ecvPluginDbNaming.h"

#ifdef AICore_ENABLED
#include "aicore/backend_capi.h"
#endif

qOCR::qOCR(QObject* parent)
    : QObject(parent), ccStdPluginInterface(":/CC/plugin/qOCR/info.json") {
    ecvPS::registerSettingsGroup(QStringLiteral("qOCR"));
    qRegisterMetaType<OCRRunSettings>("OCRRunSettings");
    qRegisterMetaType<OCRRunResult>("OCRRunResult");
    qRegisterMetaType<OCRBoxResult>("OCRBoxResult");
    m_workspaceAction = new QAction(QIcon(":/CC/plugin/qOCR/images/qOCR.svg"),
                                    tr("OCR Workspace"), this);
    m_workspaceAction->setObjectName(QStringLiteral("qOCRWorkspaceAction"));
    m_workspaceAction->setToolTip(
            tr("PP-OCR text recognition: stage models, pick a demo image, "
               "run det+rec, export the annotated result to the DB tree"));
    connect(m_workspaceAction, &QAction::triggered, this, &qOCR::showDialog);

    m_inferenceHeartbeat = new QTimer(this);
    m_inferenceHeartbeat->setInterval(10000);
    connect(m_inferenceHeartbeat, &QTimer::timeout, this, [this]() {
        if (!m_worker || !m_worker->isRunning() || !m_dialog) return;
        m_inferenceElapsedSeconds += 10;
        m_dialog->appendLog(tr("[qOCR] Task is running (%1 s elapsed)...")
                                    .arg(m_inferenceElapsedSeconds));
    });
}

QList<QAction*> qOCR::getActions() { return {m_workspaceAction}; }

void qOCR::registerCommands(ccCommandLineInterface* cmd) {
    if (cmd == nullptr) {
        return;
    }
    cmd->registerCommand(
            ccCommandLineInterface::Command::Shared(new CommandOcrRecognize));
}

void qOCR::setMainAppInterface(ecvMainAppInterface* app) {
    ccStdPluginInterface::setMainAppInterface(app);
}

void qOCR::showDialog() {
    if (!m_app) return;
    if (!m_dialog) {
        m_dialog = new OCRDialog(static_cast<QWidget*>(m_app->getMainWindow()));
        connect(m_dialog, &OCRDialog::runRequested, this, &qOCR::executeTask);
        connect(m_dialog, &OCRDialog::cancelRequested, this, &qOCR::cancelTask);
    }
    m_dialog->show();
    m_dialog->raise();
    m_dialog->activateWindow();
}

void qOCR::executeTask(const OCRRunSettings& settings) {
    if (m_worker && m_worker->isRunning()) return;

    // A fresh worker per run (the shared AICore plugin lifecycle): the old
    // worker's context is released on the main thread before replacement.
    if (m_worker) {
        m_worker->disconnect(this);
        m_worker->disconnect(m_dialog);
        m_worker->releaseContextOnMainThread();
        m_worker->deleteLater();
        m_worker = nullptr;
    }

    if (!QFileInfo::exists(settings.detPath) ||
        !QFileInfo::exists(settings.recPath) ||
        (!settings.tablePath.isEmpty() &&
         !QFileInfo::exists(settings.tablePath))) {
        m_dialog->appendLog(tr("[Error] OCR models missing on disk."));
        return;
    }

    QString workerDevice = settings.device;
#ifdef AICore_ENABLED
    if (workerDevice != QStringLiteral("cpu") &&
        aicore_warmup_backend(workerDevice.toUtf8().constData()) != 0) {
        if (aicore_is_gpu_device(workerDevice.toUtf8().constData())) {
            workerDevice = QStringLiteral("cpu");
            m_dialog->appendLog(
                    tr("[qOCR] GPU backend unavailable — using CPU for this "
                       "run."));
        }
    }
#endif

    m_currentSettings = settings;
    m_currentSettings.device = workerDevice;

    OCRWorker::Settings ws;
    ws.detPath = settings.detPath;
    ws.recPath = settings.recPath;
    ws.layoutPath = settings.layoutPath;
    ws.tablePath = settings.tablePath;
    ws.imagePaths = settings.imagePaths;
    ws.device = workerDevice;
    ws.threads = settings.threads;

    m_worker = new OCRWorker(ws, this);
    connect(m_worker, &OCRWorker::logMessage, m_dialog, &OCRDialog::appendLog,
            Qt::QueuedConnection);
    connect(m_worker, &OCRWorker::taskStage, m_dialog, &OCRDialog::setTaskStage,
            Qt::QueuedConnection);
    connect(m_worker, &OCRWorker::resultReady, this, &qOCR::onResultReady,
            Qt::QueuedConnection);
    connect(m_worker, &OCRWorker::taskFinished, this, &qOCR::onTaskFinished,
            Qt::QueuedConnection);
    m_dialog->setRunning(true);
    m_inferenceElapsedSeconds = 0;
    m_inferenceHeartbeat->start();
    m_worker->start();
}

void qOCR::cancelTask() {
    if (m_worker && m_worker->isRunning()) m_worker->requestTaskCancel();
}

void qOCR::onResultReady(const OCRRunResult& result) {
    m_dialog->showResult(result);
    if (m_currentSettings.addToDb) {
        addResultToDb(result, m_currentSettings);
    } else {
        m_dialog->appendLog(
                tr("[qOCR] Result remains in the workspace table only; "
                   "nothing was added to the DB."));
    }
}

void qOCR::addResultToDb(const OCRRunResult& result,
                         const OCRRunSettings& settings) {
    if (!m_app || result.sourceImage.isNull()) return;

    const QImage annotated = OCRRender::officialVisualization(result);

    const QString deviceTag = ecvPluginDbNaming::deviceTagFromName(
            result.resolvedDevice.isEmpty() ? settings.device
                                            : result.resolvedDevice);
    const QString modelTag =
            ecvPluginDbNaming::modelTagFromFilename(settings.recPath);
    const QString name = ecvPluginDbNaming::makeUnique(
            QStringLiteral("OCR_%1_%2").arg(modelTag, deviceTag), m_app);

    auto* img = new ccImage(annotated, name);
    img->setMetaData(QStringLiteral("OCR"), true);
    img->setMetaData(QStringLiteral("OCR/Count"), result.boxes.size());
    QStringList texts;
    for (const OCRBoxResult& box : result.boxes) {
        texts << QStringLiteral("%1 (%2)").arg(box.text).arg(box.score, 0, 'f',
                                                             3);
    }
    img->setMetaData(QStringLiteral("OCR/Texts"), texts.join('\n'));
    img->setMetaData(QStringLiteral("OCR/DetModel"), result.detModel);
    img->setMetaData(QStringLiteral("OCR/RecModel"), result.recModel);
    if (!result.tableModel.isEmpty()) {
        // Reconstructed tables travel with the overlay: the structured HTML
        // (cell spans) and the RFC 4180 CSV, one block per table.
        QStringList html, csv;
        for (const OCRTableResult& table : result.tables) {
            html << table.html;
            csv << QString::fromUtf8(table.csv);
        }
        img->setMetaData(QStringLiteral("OCR/TableModel"), result.tableModel);
        img->setMetaData(QStringLiteral("OCR/TableCount"),
                         result.tables.size());
        img->setMetaData(QStringLiteral("OCR/TableHTML"), html.join('\n'));
        img->setMetaData(QStringLiteral("OCR/TableCSV"), csv.join('\n'));
        img->setMetaData(QStringLiteral("OCR/Table (ms)"), result.tableMs);
    }
    img->setMetaData(QStringLiteral("Runtime (ms)"),
                     static_cast<double>(result.e2eMs));
    img->setMetaData(QStringLiteral("OCR/Preprocess (ms)"),
                     static_cast<double>(result.preprocessMs));
    img->setMetaData(QStringLiteral("OCR/Inference (ms)"),
                     static_cast<double>(result.inferenceMs));
    img->setMetaData(QStringLiteral("OCR/Postprocess (ms)"),
                     static_cast<double>(result.postprocessMs));
    if (!result.resolvedDevice.isEmpty()) {
        img->setMetaData(QStringLiteral("Device"), result.resolvedDevice);
    }
    img->setMetaData(QStringLiteral("Source"), result.imagePath);
    img->setMetaData(QStringLiteral("Model"),
                     QFileInfo(settings.recPath).fileName());

    // DB materialization always attempts to attach the visible overlay to the
    // active render view. The single Add result to DB choice owns both actions.
    if (ecvGenericGLDisplay* display = m_app->getActiveGLDisplay()) {
        img->setDisplay(display);
        img->setVisible(true);
        img->setEnabled(true);
    } else {
        m_dialog->appendLog(
                tr("[qOCR] No active render window — result added to the DB "
                   "tree without an immediate overlay."));
    }

    m_app->addToDB(img, true, true, false, true);
    m_app->setSelectedInDB(img, true);
    m_dialog->appendLog(
            tr("[qOCR] Added numbered result '%1' to the DB tree.").arg(name));
}

void qOCR::onTaskFinished(bool success) {
    m_inferenceHeartbeat->stop();
    m_dialog->setRunning(false);
    if (m_worker) {
        m_worker->releaseContextOnMainThread();
        m_worker->deleteLater();
        m_worker = nullptr;
    }
    if (!success) m_dialog->appendLog(tr("[Error] OCR task failed."));
}
