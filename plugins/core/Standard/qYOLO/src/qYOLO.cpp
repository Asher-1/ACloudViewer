// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qYOLO.h"

#include <ecvImage.h>
#include <ecvMainAppInterface.h>
#include <ecvPluginDbNaming.h>

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QMainWindow>
#include <QMessageBox>
#include <QTimer>
#include <QUuid>

#include "ecvPersistentSettings.h"

#ifdef AICore_ENABLED
#include "aicore/backend_capi.h"
#include "aicore/runtime_capi.h"
#endif

namespace {

bool isYOLOOutputImage(const ccImage* img) {
    if (!img) return false;
    if (img->getName().startsWith(QStringLiteral("YOLO_"))) return true;
    return img->getMetaData(QStringLiteral("YOLO")).isValid();
}

}  // namespace

qYOLO::qYOLO(QObject* parent)
    : QObject(parent), ccStdPluginInterface(":/CC/plugin/qYOLO/info.json") {
    ecvPS::registerSettingsGroup(QStringLiteral("qYOLO"));
    qRegisterMetaType<YOLORunResult>("YOLORunResult");
    qRegisterMetaType<YOLODepthResult>("YOLODepthResult");
    qRegisterMetaType<YOLODialog::Settings>("YOLODialog::Settings");
    m_action = new QAction(tr("YOLO Detect & Depth"), this);
    m_action->setToolTip(
            tr("YOLO real-time object detection / metric depth (GGML)"));
    m_action->setIcon(QIcon(":/CC/plugin/qYOLO/images/qYOLO.svg"));
    connect(m_action, &QAction::triggered, this, &qYOLO::showDialog);

    m_inferenceHeartbeat = new QTimer(this);
    m_inferenceHeartbeat->setInterval(10000);
    connect(m_inferenceHeartbeat, &QTimer::timeout, this, [this]() {
        if (!m_worker || !m_worker->isRunning() || !m_dialog) return;
        m_inferenceElapsedSeconds += 10;
        m_dialog->appendLog(tr("[YOLO] Task is running (%1 s elapsed)...")
                                    .arg(m_inferenceElapsedSeconds));
    });
}

QList<QAction*> qYOLO::getActions() { return {m_action}; }

void qYOLO::onNewSelection(const ccHObject::Container& selectedEntities) {
    m_selectedEntities = selectedEntities;
    if (!m_dialog || !m_dialog->isVisible()) return;
    const QStringList names = selectedDbImageNames();
    if (!names.isEmpty()) {
        m_dialog->applyDbTreeSelection(names);
    }
}

ccImage* qYOLO::findDbImage(const QString& name) const {
    if (!m_app) return nullptr;
    ccHObject* root = m_app->dbRootObject();
    if (!root) return nullptr;
    ccHObject::Container images;
    root->filterChildren(images, true, CV_TYPES::IMAGE, false);
    for (ccHObject* obj : images) {
        if (obj && obj->getName() == name) {
            return dynamic_cast<ccImage*>(obj);
        }
    }
    return nullptr;
}

QStringList qYOLO::selectedDbImageNames() const {
    QStringList names;
    for (ccHObject* obj : m_selectedEntities) {
        if (!obj) continue;
        if (obj->isA(CV_TYPES::IMAGE)) {
            ccImage* img = dynamic_cast<ccImage*>(obj);
            if (!img || img->data().isNull()) continue;
            if (isYOLOOutputImage(img)) continue;
            names.append(obj->getName());
        } else if (obj->isGroup()) {
            ccHObject::Container images;
            obj->filterChildren(images, true, CV_TYPES::IMAGE, false);
            for (ccHObject* child : images) {
                if (!child) continue;
                ccImage* img = dynamic_cast<ccImage*>(child);
                if (!img || img->data().isNull()) continue;
                if (isYOLOOutputImage(img)) continue;
                names.append(child->getName());
            }
        }
    }
    return names;
}

bool qYOLO::resolveInputPath(const QString& rawPath,
                             QString& outPath,
                             QString* errorMsg) {
    outPath.clear();
    if (rawPath.startsWith(QStringLiteral("db://"))) {
        const QString name = rawPath.mid(5);
        ccImage* img = findDbImage(name);
        if (!img) {
            if (errorMsg) *errorMsg = tr("DB image not found: %1").arg(name);
            return false;
        }
        if (img->data().isNull()) {
            if (errorMsg) {
                *errorMsg = tr("DB image has no pixel data: %1").arg(name);
            }
            return false;
        }
        const QString tmpDir = YOLODialog::modelCacheDir() + "/../tmp";
        QDir().mkpath(tmpDir);
        outPath = tmpDir + "/yolo-" +
                  QUuid::createUuid().toString(QUuid::WithoutBraces) + ".png";
        if (!img->data().save(outPath)) {
            if (errorMsg) {
                *errorMsg = tr("Failed to export DB image: %1").arg(name);
            }
            return false;
        }
        m_stagedInputFiles << outPath;
        return true;
    }
    if (QFile::exists(rawPath)) {
        outPath = rawPath;
        return true;
    }
    if (errorMsg) *errorMsg = tr("Input file not found: %1").arg(rawPath);
    return false;
}

void qYOLO::clearStagedInputFiles() {
    for (const QString& path : m_stagedInputFiles) {
        QFile::remove(path);
    }
    m_stagedInputFiles.clear();
}

void qYOLO::refreshDbImages() {
    if (!m_app || !m_dialog) return;
    ccHObject* root = m_app->dbRootObject();
    if (!root) {
        m_dialog->setDbImages({});
        return;
    }
    ccHObject::Container images;
    root->filterChildren(images, true, CV_TYPES::IMAGE, false);
    QList<YOLODialog::DbImageEntry> entries;
    for (ccHObject* obj : images) {
        if (!obj || !obj->isEnabled()) continue;
        ccImage* img = dynamic_cast<ccImage*>(obj);
        if (!img || img->data().isNull() || isYOLOOutputImage(img)) continue;
        YOLODialog::DbImageEntry entry;
        entry.name = obj->getName();
        entry.preview = img->data();
        entries.append(entry);
    }
    m_dialog->setDbImages(entries);
}

void qYOLO::showDialog() {
    if (!m_app) return;
    if (!m_dialog) {
        m_dialog =
                new YOLODialog(static_cast<QWidget*>(m_app->getMainWindow()));
        m_dialog->setAppInterface(m_app);
        connect(m_dialog, &YOLODialog::runRequested, this, &qYOLO::executeTask);
        connect(m_dialog, &YOLODialog::cancelRequested, this,
                &qYOLO::cancelTask);
        connect(m_dialog, &YOLODialog::refreshDbImagesRequested, this,
                [this]() { refreshDbImages(); });
        connect(m_dialog, &YOLODialog::liveCaptureReady, this,
                [this](const YOLORunResult& result) {
                    YOLODialog::Settings s = m_dialog->getSettings();
                    addResultToDb(result, s, tr("live"));
                });
        connect(m_dialog, &YOLODialog::liveDepthCaptureReady, this,
                [this](const YOLODepthResult& result) {
                    YOLODialog::Settings s = m_dialog->getSettings();
                    addDepthResultToDb(result, s, tr("live"));
                });
    }
    m_dialog->refreshModelList();
    refreshDbImages();
    const QStringList selectedNames = selectedDbImageNames();
    if (!selectedNames.isEmpty()) {
        m_dialog->applyDbTreeSelection(selectedNames);
    }
    m_dialog->show();
    m_dialog->raise();
    m_dialog->activateWindow();
}

void qYOLO::executeTask(const YOLODialog::Settings& settings) {
    if (m_worker && m_worker->isRunning()) return;

    if (m_worker) {
        m_worker->disconnect(this);
        m_worker->disconnect(m_dialog);
        m_worker->releaseContextOnMainThread();
        m_worker->deleteLater();
        m_worker = nullptr;
    }
    clearStagedInputFiles();

    if (settings.modelPath.isEmpty()) {
        m_dialog->appendLog(tr("[Error] Model required."));
        return;
    }

    QString resolvedPath;
    QString err;
    if (!resolveInputPath(settings.inputPath, resolvedPath, &err)) {
        clearStagedInputFiles();
        m_dialog->appendLog(err);
        return;
    }

    YOLODialog::Settings workerSettings = settings;
    workerSettings.inputPath = resolvedPath;

    QString workerDevice = settings.device;
#ifdef AICore_ENABLED
    if (aicore_warmup_backend(workerDevice.toUtf8().constData()) != 0) {
        if (aicore_is_gpu_device(workerDevice.toUtf8().constData())) {
            workerDevice = QStringLiteral("cpu");
            m_dialog->appendLog(
                    tr("[YOLO] GPU backend unavailable — using CPU for "
                       "this run."));
        }
    }
#endif

    YOLOWorker::Settings ws;
    ws.modelPath = settings.modelPath;
    ws.inputPath = workerSettings.inputPath;
    ws.threads = settings.threads;
    ws.device = workerDevice;
    ws.confThres = settings.confThres;
    ws.iouThres = settings.iouThres;
    ws.topK = settings.topK;

    m_currentSettings = settings;
    m_worker = new YOLOWorker(ws, this);
    connect(m_worker, &YOLOWorker::logMessage, m_dialog, &YOLODialog::appendLog,
            Qt::QueuedConnection);
    connect(m_worker, &YOLOWorker::progressUpdate, this,
            &qYOLO::onWorkerProgress, Qt::QueuedConnection);
    connect(m_worker, &YOLOWorker::resultReady, this, &qYOLO::onResultReady,
            Qt::QueuedConnection);
    connect(m_worker, &YOLOWorker::depthResultReady, this,
            &qYOLO::onDepthResultReady, Qt::QueuedConnection);
    connect(m_worker, &YOLOWorker::taskFinished, this, &qYOLO::onTaskFinished,
            Qt::QueuedConnection);
    connect(m_worker, &YOLOWorker::modelInfoReady, m_dialog,
            &YOLODialog::appendLog, Qt::QueuedConnection);
    m_dialog->setRunning(true);
    m_inferenceElapsedSeconds = 0;
    m_inferenceHeartbeat->start();
    m_worker->start();
}

void qYOLO::cancelTask() {
    if (m_worker && m_worker->isRunning()) m_worker->requestTaskCancel();
}

void qYOLO::onResultReady(const YOLORunResult& result) {
    if (!m_app) return;

    if (!m_currentSettings.addAnnotatedImageToDb ||
        result.annotatedImage.isNull()) {
        m_dialog->appendLog(tr("[YOLO] Done (no DB export selected)."));
        return;
    }

    addResultToDb(result, m_currentSettings, result.imageName);
}

void qYOLO::onDepthResultReady(const YOLODepthResult& result) {
    if (!m_app) return;

    if (!m_currentSettings.addAnnotatedImageToDb ||
        result.annotatedImage.isNull()) {
        m_dialog->appendLog(tr("[YOLO] Done (no DB export selected)."));
        return;
    }

    addDepthResultToDb(result, m_currentSettings, result.imageName);
}

void qYOLO::addResultToDb(const YOLORunResult& result,
                          const YOLODialog::Settings& settings,
                          const QString& sourceLabel) {
    if (!m_app) return;

    const QString deviceTag = ecvPluginDbNaming::deviceTagFromName(
            result.resolvedDevice.isEmpty() ? settings.device
                                            : result.resolvedDevice);
    const QString name = ecvPluginDbNaming::makeUnique(
            QStringLiteral("YOLO_%1_%2").arg(sourceLabel, deviceTag), m_app);
    auto* img = new ccImage(result.annotatedImage, name);
    img->setMetaData(QStringLiteral("YOLO"), true);
    img->setMetaData(QStringLiteral("YOLO/Task"), QStringLiteral("detect"));
    img->setMetaData(QStringLiteral("YOLO/Model"), result.modelVariant);
    img->setMetaData(QStringLiteral("YOLO/Count"),
                     static_cast<qlonglong>(result.detections.size()));
    img->setMetaData(QStringLiteral("Runtime (ms)"), result.runtimeMs);
    if (!result.imagePath.isEmpty()) {
        img->setMetaData(QStringLiteral("Source"), result.imagePath);
    } else {
        img->setMetaData(QStringLiteral("Source"), sourceLabel);
    }
    if (!result.resolvedDevice.isEmpty()) {
        img->setMetaData(QStringLiteral("Device"), result.resolvedDevice);
    }
    img->setMetaData(QStringLiteral("Model"),
                     QFileInfo(settings.modelPath).fileName());
    if (!result.resultJson.isEmpty()) {
        img->setMetaData(QStringLiteral("YOLO/Results"),
                         QString::fromUtf8(result.resultJson));
    }
    for (int i = 0; i < static_cast<int>(result.detections.size()); ++i) {
        const YOLODetection& d = result.detections[static_cast<size_t>(i)];
        const QString prefix = QStringLiteral("YOLO/Det%1/").arg(i + 1);
        img->setMetaData(prefix + QStringLiteral("class_id"),
                         static_cast<qlonglong>(d.classId));
        img->setMetaData(prefix + QStringLiteral("class_name"), d.className);
        img->setMetaData(prefix + QStringLiteral("score"),
                         static_cast<double>(d.score));
        img->setMetaData(prefix + QStringLiteral("box"),
                         QStringLiteral("[%1,%2,%3,%4]")
                                 .arg(d.x1, 0, 'f', 2)
                                 .arg(d.y1, 0, 'f', 2)
                                 .arg(d.x2, 0, 'f', 2)
                                 .arg(d.y2, 0, 'f', 2));
    }
    m_app->addToDB(img, true, true, false, true);
    m_app->setSelectedInDB(img, true);
    m_dialog->appendLog(tr("[YOLO] Added annotated image '%1'.").arg(name));
}

void qYOLO::addDepthResultToDb(const YOLODepthResult& result,
                               const YOLODialog::Settings& settings,
                               const QString& sourceLabel) {
    if (!m_app || result.annotatedImage.isNull()) return;

    const QString deviceTag = ecvPluginDbNaming::deviceTagFromName(
            result.resolvedDevice.isEmpty() ? settings.device
                                            : result.resolvedDevice);
    const QString name = ecvPluginDbNaming::makeUnique(
            QStringLiteral("YOLODepth_%1_%2").arg(sourceLabel, deviceTag),
            m_app);
    auto* img = new ccImage(result.annotatedImage, name);
    img->setMetaData(QStringLiteral("YOLO"), true);
    img->setMetaData(QStringLiteral("YOLO/Task"), QStringLiteral("depth"));
    img->setMetaData(QStringLiteral("YOLO/Model"), result.modelVariant);
    img->setMetaData(QStringLiteral("YOLO/DepthWidth"),
                     static_cast<qlonglong>(result.width));
    img->setMetaData(QStringLiteral("YOLO/DepthHeight"),
                     static_cast<qlonglong>(result.height));
    img->setMetaData(QStringLiteral("YOLO/MinDepth (m)"),
                     result.stats.minDepth);
    img->setMetaData(QStringLiteral("YOLO/MaxDepth (m)"),
                     result.stats.maxDepth);
    img->setMetaData(QStringLiteral("YOLO/MeanDepth (m)"),
                     result.stats.meanDepth);
    img->setMetaData(QStringLiteral("YOLO/P95Depth (m)"),
                     result.stats.p95Depth);
    img->setMetaData(QStringLiteral("YOLO/ValidPixels"),
                     static_cast<qlonglong>(result.stats.validPixels));
    img->setMetaData(QStringLiteral("Runtime (ms)"), result.runtimeMs);
    if (!result.imagePath.isEmpty()) {
        img->setMetaData(QStringLiteral("Source"), result.imagePath);
    } else {
        img->setMetaData(QStringLiteral("Source"), sourceLabel);
    }
    if (!result.resolvedDevice.isEmpty()) {
        img->setMetaData(QStringLiteral("Device"), result.resolvedDevice);
    }
    img->setMetaData(QStringLiteral("Model"),
                     QFileInfo(settings.modelPath).fileName());
    if (!result.resultJson.isEmpty()) {
        img->setMetaData(QStringLiteral("YOLO/DepthStats"),
                         QString::fromUtf8(result.resultJson));
    }
    m_app->addToDB(img, true, true, false, true);
    m_app->setSelectedInDB(img, true);
    m_dialog->appendLog(tr("[YOLO] Added depth image '%1'.").arg(name));
}

void qYOLO::onTaskFinished(bool success) {
    m_inferenceHeartbeat->stop();
    m_dialog->setRunning(false);
    m_dialog->enableResultButtons(success);
    if (m_worker) {
        m_worker->releaseContextOnMainThread();
        m_worker->deleteLater();
        m_worker = nullptr;
    }
    clearStagedInputFiles();
    if (!success) m_dialog->appendLog(tr("[Error] Task failed."));
}

void qYOLO::onWorkerProgress(int current, int total) {
    if (!m_dialog) return;
    m_dialog->setProgress(current, total);
    const int pct = total > 0 ? (current * 100 / total) : 0;
    QString stage;
    if (pct < 25) {
        stage = tr("Loading GGUF model...");
    } else if (pct < 75) {
        stage = tr("Running inference... (%1%)").arg(pct);
    } else if (pct < 100) {
        stage = tr("Processing result...");
    } else {
        stage = tr("Done.");
    }
    m_dialog->setTaskStage(stage, pct);
}
