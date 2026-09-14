// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qGKD.h"

#include <ecvImage.h>
#include <ecvMainAppInterface.h>
#include <ecvPluginDbNaming.h>

#include <QCoreApplication>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QMainWindow>
#include <QMessageBox>
#include <QSettings>
#include <QTimer>
#include <QUuid>

#include "ecvPersistentSettings.h"

#ifdef AICore_ENABLED
#include "aicore/backend_capi.h"
#include "aicore/gkd_capi.h"
#include "aicore/runtime_capi.h"
#include "aicore/yolo_capi.h"
#endif

namespace {

bool isGKDOutputImage(const ccImage* img) {
    if (!img) return false;
    if (img->getName().startsWith(QStringLiteral("GKD_"))) return true;
    return img->getMetaData(QStringLiteral("GKD")).isValid();
}

}  // namespace

qGKD::~qGKD() {
    // Last-resort cleanup only: in the normal exit flow aboutToQuit has
    // already released the resident contexts (see the ctor comment for
    // why that must happen while the event loop is alive).
    releaseResidentContexts();
    drainRetiredContexts();
}

void qGKD::releaseResidentContexts() {
    // A run may still be in flight (user quits while inference runs):
    // cancel it and give the worker thread a bounded window to unwind —
    // otherwise freeing the resident context here would race the
    // inference that is using it (use-after-free). If the worker will
    // not stop in time, deliberately leak the contexts instead: a leak
    // at process exit is harmless, a use-after-free is not.
    if (m_worker && m_worker->isRunning()) {
        m_worker->requestTaskCancel();
        if (!m_worker->wait(5000)) {
            m_worker->disconnect(this);
            m_worker->deleteLater();
            m_worker = nullptr;
            return;
        }
    }
    if (m_worker) {
        m_worker->disconnect(this);
        m_worker->releaseContextOnMainThread();
        m_worker->deleteLater();
        m_worker = nullptr;
    }
#ifdef AICore_ENABLED
    if (m_ctxCache.gkdCtx) {
        aicore_gkd_free(static_cast<aicore_gkd_ctx*>(m_ctxCache.gkdCtx));
        m_ctxCache.gkdCtx = nullptr;
    }
    if (m_ctxCache.yoloCtx) {
        aicore_yolo_free(static_cast<aicore_yolo_ctx*>(m_ctxCache.yoloCtx));
        m_ctxCache.yoloCtx = nullptr;
    }
#endif
}

void qGKD::drainRetiredContexts() {
#ifdef AICore_ENABLED
    // Contexts replaced by a newer load are freed here on the main thread
    // (GPU teardown must not race the render thread).
    for (void* handle : m_ctxCache.retiredGkd) {
        aicore_gkd_free(static_cast<aicore_gkd_ctx*>(handle));
    }
    m_ctxCache.retiredGkd.clear();
    for (void* handle : m_ctxCache.retiredYolo) {
        aicore_yolo_free(static_cast<aicore_yolo_ctx*>(handle));
    }
    m_ctxCache.retiredYolo.clear();
#endif
}

qGKD::qGKD(QObject* parent)
    : QObject(parent), ccStdPluginInterface(":/CC/plugin/qGKD/info.json") {
    ecvPS::registerSettingsGroup(QStringLiteral("qGKD"));
    qRegisterMetaType<GKDRunResult>("GKDRunResult");
    qRegisterMetaType<GKDWorker::Settings>("GKDWorker::Settings");
    m_action = new QAction(tr("GKD Keypoint Detection"), this);
    m_action->setToolTip(
            tr("GKDT general keypoint detection (text / visual / multi-object) "
               "via GGML"));
    m_action->setIcon(QIcon(":/CC/plugin/qGKD/images/qGKD.svg"));
    connect(m_action, &QAction::triggered, this, &qGKD::showDialog);

    m_inferenceHeartbeat = new QTimer(this);
    m_inferenceHeartbeat->setInterval(10000);
    connect(m_inferenceHeartbeat, &QTimer::timeout, this, [this]() {
        if (!m_worker || !m_worker->isRunning() || !m_dialog) return;
        m_inferenceElapsedSeconds += 10;
        m_dialog->appendLog(tr("[GKD] Task is running (%1 s elapsed)...")
                                    .arg(m_inferenceElapsedSeconds));
    });

    // Release the resident inference contexts while the event loop is
    // still alive. Measured crash: the facedetect backend registers
    // atexit(shutdown_backend), which tears the ggml CUDA runtime down
    // BEFORE _dl_fini runs the plugin destructors — a late
    // aicore_gkd_free from ~qGKD then hits a dead CUDA context and
    // ggml_abort kills the process (SIGABRT in
    // ggml_backend_buffer_free). aboutToQuit fires while CUDA is fully
    // alive, so the same frees are safe there.
    connect(QCoreApplication::instance(), &QCoreApplication::aboutToQuit, this,
            &qGKD::releaseResidentContexts);
    connect(QCoreApplication::instance(), &QCoreApplication::aboutToQuit, this,
            &qGKD::drainRetiredContexts);
}

QList<QAction*> qGKD::getActions() { return {m_action}; }

void qGKD::onNewSelection(const ccHObject::Container& selectedEntities) {
    m_selectedEntities = selectedEntities;
    if (!m_dialog || !m_dialog->isVisible()) return;
    const QStringList names = selectedDbImageNames();
    if (!names.isEmpty()) {
        m_dialog->applyDbTreeSelection(names);
    }
}

ccImage* qGKD::findDbImage(const QString& name) const {
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

QStringList qGKD::selectedDbImageNames() const {
    QStringList names;
    for (ccHObject* obj : m_selectedEntities) {
        if (!obj) continue;
        if (obj->isA(CV_TYPES::IMAGE)) {
            ccImage* img = dynamic_cast<ccImage*>(obj);
            if (!img || img->data().isNull()) continue;
            if (isGKDOutputImage(img)) continue;
            names.append(obj->getName());
        } else if (obj->isGroup()) {
            ccHObject::Container images;
            obj->filterChildren(images, true, CV_TYPES::IMAGE, false);
            for (ccHObject* child : images) {
                if (!child) continue;
                ccImage* img = dynamic_cast<ccImage*>(child);
                if (!img || img->data().isNull()) continue;
                if (isGKDOutputImage(img)) continue;
                names.append(child->getName());
            }
        }
    }
    return names;
}

bool qGKD::resolveInputPath(const QString& rawPath,
                            QString* outPath,
                            QString* errorMsg) {
    outPath->clear();
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
        const QString tmpDir = GKDDialog::modelCacheDir() + "/../tmp";
        QDir().mkpath(tmpDir);
        *outPath = tmpDir + "/gkd-" +
                   QUuid::createUuid().toString(QUuid::WithoutBraces) + ".png";
        if (!img->data().save(*outPath)) {
            if (errorMsg) {
                *errorMsg = tr("Failed to export DB image: %1").arg(name);
            }
            return false;
        }
        m_stagedInputFiles << *outPath;
        return true;
    }
    if (QFile::exists(rawPath)) {
        *outPath = rawPath;
        return true;
    }
    if (errorMsg) *errorMsg = tr("Input file not found: %1").arg(rawPath);
    return false;
}

void qGKD::clearStagedInputFiles() {
    for (const QString& path : m_stagedInputFiles) {
        QFile::remove(path);
    }
    m_stagedInputFiles.clear();
}

void qGKD::refreshDbImages() {
    if (!m_app || !m_dialog) return;
    ccHObject* root = m_app->dbRootObject();
    if (!root) {
        m_dialog->setDbImages({});
        return;
    }
    ccHObject::Container images;
    root->filterChildren(images, true, CV_TYPES::IMAGE, false);
    QList<GKDImageEntry> entries;
    for (ccHObject* obj : images) {
        if (!obj || !obj->isEnabled()) continue;
        ccImage* img = dynamic_cast<ccImage*>(obj);
        if (!img || img->data().isNull() || isGKDOutputImage(img)) continue;
        GKDImageEntry entry;
        entry.name = obj->getName();
        entry.preview = img->data();
        entries.append(entry);
    }
    m_dialog->setDbImages(entries);
}

void qGKD::showDialog() {
    if (!m_app) return;
    if (!m_dialog) {
        m_dialog = new GKDDialog(static_cast<QWidget*>(m_app->getMainWindow()));
        m_dialog->setAppInterface(m_app);
        connect(m_dialog, &GKDDialog::runRequested, this, &qGKD::executeTask);
        connect(m_dialog, &GKDDialog::cancelRequested, this, &qGKD::cancelTask);
        connect(m_dialog, &GKDDialog::refreshDbImagesRequested, this,
                [this]() { refreshDbImages(); });
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

void qGKD::executeTask(const GKDWorker::Settings& settings) {
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
    if (!resolveInputPath(settings.inputPath, &resolvedPath, &err)) {
        clearStagedInputFiles();
        m_dialog->appendLog(err);
        return;
    }

    GKDWorker::Settings workerSettings = settings;
    workerSettings.inputPath = resolvedPath;
    // Keep the loaded GKD / YOLO-World contexts resident across runs:
    // without this cache every Run click re-read 483 MiB of GGUF (plus
    // the detector + text tower in multi-object mode), which is the
    // measured root cause of the multi-second lag between runs.
    workerSettings.cache = &m_ctxCache;

    QString workerDevice = settings.device;
#ifdef AICore_ENABLED
    if (aicore_warmup_backend(workerDevice.toUtf8().constData()) != 0) {
        if (aicore_is_gpu_device(workerDevice.toUtf8().constData())) {
            workerDevice = QStringLiteral("cpu");
            m_dialog->appendLog(
                    tr("[GKD] GPU backend unavailable — using CPU for "
                       "this run."));
        }
    }
#endif
    workerSettings.device = workerDevice;

    m_currentSettings = settings;
    m_currentSettings.device = workerDevice;
    m_currentSettings.inputPath = resolvedPath;

    m_worker = new GKDWorker(workerSettings, this);
    connect(m_worker, &GKDWorker::logMessage, m_dialog, &GKDDialog::appendLog,
            Qt::QueuedConnection);
    connect(m_worker, &GKDWorker::taskStage, m_dialog, &GKDDialog::setTaskStage,
            Qt::QueuedConnection);
    connect(m_worker, &GKDWorker::resultReady, this, &qGKD::onResultReady,
            Qt::QueuedConnection);
    connect(m_worker, &GKDWorker::taskFinished, this, &qGKD::onTaskFinished,
            Qt::QueuedConnection);
    m_dialog->setRunning(true);
    m_inferenceElapsedSeconds = 0;
    m_inferenceHeartbeat->start();
    m_worker->start();
}

void qGKD::cancelTask() {
    if (m_worker && m_worker->isRunning()) m_worker->requestTaskCancel();
}

void qGKD::onResultReady(const GKDRunResult& result) {
    if (m_dialog) m_dialog->setLastRun(result);
    if (!m_app) return;
    if (m_currentSettings.addResultToDb && !result.renderedImage.isNull()) {
        addResultToDb(result, m_currentSettings);
    }
    if (!m_currentSettings.savePngDir.isEmpty()) {
        saveResultPng(result);
    }
    if (!m_currentSettings.addResultToDb &&
        m_currentSettings.savePngDir.isEmpty()) {
        m_dialog->appendLog(tr("[GKD] Done (no output selected)."));
    }
}

void qGKD::addResultToDb(const GKDRunResult& result,
                         const GKDWorker::Settings& settings) {
    if (!m_app || result.renderedImage.isNull()) return;

    const QString deviceTag = ecvPluginDbNaming::deviceTagFromName(
            result.resolvedDevice.isEmpty() ? settings.device
                                            : result.resolvedDevice);
    const QString modelTag =
            ecvPluginDbNaming::modelTagFromFilename(settings.modelPath);
    const QString name = ecvPluginDbNaming::makeUnique(
            QStringLiteral("GKD_%1_%2_%3")
                    .arg(modelTag, result.mode, deviceTag),
            m_app);
    auto* img = new ccImage(result.renderedImage, name);
    img->setMetaData(QStringLiteral("GKD"), true);
    img->setMetaData(QStringLiteral("GKD/Mode"), result.mode);
    img->setMetaData(QStringLiteral("GKD/Keypoints"), result.totalKeypoints);
    img->setMetaData(QStringLiteral("GKD/Objects"), result.sets.size());
    img->setMetaData(QStringLiteral("Runtime (ms)"), result.runtimeMs);
    img->setMetaData(QStringLiteral("GKD/Preprocess (ms)"),
                     result.preprocessMs);
    img->setMetaData(QStringLiteral("GKD/Postprocess (ms)"),
                     result.postprocessMs);
    img->setMetaData(QStringLiteral("GKD/Total (ms)"), result.totalRuntimeMs);
    if (!result.imagePath.isEmpty()) {
        img->setMetaData(QStringLiteral("Source"), result.imagePath);
    }
    if (!result.resolvedDevice.isEmpty()) {
        img->setMetaData(QStringLiteral("Device"), result.resolvedDevice);
    }
    if (!result.backend.isEmpty()) {
        img->setMetaData(QStringLiteral("Backend"), result.backend);
    }
    img->setMetaData(QStringLiteral("Model"),
                     QFileInfo(settings.modelPath).fileName());
    if (!result.infoJson.isEmpty()) {
        img->setMetaData(QStringLiteral("GKD/Info"),
                         QString::fromUtf8(result.infoJson));
    }
    m_app->addToDB(img, true, true, false, true);
    m_app->setSelectedInDB(img, true);
    m_dialog->appendLog(tr("[GKD] Added result image '%1'.").arg(name));
}

void qGKD::saveResultPng(const GKDRunResult& result) {
    if (result.renderedImage.isNull() ||
        m_currentSettings.savePngDir.isEmpty()) {
        return;
    }
    QDir dir(m_currentSettings.savePngDir);
    if (!dir.exists() && !dir.mkpath(QStringLiteral("."))) {
        m_dialog->appendLog(tr("[GKD] Cannot create output directory: %1")
                                    .arg(m_currentSettings.savePngDir));
        return;
    }
    const QString base = QFileInfo(result.imageName).completeBaseName();
    QString filePath =
            dir.filePath(QStringLiteral("GKD_%1_%2.png")
                                 .arg(base.isEmpty() ? tr("result") : base)
                                 .arg(QDateTime::currentDateTime().toString(
                                         QStringLiteral("yyyyMMdd_HHmmss"))));
    if (result.renderedImage.save(filePath)) {
        m_dialog->appendLog(tr("[GKD] Saved PNG: %1").arg(filePath));
    } else {
        m_dialog->appendLog(tr("[GKD] Failed to save PNG: %1").arg(filePath));
    }
}

void qGKD::onTaskFinished(bool success) {
    m_inferenceHeartbeat->stop();
    m_dialog->setRunning(false);
    if (m_worker) {
        m_worker->releaseContextOnMainThread();
        m_worker->deleteLater();
        m_worker = nullptr;
    }
    drainRetiredContexts();
    clearStagedInputFiles();
    if (!success) m_dialog->appendLog(tr("[Error] Task failed."));
}
