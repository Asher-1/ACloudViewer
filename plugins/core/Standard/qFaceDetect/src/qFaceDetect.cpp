// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qFaceDetect.h"

#include <ecvImage.h>
#include <ecvMainAppInterface.h>
#include <ecvPluginDbNaming.h>

#include "ecvPersistentSettings.h"

#include <QDir>
#include <QFile>
#include <QMainWindow>
#include <QMessageBox>

#ifdef AICore_ENABLED
#include "aicore/backend_capi.h"
#include "aicore/runtime_capi.h"
#endif

namespace {

bool isFaceDetectOutputImage(const ccImage* img) {
    if (!img) return false;
    if (img->getName().startsWith(QStringLiteral("FaceDetect_"))) return true;
    return img->getMetaData(QStringLiteral("FaceDetect")).isValid();
}

}  // namespace

qFaceDetect::qFaceDetect(QObject* parent)
    : QObject(parent), ccStdPluginInterface(":/CC/plugin/qFaceDetect/info.json") {
    ecvPS::registerSettingsGroup(QStringLiteral("qFaceDetect"));
    m_action = new QAction(tr("Face Detect"), this);
    m_action->setToolTip(
            tr("InsightFace-style face detection, age/gender, and verify (GGML)"));
    m_action->setIcon(QIcon(":/CC/plugin/qFaceDetect/images/qFaceDetect.svg"));
    connect(m_action, &QAction::triggered, this, &qFaceDetect::showDialog);
}

QList<QAction*> qFaceDetect::getActions() { return {m_action}; }

void qFaceDetect::onNewSelection(const ccHObject::Container& selectedEntities) {
    m_selectedEntities = selectedEntities;
    if (!m_dialog || !m_dialog->isVisible()) return;
    const QStringList names = selectedDbImageNames();
    if (!names.isEmpty()) {
        m_dialog->applyDbTreeSelection(names);
    }
}

ccImage* qFaceDetect::findDbImage(const QString& name) const {
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

QStringList qFaceDetect::selectedDbImageNames() const {
    QStringList names;
    for (ccHObject* obj : m_selectedEntities) {
        if (!obj || !obj->isA(CV_TYPES::IMAGE)) continue;
        ccImage* img = dynamic_cast<ccImage*>(obj);
        if (img && isFaceDetectOutputImage(img)) continue;
        names.append(obj->getName());
    }
    return names;
}

bool qFaceDetect::resolveInputPath(const QString& rawPath,
                                   QString& outPath,
                                   QString* errorMsg) const {
    outPath.clear();
    if (rawPath.startsWith(QStringLiteral("db://"))) {
        const QString name = rawPath.mid(5);
        ccImage* img = findDbImage(name);
        if (!img || img->data().isNull()) {
            if (errorMsg) {
                *errorMsg = tr("DB image not found or empty: %1").arg(name);
            }
            return false;
        }
        const QString tmpDir = FaceDetectDialog::modelCacheDir() + "/../tmp";
        QDir().mkpath(tmpDir);
        outPath = tmpDir + "/" + name + ".png";
        if (!img->data().save(outPath)) {
            if (errorMsg) {
                *errorMsg = tr("Failed to export DB image: %1").arg(name);
            }
            return false;
        }
        return true;
    }
    if (QFile::exists(rawPath)) {
        outPath = rawPath;
        return true;
    }
    if (errorMsg) {
        *errorMsg = tr("Input file not found: %1").arg(rawPath);
    }
    return false;
}

void qFaceDetect::refreshDbImages() {
    if (!m_app || !m_dialog) return;
    ccHObject* root = m_app->dbRootObject();
    if (!root) {
        m_dialog->setDbImages({});
        return;
    }
    ccHObject::Container images;
    root->filterChildren(images, true, CV_TYPES::IMAGE, false);
    QList<FaceDetectDialog::DbImageEntry> entries;
    for (ccHObject* obj : images) {
        if (!obj || !obj->isEnabled()) continue;
        ccImage* img = dynamic_cast<ccImage*>(obj);
        if (!img || isFaceDetectOutputImage(img)) continue;
        FaceDetectDialog::DbImageEntry entry;
        entry.name = obj->getName();
        entry.preview = img->data();
        entries.append(entry);
    }
    m_dialog->setDbImages(entries);
}

void qFaceDetect::showDialog() {
    if (!m_app) return;
    if (!m_dialog) {
        m_dialog = new FaceDetectDialog(
                static_cast<QWidget*>(m_app->getMainWindow()));
        connect(m_dialog, &FaceDetectDialog::runRequested, this,
                &qFaceDetect::executeTask);
        connect(m_dialog, &FaceDetectDialog::cancelRequested, this,
                &qFaceDetect::cancelTask);
        connect(m_dialog, &FaceDetectDialog::refreshDbImagesRequested, this,
                [this]() { refreshDbImages(); });
        connect(m_dialog, &FaceDetectDialog::liveCaptureReady, this,
                [this](const FaceDetectRunResult& result) {
                    FaceDetectDialog::Settings s = m_dialog->getSettings();
                    addResultToDb(result, s, tr("live"));
                });
        connect(m_dialog, &FaceDetectDialog::authVisualizationReady, this,
                &qFaceDetect::onAuthVisualizationReady);
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

void qFaceDetect::executeTask(const FaceDetectDialog::Settings& settings) {
    if (m_worker && m_worker->isRunning()) return;

    if (m_worker) {
        m_worker->disconnect(this);
        m_worker->disconnect(m_dialog);
        m_worker->releaseContextOnMainThread();
        m_worker->deleteLater();
        m_worker = nullptr;
    }

    QString resolvedPath;
    QString err;
    if (!resolveInputPath(settings.inputPath, resolvedPath, &err)) {
        m_dialog->appendLog(err);
        return;
    }

    FaceDetectDialog::Settings workerSettings = settings;
    workerSettings.inputPath = resolvedPath;

    if (!settings.secondInputPath.isEmpty()) {
        QString resolvedSecond;
        if (!resolveInputPath(settings.secondInputPath, resolvedSecond, &err)) {
            m_dialog->appendLog(err);
            return;
        }
        workerSettings.secondInputPath = resolvedSecond;
    }

    if (settings.modelPath.isEmpty()) {
        m_dialog->appendLog(tr("[Error] Model required."));
        return;
    }

    QString workerDevice = settings.device;
#ifdef AICore_ENABLED
    if (aicore_warmup_backend(workerDevice.toUtf8().constData()) != 0) {
        if (aicore_is_gpu_device(workerDevice.toUtf8().constData())) {
            workerDevice = QStringLiteral("cpu");
            m_dialog->appendLog(tr(
                    "[FaceDetect] GPU backend unavailable — using CPU for this run."));
        }
    }
#endif

    FaceDetectWorker::Settings ws;
    ws.modelPath = settings.modelPath;
    ws.landmarkModelPath = settings.landmarkModelPath;
    ws.inputPath = workerSettings.inputPath;
    ws.secondInputPath = workerSettings.secondInputPath;
    ws.threads = settings.threads;
    ws.device = workerDevice;
    ws.mode = static_cast<FaceDetectWorker::Mode>(static_cast<int>(settings.mode));
    ws.verifyThreshold = settings.verifyThreshold;
    ws.antiSpoof = settings.antiSpoof;
    ws.minDetectionScore = settings.minDetectionScore;

    m_currentSettings = settings;
    m_worker = new FaceDetectWorker(ws, this);
    connect(m_worker, &FaceDetectWorker::logMessage, m_dialog,
            &FaceDetectDialog::appendLog);
    connect(m_worker, &FaceDetectWorker::progressUpdate, m_dialog,
            &FaceDetectDialog::setProgress);
    connect(m_worker, &FaceDetectWorker::resultReady, this,
            &qFaceDetect::onResultReady);
    connect(m_worker, &FaceDetectWorker::taskFinished, this,
            &qFaceDetect::onTaskFinished);
    m_dialog->setRunning(true);
    m_worker->start();
}

void qFaceDetect::cancelTask() {
#ifdef AICore_ENABLED
    aicore_cancel_request();
#endif
    if (m_worker && m_worker->isRunning()) m_worker->requestInterruption();
}

void qFaceDetect::onResultReady(const FaceDetectRunResult& result) {
    if (!m_app) return;

    if (m_currentSettings.mode == FaceDetectDialog::Mode::Verify) {
        m_dialog->appendLog(
                tr("[FaceDetect] Verify complete — distance %1, matched=%2")
                        .arg(result.verifyDistance, 0, 'f', 4)
                        .arg(result.verifyMatched));
        if (!result.resultJson.isEmpty()) {
            m_dialog->appendLog(tr("[FaceDetect] Verify JSON:\n%1")
                                        .arg(QString::fromUtf8(
                                                result.resultJson)));
        }
        return;
    }

    if (!m_currentSettings.addAnnotatedImageToDb ||
        result.annotatedImage.isNull()) {
        m_dialog->appendLog(tr("[FaceDetect] Done (no DB export selected)."));
        return;
    }

    addResultToDb(result, m_currentSettings, result.imageName);
}

void qFaceDetect::addResultToDb(const FaceDetectRunResult& result,
                                const FaceDetectDialog::Settings& settings,
                                const QString& sourceLabel) {
    if (!m_app) return;

    if (result.faces.empty()) {
        m_dialog->appendLog(
                tr("[FaceDetect] No faces exported — all %1 detection(s) were "
                   "below min score %2.")
                        .arg(result.totalDetected)
                        .arg(result.minDetectionScoreUsed, 0, 'f', 2));
        return;
    }

    const QString suffix =
            result.mode == QStringLiteral("analyze")
                    ? QStringLiteral("analyze")
                    : (result.mode == QStringLiteral("dense_landmarks")
                               ? QStringLiteral("dense")
                               : QStringLiteral("detect"));
    const QString deviceTag = ecvPluginDbNaming::deviceTagFromName(
            result.resolvedDevice.isEmpty() ? settings.device
                                            : result.resolvedDevice);
    const QString name = ecvPluginDbNaming::makeUnique(
            QStringLiteral("FaceDetect_%1_%2_%3")
                    .arg(suffix, sourceLabel, deviceTag),
            m_app);
    auto* img = new ccImage(result.annotatedImage, name);
    img->setMetaData(QStringLiteral("FaceDetect"), true);
    img->setMetaData(QStringLiteral("FaceDetectMode"), result.mode);
    img->setMetaData(QStringLiteral("FaceCount"),
                     static_cast<qlonglong>(result.faces.size()));
    img->setMetaData(QStringLiteral("FaceDetect/TotalDetected"),
                     result.totalDetected);
    img->setMetaData(QStringLiteral("FaceDetect/RejectedByScore"),
                     result.rejectedByScore);
    img->setMetaData(QStringLiteral("FaceDetect/MinDetectionScore"),
                     static_cast<double>(result.minDetectionScoreUsed));
    img->setMetaData(
            QStringLiteral("FaceDetect/ScoreSemantics"),
            tr("detector confidence in [0,1]; higher is better"));
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
    if (result.mode == QStringLiteral("dense_landmarks") &&
        !settings.landmarkModelPath.isEmpty()) {
        img->setMetaData(QStringLiteral("LandmarkModel"),
                         QFileInfo(settings.landmarkModelPath).fileName());
    }
    if (!result.resultJson.isEmpty()) {
        img->setMetaData(QStringLiteral("FaceDetectResults"),
                         QString::fromUtf8(result.resultJson));
    }
    for (int i = 0; i < static_cast<int>(result.faces.size()); ++i) {
        const FaceDetectBox& f = result.faces[static_cast<size_t>(i)];
        const QString prefix =
                QStringLiteral("FaceDetect/Face%1/").arg(i + 1);
        img->setMetaData(prefix + QStringLiteral("detection_score"), f.score);
        img->setMetaData(prefix + QStringLiteral("box"),
                         QStringLiteral("[%1,%2,%3,%4]")
                                 .arg(f.x1, 0, 'f', 2)
                                 .arg(f.y1, 0, 'f', 2)
                                 .arg(f.x2, 0, 'f', 2)
                                 .arg(f.y2, 0, 'f', 2));
        QStringList lmkParts;
        for (int k = 0; k < 5; ++k) {
            lmkParts << QStringLiteral("[%1,%2]")
                                .arg(f.landmarks[k][0], 0, 'f', 2)
                                .arg(f.landmarks[k][1], 0, 'f', 2);
        }
        img->setMetaData(prefix + QStringLiteral("landmarks"),
                         lmkParts.join(QStringLiteral(", ")));
        if (result.mode == QStringLiteral("analyze") && f.age >= 0) {
            img->setMetaData(prefix + QStringLiteral("age"), f.age);
            img->setMetaData(prefix + QStringLiteral("gender"),
                             QString(QChar(f.gender)));
        }
        if (result.mode == QStringLiteral("dense_landmarks")) {
            img->setMetaData(prefix + QStringLiteral("landmarks_2d_count"),
                             static_cast<qlonglong>(f.denseLandmarks2d.size()));
            img->setMetaData(prefix + QStringLiteral("landmarks_3d_count"),
                             static_cast<qlonglong>(f.denseLandmarks3d.size()));
        }
    }
    m_app->addToDB(img, true, true, false, true);
    m_app->setSelectedInDB(img, true);
    m_dialog->appendLog(tr("[FaceDetect] Added annotated image '%1'.").arg(name));
}

void qFaceDetect::onAuthVisualizationReady(const QImage& annotated,
                                           const QString& summary) {
    if (!m_app || annotated.isNull()) return;
    const QString deviceTag = ecvPluginDbNaming::deviceTagFromName(
            m_dialog ? m_dialog->getSettings().device : QStringLiteral("auto"));
    const QString name = ecvPluginDbNaming::makeUnique(
            QStringLiteral("FaceDetect_auth_%1").arg(deviceTag), m_app);
    auto* img = new ccImage(annotated, name);
    img->setMetaData(QStringLiteral("FaceDetect"), true);
    img->setMetaData(QStringLiteral("FaceDetectMode"), QStringLiteral("auth"));
    if (!summary.isEmpty()) {
        img->setMetaData(QStringLiteral("FaceDetectAuthSummary"), summary);
    }
    m_app->addToDB(img, true, true, false, true);
    m_app->setSelectedInDB(img, true);
    if (m_dialog) {
        m_dialog->appendLog(tr("[Registry] Added auth visualization '%1'.").arg(name));
    }
}

void qFaceDetect::onTaskFinished(bool success) {
    m_dialog->setRunning(false);
    if (m_worker) {
        m_worker->releaseContextOnMainThread();
        m_worker->deleteLater();
        m_worker = nullptr;
    }
    if (!success) m_dialog->appendLog(tr("[Error] Task failed."));
}
