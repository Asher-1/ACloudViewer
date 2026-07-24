#include "qLightGlue.h"

#ifdef AICore_ENABLED
#include "aicore/backend_capi.h"
#include "aicore/lightglue_capi.h"
#endif

#include <ecvImage.h>
#include <ecvMainAppInterface.h>
#include <ecvPluginDbNaming.h>

#include <QAction>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMainWindow>
#include <QMessageBox>

#include "match_visualization.h"
#include "feature_extractor.h"

namespace {

bool isLightGlueOutputImage(const ccImage* img) {
    if (!img) return false;
    if (img->getName().startsWith(QStringLiteral("LG_"))) return true;
    return img->getMetaData(QStringLiteral("Matches")).isValid();
}

}  // namespace

qLightGlue::qLightGlue(QObject* parent)
    : QObject(parent), ccStdPluginInterface(":/CC/plugin/qLightGlue/info.json") {
    qRegisterMetaType<LightGlueRunResult>("LightGlueRunResult");
    m_action = new QAction(tr("LightGlue Feature Matching"), this);
    m_action->setToolTip(
            tr("LightGlue — match sparse features between two images"));
    m_action->setIcon(QIcon(":/CC/plugin/qLightGlue/images/qLightGlue.svg"));
    connect(m_action, &QAction::triggered, this, &qLightGlue::showDialog);
}

void qLightGlue::onNewSelection(const ccHObject::Container& selectedEntities) {
    m_selectedEntities = selectedEntities;
    if (!m_dialog || !m_dialog->isVisible()) return;

    QStringList imageNames;
    for (ccHObject* obj : selectedEntities) {
        if (!obj || !obj->isA(CV_TYPES::IMAGE)) continue;
        ccImage* img = dynamic_cast<ccImage*>(obj);
        if (img && isLightGlueOutputImage(img)) continue;
        imageNames.append(obj->getName());
    }
    if (!imageNames.isEmpty()) {
        m_dialog->applyDbTreeSelection(imageNames);
    }
}

QList<QAction*> qLightGlue::getActions() { return {m_action}; }

bool qLightGlue::warmupInferenceBackend(const QString& device,
                                        QString* logMsg) const {
#ifdef AICore_ENABLED
    const QByteArray dev =
            device.isEmpty() ? QByteArray("auto") : device.toUtf8();
    if (aicore_lightglue_warmup_backend(dev.constData()) != 0) {
        if (logMsg) {
            *logMsg =
                    tr("[Warning] GPU backend warmup failed — worker will "
                       "retry (try CPU if it crashes).");
        }
        return false;
    }
    return true;
#else
    (void)device;
    (void)logMsg;
    return false;
#endif
}

void qLightGlue::showDialog() {
    if (!m_app) return;
    if (!m_dialog) {
        m_dialog = new LightGlueDialog(m_app->getMainWindow());
        connect(m_dialog, &LightGlueDialog::runRequested, this,
                &qLightGlue::executeTask);
        connect(m_dialog, &LightGlueDialog::cancelRequested, this,
                &qLightGlue::cancelTask);
        connect(m_dialog, &LightGlueDialog::exportMatchesRequested, this,
                &qLightGlue::onExportMatches);
        connect(m_dialog, &LightGlueDialog::refreshDbImagesRequested, this,
                [this]() { refreshDbImages(); });
    }
    refreshDbImages();
    const QStringList selectedNames = selectedDbImageNames();
    if (!selectedNames.isEmpty()) {
        m_dialog->applyDbTreeSelection(selectedNames);
    }
    m_dialog->show();
    m_dialog->raise();
    m_dialog->activateWindow();
}

ccImage* qLightGlue::findDbImage(const QString& name) const {
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

QStringList qLightGlue::selectedDbImageNames() const {
    QStringList names;
    for (ccHObject* obj : m_selectedEntities) {
        if (!obj || !obj->isA(CV_TYPES::IMAGE)) continue;
        ccImage* img = dynamic_cast<ccImage*>(obj);
        if (img && isLightGlueOutputImage(img)) continue;
        names.append(obj->getName());
    }
    return names;
}

QImage qLightGlue::loadImageForPath(const QString& path) const {
    if (path.startsWith("db://")) {
        ccImage* img = findDbImage(path.mid(5));
        return img ? img->data() : QImage();
    }
    return lightglue_plugin::load_oriented_qimage(path);
}

bool qLightGlue::resolveInputPaths(const QStringList& rawPaths,
                                   QStringList& outPaths,
                                   QString* errorMsg) const {
    outPaths.clear();
    const QString tmpDir = LightGlueDialog::modelCacheDir() + "/../tmp";
    QDir().mkpath(tmpDir);

    for (const QString& raw : rawPaths) {
        if (raw.startsWith("db://")) {
            const QString name = raw.mid(5);
            ccImage* img = findDbImage(name);
            if (!img || img->data().isNull()) {
                if (errorMsg) {
                    *errorMsg = tr("DB image not found or empty: %1").arg(name);
                }
                return false;
            }
            const QString tmpPath = tmpDir + "/" + name + ".png";
            if (!img->data().save(tmpPath)) {
                if (errorMsg) {
                    *errorMsg = tr("Failed to export DB image: %1").arg(name);
                }
                return false;
            }
            outPaths << tmpPath;
        } else if (QFile::exists(raw)) {
            outPaths << raw;
        } else {
            if (errorMsg) {
                *errorMsg = tr("Input file not found: %1").arg(raw);
            }
            return false;
        }
    }
    return true;
}

void qLightGlue::refreshDbImages() {
    if (!m_app || !m_dialog) return;
    ccHObject* root = m_app->dbRootObject();
    if (!root) {
        m_dialog->setDbImages({});
        return;
    }

    ccHObject::Container images;
    root->filterChildren(images, true, CV_TYPES::IMAGE, false);

    QList<LightGlueDialog::DbImageEntry> entries;
    for (ccHObject* obj : images) {
        if (!obj || !obj->isEnabled()) continue;
        ccImage* img = dynamic_cast<ccImage*>(obj);
        if (!img || isLightGlueOutputImage(img)) continue;
        LightGlueDialog::DbImageEntry entry;
        entry.name = obj->getName();
        entry.preview = img->data();
        entries.append(entry);
    }
    m_dialog->setDbImages(entries);
}

void qLightGlue::executeTask(const LightGlueDialog::Settings& settings) {
    if (m_worker && m_worker->isRunning()) {
        QMessageBox::warning(m_dialog, tr("LightGlue"),
                             tr("A task is already running."));
        return;
    }
    if (m_worker) {
        m_worker->disconnect(this);
        m_worker->disconnect(m_dialog);
        m_worker->deleteLater();
        m_worker = nullptr;
    }

    m_lastResult = {};
    m_currentSettings = settings;
    m_dialog->enableExportButton(false);

    if (settings.modelPath.isEmpty()) {
        m_dialog->appendLog(tr("[Error] Please select a GGUF model."));
        return;
    }

    LightGlueDialog::Settings resolvedSettings = settings;
    if (settings.mode == LightGlueDialog::Mode::Match) {
        if (settings.inputPaths.size() != 2) {
            m_dialog->appendLog(
                    tr("[Error] Select exactly two images for matching."));
            return;
        }
        m_originalInputPaths = settings.inputPaths;
        QStringList resolvedPaths;
        QString err;
        if (!resolveInputPaths(settings.inputPaths, resolvedPaths, &err)) {
            m_dialog->appendLog("[Error] " + err);
            return;
        }
        resolvedSettings.inputPaths = resolvedPaths;
        m_dialog->appendLog(tr("[LG] Resolved 2 input images for matching."));

        if (resolvedSettings.matcherType == 2) {
            m_dialog->appendLog(
                    tr("[Error] ALIKED GGUF models are matcher-only. Interactive "
                       "matching requires a native ALIKED extractor (COLMAP "
                       "uses ONNX Runtime, not Python).\n"
                       "[Hint] Select **SIFT LightGlue** for end-to-end C++ "
                       "matching (OpenCV RootSIFT + GGML)."));
            return;
        }
#ifndef QLIGHTGLUE_HAS_OPENCV
        m_dialog->appendLog(
                tr("[Error] SIFT extraction requires BUILD_OPENCV=ON."));
        return;
#endif
    } else {
        m_originalInputPaths.clear();
    }

    QString warmupMsg;
    QString workerDevice = resolvedSettings.device;
    if (!warmupInferenceBackend(resolvedSettings.device, &warmupMsg)) {
        if (!warmupMsg.isEmpty()) m_dialog->appendLog(warmupMsg);
        if (aicore_is_gpu_device(workerDevice.toUtf8().constData())) {
            workerDevice = QStringLiteral("cpu");
            m_dialog->appendLog(
                    tr("[LG] GPU backend unavailable — using CPU for this run."));
        }
    } else {
        m_dialog->appendLog(tr("[LG] Inference backend ready on UI thread."));
    }

    LightGlueWorker::Settings workerSettings;
    workerSettings.mode = static_cast<LightGlueWorker::Mode>(
            static_cast<int>(resolvedSettings.mode));
    workerSettings.modelPath = resolvedSettings.modelPath;
    workerSettings.inputPaths = resolvedSettings.inputPaths;
    workerSettings.threads = resolvedSettings.threads;
    workerSettings.device = workerDevice;
    workerSettings.minScore = resolvedSettings.minScore;
    workerSettings.matcherType = resolvedSettings.matcherType;

    m_currentSettings = resolvedSettings;
    m_worker = new LightGlueWorker(workerSettings, this);
    connect(m_worker, &LightGlueWorker::logMessage, m_dialog,
            &LightGlueDialog::appendLog, Qt::QueuedConnection);
    connect(m_worker, &LightGlueWorker::progressUpdate, m_dialog,
            &LightGlueDialog::setProgress, Qt::QueuedConnection);
    connect(m_worker, &LightGlueWorker::resultReady, this,
            &qLightGlue::onResultReady, Qt::QueuedConnection);
    connect(m_worker, &LightGlueWorker::modelInfoReady, this,
            &qLightGlue::onModelInfo, Qt::QueuedConnection);
    connect(m_worker, &LightGlueWorker::taskFinished, this,
            &qLightGlue::onTaskFinished, Qt::QueuedConnection);

    m_dialog->setRunning(true);
    m_dialog->appendLog(tr("[LG] Starting task..."));
    m_worker->start();
}

void qLightGlue::cancelTask() {
    if (m_worker && m_worker->isRunning()) {
        m_worker->requestInterruption();
        m_dialog->appendLog(tr("[LG] Cancel requested..."));
    }
}

void qLightGlue::addVisualizationToDb(const LightGlueRunResult& result) {
    if (!m_app) return;

    const QString path0 =
            m_originalInputPaths.size() > 0 ? m_originalInputPaths[0]
                                            : result.imagePath0;
    const QString path1 =
            m_originalInputPaths.size() > 1 ? m_originalInputPaths[1]
                                            : result.imagePath1;
    const QImage img0 = loadImageForPath(path0);
    const QImage img1 = loadImageForPath(path1);
    if (img0.isNull() || img1.isNull()) {
        m_dialog->appendLog(
                tr("[Warning] Could not reload source images for visualization."));
        return;
    }

    const QImage viz = renderMatchVisualization(
            img0, img1, result.keypoints0, result.keypoints1, result.matches,
            result.imageWidth0, result.imageHeight0, result.imageWidth1,
            result.imageHeight1);
    if (viz.isNull()) {
        m_dialog->appendLog(tr("[Warning] Failed to render match visualization."));
        return;
    }

    const QString modelTag =
            ecvPluginDbNaming::modelTagFromFilename(m_currentSettings.modelPath);
    const QString baseName = ecvPluginDbNaming::makeUnique(
            QStringLiteral("LG_%1_%2_x_%3_%4m")
                    .arg(modelTag,
                         ecvPluginDbNaming::sanitizeSegment(result.imageName0),
                         ecvPluginDbNaming::sanitizeSegment(result.imageName1))
                    .arg(result.matches.size()),
            m_app);

    auto* matchImage = new ccImage(viz, baseName);
    matchImage->setMetaData(QStringLiteral("Matches"),
                            QVariant(result.matches.size()));
    matchImage->setMetaData(QStringLiteral("Runtime (ms)"),
                            QVariant(result.runtimeMs));
    matchImage->setMetaData(QStringLiteral("Image 0"), result.imageName0);
    matchImage->setMetaData(QStringLiteral("Image 1"), result.imageName1);
    matchImage->setMetaData(QStringLiteral("Model"),
                            QFileInfo(m_currentSettings.modelPath).fileName());
    matchImage->setVisible(true);
    matchImage->setEnabled(true);
    m_app->addToDB(matchImage, true, true, false, true);
    m_app->setSelectedInDB(matchImage, true);
    m_app->zoomOnEntities(matchImage);
    refreshDbImages();
    m_dialog->appendLog(
            tr("[LG] Added match visualization '%1' to DB (%2 matches, %3 ms)")
                    .arg(matchImage->getName())
                    .arg(result.matches.size())
                    .arg(result.runtimeMs, 0, 'f', 1));
}

void qLightGlue::onResultReady(const LightGlueRunResult& result) {
    m_lastResult = result;
    if (m_currentSettings.addResultToDb) {
        addVisualizationToDb(result);
    }
    m_dialog->enableExportButton(!result.matches.isEmpty());
}

void qLightGlue::onModelInfo(const QString& info) {
    m_dialog->appendLog(tr("[LG] Model info:\n") + info);
}

void qLightGlue::onTaskFinished(bool success) {
    if (success) {
        m_dialog->appendLog(tr("[LG] Task finished."));
        m_dialog->setProgress(100, 100);
    } else {
        m_dialog->appendLog(tr("[Error] Task failed — see log above."));
        m_dialog->setProgress(0, 100);
        m_lastResult = {};
        m_dialog->enableExportButton(false);
    }
    m_dialog->setRunning(false);
    if (m_worker) {
        m_worker->releaseContextOnMainThread();
        m_worker->deleteLater();
        m_worker = nullptr;
    }

    const QString tmpDir = LightGlueDialog::modelCacheDir() + "/../tmp";
    QDir tmp(tmpDir);
    if (tmp.exists()) {
        tmp.removeRecursively();
    }

    if (m_app) {
        m_app->updateUI();
        m_app->refreshAll();
    }
}

void qLightGlue::onExportMatches() {
    if (m_lastResult.matches.isEmpty()) {
        QMessageBox::information(m_dialog, tr("LightGlue"),
                                 tr("No matches to export."));
        return;
    }

    const QString defaultName =
            QStringLiteral("lightglue_%1_matches.json")
                    .arg(m_lastResult.sourceName.isEmpty()
                                 ? QStringLiteral("run")
                                 : m_lastResult.sourceName);
    const QString path = QFileDialog::getSaveFileName(
            m_dialog, tr("Export matches"), defaultName,
            tr("JSON files (*.json);;All files (*)"));
    if (path.isEmpty()) return;

    QJsonObject root;
    root["source"] = m_lastResult.sourceName;
    root["image0"] = m_lastResult.imageName0;
    root["image1"] = m_lastResult.imageName1;
    root["runtime_ms"] = m_lastResult.runtimeMs;
    root["keypoints0"] = m_lastResult.nKeypoints0;
    root["keypoints1"] = m_lastResult.nKeypoints1;
    QJsonArray matches;
    for (const auto& m : m_lastResult.matches) {
        QJsonObject item;
        item["idx1"] = m.idx1;
        item["idx2"] = m.idx2;
        item["score"] = static_cast<double>(m.score);
        if (m.idx1 >= 0 && m.idx1 < m_lastResult.keypoints0.size()) {
            item["x1"] = m_lastResult.keypoints0[m.idx1].x();
            item["y1"] = m_lastResult.keypoints0[m.idx1].y();
        }
        if (m.idx2 >= 0 && m.idx2 < m_lastResult.keypoints1.size()) {
            item["x2"] = m_lastResult.keypoints1[m.idx2].x();
            item["y2"] = m_lastResult.keypoints1[m.idx2].y();
        }
        matches.append(item);
    }
    root["matches"] = matches;

    QFile file(path);
    if (!file.open(QIODevice::WriteOnly)) {
        m_dialog->appendLog(tr("[Error] Failed to write %1").arg(path));
        return;
    }
    file.write(QJsonDocument(root).toJson(QJsonDocument::Indented));
    m_dialog->appendLog(tr("[LG] Exported %1 matches to %2")
                                .arg(m_lastResult.matches.size())
                                .arg(path));
}
