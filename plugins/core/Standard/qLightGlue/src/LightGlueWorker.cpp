// LightGlue worker — native feature extraction + AICore GGML matching.

#include "LightGlueWorker.h"

#include <QElapsedTimer>
#include <QFileInfo>

#ifdef AICore_ENABLED
#include "aicore/backend_capi.h"
#include "aicore/lightglue_capi.h"
#include "feature_extractor.h"

#include <QJsonDocument>
#include <QJsonObject>
#endif

LightGlueWorker::LightGlueWorker(const Settings& settings, QObject* parent)
    : QThread(parent), m_settings(settings) {
    static bool registered = false;
    if (!registered) {
        qRegisterMetaType<LightGlueRunResult>("LightGlueRunResult");
        registered = true;
    }
}

void LightGlueWorker::releaseContextOnMainThread() {
#ifdef AICore_ENABLED
    if (m_pendingCtx) {
        aicore_lightglue_free(m_pendingCtx);
        m_pendingCtx = nullptr;
    }
#endif
}

#ifdef AICore_ENABLED

namespace {

QString formatDeviceLog(const LightGlueWorker::Settings& settings) {
    const QString req = settings.device.trimmed();
    if (!req.isEmpty() &&
        req.compare(QLatin1String("auto"), Qt::CaseInsensitive) != 0) {
        return QStringLiteral("[LG] Using device: %1").arg(req);
    }
    return QStringLiteral("[LG] Using device: auto (%1)")
            .arg(QString::fromUtf8(aicore_auto_device_order()));
}

QString formatResolvedDeviceLog(aicore_lightglue_ctx* ctx) {
    if (!ctx) return {};
    char* info = aicore_lightglue_info_json(ctx);
    if (!info) return {};
    const QJsonObject obj =
            QJsonDocument::fromJson(QByteArray(info)).object();
    aicore_lightglue_free_string(info);
    const QString resolved = obj.value(QStringLiteral("device")).toString();
    if (resolved.isEmpty()) return {};
    return QStringLiteral("[LG] ggml backend ready on device: %1").arg(resolved);
}

bool extract_feature_pair(const LightGlueWorker::Settings& settings,
                          lightglue_plugin::OwnedFeatures* f0,
                          lightglue_plugin::OwnedFeatures* f1,
                          QString* log) {
    if (settings.inputPaths.size() != 2) {
        return false;
    }
    const QString p0 = settings.inputPaths[0];
    const QString p1 = settings.inputPaths[1];
    std::string err;

    if (settings.matcherType != 1) {
        if (log) {
            *log = QStringLiteral(
                    "ALIKED image matching requires a native feature extractor "
                    "(COLMAP uses ONNX Runtime for aliked-n16rot.onnx — not "
                    "Python/PyTorch).\n"
                    "GGUF aliked-lightglue-* weights are matcher-only.\n"
                    "[Hint] Select a SIFT LightGlue model for end-to-end C++ "
                    "matching (OpenCV RootSIFT + GGML).");
        }
        return false;
    }

    if (!lightglue_plugin::extract_sift_opencv(
                p0, settings.maxKeypoints, settings.maxResize, f0, &err)) {
        if (log) *log = QString::fromStdString(err);
        return false;
    }
    if (!lightglue_plugin::extract_sift_opencv(
                p1, settings.maxKeypoints, settings.maxResize, f1, &err)) {
        if (log) *log = QString::fromStdString(err);
        return false;
    }
    return true;
}

QVector<QPointF> keypoints_to_qt(const aicore_lightglue_features& f) {
    QVector<QPointF> out;
    if (!f.keypoints || f.n_keypoints <= 0) return out;
    out.reserve(f.n_keypoints);
    for (int32_t i = 0; i < f.n_keypoints; ++i) {
        out.append(QPointF(f.keypoints[i].x, f.keypoints[i].y));
    }
    return out;
}

}  // namespace

bool LightGlueWorker::runModelInfo() {
    emit logMessage(formatDeviceLog(m_settings));
    emit logMessage("[LG] Loading model: " + m_settings.modelPath);
    emit progressUpdate(20, 100);

    aicore_lightglue_options* opts = aicore_lightglue_options_new();
    if (!m_settings.device.isEmpty()) {
        aicore_lightglue_options_set_device(
                opts, m_settings.device.toStdString().c_str());
    }
    aicore_lightglue_options_set_threads(opts, m_settings.threads);
    aicore_lightglue_options_set_matcher_type(opts, m_settings.matcherType);

    aicore_lightglue_ctx* ctx = aicore_lightglue_load_opts(
            m_settings.modelPath.toStdString().c_str(), opts);
    aicore_lightglue_options_free(opts);
    if (!ctx) {
        emit logMessage("[Error] Failed to allocate LightGlue context.");
        return false;
    }
    if (const char* err = aicore_lightglue_last_error(ctx)) {
        emit logMessage(QString("[Error] Failed to load model: %1").arg(err));
        m_pendingCtx = ctx;
        return false;
    }
    emit logMessage(formatResolvedDeviceLog(ctx));

    char* json = aicore_lightglue_info_json(ctx);
    if (json) {
        emit modelInfoReady(QString::fromUtf8(json));
        aicore_lightglue_free_string(json);
    }
    m_pendingCtx = ctx;
    emit progressUpdate(100, 100);
    return true;
}

bool LightGlueWorker::runMatch() {
    if (m_settings.inputPaths.size() != 2) {
        emit logMessage("[Error] LightGlue requires exactly two input images.");
        return false;
    }

    QElapsedTimer timer;
    timer.start();

    emit progressUpdate(5, 100);
    emit logMessage(formatDeviceLog(m_settings));
    emit logMessage("[LG] Loading model: " + m_settings.modelPath);

    aicore_lightglue_options* opts = aicore_lightglue_options_new();
    if (!m_settings.device.isEmpty()) {
        aicore_lightglue_options_set_device(
                opts, m_settings.device.toStdString().c_str());
    }
    aicore_lightglue_options_set_threads(opts, m_settings.threads);
    aicore_lightglue_options_set_min_score(opts, m_settings.minScore);
    aicore_lightglue_options_set_matcher_type(opts, m_settings.matcherType);

    aicore_lightglue_ctx* ctx = aicore_lightglue_load_opts(
            m_settings.modelPath.toStdString().c_str(), opts);
    aicore_lightglue_options_free(opts);
    if (!ctx) {
        emit logMessage("[Error] Failed to allocate LightGlue context.");
        return false;
    }
    if (const char* err = aicore_lightglue_last_error(ctx)) {
        emit logMessage(QString("[Error] Failed to load model: %1").arg(err));
        m_pendingCtx = ctx;
        return false;
    }
    emit logMessage(formatResolvedDeviceLog(ctx));

    emit progressUpdate(20, 100);
    emit logMessage("[LG] Extracting RootSIFT features (OpenCV)...");

    lightglue_plugin::OwnedFeatures f0;
    lightglue_plugin::OwnedFeatures f1;
    QString extractLog;
    if (!extract_feature_pair(m_settings, &f0, &f1, &extractLog)) {
        emit logMessage(QString("[Error] Feature extraction failed: %1")
                                .arg(extractLog));
        m_pendingCtx = ctx;
        return false;
    }

    emit progressUpdate(45, 100);
    emit logMessage(QString("[LG] Matching %1 x %2 keypoints (GGML)...")
                            .arg(f0.view.n_keypoints)
                            .arg(f1.view.n_keypoints));

    aicore_lightglue_match* matches = nullptr;
    int32_t n_matches = 0;
    if (aicore_lightglue_run_match(ctx, &f0.view, &f1.view, &matches,
                                   &n_matches) != 0) {
        const char* matchErr = aicore_lightglue_last_error(ctx);
        emit logMessage(QString("[Error] Matching failed: %1")
                                .arg(matchErr ? matchErr : "unknown"));
        m_pendingCtx = ctx;
        return false;
    }

    LightGlueRunResult result;
    result.imagePath0 = m_settings.inputPaths[0];
    result.imagePath1 = m_settings.inputPaths[1];
    result.imageName0 = result.imagePath0.startsWith("db://")
                                ? result.imagePath0.mid(5)
                                : QFileInfo(result.imagePath0).completeBaseName();
    result.imageName1 = result.imagePath1.startsWith("db://")
                                ? result.imagePath1.mid(5)
                                : QFileInfo(result.imagePath1).completeBaseName();
    result.sourceName = result.imageName0 + "_x_" + result.imageName1;
    result.nKeypoints0 = f0.view.n_keypoints;
    result.nKeypoints1 = f1.view.n_keypoints;
    result.imageWidth0 = f0.view.image_width;
    result.imageHeight0 = f0.view.image_height;
    result.imageWidth1 = f1.view.image_width;
    result.imageHeight1 = f1.view.image_height;
    result.keypoints0 = keypoints_to_qt(f0.view);
    result.keypoints1 = keypoints_to_qt(f1.view);
    result.runtimeMs = timer.elapsed();
    result.matches.reserve(n_matches);
    for (int32_t i = 0; i < n_matches; ++i) {
        result.matches.append(
                {matches[i].idx1, matches[i].idx2, matches[i].score});
    }

    aicore_lightglue_free_matches(matches);
    m_pendingCtx = ctx;

    emit progressUpdate(100, 100);
    emit logMessage(QString("[LG] Found %1 mutual matches in %2 ms.")
                            .arg(n_matches)
                            .arg(result.runtimeMs, 0, 'f', 1));
    emit resultReady(result);
    return true;
}

#endif

void LightGlueWorker::run() {
#ifndef AICore_ENABLED
    emit logMessage("[Error] LightGlue not enabled at build time.");
    emit taskFinished(false);
    return;
#else
    if (isInterruptionRequested()) {
        emit taskFinished(false);
        return;
    }
    bool ok = false;
    switch (m_settings.mode) {
        case Mode::Match:
            ok = runMatch();
            break;
        case Mode::ModelInfo:
            ok = runModelInfo();
            break;
    }
    emit taskFinished(ok);
#endif
}
