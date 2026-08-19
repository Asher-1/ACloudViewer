// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "FaceDetectWorker.h"

#include <QElapsedTimer>
#include <QFileInfo>
#include <QFontMetrics>
#include <QImage>
#include <QImageReader>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QPainter>
#include <QPointF>
#include <QVector3D>
#include <algorithm>

#include "FaceDetectEmbedHelpers.h"

#ifdef AICore_ENABLED
#include "aicore/backend_capi.h"
#include "aicore/facedetect_capi.h"
#endif
#include "aicore/inference_log.h"
#include "aicore/runtime_capi.h"

FaceDetectWorker::FaceDetectWorker(const Settings& settings, QObject* parent)
    : QThread(parent), m_settings(settings) {
#ifdef AICore_ENABLED
    m_cancelToken = aicore_cancel_token_new();
#endif
    qRegisterMetaType<FaceDetectRunResult>("FaceDetectRunResult");
}

FaceDetectWorker::~FaceDetectWorker() {
#ifdef AICore_ENABLED
    aicore_cancel_token_free(m_cancelToken);
#endif
}

void FaceDetectWorker::requestTaskCancel() {
#ifdef AICore_ENABLED
    aicore_cancel_token_request(m_cancelToken);
#endif
    requestInterruption();
}

void FaceDetectWorker::releaseContextOnMainThread() {
#ifdef AICore_ENABLED
    if (m_pendingCtx) {
        aicore_facedetect_free(m_pendingCtx);
        m_pendingCtx = nullptr;
    }
    if (m_pendingLandmarkCtx) {
        aicore_facedetect_free(m_pendingLandmarkCtx);
        m_pendingLandmarkCtx = nullptr;
    }
#endif
}

#ifdef AICore_ENABLED

namespace {

QImage load_rgb_image(const QString& path) {
    QImageReader reader(path);
    reader.setAutoTransform(true);
    QImage img = reader.read();
    if (img.isNull()) return {};
    return img.convertToFormat(QImage::Format_RGB888);
}

bool verifyPathsWithFallback(aicore_facedetect_ctx* ctx,
                             const QString& pathA,
                             const QString& pathB,
                             float threshold,
                             int antiSpoof,
                             float minDetectionScore,
                             float* outDistance,
                             int* outVerified) {
    if (aicore_facedetect_verify_paths(
                ctx, pathA.toUtf8().constData(), pathB.toUtf8().constData(),
                threshold, antiSpoof, outDistance, outVerified) == 0) {
        return true;
    }
    std::vector<float> embA;
    std::vector<float> embB;
    if (!FaceDetectEmbed::embedImagePathWithFallback(ctx, pathA, &embA,
                                                     minDetectionScore) ||
        !FaceDetectEmbed::embedImagePathWithFallback(ctx, pathB, &embB,
                                                     minDetectionScore)) {
        return false;
    }
    double dot = 0.0;
    const size_t n = std::min(embA.size(), embB.size());
    for (size_t i = 0; i < n; ++i) {
        dot += static_cast<double>(embA[i]) * embB[i];
    }
    const float dist = static_cast<float>(1.0 - dot);
    *outDistance = dist;
    *outVerified = dist <= threshold ? 1 : 0;
    return true;
}

std::vector<FaceDetectBox> parse_analyze_json(const QByteArray& json) {
    return FaceDetectEmbed::parseAnalyzeJson(json);
}

std::vector<FaceDetectBox> parse_dense_json(const QByteArray& json) {
    return FaceDetectEmbed::parseDenseJson(json);
}

QImage draw_dense_annotations(const QImage& source,
                              const std::vector<FaceDetectBox>& faces,
                              float minDetectionScore) {
    // Draw directly on source — QPainter paints on RGB888 natively in Qt 5+
    // (same as annotateDetect/annotateRecognize); the extra RGB32 conversion
    // here used to copy the whole frame for nothing.
    QImage rgb = source;
    QPainter painter(&rgb);
    painter.setRenderHint(QPainter::Antialiasing, true);

    QFont labelFont = painter.font();
    labelFont.setPointSize(9);
    labelFont.setBold(true);
    painter.setFont(labelFont);

    for (const FaceDetectBox& f : faces) {
        const bool belowThreshold =
                minDetectionScore > 0.0f && f.score < minDetectionScore;
        const QColor boxColor =
                belowThreshold ? QColor(239, 68, 68) : QColor(34, 211, 238);
        const QColor labelColor =
                belowThreshold ? QColor(254, 202, 202) : QColor(233, 213, 255);

        painter.setPen(QPen(boxColor, 2));
        painter.setBrush(Qt::NoBrush);
        painter.drawRect(QRectF(f.x1, f.y1, f.x2 - f.x1, f.y2 - f.y1));

        painter.setPen(QPen(QColor(253, 224, 71), 1));
        painter.setBrush(QColor(253, 224, 71));
        for (int i = 0; i < 5; ++i) {
            painter.drawEllipse(QPointF(f.landmarks[i][0], f.landmarks[i][1]),
                                2.5, 2.5);
        }
        painter.setBrush(Qt::NoBrush);

        painter.setPen(Qt::NoPen);
        painter.setBrush(QColor(56, 189, 248, 80));
        for (const QPointF& pt : f.denseLandmarks2d) {
            painter.drawEllipse(pt, 1.5, 1.5);
        }

        painter.setBrush(QColor(167, 139, 250, 70));
        for (const QVector3D& pt : f.denseLandmarks3d) {
            painter.drawEllipse(QPointF(pt.x(), pt.y()), 1.2, 1.2);
        }
        painter.setBrush(Qt::NoBrush);

        QString label = QStringLiteral("score %1  2d=%2  3d=%3")
                                .arg(f.score, 0, 'f', 3)
                                .arg(f.denseLandmarks2d.size())
                                .arg(f.denseLandmarks3d.size());
        if (belowThreshold) {
            label += QStringLiteral("  (below min)");
        }
        const QFontMetrics fm(labelFont);
        constexpr int padH = 4;
        constexpr int padV = 3;
        const QRect textBounds = fm.boundingRect(QRect(0, 0, 10000, 10000),
                                                 Qt::TextSingleLine, label);
        const int labelW = textBounds.width() + 2 * padH;
        const int labelH = textBounds.height() + 2 * padV;
        float labelY = f.y1 - static_cast<float>(labelH) - 2.0f;
        if (labelY < 0.0f) labelY = f.y2 + 2.0f;
        const QRectF textRect(f.x1, labelY, static_cast<qreal>(labelW),
                              static_cast<qreal>(labelH));
        painter.fillRect(textRect, QColor(15, 23, 42, 200));
        painter.setPen(labelColor);
        painter.drawText(textRect.adjusted(padH, padV, -padH, -padV),
                         Qt::AlignLeft | Qt::AlignTop | Qt::TextDontClip,
                         label);
    }
    return rgb;
}

}  // namespace

bool FaceDetectWorker::runInference() {
    if (m_settings.inputPath.isEmpty()) {
        emit logMessage("[Error] No input image.");
        return false;
    }
    if (m_settings.modelPath.isEmpty()) {
        emit logMessage("[Error] Detector model required.");
        return false;
    }
    if (m_settings.mode == Mode::DenseLandmarks &&
        m_settings.landmarkModelPath.isEmpty()) {
        emit logMessage("[Error] Landmark model required for Dense Landmarks.");
        return false;
    }

    const QImage rgb = load_rgb_image(m_settings.inputPath);
    if (rgb.isNull()) {
        emit logMessage("[Error] Failed to load input image.");
        return false;
    }

    QElapsedTimer timer;
    timer.start();

    aicore_inference_log::log_device_request(QStringLiteral("FaceDetect"),
                                             m_settings.device);
    emit logMessage("[FaceDetect] Loading detector: " + m_settings.modelPath);

    aicore_facedetect_options* opts = aicore_facedetect_options_new();
    if (!opts) {
        emit logMessage("[Error] Failed to allocate FaceDetect options.");
        return false;
    }
    if (!m_settings.device.isEmpty()) {
        aicore_facedetect_options_set_device(
                opts, m_settings.device.toStdString().c_str());
    }
    aicore_facedetect_options_set_threads(opts, m_settings.threads);

    aicore_facedetect_ctx* ctx = aicore_facedetect_load_opts(
            m_settings.modelPath.toStdString().c_str(), opts);
    if (!aicore_facedetect_is_ready(ctx)) {
        aicore_facedetect_options_free(opts);
        if (ctx) {
            if (const char* err = aicore_facedetect_last_error(ctx)) {
                emit logMessage(QString("[Error] %1").arg(err));
            }
            aicore_facedetect_free(ctx);
        }
        emit logMessage(
                "[Error] Failed to create FaceDetect detector context.");
        return false;
    }

    aicore_facedetect_ctx* landmark_ctx = nullptr;
    if (m_settings.mode == Mode::DenseLandmarks) {
        emit logMessage("[FaceDetect] Loading landmark head: " +
                        m_settings.landmarkModelPath);
        landmark_ctx = aicore_facedetect_load_opts(
                m_settings.landmarkModelPath.toStdString().c_str(), opts);
        if (!aicore_facedetect_is_ready(landmark_ctx)) {
            aicore_facedetect_options_free(opts);
            aicore_facedetect_free(ctx);
            if (landmark_ctx) {
                if (const char* err =
                            aicore_facedetect_last_error(landmark_ctx)) {
                    emit logMessage(QString("[Error] %1").arg(err));
                }
                aicore_facedetect_free(landmark_ctx);
            }
            emit logMessage("[Error] Failed to create landmark context.");
            return false;
        }
    }
    aicore_facedetect_options_free(opts);
    if (const char* err = aicore_facedetect_last_error(ctx)) {
        emit logMessage(QString("[Error] %1").arg(err));
        m_pendingCtx = ctx;
        m_pendingLandmarkCtx = landmark_ctx;
        return false;
    }

    FaceDetectRunResult result;
    result.imagePath = m_settings.inputPath;
    result.imageName = QFileInfo(m_settings.inputPath).completeBaseName();
    result.secondImagePath = m_settings.secondInputPath;

    if (char* info = aicore_facedetect_info_json(ctx)) {
        const QJsonObject obj =
                QJsonDocument::fromJson(QByteArray(info)).object();
        aicore_facedetect_free_string(info);
        result.resolvedDevice = obj.value(QStringLiteral("device")).toString();
        aicore_inference_log::log_device_resolved(QStringLiteral("FaceDetect"),
                                                  result.resolvedDevice);
    }

    emit progressUpdate(20, 100);

    if (m_settings.mode == Mode::Verify) {
        if (m_settings.secondInputPath.isEmpty()) {
            emit logMessage("[Error] Verify mode requires a second image.");
            m_pendingCtx = ctx;
            m_pendingLandmarkCtx = landmark_ctx;
            return false;
        }
        float dist = 0.0f;
        int verified = 0;
        if (!verifyPathsWithFallback(
                    ctx, m_settings.inputPath, m_settings.secondInputPath,
                    m_settings.verifyThreshold, m_settings.antiSpoof ? 1 : 0,
                    m_settings.minDetectionScore, &dist, &verified)) {
            emit logMessage(
                    QString("[Error] Verify failed: %1")
                            .arg(aicore_facedetect_last_error(ctx)
                                         ? aicore_facedetect_last_error(ctx)
                                         : "unknown"));
            m_pendingCtx = ctx;
            m_pendingLandmarkCtx = landmark_ctx;
            return false;
        }
        result.verifyDistance = dist;
        result.verifyMatched = verified;
        result.mode = QStringLiteral("verify");

        // Determine whether anti-spoof vetoed a passing distance.
        const bool distancePassed = dist <= m_settings.verifyThreshold;
        const bool antiSpoofVeto =
                m_settings.antiSpoof && distancePassed && verified == 0;

        {
            QJsonObject root;
            root.insert(QStringLiteral("mode"), result.mode);
            root.insert(QStringLiteral("distance"), dist);
            root.insert(QStringLiteral("verified"), verified != 0);
            root.insert(QStringLiteral("threshold"),
                        m_settings.verifyThreshold);
            root.insert(QStringLiteral("anti_spoof"), m_settings.antiSpoof);
            if (m_settings.antiSpoof) {
                root.insert(QStringLiteral("anti_spoof_passed"),
                            !antiSpoofVeto);
            }
            root.insert(QStringLiteral("image_a"), m_settings.inputPath);
            root.insert(QStringLiteral("image_b"), m_settings.secondInputPath);
            result.resultJson =
                    QJsonDocument(root).toJson(QJsonDocument::Indented);
        }

        QString verifyMsg;
        if (verified) {
            verifyMsg = QString("[FaceDetect] Cosine distance %1 — MATCH "
                                "(threshold %2)")
                                .arg(dist, 0, 'f', 4)
                                .arg(m_settings.verifyThreshold, 0, 'f', 2);
        } else if (antiSpoofVeto) {
            verifyMsg = QString("[FaceDetect] Cosine distance %1 — distance "
                                "PASSED (threshold %2) but REJECTED by "
                                "anti-spoof (liveness check failed)")
                                .arg(dist, 0, 'f', 4)
                                .arg(m_settings.verifyThreshold, 0, 'f', 2);
        } else {
            verifyMsg = QString("[FaceDetect] Cosine distance %1 — NO MATCH "
                                "(threshold %2)")
                                .arg(dist, 0, 'f', 4)
                                .arg(m_settings.verifyThreshold, 0, 'f', 2);
        }
        emit logMessage(verifyMsg);
    } else {
        char* json = nullptr;
        if (m_settings.mode == Mode::DenseLandmarks) {
            json = aicore_facedetect_dense_landmarks_rgb_json(
                    ctx, landmark_ctx, rgb.constBits(), rgb.width(),
                    rgb.height(), m_settings.minDetectionScore);
            result.mode = QStringLiteral("dense_landmarks");
        } else if (m_settings.mode == Mode::Analyze) {
            json = aicore_facedetect_analyze_rgb_json(
                    ctx, rgb.constBits(), rgb.width(), rgb.height(), 0.f);
            result.mode = QStringLiteral("analyze");
        } else {
            json = aicore_facedetect_detect_rgb_json(ctx, rgb.constBits(),
                                                     rgb.width(), rgb.height());
            result.mode = QStringLiteral("detect");
        }
        if (json == nullptr) {
            emit logMessage(
                    QString("[Error] Inference failed: %1")
                            .arg(aicore_facedetect_last_error(ctx)
                                         ? aicore_facedetect_last_error(ctx)
                                         : "unknown"));
            m_pendingCtx = ctx;
            m_pendingLandmarkCtx = landmark_ctx;
            return false;
        }
        const QByteArray payload(json);
        aicore_facedetect_free_string(json);
        result.resultJson = payload;

        if (m_settings.mode == Mode::DenseLandmarks) {
            const auto allFaces = parse_dense_json(payload);
            result.totalDetected = static_cast<int>(allFaces.size());
            result.faces = allFaces;
            result.rejectedByScore = 0;
            result.minDetectionScoreUsed = m_settings.minDetectionScore;
            result.annotatedImage = draw_dense_annotations(
                    rgb, allFaces, m_settings.minDetectionScore);
        } else if (m_settings.mode == Mode::Analyze) {
            const auto allFaces = parse_analyze_json(payload);
            result.totalDetected = static_cast<int>(allFaces.size());
            result.minDetectionScoreUsed = m_settings.minDetectionScore;
            result.faces = allFaces;
            FaceDetectEmbed::filterFacesByScore(&result.faces,
                                                m_settings.minDetectionScore);
            result.rejectedByScore = result.totalDetected -
                                     static_cast<int>(result.faces.size());
            result.annotatedImage = FaceDetectEmbed::annotateAnalyze(
                    rgb, allFaces, m_settings.minDetectionScore);
        } else {
            const auto allFaces = FaceDetectEmbed::parseDetectJson(payload);
            result.totalDetected = static_cast<int>(allFaces.size());
            result.minDetectionScoreUsed = m_settings.minDetectionScore;
            result.faces = allFaces;
            FaceDetectEmbed::filterFacesByScore(&result.faces,
                                                m_settings.minDetectionScore);
            result.rejectedByScore = result.totalDetected -
                                     static_cast<int>(result.faces.size());
            result.annotatedImage = FaceDetectEmbed::annotateDetect(
                    rgb, allFaces, m_settings.minDetectionScore);
        }
        if (result.faces.empty()) {
            if (m_settings.mode == Mode::DenseLandmarks) {
                emit logMessage(
                        QString("[FaceDetect] No dense landmarks (min score "
                                "%1).")
                                .arg(m_settings.minDetectionScore, 0, 'f', 2));
            } else {
                emit logMessage(
                        QString("[FaceDetect] No faces passed min detection "
                                "score "
                                "%1 (detected %2, rejected %3).")
                                .arg(m_settings.minDetectionScore, 0, 'f', 2)
                                .arg(result.totalDetected)
                                .arg(result.rejectedByScore));
            }
        } else {
            if (m_settings.mode == Mode::DenseLandmarks) {
                emit logMessage(
                        QString("[FaceDetect] Dense landmarks on %1 face(s).")
                                .arg(result.faces.size()));
            } else {
                emit logMessage(
                        QString("[FaceDetect] %1 face(s) kept of %2 detected "
                                "(min score %3, rejected %4).")
                                .arg(result.faces.size())
                                .arg(result.totalDetected)
                                .arg(m_settings.minDetectionScore, 0, 'f', 2)
                                .arg(result.rejectedByScore));
            }
        }
    }

    result.runtimeMs = timer.elapsed();
    m_pendingCtx = ctx;
    m_pendingLandmarkCtx = landmark_ctx;
    emit progressUpdate(100, 100);
    QString summary;
    if (m_settings.mode == Mode::Verify) {
        summary = QStringLiteral("verify distance=%1 threshold=%2")
                          .arg(result.verifyDistance, 0, 'f', 3)
                          .arg(m_settings.verifyThreshold, 0, 'f', 2);
    } else if (m_settings.mode == Mode::DenseLandmarks) {
        summary = QStringLiteral("%1 face(s), dense 106+68")
                          .arg(result.faces.size());
    } else {
        summary = QStringLiteral("%1 face(s), min score %2")
                          .arg(result.faces.size())
                          .arg(m_settings.minDetectionScore, 0, 'f', 2);
    }
    aicore_inference_log::log_inference_done(QStringLiteral("FaceDetect"),
                                             result.resolvedDevice,
                                             result.runtimeMs, summary);
    emit resultReady(result);
    return true;
}

#endif

void FaceDetectWorker::run() {
#ifndef AICore_ENABLED
    emit logMessage("[Error] AICore not enabled.");
    emit taskFinished(false);
#else
    if (aicore_device_task_lock_cancelable(
                m_settings.device.toUtf8().constData(), m_cancelToken) != 0) {
        emit taskFinished(false);
        return;
    }
    aicore_cancel_scope_begin(m_cancelToken);
    const bool ok = runInference();
    aicore_cancel_scope_end(m_cancelToken);
    aicore_device_task_unlock();
    emit taskFinished(ok);
#endif
}
