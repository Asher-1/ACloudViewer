// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "FaceLiveDetectInferWorker.h"

#include <QFileInfo>

#include "FaceDetectEmbedHelpers.h"

#ifdef AICore_ENABLED
#include "aicore/facedetect_capi.h"
#include "aicore/runtime_capi.h"
#endif

FaceLiveDetectInferWorker::FaceLiveDetectInferWorker(QObject* parent)
    : QObject(parent) {
    qRegisterMetaType<FaceLiveDetectInferWorker::Job>(
            "FaceLiveDetectInferWorker::Job");
    qRegisterMetaType<FaceLiveDetectInferWorker::Result>(
            "FaceLiveDetectInferWorker::Result");
}

void FaceLiveDetectInferWorker::releaseModel() {
#ifdef AICore_ENABLED
    if (m_ctx) {
        aicore_facedetect_free(m_ctx);
        m_ctx = nullptr;
    }
#endif
    m_loadedModelPath.clear();
}

#ifdef AICore_ENABLED
bool FaceLiveDetectInferWorker::ensureModel(const Job& job) {
    if (job.modelPath.isEmpty() || !QFileInfo::exists(job.modelPath))
        return false;
    if (m_ctx && m_loadedModelPath == job.modelPath) return true;
    releaseModel();
    aicore_facedetect_options* opts = aicore_facedetect_options_new();
    aicore_facedetect_options_set_device(opts, job.device.toUtf8().constData());
    aicore_facedetect_options_set_threads(opts, job.threads);
    m_ctx = aicore_facedetect_load_opts(job.modelPath.toUtf8().constData(),
                                        opts);
    aicore_facedetect_options_free(opts);
    if (!m_ctx) return false;
    m_loadedModelPath = job.modelPath;
    return true;
}

bool FaceLiveDetectInferWorker::runDetectJob(const Job& job, Result* out) {
    if (!out || !ensureModel(job)) return false;
    const QImage rgb888 = job.inferRgb.convertToFormat(QImage::Format_RGB888);
    char* json = aicore_facedetect_detect_rgb_json(
            m_ctx, rgb888.constBits(), rgb888.width(), rgb888.height());
    if (!json) return false;
    const QByteArray payload(json);
    aicore_facedetect_free_string(json);

    auto allFaces = FaceDetectEmbed::parseDetectJson(payload);
    FaceDetectEmbed::scaleFaceBoxes(&allFaces, job.inferScale);

    out->snapshot.resultJson = payload;
    out->snapshot.totalDetected = static_cast<int>(allFaces.size());
    out->snapshot.minDetectionScoreUsed = job.minDetectionScore;
    out->snapshot.faces = allFaces;
    FaceDetectEmbed::filterFacesByScore(&out->snapshot.faces,
                                        job.minDetectionScore);
    out->snapshot.rejectedByScore =
            out->snapshot.totalDetected -
            static_cast<int>(out->snapshot.faces.size());
    out->snapshot.mode = QStringLiteral("detect");

    QImage annotated = FaceDetectEmbed::annotateDetect(job.displayRgb, allFaces,
                                                       job.minDetectionScore);
    if (job.inferScale != 1.f && !annotated.isNull()) {
        annotated =
                annotated.scaled(job.displayRgb.size(), Qt::IgnoreAspectRatio,
                                 Qt::SmoothTransformation);
    }
    out->snapshot.annotatedImage = annotated;
    out->displayImage = annotated;
    return true;
}

bool FaceLiveDetectInferWorker::runRecognizeJob(const Job& job, Result* out) {
    if (!out || !ensureModel(job)) return false;
    const QImage rgb888 = job.inferRgb.convertToFormat(QImage::Format_RGB888);
    char* json = aicore_facedetect_detect_rgb_json(
            m_ctx, rgb888.constBits(), rgb888.width(), rgb888.height());
    if (!json) return false;
    const QByteArray payload(json);
    aicore_facedetect_free_string(json);

    auto allFaces = FaceDetectEmbed::parseDetectJson(payload);
    FaceDetectEmbed::scaleFaceBoxes(&allFaces, job.inferScale);

    out->snapshot.resultJson = payload;
    out->snapshot.faces = allFaces;
    out->snapshot.totalDetected = static_cast<int>(allFaces.size());
    out->snapshot.minDetectionScoreUsed = job.minDetectionScore;
    FaceDetectEmbed::filterFacesByScore(&out->snapshot.faces,
                                        job.minDetectionScore);
    out->snapshot.rejectedByScore =
            out->snapshot.totalDetected -
            static_cast<int>(out->snapshot.faces.size());
    out->snapshot.mode = QStringLiteral("recognize");

    const QImage displayRgb =
            job.displayRgb.convertToFormat(QImage::Format_RGB888);
    QVector<QString> labels(static_cast<int>(allFaces.size()));
    int identified = 0;

    if (!job.registry || !job.registry->isOpen()) {
        labels.fill(QStringLiteral("Unknown"),
                    static_cast<int>(allFaces.size()));
    } else {
        for (size_t i = 0; i < allFaces.size(); ++i) {
            if (allFaces[i].score < job.minDetectionScore) {
                labels[static_cast<int>(i)] =
                        QStringLiteral("skipped (det=%1)")
                                .arg(allFaces[i].score, 0, 'f', 2);
                continue;
            }
            std::vector<float> emb;
            if (!FaceDetectEmbed::embedFaceBoxFromFrame(
                        m_ctx, displayRgb, allFaces[i], job.minDetectionScore,
                        &emb)) {
                labels[static_cast<int>(i)] = QStringLiteral("embed failed");
                continue;
            }
            labels[static_cast<int>(i)] = FaceDetectEmbed::labelForEmbedding(
                    job.registry, emb, job.matchThreshold);
            if (!labels[static_cast<int>(i)].startsWith(
                        QStringLiteral("NO MATCH")) &&
                !labels[static_cast<int>(i)].startsWith(
                        QStringLiteral("Unknown"))) {
                ++identified;
            }
        }
    }

    out->identifiedCount = identified;
    const QImage annotated = FaceDetectEmbed::annotateRecognize(
            displayRgb, allFaces, labels, job.minDetectionScore);
    out->snapshot.annotatedImage = annotated;
    out->displayImage = annotated;
    return true;
}
#endif

void FaceLiveDetectInferWorker::runJob(FaceLiveDetectInferWorker::Job job) {
    Result result;
#ifndef AICore_ENABLED
    result.ok = false;
    emit inferComplete(result);
    return;
#else
    aicore_inference_lock();
    aicore_cancel_begin();
    if (job.streamMode == StreamMode::Recognize) {
        result.ok = runRecognizeJob(job, &result);
    } else {
        result.ok = runDetectJob(job, &result);
    }
    aicore_cancel_end();
    aicore_inference_unlock();
    emit inferComplete(result);
#endif
}
