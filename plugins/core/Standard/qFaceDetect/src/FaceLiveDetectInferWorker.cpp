// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "FaceLiveDetectInferWorker.h"

#include <QFileInfo>
#include <algorithm>
#include <cmath>

#include "FaceDetectEmbedHelpers.h"
#include "FaceDetectModelContext.h"

#ifdef AICore_ENABLED
#include "aicore/facedetect_capi.h"
#include "aicore/runtime_capi.h"
#endif

namespace {

QSize displaySizeFor(const QImage& inferImage, float inferScale) {
    if (inferScale <= 0.f || std::abs(inferScale - 1.f) < 1.0e-4f) {
        return inferImage.size();
    }
    return QSize(std::max(1, static_cast<int>(std::lround(inferImage.width() /
                                                          inferScale))),
                 std::max(1, static_cast<int>(std::lround(inferImage.height() /
                                                          inferScale))));
}

}  // namespace

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
#ifdef AICore_ENABLED
    m_loadedDevice.clear();
    m_loadedThreads = 0;
#endif
}

#ifdef AICore_ENABLED
bool FaceLiveDetectInferWorker::ensureModel(const Job& job) {
    if (job.modelPath.isEmpty() || !QFileInfo::exists(job.modelPath))
        return false;
    if (aicore_facedetect_is_ready(m_ctx) &&
        m_loadedModelPath == job.modelPath && m_loadedDevice == job.device &&
        m_loadedThreads == job.threads) {
        return true;
    }
    releaseModel();
    aicore_facedetect_options* opts = aicore_facedetect_options_new();
    if (!opts) return false;
    aicore_facedetect_options_set_device(opts, job.device.toUtf8().constData());
    aicore_facedetect_options_set_threads(opts, job.threads);
    m_ctx = aicore_facedetect_load_opts(job.modelPath.toUtf8().constData(),
                                        opts);
    aicore_facedetect_options_free(opts);
    if (!aicore_facedetect_is_ready(m_ctx)) {
        releaseModel();
        return false;
    }
    m_loadedModelPath = job.modelPath;
    m_loadedDevice = job.device;
    m_loadedThreads = job.threads;
    return true;
}

bool FaceLiveDetectInferWorker::runDetectJob(const Job& job, Result* out) {
    if (!out || !ensureModel(job)) return false;
    // inferRgb is already Format_RGB888 from cvMatToQImage — no conversion.
    char* json = aicore_facedetect_detect_rgb_json(
            m_ctx, job.inferRgb.constBits(), job.inferRgb.width(),
            job.inferRgb.height());
    if (!json) return false;
    const QByteArray payload(json);
    aicore_facedetect_free_string(json);

    // Keep boxes in infer coordinates for annotation at infer resolution.
    auto allFaces = FaceDetectEmbed::parseDetectJson(payload);

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

    // Annotate at infer resolution — avoids expensive full-res conversion.
    QImage annotated = FaceDetectEmbed::annotateDetect(job.inferRgb, allFaces,
                                                       job.minDetectionScore);
    if (job.inferScale != 1.f && !annotated.isNull()) {
        annotated =
                annotated.scaled(displaySizeFor(job.inferRgb, job.inferScale),
                                 Qt::IgnoreAspectRatio, Qt::FastTransformation);
    }
    out->snapshot.annotatedImage = annotated;
    out->displayImage = annotated;

    // Scale boxes to display coordinates for snapshot consumers.
    FaceDetectEmbed::scaleFaceBoxes(&out->snapshot.faces, job.inferScale);
    return true;
}

bool FaceLiveDetectInferWorker::runRecognizeJob(const Job& job, Result* out) {
    if (!out || !ensureModel(job)) return false;
    // inferRgb is already Format_RGB888 from cvMatToQImage — no conversion.
    char* json = aicore_facedetect_detect_rgb_json(
            m_ctx, job.inferRgb.constBits(), job.inferRgb.width(),
            job.inferRgb.height());
    if (!json) return false;
    const QByteArray payload(json);
    aicore_facedetect_free_string(json);

    // Keep boxes in infer coordinates for annotation at infer resolution.
    auto allFaces = FaceDetectEmbed::parseDetectJson(payload);

    out->snapshot.resultJson = payload;
    out->snapshot.totalDetected = static_cast<int>(allFaces.size());
    out->snapshot.minDetectionScoreUsed = job.minDetectionScore;
    out->snapshot.mode = QStringLiteral("recognize");

    // Compute embeddings & labels for ALL faces (parallel to allFaces),
    // then filter both together — keeps labels aligned with snapshot.faces.
    QVector<QString> allLabels(static_cast<int>(allFaces.size()));
    int identified = 0;

    for (size_t i = 0; i < allFaces.size(); ++i) {
        if (allFaces[i].score < job.minDetectionScore) {
            allLabels[static_cast<int>(i)] =
                    QStringLiteral("skipped (det=%1)")
                            .arg(allFaces[i].score, 0, 'f', 2);
            continue;
        }
        if (!job.registry || !job.registry->isOpen()) {
            allLabels[static_cast<int>(i)] = QStringLiteral("Unknown");
            continue;
        }
        std::vector<float> emb;
        if (!FaceDetectEmbed::embedFaceBoxFromFrame(
                    m_ctx, job.inferRgb, allFaces[i], job.minDetectionScore,
                    &emb)) {
            allLabels[static_cast<int>(i)] = QStringLiteral("embed failed");
            continue;
        }
        allLabels[static_cast<int>(i)] = FaceDetectEmbed::labelForEmbedding(
                job.registry, emb, job.matchThreshold);
        if (!allLabels[static_cast<int>(i)].startsWith(
                    QStringLiteral("NO MATCH")) &&
            !allLabels[static_cast<int>(i)].startsWith(
                    QStringLiteral("Unknown"))) {
            ++identified;
        }
    }

    out->identifiedCount = identified;

    // Filter faces and labels together — labels stay aligned with faces.
    out->snapshot.faces = allFaces;
    FaceDetectEmbed::filterFacesByScore(&out->snapshot.faces,
                                        job.minDetectionScore);
    // Rebuild labels for filtered faces using score-based index mapping.
    {
        QVector<QString> filteredLabels;
        filteredLabels.reserve(static_cast<int>(out->snapshot.faces.size()));
        for (size_t i = 0; i < allFaces.size() &&
                           static_cast<int>(filteredLabels.size()) <
                                   static_cast<int>(out->snapshot.faces.size());
             ++i) {
            if (allFaces[i].score >= job.minDetectionScore) {
                filteredLabels.push_back(allLabels[static_cast<int>(i)]);
            }
        }
        out->labels = filteredLabels;
    }
    out->snapshot.rejectedByScore =
            out->snapshot.totalDetected -
            static_cast<int>(out->snapshot.faces.size());
    // Annotate at infer resolution — avoids expensive full-res conversion.
    QImage annotated = FaceDetectEmbed::annotateRecognize(
            job.inferRgb, allFaces, allLabels, job.minDetectionScore);
    if (job.inferScale != 1.f && !annotated.isNull()) {
        annotated =
                annotated.scaled(displaySizeFor(job.inferRgb, job.inferScale),
                                 Qt::IgnoreAspectRatio, Qt::FastTransformation);
    }
    out->snapshot.annotatedImage = annotated;
    out->displayImage = annotated;

    // Scale boxes to display coordinates for snapshot consumers.
    FaceDetectEmbed::scaleFaceBoxes(&out->snapshot.faces, job.inferScale);
    return true;
}
#endif

void FaceLiveDetectInferWorker::runJob(FaceLiveDetectInferWorker::Job job) {
    Result result;
    result.generation = job.generation;
#ifndef AICore_ENABLED
    result.ok = false;
    emit inferComplete(result);
    return;
#else
    FaceDetectInferenceGuard inferenceGuard(job.device);
    if (job.streamMode == StreamMode::Recognize) {
        result.ok = runRecognizeJob(job, &result);
    } else {
        result.ok = runDetectJob(job, &result);
    }
    emit inferComplete(result);
#endif
}
