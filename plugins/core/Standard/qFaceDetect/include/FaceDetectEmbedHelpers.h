// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "FaceDetectWorker.h"
#include "FaceRegistryStore.h"

#include <QByteArray>
#include <QImage>
#include <QString>
#include <QVector>
#include <vector>

struct aicore_facedetect_ctx;

namespace FaceDetectEmbed {

constexpr float kDefaultCropMarginRatio = 0.25f;

QString modelCacheDir();

struct AnnotatedFaceLabel {
    FaceDetectBox box;
    QString label;
    bool matched = false;
    bool dashed = false;
    /** When true, skip face rect and draw a top banner (whole-image embed auth). */
    bool labelOnly = false;
};

QImage padImageForDetection(const QImage& src);
FaceDetectBox expandFaceBox(const FaceDetectBox& box, float marginRatio, int imgW,
                            int imgH);
QImage cropFaceRgb(const QImage& rgb, const FaceDetectBox& face);

std::vector<FaceDetectBox> parseDetectJson(const QByteArray& json);
std::vector<FaceDetectBox> parseAnalyzeJson(const QByteArray& json);
std::vector<FaceDetectBox> parseDenseJson(const QByteArray& json);
void filterFacesByScore(std::vector<FaceDetectBox>* faces, float minScore);
void scaleFaceBoxes(std::vector<FaceDetectBox>* faces, float scale);

QString formatMatchLabel(const QString& name, float distance);
QString formatNoMatchLabel(float nearestDistance,
                           const QString& nearestName = QString());
QString labelForEmbedding(const FaceRegistryStore* registry,
                          const std::vector<float>& embedding,
                          float matchThreshold);

#ifdef AICore_ENABLED
/** Load RGB via AICore (libjpeg/stb, cv2.imread parity). Use for detect+embed on
 *  the same buffer — avoids Qt JPEG decode / EXIF drift vs detect_path_json. */
QImage loadRgbForInference(const QString& path);
std::vector<FaceDetectBox> detectBoxesFromRgb(aicore_facedetect_ctx* ctx,
                                              const QImage& rgb);
bool embedCropWithFallback(aicore_facedetect_ctx* ctx, const QImage& crop,
                           std::vector<float>* out, float minDetectionScore);
bool embedImagePathWithFallback(aicore_facedetect_ctx* ctx, const QString& path,
                                std::vector<float>* out, float minDetectionScore);
/** Detect → largest face → embedFaceBoxFromFrame (same path as group-photo auth).
 *  Sets \p usedTemplateFallback true when portrait template alignment was used. */
bool embedImagePathDetectAligned(aicore_facedetect_ctx* ctx, const QString& path,
                                 std::vector<float>* out, float minDetectionScore,
                                 bool* usedTemplateFallback = nullptr);
bool embedFaceBoxFromFrame(aicore_facedetect_ctx* ctx, const QImage& rgb,
                           const FaceDetectBox& box, float minDetectionScore,
                           std::vector<float>* out);
#endif

QImage annotateDetect(const QImage& source, const std::vector<FaceDetectBox>& faces,
                      float minDetectionScore);
QImage annotateAnalyze(const QImage& source, const std::vector<FaceDetectBox>& faces,
                       float minDetectionScore);
QImage annotateLabeledFaces(const QImage& source,
                            const std::vector<AnnotatedFaceLabel>& faces);
QImage annotateRecognize(const QImage& source, const std::vector<FaceDetectBox>& faces,
                         const QVector<QString>& labels, float minDetectionScore);

}  // namespace FaceDetectEmbed
