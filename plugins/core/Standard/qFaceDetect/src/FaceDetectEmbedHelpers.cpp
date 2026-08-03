// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "FaceDetectEmbedHelpers.h"

#include <CVLog.h>

#include <QDir>
#include <QFontMetrics>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QPainter>
#include <QVector3D>
#include <algorithm>
#include <cmath>
#include <cstring>

#ifdef AICore_ENABLED
#include "aicore/facedetect_capi.h"
#endif

namespace FaceDetectEmbed {

QString modelCacheDir() {
#ifdef AICore_ENABLED
    char* dir = aicore_facedetect_model_cache_dir();
    if (dir) {
        const QString result = QString::fromUtf8(dir);
        aicore_facedetect_free_string(dir);
        return result;
    }
#endif
    return QDir::homePath() +
           QStringLiteral("/cloudViewer_data/extract/facedetect_models");
}

QImage padImageForDetection(const QImage& src) {
    const QImage rgb = src.convertToFormat(QImage::Format_RGB888);
    if (rgb.isNull()) return {};
    const int border = std::max(32, std::max(rgb.width(), rgb.height()) / 2);
    QImage out(rgb.width() + 2 * border, rgb.height() + 2 * border,
               QImage::Format_RGB888);
    out.fill(Qt::black);
    QPainter painter(&out);
    painter.drawImage(border, border, rgb);
    return out;
}

FaceDetectBox expandFaceBox(const FaceDetectBox& box,
                            float marginRatio,
                            int imgW,
                            int imgH) {
    const float w = box.x2 - box.x1;
    const float h = box.y2 - box.y1;
    const float mx = w * marginRatio;
    const float my = h * marginRatio;
    FaceDetectBox out = box;
    out.x1 = std::max(0.f, box.x1 - mx);
    out.y1 = std::max(0.f, box.y1 - my);
    out.x2 = std::min(static_cast<float>(imgW), box.x2 + mx);
    out.y2 = std::min(static_cast<float>(imgH), box.y2 + my);
    return out;
}

QImage cropFaceRgb(const QImage& rgb, const FaceDetectBox& face) {
    const int x = std::max(0, static_cast<int>(face.x1));
    const int y = std::max(0, static_cast<int>(face.y1));
    const int w =
            std::min(rgb.width() - x, static_cast<int>(face.x2 - face.x1));
    const int h =
            std::min(rgb.height() - y, static_cast<int>(face.y2 - face.y1));
    if (w <= 8 || h <= 8) return {};
    return rgb.copy(x, y, w, h).convertToFormat(QImage::Format_RGB888);
}

std::vector<FaceDetectBox> parseDetectJson(const QByteArray& json) {
    std::vector<FaceDetectBox> out;
    const QJsonDocument doc = QJsonDocument::fromJson(json);
    if (!doc.isObject()) return out;
    const QJsonArray faces =
            doc.object().value(QStringLiteral("faces")).toArray();
    out.reserve(static_cast<size_t>(faces.size()));
    for (const QJsonValue& fv : faces) {
        const QJsonObject fo = fv.toObject();
        FaceDetectBox box;
        box.score = static_cast<float>(
                fo.value(QStringLiteral("score")).toDouble());
        const QJsonArray bb = fo.value(QStringLiteral("box")).toArray();
        if (bb.size() >= 4) {
            box.x1 = static_cast<float>(bb.at(0).toDouble());
            box.y1 = static_cast<float>(bb.at(1).toDouble());
            box.x2 = static_cast<float>(bb.at(2).toDouble());
            box.y2 = static_cast<float>(bb.at(3).toDouble());
        }
        const QJsonArray lmk = fo.value(QStringLiteral("landmarks")).toArray();
        for (int i = 0; i < 5 && i < lmk.size(); ++i) {
            const QJsonArray pt = lmk.at(i).toArray();
            if (pt.size() >= 2) {
                box.landmarks[i][0] = static_cast<float>(pt.at(0).toDouble());
                box.landmarks[i][1] = static_cast<float>(pt.at(1).toDouble());
            }
        }
        out.push_back(box);
    }
    return out;
}

std::vector<FaceDetectBox> parseAnalyzeJson(const QByteArray& json) {
    std::vector<FaceDetectBox> out;
    const QJsonDocument doc = QJsonDocument::fromJson(json);
    if (!doc.isObject()) return out;
    const QJsonArray faces =
            doc.object().value(QStringLiteral("faces")).toArray();
    out.reserve(static_cast<size_t>(faces.size()));
    for (const QJsonValue& fv : faces) {
        const QJsonObject fo = fv.toObject();
        FaceDetectBox box;
        box.score = static_cast<float>(
                fo.value(QStringLiteral("score")).toDouble());
        const QJsonArray bb = fo.value(QStringLiteral("box")).toArray();
        if (bb.size() >= 4) {
            box.x1 = static_cast<float>(bb.at(0).toDouble());
            box.y1 = static_cast<float>(bb.at(1).toDouble());
            box.x2 = static_cast<float>(bb.at(2).toDouble());
            box.y2 = static_cast<float>(bb.at(3).toDouble());
        }
        box.age = fo.value(QStringLiteral("age")).toInt(-1);
        const QString gender = fo.value(QStringLiteral("gender")).toString();
        box.gender = gender.isEmpty() ? '?' : gender.at(0).toLatin1();
        out.push_back(box);
    }
    return out;
}

std::vector<FaceDetectBox> parseDenseJson(const QByteArray& json) {
    std::vector<FaceDetectBox> out;
    const QJsonDocument doc = QJsonDocument::fromJson(json);
    if (!doc.isObject()) return out;
    const QJsonArray faces =
            doc.object().value(QStringLiteral("faces")).toArray();
    out.reserve(static_cast<size_t>(faces.size()));
    for (const QJsonValue& fv : faces) {
        const QJsonObject fo = fv.toObject();
        FaceDetectBox box;
        box.score = static_cast<float>(
                fo.value(QStringLiteral("score")).toDouble());
        const QJsonArray bb = fo.value(QStringLiteral("box")).toArray();
        if (bb.size() >= 4) {
            box.x1 = static_cast<float>(bb.at(0).toDouble());
            box.y1 = static_cast<float>(bb.at(1).toDouble());
            box.x2 = static_cast<float>(bb.at(2).toDouble());
            box.y2 = static_cast<float>(bb.at(3).toDouble());
        }
        const QJsonArray lmk5 =
                fo.value(QStringLiteral("landmarks_5")).toArray();
        if (lmk5.isEmpty()) {
            const QJsonArray lmk =
                    fo.value(QStringLiteral("landmarks")).toArray();
            for (int i = 0; i < 5 && i < lmk.size(); ++i) {
                const QJsonArray pt = lmk.at(i).toArray();
                if (pt.size() >= 2) {
                    box.landmarks[i][0] =
                            static_cast<float>(pt.at(0).toDouble());
                    box.landmarks[i][1] =
                            static_cast<float>(pt.at(1).toDouble());
                }
            }
        } else {
            for (int i = 0; i < 5 && i < lmk5.size(); ++i) {
                const QJsonArray pt = lmk5.at(i).toArray();
                if (pt.size() >= 2) {
                    box.landmarks[i][0] =
                            static_cast<float>(pt.at(0).toDouble());
                    box.landmarks[i][1] =
                            static_cast<float>(pt.at(1).toDouble());
                }
            }
        }
        const QJsonArray lmk2d =
                fo.value(QStringLiteral("landmarks_2d")).toArray();
        box.denseLandmarks2d.reserve(static_cast<size_t>(lmk2d.size()));
        for (const QJsonValue& pv : lmk2d) {
            const QJsonArray pt = pv.toArray();
            if (pt.size() >= 2) {
                box.denseLandmarks2d.emplace_back(
                        static_cast<qreal>(pt.at(0).toDouble()),
                        static_cast<qreal>(pt.at(1).toDouble()));
            }
        }
        const QJsonArray lmk3d =
                fo.value(QStringLiteral("landmarks_3d")).toArray();
        box.denseLandmarks3d.reserve(static_cast<size_t>(lmk3d.size()));
        for (const QJsonValue& pv : lmk3d) {
            const QJsonArray pt = pv.toArray();
            if (pt.size() >= 3) {
                box.denseLandmarks3d.emplace_back(
                        static_cast<float>(pt.at(0).toDouble()),
                        static_cast<float>(pt.at(1).toDouble()),
                        static_cast<float>(pt.at(2).toDouble()));
            }
        }
        out.push_back(box);
    }
    return out;
}

void filterFacesByScore(std::vector<FaceDetectBox>* faces, float minScore) {
    if (!faces || minScore <= 0.f) return;
    faces->erase(std::remove_if(faces->begin(), faces->end(),
                                [minScore](const FaceDetectBox& f) {
                                    return f.score < minScore;
                                }),
                 faces->end());
}

void scaleFaceBoxes(std::vector<FaceDetectBox>* faces, float scale) {
    if (!faces || scale <= 0.f || std::abs(scale - 1.f) < 1e-4f) return;
    const float inv = 1.f / scale;
    for (FaceDetectBox& f : *faces) {
        f.x1 *= inv;
        f.y1 *= inv;
        f.x2 *= inv;
        f.y2 *= inv;
        for (int i = 0; i < 5; ++i) {
            f.landmarks[i][0] *= inv;
            f.landmarks[i][1] *= inv;
        }
    }
}

QString formatMatchLabel(const QString& name, float distance) {
    return QStringLiteral("%1 (d=%2)").arg(name).arg(distance, 0, 'f', 3);
}

QString formatNoMatchLabel(float nearestDistance, const QString& nearestName) {
    if (!nearestName.isEmpty()) {
        return QStringLiteral("NO MATCH (d=%1, %2)")
                .arg(nearestDistance, 0, 'f', 3)
                .arg(nearestName);
    }
    return QStringLiteral("NO MATCH (d=%1)").arg(nearestDistance, 0, 'f', 3);
}

QString labelForEmbedding(const FaceRegistryStore* registry,
                          const std::vector<float>& embedding,
                          float matchThreshold) {
    if (!registry || !registry->isOpen() || embedding.empty()) {
        return QStringLiteral("Unknown");
    }
    const auto match = registry->bestMatch(embedding, matchThreshold);
    if (match) {
        return formatMatchLabel(match->entry.name, match->distance);
    }
    const auto nearest = registry->nearestMatch(embedding);
    if (nearest) {
        return formatNoMatchLabel(nearest->distance, nearest->entry.name);
    }
    return QStringLiteral("Unknown");
}

#ifdef AICore_ENABLED
const FaceDetectBox* pickPrimaryFaceBox(const std::vector<FaceDetectBox>& boxes,
                                        float minDetectionScore);

QImage loadRgbForInference(const QString& path) {
    uint8_t* rgb = nullptr;
    int32_t w = 0;
    int32_t h = 0;
    const int rc = aicore_facedetect_load_path_rgb(path.toUtf8().constData(),
                                                   &rgb, &w, &h);
    if (rc != 0 || rgb == nullptr || w <= 0 || h <= 0) {
        if (rgb) aicore_facedetect_free_vec(reinterpret_cast<float*>(rgb));
        return {};
    }

    QImage img(w, h, QImage::Format_RGB888);
    const int rowBytes = w * 3;
    const size_t totalBytes =
            static_cast<size_t>(rowBytes) * static_cast<size_t>(h);
    if (img.bytesPerLine() == rowBytes) {
        std::memcpy(img.bits(), rgb, totalBytes);
    } else {
        for (int y = 0; y < h; ++y) {
            std::memcpy(img.scanLine(y),
                        rgb + static_cast<size_t>(y) * rowBytes,
                        static_cast<size_t>(rowBytes));
        }
    }
    aicore_facedetect_free_vec(reinterpret_cast<float*>(rgb));
    return img;
}

const uint8_t* tightRgb888Bytes(const QImage& rgb,
                                int* outW,
                                int* outH,
                                QByteArray* storage) {
    const QImage rgb888 = rgb.convertToFormat(QImage::Format_RGB888);
    if (rgb888.isNull()) return nullptr;
    const int w = rgb888.width();
    const int h = rgb888.height();
    if (outW) *outW = w;
    if (outH) *outH = h;
    const int rowBytes = w * 3;
    if (rgb888.bytesPerLine() == rowBytes) {
        return rgb888.constBits();
    }
    storage->resize(rowBytes * h);
    for (int y = 0; y < h; ++y) {
        std::memcpy(storage->data() + static_cast<size_t>(y) * rowBytes,
                    rgb888.constScanLine(y), static_cast<size_t>(rowBytes));
    }
    return reinterpret_cast<const uint8_t*>(storage->data());
}

std::vector<FaceDetectBox> detectBoxesFromRgb(aicore_facedetect_ctx* ctx,
                                              const QImage& rgb) {
    if (!ctx || rgb.isNull()) return {};
    QByteArray tight;
    int w = 0;
    int h = 0;
    const uint8_t* bytes = tightRgb888Bytes(rgb, &w, &h, &tight);
    if (!bytes) return {};
    char* json = aicore_facedetect_detect_rgb_json(ctx, bytes, w, h);
    const QByteArray payload = json ? QByteArray(json) : QByteArray();
    if (json) aicore_facedetect_free_string(json);
    return parseDetectJson(payload);
}

namespace {

bool faceBoxHasLandmarks(const FaceDetectBox& box) {
    for (int i = 0; i < 5; ++i) {
        if (box.landmarks[i][0] != 0.f || box.landmarks[i][1] != 0.f) {
            return true;
        }
    }
    return false;
}

void offsetFaceBoxes(std::vector<FaceDetectBox>* faces, float dx, float dy) {
    if (!faces) return;
    for (FaceDetectBox& box : *faces) {
        box.x1 += dx;
        box.y1 += box.x2 += dx;
        box.y2 += dy;
        for (int i = 0; i < 5; ++i) {
            box.landmarks[i][0] += dx;
            box.landmarks[i][1] += dy;
        }
    }
}

bool embedRgbLandmarks(aicore_facedetect_ctx* ctx,
                       const QImage& rgb,
                       const FaceDetectBox& box,
                       std::vector<float>* out) {
    if (!ctx || !out || rgb.isNull() || !faceBoxHasLandmarks(box)) return false;

    QByteArray tight;
    int w = 0;
    int h = 0;
    const uint8_t* bytes = tightRgb888Bytes(rgb, &w, &h, &tight);
    if (!bytes) return false;

    float landmarks[10];
    for (int i = 0; i < 5; ++i) {
        landmarks[i * 2 + 0] = box.landmarks[i][0];
        landmarks[i * 2 + 1] = box.landmarks[i][1];
    }

    float* vec = nullptr;
    int dim = 0;
    const int rc = aicore_facedetect_embed_rgb_landmarks(ctx, bytes, w, h,
                                                         landmarks, &vec, &dim);
    if (rc == 0 && vec && dim > 0) {
        out->assign(vec, vec + dim);
        aicore_facedetect_free_vec(vec);
        return true;
    }
    if (vec) aicore_facedetect_free_vec(vec);
    return false;
}

bool tryDetectAlignedEmbed(aicore_facedetect_ctx* ctx,
                           const QImage& rgb,
                           float minDetectionScore,
                           std::vector<float>* out,
                           const QString& logTag,
                           int faceCount) {
    if (rgb.isNull()) return false;
    if (const FaceDetectBox* primary = pickPrimaryFaceBox(
                detectBoxesFromRgb(ctx, rgb), minDetectionScore)) {
        if (embedFaceBoxFromFrame(ctx, rgb, *primary, minDetectionScore, out)) {
            CVLog::Print(QString("[FaceDetect] embed %1: detect-aligned "
                                 "(score=%2, %3 face(s))")
                                 .arg(logTag)
                                 .arg(primary->score, 0, 'f', 3)
                                 .arg(faceCount));
            return true;
        }
    }
    return false;
}

}  // namespace

bool embedCropWithFallback(aicore_facedetect_ctx* ctx,
                           const QImage& crop,
                           std::vector<float>* out,
                           float minDetectionScore) {
    if (!ctx || !out) return false;

    auto tryEmbedRgb = [&](const QImage& img, float minScore) -> bool {
        QByteArray tight;
        int w = 0;
        int h = 0;
        const uint8_t* bytes = tightRgb888Bytes(img, &w, &h, &tight);
        if (!bytes) return false;
        float* vec = nullptr;
        int dim = 0;
        const int rc = aicore_facedetect_embed_rgb(ctx, bytes, w, h, minScore,
                                                   &vec, &dim);
        if (rc == 0 && vec && dim > 0) {
            out->assign(vec, vec + dim);
            aicore_facedetect_free_vec(vec);
            return true;
        }
        if (vec) aicore_facedetect_free_vec(vec);
        return false;
    };

    if (tryEmbedRgb(crop, minDetectionScore) || tryEmbedRgb(crop, 0.f))
        return true;
    const QImage padded = padImageForDetection(crop);
    return tryEmbedRgb(padded, 0.f) || tryEmbedRgb(crop, 0.f);
}

bool embedImagePathWithFallback(aicore_facedetect_ctx* ctx,
                                const QString& path,
                                std::vector<float>* out,
                                float minDetectionScore) {
    if (!ctx || !out) return false;
    for (float minScore : {minDetectionScore, 0.f}) {
        float* vec = nullptr;
        int dim = 0;
        const int rc = aicore_facedetect_embed_path(
                ctx, path.toUtf8().constData(), minScore, &vec, &dim);
        if (rc == 0 && vec && dim > 0) {
            out->assign(vec, vec + dim);
            aicore_facedetect_free_vec(vec);
            return true;
        }
        if (vec) aicore_facedetect_free_vec(vec);
    }
    QImage rgb = loadRgbForInference(path);
    if (rgb.isNull()) return false;
    if (embedCropWithFallback(ctx, rgb, out, minDetectionScore)) return true;
    return embedCropWithFallback(ctx, padImageForDetection(rgb), out, 0.f);
}

const FaceDetectBox* pickPrimaryFaceBox(const std::vector<FaceDetectBox>& boxes,
                                        float minDetectionScore) {
    const FaceDetectBox* best = nullptr;
    float bestArea = 0.f;
    for (const FaceDetectBox& box : boxes) {
        if (minDetectionScore > 0.f && box.score < minDetectionScore) continue;
        const float area =
                std::max(0.f, box.x2 - box.x1) * std::max(0.f, box.y2 - box.y1);
        if (!best || area > bestArea) {
            best = &box;
            bestArea = area;
        }
    }
    return best;
}

bool embedImagePathDetectAligned(aicore_facedetect_ctx* ctx,
                                 const QString& path,
                                 std::vector<float>* out,
                                 float minDetectionScore,
                                 bool* usedTemplateFallback) {
    if (!ctx || !out || path.isEmpty()) return false;
    if (usedTemplateFallback) *usedTemplateFallback = false;

    const QString fileName = QFileInfo(path).fileName();
    QImage rgb = loadRgbForInference(path);

    if (tryDetectAlignedEmbed(
                ctx, rgb, minDetectionScore, out, fileName,
                static_cast<int>(detectBoxesFromRgb(ctx, rgb).size()))) {
        return true;
    }

    if (!rgb.isNull()) {
        const int border =
                std::max(32, std::max(rgb.width(), rgb.height()) / 2);
        const QImage padded = padImageForDetection(rgb);
        std::vector<FaceDetectBox> paddedBoxes =
                detectBoxesFromRgb(ctx, padded);
        offsetFaceBoxes(&paddedBoxes, static_cast<float>(-border),
                        static_cast<float>(-border));
        if (const FaceDetectBox* primary =
                    pickPrimaryFaceBox(paddedBoxes, minDetectionScore)) {
            if (embedFaceBoxFromFrame(ctx, rgb, *primary, minDetectionScore,
                                      out)) {
                CVLog::Print(QString("[FaceDetect] embed %1: detect-aligned "
                                     "padded (score=%2, %3 face(s))")
                                     .arg(fileName)
                                     .arg(primary->score, 0, 'f', 3)
                                     .arg(paddedBoxes.size()));
                return true;
            }
        }
    }

    if (usedTemplateFallback) *usedTemplateFallback = true;
    CVLog::Print(QString("[FaceDetect] embed %1: falling back to template/full "
                         "embed (detect failed on original + padded)")
                         .arg(fileName));
    return embedImagePathWithFallback(ctx, path, out, minDetectionScore);
}

bool embedFaceBoxFromFrame(aicore_facedetect_ctx* ctx,
                           const QImage& rgb,
                           const FaceDetectBox& box,
                           float minDetectionScore,
                           std::vector<float>* out) {
    if (rgb.isNull()) return false;
    if (faceBoxHasLandmarks(box) && embedRgbLandmarks(ctx, rgb, box, out)) {
        return true;
    }

    const FaceDetectBox expanded = expandFaceBox(box, kDefaultCropMarginRatio,
                                                 rgb.width(), rgb.height());
    const QImage crop = cropFaceRgb(rgb, expanded);
    if (crop.isNull()) return false;
    return embedCropWithFallback(ctx, crop, out, minDetectionScore);
}
#endif

QImage annotateDetect(const QImage& source,
                      const std::vector<FaceDetectBox>& faces,
                      float minDetectionScore) {
    QImage rgb = source.convertToFormat(QImage::Format_RGB32);
    QPainter painter(&rgb);
    painter.setRenderHint(QPainter::Antialiasing, true);

    QFont labelFont = painter.font();
    labelFont.setPointSize(9);
    labelFont.setBold(true);
    painter.setFont(labelFont);

    for (const FaceDetectBox& f : faces) {
        const bool belowThreshold =
                minDetectionScore > 0.f && f.score < minDetectionScore;
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

        const QString scoreText = QString::number(f.score, 'f', 3);
        const QString label =
                belowThreshold
                        ? QStringLiteral("score %1 (below min)").arg(scoreText)
                        : QStringLiteral("score %1").arg(scoreText);
        const QFontMetrics fm(labelFont);
        constexpr int padH = 4;
        constexpr int padV = 3;
        const QRect textBounds = fm.boundingRect(QRect(0, 0, 10000, 10000),
                                                 Qt::TextSingleLine, label);
        const int labelW = textBounds.width() + 2 * padH;
        const int labelH = textBounds.height() + 2 * padV;
        float labelY = f.y1 - static_cast<float>(labelH) - 2.f;
        if (labelY < 0.f) labelY = f.y2 + 2.f;
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

QImage annotateAnalyze(const QImage& source,
                       const std::vector<FaceDetectBox>& faces,
                       float minDetectionScore) {
    QImage rgb = source.convertToFormat(QImage::Format_RGB32);
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

        QString label = QStringLiteral("score %1").arg(f.score, 0, 'f', 3);
        if (belowThreshold) {
            label += QStringLiteral("  (below min)");
        }
        if (f.age >= 0) {
            label += QStringLiteral("  age %1  %2")
                             .arg(f.age)
                             .arg(QChar(f.gender));
        }

        const QFontMetrics fm(labelFont);
        constexpr int padH = 4;
        constexpr int padV = 3;
        const QRect textBounds = fm.boundingRect(QRect(0, 0, 10000, 10000),
                                                 Qt::TextSingleLine, label);
        const int labelW = textBounds.width() + 2 * padH;
        const int labelH = textBounds.height() + 2 * padV;
        float labelY = f.y1 - static_cast<float>(labelH) - 2.0f;
        if (labelY < 0.0f) {
            labelY = f.y2 + 2.0f;
        }
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

QImage annotateLabeledFaces(const QImage& source,
                            const std::vector<AnnotatedFaceLabel>& faces) {
    QImage canvas = source.convertToFormat(QImage::Format_RGB888);
    if (canvas.isNull()) return {};
    QPainter painter(&canvas);
    painter.setRenderHint(QPainter::Antialiasing, true);
    QFont font = painter.font();
    font.setPointSize(10);
    font.setBold(true);
    painter.setFont(font);

    for (const AnnotatedFaceLabel& face : faces) {
        const QString text =
                face.label.isEmpty() ? QStringLiteral("?") : face.label;
        const QColor color =
                face.matched ? QColor(34, 197, 94) : QColor(239, 68, 68);

        if (face.labelOnly) {
            const QRect textBg =
                    painter.fontMetrics().boundingRect(text).adjusted(-8, -4, 8,
                                                                      4);
            QRect banner = textBg;
            banner.moveTopLeft(QPoint(8, 8));
            painter.fillRect(banner, color);
            painter.setPen(Qt::black);
            painter.drawText(banner, Qt::AlignCenter, text);
            continue;
        }

        const QRect rect(static_cast<int>(face.box.x1),
                         static_cast<int>(face.box.y1),
                         static_cast<int>(face.box.x2 - face.box.x1),
                         static_cast<int>(face.box.y2 - face.box.y1));
        QPen pen(color, 2);
        if (face.dashed) pen.setStyle(Qt::DashLine);
        painter.setPen(pen);
        painter.setBrush(Qt::NoBrush);
        painter.drawRect(rect);

        const QRect textBg =
                painter.fontMetrics().boundingRect(text).adjusted(-4, -2, 4, 2);
        QRect labelRect = textBg;
        labelRect.moveTopLeft(rect.topLeft() + QPoint(0, -textBg.height()));
        if (labelRect.top() < 0) {
            labelRect.moveTop(rect.bottom() + 2);
        }
        painter.fillRect(labelRect, color);
        painter.setPen(Qt::black);
        painter.drawText(labelRect, Qt::AlignCenter, text);
    }
    return canvas;
}

QImage annotateRecognize(const QImage& source,
                         const std::vector<FaceDetectBox>& faces,
                         const QVector<QString>& labels,
                         float minDetectionScore) {
    QImage rgb = source.convertToFormat(QImage::Format_RGB32);
    QPainter painter(&rgb);
    painter.setRenderHint(QPainter::Antialiasing, true);

    QFont labelFont = painter.font();
    labelFont.setPointSize(10);
    labelFont.setBold(true);
    painter.setFont(labelFont);

    for (size_t i = 0; i < faces.size(); ++i) {
        const FaceDetectBox& f = faces[i];
        const QString identity = (i < static_cast<size_t>(labels.size()) &&
                                  !labels[static_cast<int>(i)].isEmpty())
                                         ? labels[static_cast<int>(i)]
                                         : QStringLiteral("Unknown");
        const bool isUnknown = identity.startsWith(QStringLiteral("Unknown"),
                                                   Qt::CaseInsensitive) ||
                               identity.startsWith(QStringLiteral("NO MATCH"),
                                                   Qt::CaseInsensitive);
        const bool belowThreshold =
                minDetectionScore > 0.f && f.score < minDetectionScore;

        const QColor boxColor =
                isUnknown ? QColor(239, 68, 68)
                          : (belowThreshold ? QColor(251, 191, 36)
                                            : QColor(74, 222, 128));
        const QColor labelColor =
                isUnknown ? QColor(254, 202, 202) : QColor(220, 252, 231);

        painter.setPen(QPen(boxColor, 2));
        painter.setBrush(Qt::NoBrush);
        painter.drawRect(QRectF(f.x1, f.y1, f.x2 - f.x1, f.y2 - f.y1));

        painter.setPen(QPen(QColor(253, 224, 71), 1));
        painter.setBrush(QColor(253, 224, 71));
        for (int j = 0; j < 5; ++j) {
            painter.drawEllipse(QPointF(f.landmarks[j][0], f.landmarks[j][1]),
                                2.5, 2.5);
        }
        painter.setBrush(Qt::NoBrush);

        const QFontMetrics fm(labelFont);
        constexpr int padH = 5;
        constexpr int padV = 3;
        const QRect textBounds = fm.boundingRect(QRect(0, 0, 10000, 10000),
                                                 Qt::TextSingleLine, identity);
        const int labelW = textBounds.width() + 2 * padH;
        const int labelH = textBounds.height() + 2 * padV;
        float labelY = f.y1 - static_cast<float>(labelH) - 2.f;
        if (labelY < 0.f) labelY = f.y2 + 2.f;
        const QRectF textRect(f.x1, labelY, static_cast<qreal>(labelW),
                              static_cast<qreal>(labelH));
        painter.fillRect(textRect, QColor(15, 23, 42, 220));
        painter.setPen(labelColor);
        painter.drawText(textRect.adjusted(padH, padV, -padH, -padV),
                         Qt::AlignLeft | Qt::AlignTop | Qt::TextDontClip,
                         identity);
    }
    return rgb;
}

}  // namespace FaceDetectEmbed
