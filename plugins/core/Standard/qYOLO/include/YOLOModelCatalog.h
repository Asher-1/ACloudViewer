// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QByteArray>
#include <QImage>
#include <QString>
#include <QVector>
#include <cstdint>

/** Single parsed detection (from the AICore YOLO JSON envelope). */
struct YOLODetection {
    uint32_t classId = 0;
    QString className;
    float score = 0.0f;
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
};

/** Depth statistics parsed from aicore_yolo_last_depth_json. */
struct YOLODepthStats {
    int width = 0;          // depth map (= original image) width
    int height = 0;         // depth map (= original image) height
    double minDepth = 0.0;  // meters, over valid pixels
    double maxDepth = 0.0;
    double meanDepth = 0.0;
    double p95Depth = 0.0;  // robust far bound for colorization
    long long validPixels = 0;
};

/** Result envelope of one YOLO detect inference. */
struct YOLORunResult {
    QString imagePath;
    QString imageName;
    QImage annotatedImage;
    QVector<YOLODetection> detections;
    double runtimeMs = 0.0;
    int totalDetected = 0;
    QString modelVariant;
    int imageSize = 0;
    int numClasses = 0;
    bool end2end = false;
    QString resolvedDevice;
    QString modelPath;
    QByteArray resultJson;
};

/** Result envelope of one YOLO depth inference (typed float map + stats). */
struct YOLODepthResult {
    QString imagePath;
    QString imageName;
    QImage annotatedImage;    // turbo colorized depth
    QVector<float> depthMap;  // row-major, width * height floats (meters)
    int width = 0;
    int height = 0;
    YOLODepthStats stats;
    double runtimeMs = 0.0;
    QString modelVariant;
    int imageSize = 0;
    QString resolvedDevice;
    QString modelPath;
    QByteArray resultJson;  // last_depth_json stats envelope
};

Q_DECLARE_METATYPE(YOLORunResult)
Q_DECLARE_METATYPE(YOLODepthResult)

/** Catalog entry mirroring aicore_yolo_model_entry. */
struct YOLOModelEntry {
    QString filename;
    QString downloadUrl;
    QString displayName;
    QString quantNote;
    QString licenseNote;
    bool depthCapable = false;
    bool end2end = false;
};

namespace YOLOHelpers {

/** Enumerate the published catalog from AICore. */
QVector<YOLOModelEntry> catalogModels();
/** All detection-capable catalog entries (depth_capable == false). */
QVector<YOLOModelEntry> detectionModels();
/** All depth-capable catalog entries. */
QVector<YOLOModelEntry> depthModels();
/** Lookup by GGUF filename; returns false when unknown. */
bool findModelByFilename(const QString& filename, YOLOModelEntry* out);

/** Model cache directory for qYOLO (aicore_yolo_model_cache_dir). */
QString modelCacheDir();

/** Build the user-facing catalog label without duplicating a quantization
 *  note that is already part of displayName. */
QString modelDisplayLabel(const YOLOModelEntry& entry);

/** Return tightly packed RGB888 pixels for AICore's stride-less C API.
 *  scratch owns the returned bytes only when QImage row padding is present. */
const uchar* packedRgb888Data(const QImage& image, QByteArray* scratch);

/** Parse the AICore YOLO detect JSON envelope into a run result. Returns
 *  true on success; the detections array may be empty (no objects). */
bool parseDetectionsJson(const QByteArray& json, YOLORunResult* out);

/** Parse the aicore_yolo_last_depth_json statistics envelope. */
bool parseDepthStatsJson(const QByteArray& json, YOLODepthStats* out);

/** Draw bounding boxes + class/score labels onto the image. Pure pixel
 *  logic — unit tested without AICore. */
void drawDetections(QImage* image,
                    const QVector<YOLODetection>& detections,
                    int thickness = 3);

/** Turbo-style colorization of a metric depth map (near = blue, far = red;
 *  same mapping as drawDepthLegend). When minDepth >= maxDepth the valid
 *  (finite, > 0) pixel range is computed automatically (min .. p95).
 *  Invalid pixels render black. Returns a null image on invalid input. */
QImage depthColorImage(const float* depth,
                       int width,
                       int height,
                       double minDepth = 0.0,
                       double maxDepth = 0.0);

/** Colorbar legend (top-right corner) with min/max labels in meters. */
void drawDepthLegend(QImage* image, double minDepth, double maxDepth);

/** Deterministic per-class palette (20 colors, COCO-consistent). */
QRgb classColor(uint32_t classId);

/** True when the model filename looks like a depth variant. */
bool filenameIsDepth(const QString& filename);

}  // namespace YOLOHelpers
