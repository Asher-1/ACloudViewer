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

/** One typed detection (shared by detect/segment/pose accessors). */
struct YOLODetection {
    uint32_t classId = 0;
    QString className;
    float score = 0.0f;
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
};

/** COCO-17 keypoint in source-image pixels. */
struct YOLOKeypoint {
    float x = 0.0f;
    float y = 0.0f;
    float visibility = 1.0f;
};

/** One pose detection: box + decoded keypoints. */
struct YOLOKeypointSet {
    YOLODetection det;
    QVector<YOLOKeypoint> kpts;
};

/** Oriented box in source-image pixels (angle radians, unrotated w/h). */
struct YOLOObbBox {
    float cx = 0.0f;
    float cy = 0.0f;
    float w = 0.0f;
    float h = 0.0f;
    float angle = 0.0f;  // radians
    float score = 0.0f;
    uint32_t classId = 0;
    QString className;
};

/** One classification entry (softmax probability). */
struct YOLOClassProb {
    uint32_t classId = 0;
    QString className;
    float prob = 0.0f;
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

/** Binary instance mask of one segmented detection (source-image pixels,
 *  full image size, one byte per pixel: 0 = background, 1 = foreground).
 *  AICore remaps masks from the letterbox canvas to the original image
 *  space, so they align 1:1 with the detection boxes. */
struct YOLOSegMask {
    QByteArray bits;  // w * h bytes
    int w = 0;
    int h = 0;
};

/** Result envelope of one YOLO detect inference. */
struct YOLORunResult {
    QString imagePath;
    QString imageName;
    QImage annotatedImage;
    QVector<YOLODetection> detections;
    QVector<YOLOSegMask> masks;  // valid when the model task is "segment"
    QString task;                // "detect" | "segment" | "pose" | "obb" |
                                 // "semantic" | "classify" (world/yoloe map
                                 // to detect/segment)
    double runtimeMs = 0.0;
    int totalDetected = 0;
    QString modelVariant;
    int imageSize = 0;
    int numClasses = 0;
    bool end2end = false;
    QString resolvedDevice;
    QString modelPath;
    QByteArray resultJson;

    // ---- pose results (task == "pose") ----
    QVector<YOLOKeypointSet> keypointSets;
    int kptCount = 0;  // keypoints per set (17 for the COCO models)
    // ---- obb results (task == "obb") ----
    QVector<YOLOObbBox> obbBoxes;
    // ---- classify results (task == "classify") ----
    QVector<YOLOClassProb> classifications;  // full softmax table
    // ---- semantic results (task == "semantic") ----
    QByteArray semanticClassMap;  // width*height bytes, one class id/pixel
    int semanticWidth = 0;
    int semanticHeight = 0;
    int semanticNumClasses = 0;
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
    // GGUF task: "detect" | "segment" | "depth" | "pose" | "obb" |
    // "semantic" | "classify" | "text". The model combo of each task tab is
    // filtered on this field (plus textInput for the open-vocabulary
    // families), so a detect tab never offers a segment model and the
    // world/yoloe tabs never offer closed-set models.
    QString task;
    bool depthCapable = false;
    bool end2end = false;
    bool textInput = false;  // YOLO-World / YOLOE / text towers
};

namespace YOLOHelpers {

/** Enumerate the published catalog from AICore. */
QVector<YOLOModelEntry> catalogModels();
/** All pure object-detection catalog entries (closed-set, task ==
 *  "detect"). */
QVector<YOLOModelEntry> detectionModels();
/** All instance-segmentation catalog entries (closed-set, task ==
 *  "segment"). */
QVector<YOLOModelEntry> segmentModels();
/** All metric-depth catalog entries (task == "depth"). */
QVector<YOLOModelEntry> depthModels();
/** All keypoint-pose entries (task == "pose"). */
QVector<YOLOModelEntry> poseModels();
/** All oriented-box entries (task == "obb"). */
QVector<YOLOModelEntry> obbModels();
/** All classification entries (task == "classify"). */
QVector<YOLOModelEntry> classifyModels();
/** All semantic-segmentation entries (task == "semantic"). */
QVector<YOLOModelEntry> semanticModels();
/** YOLO-World open-vocabulary detectors (CLIP text tower). */
QVector<YOLOModelEntry> worldModels();
/** YOLOE open-vocabulary segmenters (MobileCLIP text tower). */
QVector<YOLOModelEntry> yoloeModels();
/** Text-encoder towers (CLIP ViT-B/32, MobileCLIP2-B). */
QVector<YOLOModelEntry> textModels();
/** Filter the full catalog on a tab task id ("detect"|"segment"|"depth"|
 *  "pose"|"obb"|"classify"|"semantic"|"world"|"yoloe"|"text"). */
QVector<YOLOModelEntry> taskModels(const QString& task);
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

/** Draw instance masks as a translucent per-class tint over the image,
 *  then the detection boxes/labels on top. masks and detections are
 *  index-aligned (mask i belongs to detection i). */
void drawSegmentation(QImage* image,
                      const QVector<YOLOSegMask>& masks,
                      const QVector<YOLODetection>& detections,
                      int thickness = 2);

/** Draw pose results: COCO-17 skeleton lines between visible keypoints,
 *  keypoint dots and the box/label (keypoints in source pixels).
 *  keypointSets may be empty (draws nothing). */
void drawPose(QImage* image,
              const QVector<YOLOKeypointSet>& keypointSets,
              int thickness = 2);

/** Draw oriented boxes as rotated rectangles with a center mark and label
 *  (coordinates in source pixels). */
void drawObb(QImage* image,
             const QVector<YOLOObbBox>& boxes,
             int thickness = 2);

/** Blend the semantic class map (width*height bytes, one class id per
 *  source pixel) over the image at 50% alpha using the Cityscapes-19
 *  palette (deterministic fallback beyond 19 classes). */
void drawSemantic(QImage* image,
                  const QByteArray& classMap,
                  int width,
                  int height,
                  int numClasses);

/** Draw a top-k classification banner (top-left). */
void drawClassifications(QImage* image,
                         const QVector<YOLOClassProb>& classifications,
                         int topK = 5);

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
