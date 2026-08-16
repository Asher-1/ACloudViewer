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
#include <QStringList>
#include <QVector>

#include <cstdint>

/** Single parsed detection (from the AICore RF-DETR JSON envelope). */
struct RFDetrDetection {
    uint32_t classId = 0;
    QString className;
    float score = 0.0f;
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
    /** PNG-encoded binary mask (empty for detection-only models). */
    QByteArray maskPng;
};

/** Result envelope of one RF-DETR inference. */
struct RFDetrRunResult {
    QString imagePath;
    QString imageName;
    QImage annotatedImage;
    QVector<RFDetrDetection> detections;
    double runtimeMs = 0.0;
    int totalDetected = 0;
    QString modelVariant;
    int imageSize = 0;
    int numClasses = 0;
    bool segmentation = false;
    QString resolvedDevice;
    QString modelPath;
    QByteArray resultJson;
};

Q_DECLARE_METATYPE(RFDetrRunResult)

/** Catalog entry mirroring aicore_rfdetr_model_entry. */
struct RFDetrModelEntry {
    QString filename;
    QString downloadUrl;
    QString displayName;
    QString quantNote;
    QString licenseNote;
    bool segmentationCapable = false;
};

namespace RFDetrHelpers {

/** Enumerate the published catalog from AICore. */
QVector<RFDetrModelEntry> catalogModels();
/** All detection-capable catalog entries (segmentation == false). */
QVector<RFDetrModelEntry> detectionModels();
/** All segmentation-capable catalog entries. */
QVector<RFDetrModelEntry> segmentationModels();
/** Lookup by GGUF filename; returns false when unknown. */
bool findModelByFilename(const QString& filename, RFDetrModelEntry* out);

/** Model cache directory for qRFDetr (aicore_rfdetr_model_cache_dir). */
QString modelCacheDir();

/** Parse the AICore RF-DETR JSON envelope into a run result. Returns true on
 *  success; the envelope's detections array may be empty (no objects). */
bool parseDetectionsJson(const QByteArray& json, RFDetrRunResult* out);

/** Draw bounding boxes + class/score labels (and the per-detection mask tint
 *  when the detection carries one) onto the image. Pure pixel logic — unit
 *  tested without AICore. */
void drawDetections(QImage* image, const QVector<RFDetrDetection>& detections,
                    float maskAlpha = 0.35f, int thickness = 3);

/** Deterministic per-class palette (20 colors, BGR-friendly). */
QRgb classColor(uint32_t classId);

/** True when the model filename looks like a segmentation variant. */
bool filenameIsSegmentation(const QString& filename);

}  // namespace RFDetrHelpers
