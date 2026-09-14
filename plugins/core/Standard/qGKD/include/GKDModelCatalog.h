// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QByteArray>
#include <QColor>
#include <QImage>
#include <QString>
#include <QStringList>
#include <QVector>
#include <cstdint>

#include "aicore/image_view.h"

/** One detected keypoint (source-image pixels + heatmap score). */
struct GKDKeypoint {
    float x = 0.0f;
    float y = 0.0f;
    float score = 0.0f;
    QString prompt; /**< text prompt when one drove this row (else empty) */
};
Q_DECLARE_METATYPE(GKDKeypoint)

/** Keypoints of one object/ROI. */
struct GKDKeypointSet {
    QString label; /**< object class (multi-object) or prompt source label */
    bool hasBox = false;
    float x1 = 0.0f, y1 = 0.0f, x2 = 0.0f, y2 = 0.0f;
    QVector<GKDKeypoint> keypoints;
};
Q_DECLARE_METATYPE(GKDKeypointSet)

/** Result envelope of one qGKD run. */
struct GKDRunResult {
    QString imagePath;
    QString imageName;
    /** Input image with the keypoints drawn. */
    QImage renderedImage;
    QVector<GKDKeypointSet> sets;
    int totalKeypoints = 0;
    /** Kept above the display threshold while rendering. */
    int shownKeypoints = 0;
    double runtimeMs = 0.0;
    double preprocessMs = 0.0;
    double postprocessMs = 0.0;
    double totalRuntimeMs = 0.0;
    /** "text" | "visual" | "multimodal" | "multi-object" */
    QString mode;
    QString backend;
    QString resolvedDevice;
    QString modelPath;
    QByteArray infoJson;
};
Q_DECLARE_METATYPE(GKDRunResult)

/** Catalog entry mirroring aicore_gkd_model_entry. */
struct GKDModelEntry {
    QString filename;
    QString downloadUrl;
    QString displayName;
    QString quantNote;
    QString licenseNote;
    qint64 sizeBytes = 0;
};

/** DB-tree image entry offered by the dialog picker. */
struct GKDImageEntry {
    QString name;
    QImage preview;
};

namespace GKDHelpers {

/** Enumerate the published GKD catalog from AICore. */
QVector<GKDModelEntry> catalogModels();
/** Index of the catalog-declared default row; -1 when AICore is off. */
int catalogDefaultIndex();
/** Lookup by GGUF filename; returns false when unknown. */
bool findModelByFilename(const QString& filename, GKDModelEntry* out);

/** Model cache directory for qGKD (aicore_gkd_model_cache_dir). */
QString modelCacheDir();

/** Build the user-facing catalog label without duplicating a quantization
 *  note that is already part of displayName. */
QString modelDisplayLabel(const GKDModelEntry& entry);

/** Build a borrowed stride-aware AICore view. Unsupported Qt formats are
 *  converted once in-place to RGBA8888; common camera formats stay zero-copy.
 */
bool imageView(QImage* image, aicore_image_view* out);

/** Parse "x1,y1 x2,y2 ..." into a coordinate list. Returns false when the
 *  text is non-empty but malformed, or holds an odd number of coordinates.
 *  An empty/whitespace text yields an empty list and true. Pure text logic —
 *  unit tested without AICore. */
bool parseCoordinatePairs(const QString& text, QVector<QPointF>* out);

/** Split a prompt line into non-empty trimmed prompts (comma separated;
 *  quoted segments keep their commas). Pure text logic — unit tested
 *  without AICore. */
QStringList splitPrompts(const QString& text);

/** Deterministic display color for an object/keypoint group id. */
QColor groupColor(int groupId);

/** Draw keypoint sets (dots + prompt labels + optional boxes) over a copy of
 *  the source image. Keypoints with score < minScore are skipped. Pure pixel
 *  logic — unit tested without AICore. */
QImage renderResult(const QImage& source,
                    const QVector<GKDKeypointSet>& sets,
                    float minScore);

}  // namespace GKDHelpers
