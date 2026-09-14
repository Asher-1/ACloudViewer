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
    /** Detector confidence of the source box (multi-object); 1.0 for
     *  single-object ROIs, matching the official COCO bbox_scores. */
    float bboxScore = 1.0f;
    QVector<GKDKeypoint> keypoints;
    /** Official-demo skeleton bones as flat source-image segments
     *  (x1,y1,x2,y2 per bone); empty when the scenario has no official
     *  skeleton or a bone endpoint fell below the display threshold. */
    QVector<QPointF> bones;
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
    /** Prompt texts and preset skeleton of the run (COCO export). */
    QStringList kpsTexts;
    QString skeleton;
    /** Source image size (COCO images entry). */
    int imageWidth = 0;
    int imageHeight = 0;
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

/** One dialog mode's official-demo "Use test data" preset: the
 *  scenario-matched sample image(s) plus the exact prompt values of the
 *  upstream General-Keypoint-Detection README demos, so one click
 *  reproduces a published result. Empty fields clear the corresponding
 *  widget (a fill fully defines the mode's prompt state). */
struct GKDModePreset {
    QString queryImage;    /**< sample query-image file name */
    QString kpsTexts;      /**< keypoint texts (empty = clear) */
    QString supportImage;  /**< 1-shot support-image file name (empty = none) */
    QString supportKps;    /**< "x1,y1 x2,y2 …" (empty = clear) */
    QString objectClasses; /**< multi-object class list (empty = clear) */
    /** Multi-object scene confidence cut applied on fill (<= 0 = leave the
     *  detector spin untouched). Dense/small-target scenes measure better
     *  recall at lower cuts (fish school 25 -> 46 boxes at 0.10, all real
     *  fish on visual check), while closed-vocabulary scenes keep the
     *  global 0.25 default. */
    float yoloConf = -1.0f;
    /** Official-demo skeleton as 1-based keypoint-index pairs
     *  ("1-2 1-3 ...", from the upstream predefined_keypoints.py
     *  schemas / README --skeleton flags; empty = render points only).
     *  Only scenarios with an official skeleton carry one. */
    QString skeleton;
};

namespace GKDHelpers {

/** Enumerate the published GKD catalog from AICore. */
QVector<GKDModelEntry> catalogModels();
/** Index of the catalog-declared default row; -1 when AICore is off. */
int catalogDefaultIndex();
/** Lookup by GGUF filename; returns false when unknown. */
bool findModelByFilename(const QString& filename, GKDModelEntry* out);

/** The YOLO-World detector family of the existing yolo task catalog
 *  (multi-object box stage). The default index prefers yolov8l-world:
 *  measured duplicate-free with calibrated scores where s-world emits
 *  near-tie duplicate boxes once the confidence cut drops. */
QVector<GKDModelEntry> yoloWorldModels();
int yoloWorldDefaultIndex();
/** The text-encoder towers (CLIP / MobileCLIP) that encode the
 *  open-vocabulary class names of a YOLO-World detector. */
QVector<GKDModelEntry> yoloTextModels();
int yoloTextDefaultIndex();

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

/** Parse a skeleton string "1-2 1-3 ..." into 1-based index pairs.
 * Malformed tokens are skipped; an empty/whitespace text yields an
 * empty list. Pure text logic — unit tested without AICore. */
QVector<QPair<int, int>> parseSkeleton(const QString& text);

/** Resolve skeleton pairs into flat source-image segments (x1,y1,x2,y2
 *  per bone, appended in order) over the full keypoint list. A bone is
 *  drawn only when both endpoints score >= \p minScore: by prompt name
 *  when \p prompts is non-empty (index i refers to prompts[i-1]),
 *  otherwise by result index (the visual-prompt mode has no texts, so
 *  order is the support-point order). Pure logic — unit tested without
 *  AICore. */
QVector<QPointF> buildBones(const QStringList& prompts,
                            const QVector<GKDKeypoint>& keypoints,
                            const QVector<QPair<int, int>>& skeleton,
                            float minScore);

/** Prompt-mode ids of the qGKD dialog, in stack-page order ("text",
 *  "visual", "multimodal", "multi"). Pure string logic — unit tested
 *  without AICore. */
QStringList promptModes();

/** True when the mode's panel shows the YOLO-World detector rows
 *  (multi-object composition). */
bool modeUsesYolo(const QString& mode);

/** The official-demo preset of one mode (values pinned from the upstream
 *  General-Keypoint-Detection README demos). Pure data logic — unit
 *  tested without AICore. */
GKDModePreset modePreset(const QString& mode);

/** All scenario presets of one mode, in "Try sample data" rotation order.
 *  Single-prompt modes return one entry; the multi-object mode returns
 *  every bundled multi-target scene (alpaca herd / bronze statues / pigs
 *  / fish school), so successive clicks walk the whole dataset. Pure
 *  data logic — unit tested without AICore. */
QVector<GKDModePreset> modePresets(const QString& mode);

/** Deterministic display color for an object/keypoint group id. */
QColor groupColor(int groupId);

/** Greedy label de-overlap: places each rect top-to-bottom, pushing a
 *  rect down (2 px gutter) until it no longer intersects an already
 *  placed one. \p canvas bounds the cascade: a label is clamped into
 *  the canvas up front and pinned to the bottom edge if the push-down
 *  would run it out of the image (staying inside the canvas, possibly
 *  overlapping, beats being cropped away). Order-preserving,
 *  deterministic; O(n²) worst case which is fine for the ≤ ~150 labels
 *  a run can show. Pure geometry — unit tested without AICore. */
QVector<QRect> placeLabels(const QVector<QRect>& preferred,
                           const QSize& canvas);

/** Draw keypoint sets (dots + optional boxes + skeleton bones + optional
 *  per-point prompt labels) over a copy of the source image. Keypoints
 *  with score < minScore are skipped. When \p pointLabels is true each
 *  shown keypoint carries its "prompt score" label with official-style
 *  text sizing (scaled to the image width) and a capacity budget: on
 *  crowded scenes the font shrinks and the lowest-scoring labels are
 *  dropped rather than covering the scene; labels flip to the left of
 *  their dot near the right edge and never leave the canvas, and
 *  moved labels get a thin leader arrow. Pure pixel logic — unit
 *  tested without AICore. */
QImage renderResult(const QImage& source,
                    const QVector<GKDKeypointSet>& sets,
                    float minScore,
                    bool pointLabels = false);

/** Serialize one run as an official-style COCO prediction JSON (categories
 *  with keypoint names + 1-based skeleton, images, annotations with xywh
 *  boxes, detector scores, and x,y,score triplets). Keypoints below the
 *  display threshold are absent from the source sets and therefore from the
 *  export. Pure logic — unit tested without AICore. */
QString buildCocoJson(const GKDRunResult& result);

}  // namespace GKDHelpers
