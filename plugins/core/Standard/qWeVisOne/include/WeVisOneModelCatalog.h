// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Read-only bridge to the WeVisOne-VL task model catalog exported by
// libAICore. The catalog is the single URL/digest owner; this header only
// adapts it to Qt types for the dialog combos and the batch command.

#pragma once

#include <QImage>
#include <QRectF>
#include <QSize>
#include <QString>
#include <QStringList>
#include <QVector>

class QJsonArray;

struct WeVisOneModelEntry {
    QString filename;
    QString downloadUrl;
    QString displayName;
    QString family;  // "2b" or "4b"
    int role = 0;    // 0 vlm, 1 mmproj, 2 decoder (aicore_wevisone_model_role)
    QString quantization;
    QString sha256;
    qlonglong sizeBytes = 0;
};

namespace WeVisOneHelpers {

/** Catalog rows (all roles) from aicore_wevisone_model_*; empty when
 *  AICore is not available. */
QVector<WeVisOneModelEntry> catalogModels();

/** Index of the "(recommended)" row the catalog declares. */
int catalogDefaultIndex();

/** Catalog row whose filename matches (false when not found). */
bool findModelByFilename(const QString& filename, WeVisOneModelEntry* out);

/** Local model cache directory (shared data root; empty when unavailable). */
QString modelCacheDir();

/** Distinct checkpoint families of the catalog in first-seen order
 *  ("2b", "4b") for the size combo. */
QStringList catalogFamilies();

/** JSON array of the catalog rows (batch command RESULT_JSON schema). */
QJsonArray catalogJson();

/** Parses a visual-prompt reference-box spec in the upstream CLI
 *  --ref-boxes format ("x1,y1,x2,y2;x1,y1,x2,y2" in original-image
 *  pixels) into a flat x1,y1,x2,y2 float array for
 *  aicore_wevisone_run_image. Empty text yields an empty vector;
 *  malformed quadruples set \p error and return an empty vector.
 *  Shared by the dialog worker and the headless batch command. */
QVector<float> parseRefBoxes(const QString& spec, QString* error);

/** Renders a structured model answer as human-readable multi-line text for
 *  the console log: one aligned line per <ref>label</ref> group (labels
 *  padded to the widest so coordinate columns line up), "label: [x1,y1,x2,y2],
 *  ..." (box) or "label: (x,y), ..." (point). When \p imageSize is valid the
 *  model's 0-1000 coordinate tokens are converted to original-image pixels
 *  (the same frame as the exported annotated image); otherwise the verbatim
 *  0-1000 values are shown. Free text outside the groups (e.g. the counting
 *  tail "Total number: 4.") is kept as its own line. Plain-text answers
 *  (ocr / freeform) pass through trimmed. The raw token string stays
 *  available via the typed result for downstream parsing; this is
 *  presentation only. */
QString formatAnswerForLog(const QString& answer,
                           const QSize& imageSize = QSize());

/** The model's stated counting total ("Total number: N.", upstream
 *  demo/app.py _COUNT_RE), or -1 when absent. For count_box/count_point this
 *  is the authoritative count; the parsed box list is localization evidence
 *  and may be shorter than the stated total. */
int statedTotalCount(const QString& answer);

/** One runnable demo sample read from the dataset's own manifest.json. */
struct WeVisOneDemoSample {
    QString image;     // member filename, relative to the extract dir
    QString query;     // manifest-verbatim query text
    QString refBoxes;  // upstream --ref-boxes spec, empty when none
};

/** Reads the recommended demo for \p task from the dataset's own
 *  manifest.json. Priority: (1) chess.jpg when the manifest offers it for
 *  the task — the classic chessboard case is the canonical multi-object
 *  demo for referring/count_box/count_point; (2) otherwise the first
 *  WeVisOne-specific case-study original (an image outside the seven
 *  shared official demo photos) so the first fill showcases WeVisOne's
 *  own data; (3) otherwise the task's first sample. Returns false when the
 *  manifest is absent/unreadable or the task has no sample (callers then
 *  use their built-in offline fallback). */
bool readDemoSample(const QString& manifestPath,
                    const QString& task,
                    WeVisOneDemoSample* out);

/** The eleven released task families (prompts.py), in the upstream
 *  README order. */
QStringList taskNames();

/** Point-marker radius for an image width, mirroring the official python
 *  overlay (wevisone_vl/visualization.py render_overlay:
 *  radius = max(6, round(width/85))). The C++ export helper draws
 *  degenerate point boxes as 3x3 rectangles only — invisible on large
 *  images — so the plugin overlay uses the python dot geometry. */
int pointMarkerRadius(int imageWidth);

/** White outline thickness of the point marker
 *  (visualization.py: max(2, round(radius/3))). */
int pointMarkerOutline(int radius);

/** Qt overlay of a run result on \p input: green (0,220,0) matte composite
 *  at a*0.55 (nearest-neighbor matte->canvas mapping, 0.01 threshold) and
 *  red width-3 box strokes, pixel-identical to the task's render_result_png
 *  export helper; degenerate [x,y,x,y] point boxes (the count_point / point
 *  grounding payload) become visible filled dots with a white outline ring
 *  sized by pointMarkerRadius/pointMarkerOutline. \p matte may be empty
 *  (no masks). Direct RGB888 blending keeps round(p*(1-a)+c*a) exact —
 *  QPainter anti-aliasing would diverge from the upstream visualization. */
QImage annotateResult(const QImage& input,
                      const QVector<QRectF>& boxes,
                      const QVector<float>& matte,
                      int matteWidth,
                      int matteHeight);

}  // namespace WeVisOneHelpers