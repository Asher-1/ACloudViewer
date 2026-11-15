// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Read-only bridge to the VLX-Seek task model catalog exported by libAICore.
// The catalog is the single URL/digest owner; this header only adapts it to
// Qt types for the dialog combos and the batch command.

#pragma once

#include <QString>
#include <QStringList>
#include <QVector>

class QJsonArray;

struct VLXSeekModelEntry {
    QString filename;
    QString downloadUrl;
    QString displayName;
    QString family;
    int role = 0;  // 0 wedetect, 1 vision, 2 llm (aicore_vlx_seek_model_role)
    QString quantization;
    QString sha256;
    qlonglong sizeBytes = 0;
};

namespace VLXSeekHelpers {

/** Catalog rows (all roles) from aicore_vlx_seek_model_*; empty when
 *  AICore is not available. */
QVector<VLXSeekModelEntry> catalogModels();

/** Index of the "(recommended)" row the catalog declares. */
int catalogDefaultIndex();

/** Catalog row whose filename matches (false when not found). */
bool findModelByFilename(const QString& filename, VLXSeekModelEntry* out);

/** Local model cache directory (shared data root; empty when unavailable). */
QString modelCacheDir();

/** Rows of one role as display labels for a combo (keeps identities). */
QStringList roleDisplayLabels(int role);

/** JSON array of the catalog rows (batch command RESULT_JSON schema). */
QJsonArray catalogJson();

/** Parse a comma-separated zero-based proposal-index list ("2, 3") for the
 *  brief_region_caption / region_ocr tasks. Empty text yields an empty
 *  vector; malformed or negative entries set \p error and return an empty
 *  vector. Shared by the dialog worker and the headless batch command. */
QVector<int32_t> parseTargetIndexes(const QString& text, QString* error);

/** Parses an external proposal-boxes JSON file (upstream --bbox-list
 *  format: [[x1,y1,x2,y2],...] in original-image pixels, <= 100 boxes)
 *  into a flat x1,y1,x2,y2 float array for aicore_vlx_seek_run_image.
 *  Returns an empty vector with \p error set on any failure. */
QVector<float> parseExternalBoxes(const QString& path, QString* error);

/** Stable download URL of an OCR catalog asset, resolved through the public
 *  AICore OCR catalog API (empty when the asset is unknown). */
QString ocrModelDownloadUrl(const QString& filename);

/** Resolved PP-OCR det/rec assets for the region_ocr OCR text-detector
 *  proposer. The preset pairing comes from the public OCR preset API and the
 *  cache directory from the OCR catalog; filenames are pinned-quantization
 *  release assets (empty fields = the public API was unavailable). */
struct OcrProposerModels {
    QString detFileName;
    QString recFileName;
    QString cacheDir;
};
OcrProposerModels ocrProposerModels();

/** Converts OCR text-detector quads (\p quadCount * [4][2] floats, row
 *  major) into the axis-aligned [x1,y1,x2,y2] proposal boxes the VLX-Seek
 *  runtime consumes: degenerate quads are skipped and the list is capped at
 *  the 100-proposal C API contract (the built-in WeDetect-Uni proposer
 *  supplies exactly 100). Returns false when no usable box remains. */
bool ocrQuadsToBoxes(const float* quads, int quadCount, QVector<float>* boxes);

/** Strips the VLX-Seek internal XML tags for human-readable display. The
 *  raw output uses <ground>text</ground><objects><objN>...</objects> groups
 *  (one group per grounded text — multi-region answers carry several),
 *  optional <think>...</think> reasoning and an <answer>...</answer> block.
 *  Every <objects> group is replaced by its own [objN, ...] list so the
 *  interleaved ground texts survive; the answer's objN numbers match the
 *  proposal boxes (target-index numbering). With \p keepReasoning the
 *  <think> block is preserved before the answer content (console lines);
 *  otherwise it is dropped (compact status lines). */
QString stripAnswerTags(const QString& answer, bool keepReasoning = false);

}  // namespace VLXSeekHelpers
