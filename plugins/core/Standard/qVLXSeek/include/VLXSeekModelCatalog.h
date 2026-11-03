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

}  // namespace VLXSeekHelpers
