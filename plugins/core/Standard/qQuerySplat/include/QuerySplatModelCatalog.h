// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
#pragma once

#include <QString>
#include <QStringList>
#include <QVector>

/** Catalog entry mirroring aicore_querysplat_model_entry. */
struct QuerySplatModelEntry {
    QString filename;
    QString downloadUrl;
    QString displayName;
    QString role;  // "decoder" | "lpips"
    QString quantNote;
    QString licenseNote;
    QString variant;
    qint64 sizeBytes = 0;
};

namespace QuerySplatHelpers {

/** Enumerate the published catalog from AICore (decoder f32/f16/q8_0 +
 *  the LPIPS TTO metric weights). */
QVector<QuerySplatModelEntry> catalogModels();

/** Published decoder rows in catalog order (q8_0 first = the default). */
QVector<QuerySplatModelEntry> decoderModels();

/** Index (within catalogModels()) of the default decoder row. */
int defaultDecoderIndex();

/** Shared decoder cache dir (aicore_querysplat_model_resolve_cache_path
 *  root; CLOUDVIEWER_DATA_ROOT aware). */
QString decoderCacheDir();

/** Resolved encoder path: the map_anything shared VGGT-Omega 1B/512 q8_0
 *  (single source of truth for the encoder GGUFs). Empty when the shared
 *  catalog is unavailable. */
QString encoderDefaultPath();

QString findModelByFilename(const QString& filename, QuerySplatModelEntry* out);

}  // namespace QuerySplatHelpers
