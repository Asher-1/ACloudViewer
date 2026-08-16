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

/** Result envelope of one RMBG-2.0 background removal. */
struct RMBGRunResult {
    QString imagePath;
    QString imageName;
    /** RGBA image with the background removed (transparent pixels). */
    QImage resultImage;
    double runtimeMs = 0.0;
    /** Mean alpha over the image (0..1); 1.0 = fully opaque. */
    double alphaMean = 0.0;
    /** Fraction of pixels with alpha >= 128 (foreground coverage). */
    double foregroundRatio = 0.0;
    QString modelVariant;
    int inputSize = 0;
    QString backend;
    QString resolvedDevice;
    QString modelPath;
    QByteArray infoJson;
};

Q_DECLARE_METATYPE(RMBGRunResult)

/** Catalog entry mirroring aicore_rmbg_model_entry. */
struct RMBGModelEntry {
    QString filename;
    QString downloadUrl;
    QString displayName;
    QString quantNote;
    QString licenseNote;
};

namespace RMBGHelpers {

/** Enumerate the published catalog from AICore. */
QVector<RMBGModelEntry> catalogModels();
/** Lookup by GGUF filename; returns false when unknown. */
bool findModelByFilename(const QString& filename, RMBGModelEntry* out);

/** Model cache directory for qRMBG (aicore_rmbg_model_cache_dir). */
QString modelCacheDir();

/** Parse the AICore RMBG info JSON into a run result. Returns true on
 *  success. */
bool parseInfoJson(const QByteArray& json, RMBGRunResult* out);

/** Alpha statistics over the RGBA result image. Pure pixel logic — unit
 *  tested without AICore. */
void computeAlphaStats(const QImage& rgba, double* alphaMean,
                       double* foregroundRatio);

/** Render a checkerboard pattern of the given size (transparent preview
 *  background). */
QImage makeCheckerboard(const QSize& size, int cellSize = 8);

/** Composite the RGBA result over a checkerboard background (preview /
 *  thumbnail rendering). Returns a fully opaque image of the same size. */
QImage compositeOnCheckerboard(const QImage& rgba, int cellSize = 8);

/** Format the alpha stats for a status line ("alpha 82.3%, fg 45.6%"). */
QString formatAlphaStats(double alphaMean, double foregroundRatio);

}  // namespace RMBGHelpers
