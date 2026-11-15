// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QColor>
#include <QMap>
#include <QString>
#include <QVector>
#include <functional>

#include "PointCloudDocument.h"

struct PointCloudDatasetExportOptions {
    QString outputPath;
    QString format;
    QVector<PointCloudFrame> frames;
    QMap<int, QString> classNames;
    QMap<int, QColor> classColors;
    bool includePointCloud = true;
};

// Writes one of the upstream ZIP contracts: datumaro, kitti_raw, or
// sly_pointcloud. The output is staged in a sibling temporary file and is
// replaced only after the complete archive and cancellation checks succeed.
bool exportPointCloudDataset(const PointCloudDatasetExportOptions& options,
                             const std::function<bool(int, int)>& progress = {},
                             QString* error = nullptr);
