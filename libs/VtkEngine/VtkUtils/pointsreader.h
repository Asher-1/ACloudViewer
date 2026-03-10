// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/// @file pointsreader.h
/// @brief Reads 3D points from file in background thread.

#include <QRunnable>

#include "point3f.h"
#include "qVTK.h"
#include "signalledrunable.h"
#include "utils.h"

namespace VtkUtils {

/// @class PointsReader
/// @brief Reads point cloud from file; runs as SignalledRunnable.
class QVTK_ENGINE_LIB_API PointsReader : public SignalledRunnable {
    Q_OBJECT
public:
    /// @param file Path to points file
    explicit PointsReader(const QString& file);

    void run();

    /// @return Loaded points (after run completes)
    const QList<Point3F>& points() const;

private:
    QString m_file;
    QList<Point3F> m_points;
};

}  // namespace VtkUtils
