// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/// @file pointstomodelconverter.h
/// @brief Converts Point3F list to TableModel in background thread.

#include <QObject>
#include <QRunnable>

#include "point3f.h"
#include "qVTK.h"
#include "signalledrunable.h"
#include "utils.h"

namespace VtkUtils {

class TableModel;
/// @class PointsToModelConverter
/// @brief Populates TableModel from points; runs as SignalledRunnable.
class QVTK_ENGINE_LIB_API PointsToModelConverter : public SignalledRunnable {
    Q_OBJECT
public:
    /// @param points Source points
    /// @param model Target table model
    PointsToModelConverter(const QList<Point3F>& points, TableModel* model);

    void run();

private:
    QList<Point3F> m_points;
    TableModel* m_model = nullptr;
};

}  // namespace VtkUtils
