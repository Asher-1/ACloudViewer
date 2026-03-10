// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/// @file modeltopointsconverter.h
/// @brief Converts TableModel to Point3F list and vertices in background
/// thread.

#include <QList>
#include <QVector>

#include "point3f.h"
#include "signalledrunable.h"
#include "tablemodel.h"
#include "utils.h"

namespace VtkUtils {

/// @class ModelToPointsConverter
/// @brief Extracts points and triangle vertices from TableModel; runs as
/// SignalledRunnable.
class QVTK_ENGINE_LIB_API ModelToPointsConverter : public SignalledRunnable {
    Q_OBJECT
public:
    /// @param model Source table model
    explicit ModelToPointsConverter(TableModel* model);

    /// @return Extracted 3D points (after run completes)
    QList<Point3F> points() const;
    /// @return Triangle vertex indices (after run completes)
    QVector<Tuple3ui> vertices() const;

    void run();

private:
    TableModel* m_model = nullptr;
    QList<Point3F> m_points;
};

}  // namespace VtkUtils
