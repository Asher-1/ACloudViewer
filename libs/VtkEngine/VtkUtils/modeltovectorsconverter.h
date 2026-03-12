// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/// @file modeltovectorsconverter.h
/// @brief Converts TableModel to Vector4F list in background thread.

#include "signalledrunable.h"
#include "utils.h"
#include "vector4f.h"

namespace VtkUtils {
class TableModel;
/// @class ModelToVectorsConverter
/// @brief Extracts Vector4F rows from TableModel; runs as SignalledRunnable.
class QVTK_ENGINE_LIB_API ModelToVectorsConverter : public SignalledRunnable {
    Q_OBJECT
public:
    /// @param model Source table model
    ModelToVectorsConverter(TableModel* model);

    void run();

    /// @return Extracted vectors (after run completes)
    QList<Vector4F> vectors() const;

private:
    QList<Vector4F> m_vectors;
    TableModel* m_model = nullptr;
};

}  // namespace VtkUtils
