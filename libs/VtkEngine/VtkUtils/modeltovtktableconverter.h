// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/// @file modeltovtktableconverter.h
/// @brief Converts TableModel to vtkTable in background thread.

#include "qVTK.h"
#include "signalledrunable.h"

class vtkTable;
namespace VtkUtils {
class TableModel;
/// @class ModelToVtkTableConverter
/// @brief Converts TableModel data to vtkTable; runs as SignalledRunnable.
class QVTK_ENGINE_LIB_API ModelToVtkTableConverter : public SignalledRunnable {
    Q_OBJECT
public:
    /// @param model Source table model
    explicit ModelToVtkTableConverter(TableModel* model);

    /// @param labels Column labels for VTK table
    void setLabels(const QStringList& labels);
    /// @return Current column labels
    QStringList labels() const;

    void run();

    /// @return Converted vtkTable (after run completes)
    vtkTable* table() const;

private:
    TableModel* m_model = nullptr;
    vtkTable* m_table = nullptr;
    QStringList m_labels;
};

}  // namespace VtkUtils
