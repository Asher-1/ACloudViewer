// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/// @file vtkplotpiewidget.h
/// @brief VTK-based pie chart widget.

#include "vtkplotwidget.h"

namespace VtkUtils {
class VtkPlotPieWidgetPrivate;
/// @class VtkPlotPieWidget
/// @brief Pie chart plotting widget using VTK chart.
class QVTK_ENGINE_LIB_API VtkPlotPieWidget : public VtkPlotWidget {
    Q_OBJECT
public:
    explicit VtkPlotPieWidget(QWidget* parent = nullptr);
    ~VtkPlotPieWidget();

    /// @return The pie chart context item
    vtkContextItem* chart() const;

private:
    VtkPlotPieWidgetPrivate* d_ptr;
    Q_DISABLE_COPY(VtkPlotPieWidget)
};

}  // namespace VtkUtils
