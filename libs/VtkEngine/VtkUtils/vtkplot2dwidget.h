// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/// @file vtkplot2dwidget.h
/// @brief VTK-based 2D chart/plot widget (XY charts).

#include "qVTK.h"
#include "vtkplotwidget.h"

namespace VtkUtils {
class VtkPlot2DWidgetPrivate;
/// @class VtkPlot2DWidget
/// @brief 2D plotting widget using VTK chart for XY line/bar charts.
class QVTK_ENGINE_LIB_API VtkPlot2DWidget : public VtkPlotWidget {
    Q_OBJECT
public:
    explicit VtkPlot2DWidget(QWidget* parent = nullptr);
    ~VtkPlot2DWidget();

    /// @return The 2D chart context item
    vtkContextItem* chart() const;

private:
    VtkPlot2DWidgetPrivate* d_ptr;
    Q_DISABLE_COPY(VtkPlot2DWidget)
};

}  // namespace VtkUtils
