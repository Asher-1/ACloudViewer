// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/// @file vtkplot3dwidget.h
/// @brief VTK-based 3D chart/plot widget.

#include "qVTK.h"
#include "vtkplotwidget.h"

namespace VtkUtils {
class VtkPlot3DWidgetPrivate;
/// @class VtkPlot3DWidget
/// @brief 3D plotting widget using VTK chart.
class QVTK_ENGINE_LIB_API VtkPlot3DWidget : public VtkPlotWidget {
    Q_OBJECT
public:
    explicit VtkPlot3DWidget(QWidget* parent = nullptr);
    ~VtkPlot3DWidget();

    /// @return The 3D chart context item
    vtkContextItem* chart() const;

private:
    VtkPlot3DWidgetPrivate* d_ptr;
    Q_DISABLE_COPY(VtkPlot3DWidget)
};

}  // namespace VtkUtils
