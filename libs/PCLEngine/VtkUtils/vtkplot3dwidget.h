// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "qPCL.h"
#include "vtkplotwidget.h"

namespace VtkUtils {
class VtkPlot3DWidgetPrivate;
class QPCL_ENGINE_LIB_API VtkPlot3DWidget : public VtkPlotWidget {
    Q_OBJECT
public:
    explicit VtkPlot3DWidget(QWidget* parent = nullptr);
    ~VtkPlot3DWidget();

    vtkContextItem* chart() const;

private:
    VtkPlot3DWidgetPrivate* d_ptr;
    Q_DISABLE_COPY(VtkPlot3DWidget)
};

}  // namespace VtkUtils
