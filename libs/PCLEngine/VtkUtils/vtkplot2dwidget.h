// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef VTKPLOT2DWIDGET_H
#define VTKPLOT2DWIDGET_H

#include "../qPCL.h"
#include "vtkplotwidget.h"

namespace VtkUtils {
class VtkPlot2DWidgetPrivate;
class QPCL_ENGINE_LIB_API VtkPlot2DWidget : public VtkPlotWidget {
    Q_OBJECT
public:
    explicit VtkPlot2DWidget(QWidget* parent = nullptr);
    ~VtkPlot2DWidget();

    vtkContextItem* chart() const;

private:
    VtkPlot2DWidgetPrivate* d_ptr;
    Q_DISABLE_COPY(VtkPlot2DWidget)
};

}  // namespace VtkUtils
#endif  // VTKPLOT2DWIDGET_H
