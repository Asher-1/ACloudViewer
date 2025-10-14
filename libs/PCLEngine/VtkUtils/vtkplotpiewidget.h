// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef VTKPLOTPIEWIDGET_H
#define VTKPLOTPIEWIDGET_H

#include "vtkplotwidget.h"

namespace VtkUtils {
class VtkPlotPieWidgetPrivate;
class QPCL_ENGINE_LIB_API VtkPlotPieWidget : public VtkPlotWidget {
    Q_OBJECT
public:
    explicit VtkPlotPieWidget(QWidget* parent = nullptr);
    ~VtkPlotPieWidget();

    vtkContextItem* chart() const;

private:
    VtkPlotPieWidgetPrivate* d_ptr;
    Q_DISABLE_COPY(VtkPlotPieWidget)
};

}  // namespace VtkUtils
#endif  // VTKPLOTPIEWIDGET_H
