// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "vtkplotwidget.h"

#include <VtkUtils/vtkutils.h>
#include <vtkAxis.h>
#include <vtkChartXY.h>
#include <vtkChartXYZ.h>
#include <vtkContextScene.h>
#include <vtkContextView.h>
#include <vtkFloatArray.h>
#include <vtkPlot.h>
#include <vtkPlotLine3D.h>
#include <vtkPlotPoints3D.h>
#include <vtkPlotSurface.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkTable.h>
#include <vtkVariant.h>

namespace VtkUtils {

class VtkPlotWidgetPrivate {
public:
    ~VtkPlotWidgetPrivate();

    vtkContextView* contextView;
};

VtkPlotWidgetPrivate::~VtkPlotWidgetPrivate() {
    VtkUtils::vtkSafeDelete(contextView);
}

VtkPlotWidget::VtkPlotWidget(QWidget* parent) : QVTKOpenGLNativeWidget(parent) {
    setWindowTitle(tr("ChartXY"));
    d_ptr = new VtkPlotWidgetPrivate;
    init();
}

VtkPlotWidget::~VtkPlotWidget() { delete d_ptr; }

void VtkPlotWidget::init() {
    d_ptr->contextView = vtkContextView::New();
    d_ptr->contextView->SetRenderWindow(this->GetRenderWindow());
    d_ptr->contextView->GetRenderWindow()->SetMultiSamples(0);
    d_ptr->contextView->GetRenderWindow()->Render();
}

vtkContextView* VtkPlotWidget::contextView() const {
    return d_ptr->contextView;
}

}  // namespace VtkUtils
