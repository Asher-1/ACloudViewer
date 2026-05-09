// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "vtkSliceViewWidget.h"
#include "vtkMultiSliceAxisWidget.h"

#include <Visualization/vtkGLView.h>
#include <VTKExtensions/Widgets/QVTKWidgetCustom.h>

#include <QGridLayout>
#include <QVBoxLayout>

vtkSliceViewWidget::vtkSliceViewWidget(vtkGLView* view, QWidget* parent)
    : QWidget(parent), m_view(view) {
    // ParaView Slice View layout:
    //       [  Y-axis (top)  ]
    // [X-axis] [  3D view  ] [Z-axis]
    auto* grid = new QGridLayout(this);
    grid->setContentsMargins(0, 0, 0, 0);
    grid->setSpacing(0);

    m_yAxis = new vtkMultiSliceAxisWidget(
            vtkMultiSliceAxisWidget::Y_AXIS, this);
    m_xAxis = new vtkMultiSliceAxisWidget(
            vtkMultiSliceAxisWidget::X_AXIS, this);
    m_zAxis = new vtkMultiSliceAxisWidget(
            vtkMultiSliceAxisWidget::Z_AXIS, this);

    grid->addWidget(m_yAxis, 0, 0, 1, 3);        // top row spans all
    grid->addWidget(m_xAxis, 1, 0);               // left
    grid->addWidget(view->getVtkWidget(), 1, 1);  // center
    grid->addWidget(m_zAxis, 1, 2);               // right

    grid->setColumnStretch(1, 1);
    grid->setRowStretch(1, 1);

    connect(m_xAxis, &vtkMultiSliceAxisWidget::slicePositionsChanged, this,
            &vtkSliceViewWidget::onSlicePositionsChanged);
    connect(m_yAxis, &vtkMultiSliceAxisWidget::slicePositionsChanged, this,
            &vtkSliceViewWidget::onSlicePositionsChanged);
    connect(m_zAxis, &vtkMultiSliceAxisWidget::slicePositionsChanged, this,
            &vtkSliceViewWidget::onSlicePositionsChanged);
}

vtkSliceViewWidget::~vtkSliceViewWidget() = default;

void vtkSliceViewWidget::setDataBounds(const double bounds[6]) {
    m_xAxis->setDataRange(bounds[0], bounds[1]);
    m_yAxis->setDataRange(bounds[2], bounds[3]);
    m_zAxis->setDataRange(bounds[4], bounds[5]);
}

void vtkSliceViewWidget::setOutlineVisible(bool visible) {
    m_outlineVisible = visible;
    if (m_view) {
        m_view->setOutlineVisible(visible);
    }
}

void vtkSliceViewWidget::onSlicePositionsChanged(
        int axis, const QVector<double>& positions) {
    if (!m_view) return;
    std::vector<double> vec(positions.begin(), positions.end());
    m_view->setMultiSlicePositions(axis, vec);
}
