// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Wraps a vtkGLView with 3 axis slider strips on the edges,
// matching ParaView's Slice View layout with pqMultiSliceAxisWidget.

#pragma once

#include "qVTK.h"

#include <QWidget>
#include <vector>

class QLabel;
class vtkGLView;
class vtkMultiSliceAxisWidget;

class QVTK_ENGINE_LIB_API vtkSliceViewWidget : public QWidget {
    Q_OBJECT

public:
    explicit vtkSliceViewWidget(vtkGLView* view, QWidget* parent = nullptr);
    ~vtkSliceViewWidget() override;

    vtkGLView* view() const { return m_view; }

    void setDataBounds(const double bounds[6]);

    bool outlineVisible() const { return m_outlineVisible; }
    void setOutlineVisible(bool visible);

private slots:
    void onSlicePositionsChanged(int axis, const QVector<double>& positions);

private:
    vtkGLView* m_view = nullptr;
    vtkMultiSliceAxisWidget* m_xAxis = nullptr;
    vtkMultiSliceAxisWidget* m_yAxis = nullptr;
    vtkMultiSliceAxisWidget* m_zAxis = nullptr;
    QLabel* m_statusLabel = nullptr;
    bool m_outlineVisible = true;
};
