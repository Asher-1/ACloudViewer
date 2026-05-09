// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Adapted from ParaView's pqMultiSliceAxisWidget.
// Draws a ruler-like axis strip with draggable slice markers.
// Three instances are placed on the top/left/right edges of the Slice View.

#pragma once

#include "qVTK.h"

#include <QWidget>
#include <QVector>

class QVTK_ENGINE_LIB_API vtkMultiSliceAxisWidget : public QWidget {
    Q_OBJECT

public:
    enum Axis { X_AXIS = 0, Y_AXIS = 1, Z_AXIS = 2 };

    explicit vtkMultiSliceAxisWidget(Axis axis, QWidget* parent = nullptr);
    ~vtkMultiSliceAxisWidget() override;

    void setDataRange(double minVal, double maxVal);
    double dataMin() const { return m_dataMin; }
    double dataMax() const { return m_dataMax; }

    void setSlicePositions(const QVector<double>& positions);
    QVector<double> slicePositions() const { return m_positions; }

    void setAxisTitle(const QString& title) { m_title = title; update(); }
    QString axisTitle() const { return m_title; }

    Axis axis() const { return m_axis; }

    QSize sizeHint() const override;
    QSize minimumSizeHint() const override;

signals:
    void sliceAdded(int axis, double position);
    void sliceRemoved(int axis, int index);
    void sliceMoved(int axis, int index, double newPosition);
    void slicePositionsChanged(int axis, const QVector<double>& positions);

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void mouseDoubleClickEvent(QMouseEvent* event) override;
    void contextMenuEvent(QContextMenuEvent* event) override;

private:
    double posToValue(int pixel) const;
    int valueToPos(double value) const;
    QRect axisRect() const;
    int hitMarker(const QPoint& pos) const;

    Axis m_axis;
    QString m_title;
    double m_dataMin = -1.0;
    double m_dataMax = 1.0;
    QVector<double> m_positions;
    int m_dragIndex = -1;
    bool m_dragging = false;
    static constexpr int kThickness = 28;
    static constexpr int kMarkerHalf = 5;
};
