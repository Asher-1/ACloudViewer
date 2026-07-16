// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QColor>
#include <QPointF>
#include <QVector>
#include <QWidget>

class ccPointCloud;
class ccPolyline;

//! A 2D marker point drawn as an overlay on the scatter plot
struct Marker2D {
    double x = 0, y = 0;  //!< position in data (world) coordinates
    QColor color;         //!< marker color
    double radius = 5.0;  //!< marker radius in pixels
};

//! Native QPainter-based 2D scatter plot widget for qCanupo classifier
//! training.
/** This widget renders the 2D projection of multi-scale descriptors
    (the "2D view" in the CANUPO training dialog). It replaces the previous
    VTK/OpenGL-based view to avoid context conflicts in embedded Qt dialogs.

    Features:
    - Renders point cloud data (class 1 / class 2 / evaluation points)
    - Renders the classification boundary polyline
    - Supports interactive pan (middle-drag) and zoom (mouse wheel)
    - Emits signals for left/right click (used for boundary editing)

    Coordinate system:
    - Data coordinates: the raw (x,y) descriptor values
    - Screen coordinates: widget pixel positions
    - Transform: screen = (world - center) * scale + widget_center
**/
class qCanupo2DWidget : public QWidget {
    Q_OBJECT

public:
    explicit qCanupo2DWidget(QWidget* parent = nullptr);

    //! Set the point cloud data to display (class points as 2D scatter)
    void setCloud(ccPointCloud* cloud);
    //! Set the classification boundary polyline
    void setPolyline(ccPolyline* poly);
    //! Set the rendering point size in pixels
    void setPointSize(int size);
    //! Remove all overlay markers
    void clearMarkers();
    //! Add a single overlay marker (e.g. reference point for classification)
    void addMarker(double x, double y, const QColor& color, double radius);

    //! Converts screen pixel coordinates to world (data) coordinates
    QPointF screenToWorld(int x, int y) const;
    //! Returns the size of one pixel in world units (for picking tolerance)
    double pixelSize() const;

    //! Auto-fit the view to show all data (called after initial data load)
    void zoomFit();

signals:
    //! Emitted on left mouse button press (for boundary point insertion)
    void leftButtonClicked(int x, int y);
    //! Emitted on right mouse button press (for boundary point removal)
    void rightButtonClicked(int x, int y);
    //! Emitted on mouse move (for interactive boundary dragging)
    void mouseMoved(int x, int y, Qt::MouseButtons buttons);
    //! Emitted when any mouse button is released
    void buttonReleased();

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;

private:
    //! Convert data coordinates to screen pixel coordinates
    QPointF worldToScreen(double wx, double wy) const;
    //! Recalculate transform parameters after resize or zoom
    void updateTransform();

    ccPointCloud* m_cloud =
            nullptr;               //!< current point cloud (2D descriptor data)
    ccPolyline* m_poly = nullptr;  //!< classification boundary polyline
    int m_pointSize = 1;           //!< point rendering size in pixels
    QVector<Marker2D> m_markers;   //!< overlay markers (ref points, etc.)

    // View transform parameters: screen = (world - center) * scale +
    // widget_center
    double m_centerX = 0.0;  //!< view center X in world coordinates
    double m_centerY = 0.0;  //!< view center Y in world coordinates
    double m_scale = 1.0;    //!< zoom scale factor (pixels per world unit)

    // Interactive pan state
    bool m_panning = false;  //!< true while middle-button panning
    QPoint m_lastMousePos;   //!< last mouse position during pan
};
