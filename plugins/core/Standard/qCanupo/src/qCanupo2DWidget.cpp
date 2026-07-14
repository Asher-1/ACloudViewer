// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qCanupo2DWidget.h"

#include <QtCompat.h>
#include <ecvPointCloud.h>
#include <ecvPolyline.h>

#include <QMouseEvent>
#include <QPainter>
#include <QWheelEvent>
#include <cmath>

qCanupo2DWidget::qCanupo2DWidget(QWidget* parent) : QWidget(parent) {
    setMouseTracking(true);  // needed for real-time mouse move signals
    setFocusPolicy(Qt::StrongFocus);
    setMinimumSize(200, 150);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    setBackgroundRole(QPalette::Base);
    setAutoFillBackground(true);

    // White background matching CloudCompare's GL clear color
    QPalette pal = palette();
    pal.setColor(QPalette::Base, Qt::white);
    setPalette(pal);
}

void qCanupo2DWidget::setCloud(ccPointCloud* cloud) {
    m_cloud = cloud;
    update();
}

void qCanupo2DWidget::setPolyline(ccPolyline* poly) {
    m_poly = poly;
    update();
}

void qCanupo2DWidget::setPointSize(int size) {
    m_pointSize = std::max(1, size);
    update();
}

void qCanupo2DWidget::clearMarkers() { m_markers.clear(); }

void qCanupo2DWidget::addMarker(double x,
                                double y,
                                const QColor& color,
                                double radius) {
    m_markers.append({x, y, color, radius});
    update();
}

// Transform: screen_x = (world_x - centerX) * scale + widget_width/2
//            screen_y = widget_height/2 - (world_y - centerY) * scale
// (Y is flipped: world Y-up, screen Y-down)
QPointF qCanupo2DWidget::screenToWorld(int x, int y) const {
    double wx = (x - width() / 2.0) / m_scale + m_centerX;
    double wy = (height() / 2.0 - y) / m_scale + m_centerY;
    return QPointF(wx, wy);
}

QPointF qCanupo2DWidget::worldToScreen(double wx, double wy) const {
    double sx = (wx - m_centerX) * m_scale + width() / 2.0;
    double sy = height() / 2.0 - (wy - m_centerY) * m_scale;
    return QPointF(sx, sy);
}

double qCanupo2DWidget::pixelSize() const {
    return (m_scale > 0) ? 1.0 / m_scale : 1.0;
}

void qCanupo2DWidget::zoomFit() {
    if (!m_cloud || m_cloud->size() == 0) return;

    // Compute bounding box of all visible data (points + polyline)
    double xMin = 1e30, xMax = -1e30, yMin = 1e30, yMax = -1e30;
    for (unsigned i = 0; i < m_cloud->size(); ++i) {
        const CCVector3* P = m_cloud->getPoint(i);
        if (P->x < xMin) xMin = P->x;
        if (P->x > xMax) xMax = P->x;
        if (P->y < yMin) yMin = P->y;
        if (P->y > yMax) yMax = P->y;
    }

    if (m_poly) {
        for (unsigned i = 0; i < m_poly->size(); ++i) {
            const CCVector3* P = m_poly->getPoint(i);
            if (P->x < xMin) xMin = P->x;
            if (P->x > xMax) xMax = P->x;
            if (P->y < yMin) yMin = P->y;
            if (P->y > yMax) yMax = P->y;
        }
    }

    double dataW = xMax - xMin;
    double dataH = yMax - yMin;
    if (dataW < 1e-10) dataW = 1.0;
    if (dataH < 1e-10) dataH = 1.0;

    // Center the view on the data centroid
    m_centerX = (xMin + xMax) / 2.0;
    m_centerY = (yMin + yMax) / 2.0;

    // Scale to fit with 5% margin, preserving aspect ratio
    double margin = 1.05;
    double scaleX = width() / (dataW * margin);
    double scaleY = height() / (dataH * margin);
    m_scale = std::min(scaleX, scaleY);

    update();
}

void qCanupo2DWidget::paintEvent(QPaintEvent* /*event*/) {
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, m_pointSize > 2);

    // 1) Draw scatter points (class 1 + class 2 + evaluation, distinguished by
    // color)
    //    Point.x/y hold the 2D projected descriptor coordinates from the CANUPO
    //    paper.
    if (m_cloud && m_cloud->size() > 0) {
        const bool hasColors = m_cloud->hasColors();
        const int ps = m_pointSize;

        for (unsigned i = 0; i < m_cloud->size(); ++i) {
            const CCVector3* P = m_cloud->getPoint(i);
            QPointF sp = worldToScreen(P->x, P->y);

            // Frustum culling: skip points outside the visible area
            if (sp.x() < -ps || sp.x() > width() + ps || sp.y() < -ps ||
                sp.y() > height() + ps)
                continue;

            QColor col(0, 0, 0);
            if (hasColors) {
                const ecvColor::Rgb& rgb = m_cloud->getPointColor(i);
                col = QColor(rgb.r, rgb.g, rgb.b);
            }

            if (ps <= 2) {
                painter.setPen(col);
                painter.drawPoint(sp.toPoint());
            } else {
                painter.setPen(Qt::NoPen);
                painter.setBrush(col);
                painter.drawEllipse(sp, ps / 2.0, ps / 2.0);
            }
        }
    }

    // 2) Draw reference point markers (positive/negative class reference
    // points)
    for (const auto& mk : m_markers) {
        QPointF sp = worldToScreen(mk.x, mk.y);
        painter.setPen(QPen(mk.color.darker(120), 1.5));
        painter.setBrush(
                QColor(mk.color.red(), mk.color.green(), mk.color.blue(), 160));
        painter.drawEllipse(sp, mk.radius, mk.radius);
    }

    // 3) Draw classification boundary polyline (the separator between classes)
    //    Magenta color matching CloudCompare's default boundary color.
    if (m_poly && m_poly->size() > 1) {
        QPen polyPen(QColor(255, 0, 255), 2.0);
        painter.setPen(polyPen);
        painter.setBrush(Qt::NoBrush);

        QVector<QPointF> screenPts;
        screenPts.reserve(m_poly->size());
        for (unsigned i = 0; i < m_poly->size(); ++i) {
            const CCVector3* P = m_poly->getPoint(i);
            screenPts.append(worldToScreen(P->x, P->y));
        }
        painter.drawPolyline(screenPts.data(), screenPts.size());

        // Draw editable vertices (semi-transparent, for user interaction)
        painter.setPen(QPen(QColor(255, 0, 255), 1.0));
        painter.setBrush(QColor(255, 0, 255, 128));
        for (const auto& sp : screenPts) {
            painter.drawEllipse(sp, 4.0, 4.0);
        }
    }
}

void qCanupo2DWidget::mousePressEvent(QMouseEvent* event) {
    if (event->button() == Qt::MiddleButton) {
        // Middle button: always pan
        m_panning = true;
        m_lastMousePos = event->pos();
        setCursor(Qt::ClosedHandCursor);
    } else if (event->button() == Qt::LeftButton) {
        if (event->modifiers() & Qt::ShiftModifier) {
            // Shift+Left: alternate pan mode (for users without middle button)
            m_panning = true;
            m_lastMousePos = event->pos();
            setCursor(Qt::ClosedHandCursor);
        } else {
            // Left click: add/select boundary vertex
            emit leftButtonClicked(event->x(), event->y());
        }
    } else if (event->button() == Qt::RightButton) {
        // Right click: remove boundary vertex
        emit rightButtonClicked(event->x(), event->y());
    }
}

void qCanupo2DWidget::mouseReleaseEvent(QMouseEvent* event) {
    if (m_panning) {
        m_panning = false;
        setCursor(Qt::ArrowCursor);
    }
    if (event->button() == Qt::LeftButton) {
        emit buttonReleased();
    }
}

void qCanupo2DWidget::mouseMoveEvent(QMouseEvent* event) {
    if (m_panning) {
        QPoint delta = event->pos() - m_lastMousePos;
        m_centerX -= delta.x() / m_scale;
        m_centerY += delta.y() / m_scale;
        m_lastMousePos = event->pos();
        update();
    } else {
        emit mouseMoved(event->x(), event->y(), event->buttons());
    }
}

void qCanupo2DWidget::wheelEvent(QWheelEvent* event) {
    double factor = (event->angleDelta().y() > 0) ? 1.15 : (1.0 / 1.15);

    // Remember world position under cursor before zoom
    const QPointF pos = qtCompatWheelEventPos(event);
    QPointF worldPos = screenToWorld(pos.x(), pos.y());

    m_scale *= factor;

    // After scaling, recalculate what world position the cursor now maps to,
    // then shift the center so the original world point stays under the cursor.
    double newWx = (pos.x() - width() / 2.0) / m_scale + m_centerX;
    double newWy = (height() / 2.0 - pos.y()) / m_scale + m_centerY;
    m_centerX += (worldPos.x() - newWx);
    m_centerY += (worldPos.y() - newWy);

    update();
}

void qCanupo2DWidget::resizeEvent(QResizeEvent* event) {
    QWidget::resizeEvent(event);
}
