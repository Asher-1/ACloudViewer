// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Adapted from ParaView's pqMultiSliceAxisWidget / vtkMultiSliceContextItem.

#include "vtkMultiSliceAxisWidget.h"

#include <QContextMenuEvent>
#include <QMenu>
#include <QMouseEvent>
#include <QPainter>

static const QColor kAxisColors[] = {
        QColor(220, 60, 60),   // X — red
        QColor(60, 180, 60),   // Y — green
        QColor(60, 60, 220),   // Z — blue
};

vtkMultiSliceAxisWidget::vtkMultiSliceAxisWidget(Axis axis, QWidget* parent)
    : QWidget(parent), m_axis(axis) {
    switch (axis) {
        case X_AXIS: m_title = tr("X"); break;
        case Y_AXIS: m_title = tr("Y"); break;
        case Z_AXIS: m_title = tr("Z"); break;
    }
    setMouseTracking(true);
    setCursor(Qt::CrossCursor);
}

vtkMultiSliceAxisWidget::~vtkMultiSliceAxisWidget() = default;

void vtkMultiSliceAxisWidget::setDataRange(double minVal, double maxVal) {
    if (minVal >= maxVal) return;
    m_dataMin = minVal;
    m_dataMax = maxVal;
    update();
}

void vtkMultiSliceAxisWidget::setSlicePositions(
        const QVector<double>& positions) {
    m_positions = positions;
    update();
}

QSize vtkMultiSliceAxisWidget::sizeHint() const {
    if (m_axis == X_AXIS) return {200, kThickness};
    return {kThickness, 200};
}

QSize vtkMultiSliceAxisWidget::minimumSizeHint() const {
    if (m_axis == X_AXIS) return {40, kThickness};
    return {kThickness, 40};
}

QRect vtkMultiSliceAxisWidget::axisRect() const {
    const int margin = 6;
    if (m_axis == X_AXIS) {
        return QRect(margin, 0, width() - 2 * margin, kThickness);
    }
    return QRect(0, margin, kThickness, height() - 2 * margin);
}

double vtkMultiSliceAxisWidget::posToValue(int pixel) const {
    QRect r = axisRect();
    double t;
    if (m_axis == X_AXIS) {
        t = double(pixel - r.left()) / double(r.width());
    } else {
        t = 1.0 - double(pixel - r.top()) / double(r.height());
    }
    return m_dataMin + t * (m_dataMax - m_dataMin);
}

int vtkMultiSliceAxisWidget::valueToPos(double value) const {
    QRect r = axisRect();
    double t = (value - m_dataMin) / (m_dataMax - m_dataMin);
    if (m_axis == X_AXIS) {
        return r.left() + int(t * r.width());
    }
    return r.bottom() - int(t * r.height());
}

int vtkMultiSliceAxisWidget::hitMarker(const QPoint& pos) const {
    for (int i = 0; i < m_positions.size(); ++i) {
        int mp = valueToPos(m_positions[i]);
        int d = (m_axis == X_AXIS) ? qAbs(pos.x() - mp)
                                   : qAbs(pos.y() - mp);
        if (d <= kMarkerHalf + 2) return i;
    }
    return -1;
}

void vtkMultiSliceAxisWidget::paintEvent(QPaintEvent* /*event*/) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);

    QRect r = axisRect();
    QColor axColor = kAxisColors[static_cast<int>(m_axis)];

    p.fillRect(rect(), QColor(45, 45, 45));
    p.setPen(QPen(axColor.darker(150), 1));
    if (m_axis == X_AXIS) {
        int cy = r.center().y();
        p.drawLine(r.left(), cy, r.right(), cy);
    } else {
        int cx = r.center().x();
        p.drawLine(cx, r.top(), cx, r.bottom());
    }

    double range = m_dataMax - m_dataMin;
    if (range <= 0) return;

    p.setPen(QColor(180, 180, 180));
    QFont f = font();
    f.setPixelSize(9);
    p.setFont(f);

    int numTicks = (m_axis == X_AXIS) ? qMax(2, r.width() / 60)
                                      : qMax(2, r.height() / 40);
    for (int i = 0; i <= numTicks; ++i) {
        double t = double(i) / numTicks;
        double val = m_dataMin + t * range;
        int pix = valueToPos(val);
        QString label = QString::number(val, 'g', 3);

        if (m_axis == X_AXIS) {
            p.drawLine(pix, r.top() + 2, pix, r.bottom() - 2);
            p.drawText(pix - 20, r.bottom() - 2, 40, 12,
                       Qt::AlignHCenter | Qt::AlignTop, label);
        } else {
            p.drawLine(r.left() + 2, pix, r.right() - 2, pix);
            p.save();
            p.translate(r.left() - 2, pix);
            p.rotate(-90);
            p.drawText(-20, -12, 40, 12, Qt::AlignHCenter | Qt::AlignBottom,
                       label);
            p.restore();
        }
    }

    f.setPixelSize(10);
    f.setBold(true);
    p.setFont(f);
    p.setPen(axColor);
    if (m_axis == X_AXIS) {
        p.drawText(r, Qt::AlignRight | Qt::AlignTop, m_title);
    } else {
        p.save();
        p.translate(r.right() - 2, r.top() + 2);
        p.rotate(90);
        p.drawText(0, -12, 40, 12, Qt::AlignLeft | Qt::AlignBottom, m_title);
        p.restore();
    }

    for (int i = 0; i < m_positions.size(); ++i) {
        int mp = valueToPos(m_positions[i]);
        QColor mc = axColor;
        if (m_dragIndex == i) mc = mc.lighter(150);

        p.setPen(QPen(mc, 2));
        p.setBrush(mc);

        QPolygon triangle;
        if (m_axis == X_AXIS) {
            int cy = r.center().y();
            triangle << QPoint(mp, cy - kMarkerHalf)
                     << QPoint(mp - kMarkerHalf, cy + kMarkerHalf)
                     << QPoint(mp + kMarkerHalf, cy + kMarkerHalf);
        } else {
            int cx = r.center().x();
            triangle << QPoint(cx - kMarkerHalf, mp)
                     << QPoint(cx + kMarkerHalf, mp - kMarkerHalf)
                     << QPoint(cx + kMarkerHalf, mp + kMarkerHalf);
        }
        p.drawPolygon(triangle);
    }
}

void vtkMultiSliceAxisWidget::mousePressEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        int idx = hitMarker(event->pos());
        if (idx >= 0) {
            m_dragIndex = idx;
            m_dragging = true;
            return;
        }
    }
    QWidget::mousePressEvent(event);
}

void vtkMultiSliceAxisWidget::mouseMoveEvent(QMouseEvent* event) {
    if (m_dragging && m_dragIndex >= 0 && m_dragIndex < m_positions.size()) {
        int pixel = (m_axis == X_AXIS) ? event->pos().x() : event->pos().y();
        double newVal = posToValue(pixel);
        newVal = qBound(m_dataMin, newVal, m_dataMax);
        m_positions[m_dragIndex] = newVal;
        update();
        emit sliceMoved(static_cast<int>(m_axis), m_dragIndex, newVal);
        emit slicePositionsChanged(static_cast<int>(m_axis), m_positions);
        return;
    }

    int idx = hitMarker(event->pos());
    setCursor(idx >= 0 ? Qt::SizeAllCursor : Qt::CrossCursor);
    QWidget::mouseMoveEvent(event);
}

void vtkMultiSliceAxisWidget::mouseReleaseEvent(QMouseEvent* event) {
    if (m_dragging) {
        m_dragging = false;
        m_dragIndex = -1;
        update();
        return;
    }
    QWidget::mouseReleaseEvent(event);
}

void vtkMultiSliceAxisWidget::mouseDoubleClickEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        int pixel = (m_axis == X_AXIS) ? event->pos().x() : event->pos().y();
        double val = posToValue(pixel);
        val = qBound(m_dataMin, val, m_dataMax);
        m_positions.append(val);
        update();
        emit sliceAdded(static_cast<int>(m_axis), val);
        emit slicePositionsChanged(static_cast<int>(m_axis), m_positions);
    }
}

void vtkMultiSliceAxisWidget::contextMenuEvent(QContextMenuEvent* event) {
    int idx = hitMarker(event->pos());
    if (idx < 0) return;

    QMenu menu(this);
    auto* removeAction = menu.addAction(tr("Remove Slice"));
    auto* chosen = menu.exec(event->globalPos());
    if (chosen == removeAction && idx < m_positions.size()) {
        m_positions.removeAt(idx);
        update();
        emit sliceRemoved(static_cast<int>(m_axis), idx);
        emit slicePositionsChanged(static_cast<int>(m_axis), m_positions);
    }
}
