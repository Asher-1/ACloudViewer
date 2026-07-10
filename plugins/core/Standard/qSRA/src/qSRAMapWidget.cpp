// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qSRAMapWidget.h"

#include <QMouseEvent>
#include <QPaintEvent>
#include <QPainter>
#include <QResizeEvent>
#include <QWheelEvent>
#include <algorithm>
#include <cmath>

qSRAMapWidget::qSRAMapWidget(QWidget* parent) : QWidget(parent) {
    setMouseTracking(true);
    setFocusPolicy(Qt::StrongFocus);
    setMinimumSize(400, 300);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    setBackgroundRole(QPalette::Base);
    setAutoFillBackground(true);

    QPalette pal = palette();
    pal.setColor(QPalette::Base, Qt::white);
    setPalette(pal);
}

void qSRAMapWidget::setMapImage(const QImage& image) {
    m_mapImage = image;
    update();
}

void qSRAMapWidget::setMapBounds(double xMin,
                                 double xMax,
                                 double yMin,
                                 double yMax) {
    m_xMin = xMin;
    m_xMax = xMax;
    m_yMin = yMin;
    m_yMax = yMax;
    update();
}

void qSRAMapWidget::setGridVisible(bool visible) {
    m_gridVisible = visible;
    update();
}

void qSRAMapWidget::setGridColor(const QColor& color) {
    m_gridColor = color;
    update();
}

void qSRAMapWidget::setGridSteps(double xStep, double yStep) {
    m_gridXStep = xStep;
    m_gridYStep = yStep;
    update();
}

void qSRAMapWidget::setXLabels(const QVector<Label>& labels) {
    m_xLabels = labels;
    update();
}

void qSRAMapWidget::setYLabels(const QVector<Label>& labels) {
    m_yLabels = labels;
    update();
}

void qSRAMapWidget::setLabelFontSize(int size) {
    m_labelFontSize = std::max(6, size);
    update();
}

void qSRAMapWidget::setOverlaySymbols(const QVector<Symbol>& symbols) {
    m_overlaySymbols = symbols;
    update();
}

void qSRAMapWidget::clearOverlaySymbols() {
    m_overlaySymbols.clear();
    update();
}

void qSRAMapWidget::setColorScale(double minVal,
                                  double maxVal,
                                  const QImage& scaleBar) {
    m_scaleMinVal = minVal;
    m_scaleMaxVal = maxVal;
    m_colorScaleBar = scaleBar;
    update();
}

void qSRAMapWidget::setColorScaleVisible(bool visible) {
    m_colorScaleVisible = visible;
    update();
}

// Returns the usable drawing area (widget minus margins for labels/scale bar)
QRectF qSRAMapWidget::mapViewport(int widgetWidth, int widgetHeight) const {
    return QRectF(MARGIN_LEFT, MARGIN_TOP,
                  std::max(1.0, static_cast<double>(widgetWidth) - MARGIN_LEFT -
                                        MARGIN_RIGHT),
                  std::max(1.0, static_cast<double>(widgetHeight) - MARGIN_TOP -
                                        MARGIN_BOTTOM));
}

// Transform: screen = (world - center) * scale + viewport_center
// Y is flipped (world Y-up, screen Y-down)
QPointF qSRAMapWidget::worldToScreen(double wx,
                                     double wy,
                                     int widgetWidth,
                                     int widgetHeight) const {
    const QRectF vp = mapViewport(widgetWidth, widgetHeight);
    const double vpCenterX = vp.left() + vp.width() / 2.0;
    const double vpCenterY = vp.top() + vp.height() / 2.0;

    const double sx = (wx - m_centerX) * m_scale + vpCenterX;
    const double sy = vpCenterY - (wy - m_centerY) * m_scale;
    return QPointF(sx, sy);
}

QPointF qSRAMapWidget::screenToWorld(double sx,
                                     double sy,
                                     int widgetWidth,
                                     int widgetHeight) const {
    const QRectF vp = mapViewport(widgetWidth, widgetHeight);
    const double vpCenterX = vp.left() + vp.width() / 2.0;
    const double vpCenterY = vp.top() + vp.height() / 2.0;

    const double wx = (sx - vpCenterX) / m_scale + m_centerX;
    const double wy = m_centerY - (sy - vpCenterY) / m_scale;
    return QPointF(wx, wy);
}

void qSRAMapWidget::zoomFit() {
    double mapWorldW = m_xMax - m_xMin;
    double mapWorldH = m_yMax - m_yMin;
    if (mapWorldW < 1e-10) mapWorldW = 1.0;
    if (mapWorldH < 1e-10) mapWorldH = 1.0;

    m_centerX = (m_xMin + m_xMax) / 2.0;
    m_centerY = (m_yMin + m_yMax) / 2.0;

    const QRectF vp = mapViewport(width(), height());
    const double scaleX = vp.width() / mapWorldW;
    const double scaleY = vp.height() / mapWorldH;
    m_scale = std::min(scaleX, scaleY);

    update();
    emit viewChanged();
}

QImage qSRAMapWidget::exportAsImage() const {
    QImage image(size(), QImage::Format_ARGB32);
    image.fill(Qt::white);

    QPainter painter(&image);
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.setRenderHint(QPainter::TextAntialiasing, true);
    paintContent(painter, image.width(), image.height());
    return image;
}

// Main rendering routine - shared between on-screen painting and image export.
// Drawing order: map image -> grid -> border -> labels -> symbols -> color
// scale
void qSRAMapWidget::paintContent(QPainter& painter,
                                 int widgetWidth,
                                 int widgetHeight) const {
    painter.fillRect(0, 0, widgetWidth, widgetHeight, Qt::white);

    const QRectF vp = mapViewport(widgetWidth, widgetHeight);

    // 1) Draw the distance map raster image (color-coded radial deviations)
    if (!m_mapImage.isNull()) {
        const QPointF screenBL =
                worldToScreen(m_xMin, m_yMin, widgetWidth, widgetHeight);
        const QPointF screenTR =
                worldToScreen(m_xMax, m_yMax, widgetWidth, widgetHeight);

        const double left = std::min(screenBL.x(), screenTR.x());
        const double right = std::max(screenBL.x(), screenTR.x());
        const double top = std::min(screenBL.y(), screenTR.y());
        const double bottom = std::max(screenBL.y(), screenTR.y());

        const QRectF targetRect(left, top, right - left, bottom - top);

        painter.save();
        painter.setClipRect(vp);
        painter.drawImage(targetRect, m_mapImage);
        painter.restore();
    }

    // 2) Grid overlay (angular/height step lines)
    if (m_gridVisible && m_gridXStep > 0.0 && m_gridYStep > 0.0) {
        QPen gridPen(m_gridColor, 1.0, Qt::SolidLine);
        painter.setPen(gridPen);

        painter.save();
        painter.setClipRect(vp);

        if (m_xMax >= m_xMin) {
            const double xStart = m_xMin;
            const int xCount = static_cast<int>(std::floor((m_xMax - m_xMin) /
                                                           m_gridXStep)) +
                               1;
            for (int i = 0; i <= xCount; ++i) {
                const double x = xStart + static_cast<double>(i) * m_gridXStep;
                if (x > m_xMax + 1e-9) break;
                const QPointF p1 =
                        worldToScreen(x, m_yMin, widgetWidth, widgetHeight);
                const QPointF p2 =
                        worldToScreen(x, m_yMax, widgetWidth, widgetHeight);
                painter.drawLine(p1, p2);
            }
        }

        if (m_yMax >= m_yMin) {
            const double yStart = m_yMin;
            const int yCount = static_cast<int>(std::floor((m_yMax - m_yMin) /
                                                           m_gridYStep)) +
                               1;
            for (int i = 0; i <= yCount; ++i) {
                const double y = yStart + static_cast<double>(i) * m_gridYStep;
                if (y > m_yMax + 1e-9) break;
                const QPointF p1 =
                        worldToScreen(m_xMin, y, widgetWidth, widgetHeight);
                const QPointF p2 =
                        worldToScreen(m_xMax, y, widgetWidth, widgetHeight);
                painter.drawLine(p1, p2);
            }
        }

        painter.restore();
    }

    // 3) Map border frame
    painter.setPen(QPen(QColor(180, 180, 180), 1.0));
    painter.setBrush(Qt::NoBrush);
    painter.drawRect(vp);

    // 4) Axis labels (X = angular position at bottom, Y = height at left)
    QFont labelFont = painter.font();
    labelFont.setPointSize(m_labelFontSize);
    painter.setFont(labelFont);
    painter.setPen(Qt::black);

    const QFontMetrics fm(labelFont);

    for (const Label& lbl : m_xLabels) {
        const QPointF sp =
                worldToScreen(lbl.position, m_yMin, widgetWidth, widgetHeight);
        const int textWidth = fm.horizontalAdvance(lbl.text) + 8;
        const QRect textRect(static_cast<int>(sp.x() - textWidth / 2),
                             static_cast<int>(vp.bottom() + 2), textWidth,
                             MARGIN_BOTTOM - 4);
        painter.drawText(textRect, Qt::AlignHCenter | Qt::AlignTop, lbl.text);
    }

    for (const Label& lbl : m_yLabels) {
        const QPointF sp =
                worldToScreen(m_xMin, lbl.position, widgetWidth, widgetHeight);
        const int textWidth = MARGIN_LEFT - 6;
        const QRect textRect(2, static_cast<int>(sp.y() - fm.height() / 2),
                             textWidth, fm.height());
        painter.drawText(textRect, Qt::AlignRight | Qt::AlignVCenter, lbl.text);
    }

    // 5) Overlay symbols (imported measurement points from external files)
    //    Rendered as diamond+cross shapes, matching CloudCompare's symbol
    //    style.
    if (!m_overlaySymbols.isEmpty()) {
        for (const Symbol& sym : m_overlaySymbols) {
            QPointF sp = worldToScreen(sym.x, sym.y, widgetWidth, widgetHeight);
            if (!vp.contains(sp)) continue;

            double r = sym.size / 2.0;
            painter.setPen(QPen(sym.color, 1.5));
            painter.setBrush(Qt::NoBrush);
            QPointF diamond[4] = {
                    QPointF(sp.x(), sp.y() - r), QPointF(sp.x() + r, sp.y()),
                    QPointF(sp.x(), sp.y() + r), QPointF(sp.x() - r, sp.y())};
            painter.drawPolygon(diamond, 4);
            // Cross inside
            painter.drawLine(QPointF(sp.x() - r * 0.5, sp.y()),
                             QPointF(sp.x() + r * 0.5, sp.y()));
            painter.drawLine(QPointF(sp.x(), sp.y() - r * 0.5),
                             QPointF(sp.x(), sp.y() + r * 0.5));

            if (!sym.label.isEmpty()) {
                painter.setPen(sym.color);
                painter.drawText(static_cast<int>(sp.x() + r + 2),
                                 static_cast<int>(sp.y() + 4), sym.label);
            }
        }
    }

    // 6) Color scale bar (vertical gradient with min/max value labels, right
    // side)
    if (m_colorScaleVisible && !m_colorScaleBar.isNull()) {
        constexpr int barWidth = 20;
        constexpr int barMargin = 8;

        const double barLeft = widgetWidth - MARGIN_RIGHT + barMargin;
        const QRectF barRect(barLeft, vp.top(), barWidth, vp.height());

        painter.drawImage(barRect, m_colorScaleBar);

        const QString minText = QString::number(m_scaleMinVal, 'g', 4);
        const QString maxText = QString::number(m_scaleMaxVal, 'g', 4);

        const int textX = static_cast<int>(barRect.right()) + 4;
        painter.drawText(textX, static_cast<int>(barRect.bottom()), minText);
        painter.drawText(textX, static_cast<int>(barRect.top()) + fm.ascent(),
                         maxText);
    }
}

void qSRAMapWidget::paintEvent(QPaintEvent* /*event*/) {
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.setRenderHint(QPainter::TextAntialiasing, true);
    paintContent(painter, width(), height());
}

void qSRAMapWidget::mousePressEvent(QMouseEvent* event) {
    if (event->button() == Qt::MiddleButton) {
        m_panning = true;
        m_lastMousePos = event->pos();
        setCursor(Qt::ClosedHandCursor);
    } else if (event->button() == Qt::LeftButton &&
               (event->modifiers() & Qt::ShiftModifier)) {
        m_panning = true;
        m_lastMousePos = event->pos();
        setCursor(Qt::ClosedHandCursor);
    }
}

void qSRAMapWidget::mouseReleaseEvent(QMouseEvent* /*event*/) {
    if (m_panning) {
        m_panning = false;
        setCursor(Qt::ArrowCursor);
    }
}

void qSRAMapWidget::mouseMoveEvent(QMouseEvent* event) {
    if (m_panning) {
        const QPoint delta = event->pos() - m_lastMousePos;
        m_centerX -= delta.x() / m_scale;
        m_centerY += delta.y() / m_scale;
        m_lastMousePos = event->pos();
        update();
        emit viewChanged();
    }
}

void qSRAMapWidget::wheelEvent(QWheelEvent* event) {
    const double factor = (event->angleDelta().y() > 0) ? 1.15 : (1.0 / 1.15);

    const QPointF worldPos = screenToWorld(
            event->position().x(), event->position().y(), width(), height());

    m_scale *= factor;
    if (m_scale < 1e-6) m_scale = 1e-6;

    const QPointF newWorldPos = screenToWorld(
            event->position().x(), event->position().y(), width(), height());
    m_centerX += worldPos.x() - newWorldPos.x();
    m_centerY += worldPos.y() - newWorldPos.y();

    update();
    emit viewChanged();
}

void qSRAMapWidget::resizeEvent(QResizeEvent* event) {
    QWidget::resizeEvent(event);
}
