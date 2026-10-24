// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "AnnotationNavigator.h"

#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <algorithm>

AnnotationNavigator::AnnotationNavigator(QWidget* parent) : QWidget(parent) {
    setMinimumSize(180, 180);
    setMouseTracking(true);
    setWindowTitle(tr("Navigator"));
}

void AnnotationNavigator::setImage(const QImage& image) {
    m_image = image;
    update();
}

void AnnotationNavigator::setShapes(const QVector<AnnotationShape>& shapes) {
    m_shapes = shapes;
    update();
}

void AnnotationNavigator::setViewport(const QRectF& viewport) {
    m_viewport = viewport;
    update();
}

QRectF AnnotationNavigator::imageRect() const {
    if (m_image.isNull()) return QRectF();
    const double sx = std::max(1, width() - 20) / double(m_image.width());
    const double sy = std::max(1, height() - 20) / double(m_image.height());
    const double scale = std::min(sx, sy);
    const QSizeF size(m_image.width() * scale, m_image.height() * scale);
    return QRectF((width() - size.width()) / 2.0,
                  (height() - size.height()) / 2.0, size.width(),
                  size.height());
}

QPointF AnnotationNavigator::imageToWidget(const QPointF& point) const {
    const QRectF target = imageRect();
    return target.topLeft() +
           QPointF(point.x() * target.width() / m_image.width(),
                   point.y() * target.height() / m_image.height());
}

void AnnotationNavigator::paintEvent(QPaintEvent*) {
    QPainter painter(this);
    painter.fillRect(rect(), QColor(38, 42, 48));
    if (m_image.isNull()) return;
    const QRectF target = imageRect();
    painter.setRenderHint(QPainter::SmoothPixmapTransform);
    painter.drawImage(target, m_image);

    painter.setRenderHint(QPainter::Antialiasing);
    const double sx = target.width() / m_image.width();
    const double sy = target.height() / m_image.height();
    for (const AnnotationShape& shape : m_shapes) {
        if (!shape.visible || shape.points.isEmpty()) continue;
        QPainterPath path;
        path.moveTo(imageToWidget(shape.points.first()));
        for (int i = 1; i < shape.points.size(); ++i)
            path.lineTo(imageToWidget(shape.points[i]));
        if (shape.type == QStringLiteral("rectangle") ||
            shape.type == QStringLiteral("polygon") ||
            shape.type == QStringLiteral("quadrilateral") ||
            shape.type == QStringLiteral("rotation") ||
            shape.type == QStringLiteral("cuboid"))
            path.closeSubpath();
        QPen pen(QColor(80, 255, 120, 210),
                 std::max(1.0, 1.5 * std::min(sx, sy)));
        painter.setPen(pen);
        painter.setBrush(Qt::NoBrush);
        painter.drawPath(path);
    }

    QRectF viewport = m_viewport;
    if (viewport.isEmpty())
        viewport = QRectF(0, 0, m_image.width(), m_image.height());
    QRectF viewportWidget(
            imageToWidget(viewport.topLeft()),
            QSizeF(viewport.width() * sx, viewport.height() * sy));
    painter.setPen(QPen(QColor(255, 70, 70), 2.0));
    painter.setBrush(Qt::NoBrush);
    painter.drawRect(viewportWidget.intersected(target));
}

void AnnotationNavigator::mousePressEvent(QMouseEvent* event) {
    if (event->button() != Qt::LeftButton || m_image.isNull()) return;
    const QRectF target = imageRect();
    if (!target.contains(event->pos())) return;
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
    const QPointF position = event->position();
#else
    const QPointF position = event->localPos();
#endif
    const double x = std::clamp((position.x() - target.left()) / target.width(),
                                0.0, 1.0);
    const double y = std::clamp((position.y() - target.top()) / target.height(),
                                0.0, 1.0);
    emit navigationRequested(QPointF(x, y));
    event->accept();
}
