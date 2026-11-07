// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QImage>
#include <QPointF>
#include <QRectF>
#include <QWidget>

#include "AnnotationCanvas.h"

class QMouseEvent;

class AnnotationNavigator final : public QWidget {
    Q_OBJECT
public:
    explicit AnnotationNavigator(QWidget* parent = nullptr);

    void setImage(const QImage& image);
    void setShapes(const QVector<AnnotationShape>& shapes);
    void setViewport(const QRectF& viewport);

signals:
    void navigationRequested(const QPointF& imageRatio);

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;

private:
    QRectF imageRect() const;
    QPointF imageToWidget(const QPointF& point) const;

    QImage m_image;
    QVector<AnnotationShape> m_shapes;
    QRectF m_viewport;
};
