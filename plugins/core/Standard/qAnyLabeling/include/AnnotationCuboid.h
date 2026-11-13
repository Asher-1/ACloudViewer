// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QPainterPath>
#include <QPointF>
#include <QVector>

// Image-space cuboid geometry. The eight stored vertices keep the upstream
// front TL/TR/BR/BL, back TL/TR/BR/BL order; edge centers are virtual controls.
namespace annotationCuboid {
enum Control {
    FrontLeftCenter = 8,
    FrontRightCenter,
    FrontTopCenter,
    FrontBottomCenter,
    BackLeftCenter,
    BackRightCenter
};

enum class Face { None, Front, Left, Right, Back };

QVector<int> visibleControls(const QVector<QPointF>& points);
QPointF controlPoint(const QVector<QPointF>& points, int control);
int nearestControl(const QVector<QPointF>& points,
                   const QPointF& position,
                   double radius);
QPainterPath facePath(const QVector<QPointF>& points, Face face);
Face hitFace(const QVector<QPointF>& points, const QPointF& position);

// Return true only for an actual geometry change. Constraint size is measured
// in image pixels, independently of zoom, matching the source min_depth rule.
bool moveControl(QVector<QPointF>& points,
                 int control,
                 const QPointF& position,
                 double minDepth);
bool moveFace(QVector<QPointF>& points,
              Face face,
              const QPointF& offset,
              double minDepth);
}  // namespace annotationCuboid
