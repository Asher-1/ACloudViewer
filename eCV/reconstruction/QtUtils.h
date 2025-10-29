// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>
#include <QtCore>
#include <QtOpenGL>

#include "feature/types.h"
#include "util/bitmap.h"
#include "util/types.h"

namespace cloudViewer {

Eigen::Matrix4f QMatrixToEigen(const QMatrix4x4& matrix);

QMatrix4x4 EigenToQMatrix(const Eigen::Matrix4f& matrix);

QImage BitmapToQImageRGB(const colmap::Bitmap& bitmap);

void DrawKeypoints(QPixmap* image,
                   const colmap::FeatureKeypoints& points,
                   const QColor& color = Qt::red);

QPixmap ShowImagesSideBySide(const QPixmap& image1, const QPixmap& image2);

QPixmap DrawMatches(const QPixmap& image1,
                    const QPixmap& image2,
                    const colmap::FeatureKeypoints& points1,
                    const colmap::FeatureKeypoints& points2,
                    const colmap::FeatureMatches& matches,
                    const QColor& keypoints_color = Qt::red);

}  // namespace cloudViewer
