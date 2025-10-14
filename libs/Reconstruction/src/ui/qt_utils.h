// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef COLMAP_SRC_UI_QT_UTILS_H_
#define COLMAP_SRC_UI_QT_UTILS_H_

#include <Eigen/Core>
#include <QtCore>
#include <QtOpenGL>

#include "feature/types.h"
#include "util/bitmap.h"
#include "util/types.h"

namespace colmap {

Eigen::Matrix4f QMatrixToEigen(const QMatrix4x4& matrix);

QMatrix4x4 EigenToQMatrix(const Eigen::Matrix4f& matrix);

QImage BitmapToQImageRGB(const Bitmap& bitmap);

void DrawKeypoints(QPixmap* image,
                   const FeatureKeypoints& points,
                   const QColor& color = Qt::red);

QPixmap ShowImagesSideBySide(const QPixmap& image1, const QPixmap& image2);

QPixmap DrawMatches(const QPixmap& image1,
                    const QPixmap& image2,
                    const FeatureKeypoints& points1,
                    const FeatureKeypoints& points2,
                    const FeatureMatches& matches,
                    const QColor& keypoints_color = Qt::red);

}  // namespace colmap

#endif  // COLMAP_SRC_UI_QT_UTILS_H_
