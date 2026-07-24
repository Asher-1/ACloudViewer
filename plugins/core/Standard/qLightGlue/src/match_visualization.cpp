// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "match_visualization.h"

#include <QPainter>

namespace {

QImage scale_to_height(const QImage& src, int target_h) {
    if (src.isNull() || target_h <= 0) return src;
    if (src.height() <= target_h) return src;
    return src.scaledToHeight(target_h, Qt::SmoothTransformation);
}

QPointF map_keypoint(const QPointF& kp,
                     const QSize& sourceSize,
                     const QSize& targetSize) {
    if (sourceSize.width() <= 0 || sourceSize.height() <= 0) return kp;
    const double sx =
            static_cast<double>(targetSize.width()) / sourceSize.width();
    const double sy =
            static_cast<double>(targetSize.height()) / sourceSize.height();
    return QPointF(kp.x() * sx, kp.y() * sy);
}

}  // namespace

QImage renderMatchVisualization(const QImage& image0,
                                const QImage& image1,
                                const QVector<QPointF>& keypoints0,
                                const QVector<QPointF>& keypoints1,
                                const QVector<LightGlueMatchResult>& matches,
                                int compositeWidth0,
                                int compositeHeight0,
                                int compositeWidth1,
                                int compositeHeight1) {
    QImage left = image0;
    QImage right = image1;
    if (left.isNull() || right.isNull()) {
        return {};
    }

    const int target_h = 480;
    left = scale_to_height(left, target_h);
    right = scale_to_height(right, target_h);

    const int gap = 8;
    const int total_w = left.width() + gap + right.width();
    const int total_h = left.height();

    QImage canvas(total_w, total_h, QImage::Format_RGB32);
    canvas.fill(Qt::black);

    QPainter painter(&canvas);
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.drawImage(0, 0, left);
    painter.drawImage(left.width() + gap, 0, right);

    const QSize src0(compositeWidth0 > 0 ? compositeWidth0 : left.width(),
                     compositeHeight0 > 0 ? compositeHeight0 : left.height());
    const QSize src1(compositeWidth1 > 0 ? compositeWidth1 : right.width(),
                     compositeHeight1 > 0 ? compositeHeight1 : right.height());
    const QSize dst0(left.width(), left.height());
    const QSize dst1(right.width(), right.height());
    const int x_offset1 = left.width() + gap;

    QPen linePen(QColor(0, 255, 0), 1);
    painter.setPen(linePen);
    for (const auto& m : matches) {
        if (m.idx1 < 0 || m.idx1 >= keypoints0.size() || m.idx2 < 0 ||
            m.idx2 >= keypoints1.size()) {
            continue;
        }
        const QPointF p0 = map_keypoint(keypoints0[m.idx1], src0, dst0);
        const QPointF p1 = map_keypoint(keypoints1[m.idx2], src1, dst1) +
                           QPointF(x_offset1, 0);
        painter.drawLine(p0, p1);
    }

    painter.setBrush(QColor(0, 255, 0));
    painter.setPen(Qt::NoPen);
    const int r = 2;
    for (const QPointF& kp : keypoints0) {
        const QPointF p = map_keypoint(kp, src0, dst0);
        painter.drawEllipse(p, r, r);
    }
    for (const QPointF& kp : keypoints1) {
        const QPointF p = map_keypoint(kp, src1, dst1) + QPointF(x_offset1, 0);
        painter.drawEllipse(p, r, r);
    }

    painter.end();
    return canvas;
}
