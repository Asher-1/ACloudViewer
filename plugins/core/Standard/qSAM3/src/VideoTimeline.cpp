// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "VideoTimeline.h"

#include <QMouseEvent>
#include <QPainter>

namespace {
constexpr int kBarHeight = 14;
constexpr int kBandHeight = 6;
constexpr int kBandGap = 2;
constexpr int kLeftMargin = 6;
constexpr int kRightMargin = 26;  // room for the "#id" labels
}  // namespace

VideoTimeline::VideoTimeline(QWidget* parent) : QWidget(parent) {
    setMinimumHeight(40);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
}

void VideoTimeline::setFrameCount(int count) {
    m_frameCount = count;
    update();
}

void VideoTimeline::setCurrentFrame(int frame) {
    m_currentFrame = frame;
    update();
}

void VideoTimeline::setProcessedMax(int frame) {
    m_processedMax = frame;
    update();
}

void VideoTimeline::setTimeline(const QVector<VideoTimelineEntry>& entries) {
    m_entries = entries;
    update();
}

void VideoTimeline::setInstances(const QVector<int>& ids,
                                 const QVector<QColor>& colors) {
    m_instanceIds = ids;
    m_colors = colors;
    update();
}

int VideoTimeline::frameAt(const QPoint& pos) const {
    if (m_frameCount <= 0) return 0;
    const int barW = width() - kLeftMargin - kRightMargin;
    if (barW <= 0) return 0;
    const double rel = (pos.x() - kLeftMargin) / static_cast<double>(barW);
    const double clamped = qBound(0.0, rel, 1.0);
    return static_cast<int>(clamped * (m_frameCount - 1) + 0.5);
}

void VideoTimeline::mousePressEvent(QMouseEvent* e) {
    if (e->button() == Qt::LeftButton && m_frameCount > 0) {
        m_dragging = true;
        emit seekRequested(frameAt(e->pos()));
    }
}

void VideoTimeline::mouseMoveEvent(QMouseEvent* e) {
    if (m_dragging) {
        emit seekRequested(frameAt(e->pos()));
    }
}

void VideoTimeline::mouseReleaseEvent(QMouseEvent* e) {
    m_dragging = false;
    Q_UNUSED(e);
}

void VideoTimeline::leaveEvent(QEvent* e) {
    m_dragging = false;
    QWidget::leaveEvent(e);
}

void VideoTimeline::paintEvent(QPaintEvent*) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing, false);
    if (m_frameCount <= 0) {
        p.setPen(QColor(120, 120, 130));
        p.drawText(rect(), Qt::AlignCenter,
                   tr("Open a video file to show the timeline"));
        return;
    }

    const int barW = width() - kLeftMargin - kRightMargin;
    const int barX = kLeftMargin;
    const int barY = 2;
    const double pxPerFrame = barW / static_cast<double>(m_frameCount);

    // Background bar.
    p.setPen(Qt::NoPen);
    p.setBrush(QColor(40, 40, 40));
    p.drawRoundedRect(QRect(barX, barY, barW, kBarHeight), 3, 3);

    // Processed range highlight.
    if (m_processedMax >= 0) {
        p.setBrush(QColor(60, 60, 70));
        p.drawRoundedRect(
                QRect(barX, barY,
                      static_cast<int>((m_processedMax + 1) * pxPerFrame),
                      kBarHeight),
                3, 3);
    }

    // Playhead.
    const int phX =
            barX +
            static_cast<int>(m_currentFrame /
                             static_cast<double>(qMax(m_frameCount - 1, 1)) *
                             barW);
    p.fillRect(QRect(phX - 1, barY, 3, kBarHeight), QColor(255, 255, 255, 230));
    p.setPen(QColor(200, 200, 200, 220));
    p.drawText(QPointF(phX + 5, barY + kBarHeight - 2),
               QString::number(m_currentFrame));

    // Instance presence bands.
    int rowY = barY + kBarHeight + 4;
    for (int i = 0; i < m_colors.size(); ++i) {
        drawBand(p, rowY, i, m_colors[i]);
        rowY += kBandHeight + kBandGap;
    }
    const int bandsH = rowY - (barY + kBarHeight + 4) + 4;
    setMinimumHeight(kBarHeight + 6 + bandsH);
}

void VideoTimeline::drawBand(QPainter& p,
                             int row,
                             int idIndex,
                             const QColor& color) {
    const int barW = width() - kLeftMargin - kRightMargin;
    const int barX = kLeftMargin;
    const int y = row;
    const double pxPerFrame = barW / static_cast<double>(m_frameCount);

    // Tracker IDs are not guaranteed to be contiguous after resets/refines.
    // Use the explicit sorted ID list supplied by VideoTab rather than
    // inferring an ID from the display-row index.
    const int instId = m_instanceIds.value(idIndex, idIndex + 1);
    // Dim background for the full band.
    p.setBrush(QColor(color.red() * 0.3, color.green() * 0.3,
                      color.blue() * 0.3, 100));
    p.drawRoundedRect(QRect(barX, y, barW, kBandHeight), 2, 2);

    // Bright segments where the instance is present (batched runs).
    int segStart = -1;
    for (int f = 0; f < m_entries.size(); ++f) {
        bool present = false;
        for (const auto& inst : m_entries[f].instances) {
            if (inst.first == instId) {
                present = true;
                break;
            }
        }
        if (present && segStart < 0) segStart = f;
        if ((!present || f == m_entries.size() - 1) && segStart >= 0) {
            const int segEnd = present ? f + 1 : f;
            int x0 = barX + static_cast<int>(segStart * pxPerFrame);
            int x1 = barX + static_cast<int>(segEnd * pxPerFrame);
            if (x1 - x0 < 2) x1 = x0 + 2;
            p.setBrush(color);
            p.drawRoundedRect(QRect(x0, y, x1 - x0, kBandHeight), 2, 2);
            segStart = -1;
        }
    }

    // Instance label on the right.
    p.setPen(color);
    p.drawText(QPointF(barX + barW + 4, y + kBandHeight - 1),
               QString("#%1").arg(instId));
}
