// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// Timeline widget for the qSAM3 video tab.
//
// Qt re-implementation of the upstream main_video.cpp timeline: a scrubber
// bar (processed range + playhead) plus one colored band per tracked
// instance showing which frames it is present in. Clicking or dragging on
// the bar seeks.

#pragma once

#include <QColor>
#include <QVector>
#include <QWidget>

struct VideoTimelineEntry {
    QVector<QPair<int, float>> instances;  // (instance id, score) on that frame
};

class VideoTimeline : public QWidget {
    Q_OBJECT
public:
    explicit VideoTimeline(QWidget* parent = nullptr);

    void setFrameCount(int count);
    void setCurrentFrame(int frame);
    void setProcessedMax(int frame);  // highest frame tracked so far
    void setTimeline(const QVector<VideoTimelineEntry>& entries);
    void setInstances(const QVector<int>& ids, const QVector<QColor>& colors);

signals:
    void seekRequested(int frame);

protected:
    void paintEvent(QPaintEvent* e) override;
    void mousePressEvent(QMouseEvent* e) override;
    void mouseMoveEvent(QMouseEvent* e) override;
    void mouseReleaseEvent(QMouseEvent* e) override;
    void leaveEvent(QEvent* e) override;

private:
    int frameAt(const QPoint& pos) const;
    void drawBand(QPainter& p, int row, int id, const QColor& color);

    int m_frameCount = 0;
    int m_currentFrame = 0;
    int m_processedMax = -1;
    QVector<VideoTimelineEntry> m_entries;
    QVector<int> m_instanceIds;
    QVector<QColor> m_colors;
    bool m_dragging = false;
};
