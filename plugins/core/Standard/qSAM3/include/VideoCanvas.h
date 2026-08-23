// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// Video canvas for the qSAM3 video tab.
//
// Shows the current video frame with per-instance mask overlays (each
// tracked instance tinted with its own color, like the upstream
// examples/main_video.cpp build_frame_overlay) and interactive annotation:
//   - left-click on a tracked mask  -> instanceClicked(id) (refine)
//   - left-click elsewhere          -> start box drag -> boxDrawn()
//   - right-click                   -> negPointAdded(image coords) (refine)

#pragma once

#include <QColor>
#include <QImage>
#include <QLabel>
#include <QPointF>
#include <QRectF>
#include <QVector>

struct VideoInstanceBox {
    int id = -1;
    QRectF box;
    QColor color;
    float score = 0.0f;
};

class VideoCanvas : public QLabel {
    Q_OBJECT
public:
    explicit VideoCanvas(QWidget* parent = nullptr);

    void setFrame(const QImage& frame);
    void setInstances(const QVector<VideoInstanceBox>& boxes,
                      const QVector<QImage>& masks);
    void setPromptPoints(const QVector<QPointF>& pos,
                         const QVector<QPointF>& neg);
    void clearAll();
    void setInteractive(bool on);

    bool isInsideImage(const QPointF& p) const;
    QPointF screenToImage(const QPointF& screen) const;
    const QVector<QPointF>& posPoints() const { return m_posPoints; }
    const QVector<QPointF>& negPoints() const { return m_negPoints; }
    bool hasBox() const { return m_hasBox; }
    QRectF box() const { return m_box; }
    void clearBox();
    QSize imageSize() const;

    /** Hit-tests image coords against the current instance masks; returns
     *  the instance id or -1. */
    int hitTestInstance(const QPointF& p) const;

signals:
    void boxDrawn();
    void instanceClicked(int instanceId);
    void posPointAdded(const QPointF& p);
    void negPointAdded(const QPointF& p);

protected:
    void mousePressEvent(QMouseEvent* e) override;
    void mouseMoveEvent(QMouseEvent* e) override;
    void mouseReleaseEvent(QMouseEvent* e) override;
    void paintEvent(QPaintEvent* e) override;
    void resizeEvent(QResizeEvent* e) override;

private:
    void redraw();
    void drawFrameAndMasks(QPainter& p, QImage& canvas);

    bool m_interactive = false;
    QImage m_frame;       // current video frame (RGB32)
    QImage m_composited;  // frame + mask tints, at frame resolution
    QVector<VideoInstanceBox> m_boxes;
    QVector<QImage> m_masks;  // parallel to m_boxes, 0/255 grayscale
    QVector<QPointF> m_posPoints;
    QVector<QPointF> m_negPoints;

    bool m_hasBox = false;
    QRectF m_box;
    bool m_dragging = false;
    QPointF m_dragStart;
    QRectF m_dragRect;
};
