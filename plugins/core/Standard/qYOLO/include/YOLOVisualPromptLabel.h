// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QLabel>
#include <QList>
#include <QRectF>

/** Interactive preview label for YOLOE visual prompts (SAVPE): shows the
 *  input image and lets the user draw one example box per target with the
 *  mouse (the qSAM3-style prompt interaction). Boxes are kept in FULL-IMAGE
 *  pixel coordinates and rendered as overlays labeled object0..objectN-1,
 *  mirroring the official YOLOE visual-prompt semantics (visual prompts
 *  group examples; they do not carry names). */
class YOLOVisualPromptLabel : public QLabel {
    Q_OBJECT

public:
    explicit YOLOVisualPromptLabel(QWidget* parent = nullptr);

    /** Show \p image as the prompt canvas (scaled to fit \p displaySize). */
    void setPromptImage(const QImage& image, const QSize& displaySize);
    void clearPrompt();

    /** Enable/disable rubber-band box drawing (off = plain preview). */
    void setDrawingEnabled(bool enabled);
    bool drawingEnabled() const { return m_drawingEnabled; }

    /** Boxes in full-image pixel coordinates ([x1, y1, x2, y2] order). */
    QList<QRectF> boxes() const { return m_boxes; }
    void setBoxes(const QList<QRectF>& boxes);
    void removeLast();
    void clearBoxes();

    int boxCount() const { return m_boxes.size(); }

signals:
    void boxesChanged();

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;

private:
    /** Widget position -> full-image pixel coordinate (identity mapping
     *  when no image is shown). */
    QPointF toImageCoords(const QPointF& widgetPos) const;
    QRectF toWidgetRect(const QRectF& imageRect) const;
    void updateLabelFromBoxes();

    QImage m_image;            // full-resolution prompt canvas
    QSizeF m_imageSize;        // full-image size (for coord mapping)
    QRectF m_pixmapRect;       // displayed pixmap rect inside the widget
    QList<QRectF> m_boxes;     // full-image pixel coordinates
    bool m_drawingEnabled = false;
    bool m_rubberBandActive = false;
    QPointF m_rubberBandStart;     // image coords
    QRectF m_rubberBandCurrent;    // image coords
};
