// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "VideoCanvas.h"

#include <ecvAICoreUiHelper.h>

#include <QMouseEvent>
#include <QPainter>
#include <QPixmap>
#include <QResizeEvent>

VideoCanvas::VideoCanvas(QWidget* parent) : QLabel(parent) {
    setMinimumSize(ecvAICoreUi::dpiScaled(320), ecvAICoreUi::dpiScaled(240));
    setAlignment(Qt::AlignCenter);
    setStyleSheet(
            "QLabel { background: #1a1a26; border: 1px solid #333;"
            " border-radius: 4px; color: #666; font-size: 13px; }");
    setText(tr("Open a video, pick a model and click Load.\n"
               "Then pause, click on an object / drag a box to add an "
               "instance, and press Play."));
    setMouseTracking(true);
}

void VideoCanvas::setFrame(const QImage& frame) {
    m_frame = frame;
    redraw();
}

void VideoCanvas::setInstances(const QVector<VideoInstanceBox>& boxes,
                               const QVector<QImage>& masks) {
    m_boxes = boxes;
    m_masks = masks;
    redraw();
}

void VideoCanvas::setPromptPoints(const QVector<QPointF>& pos,
                                  const QVector<QPointF>& neg) {
    m_posPoints = pos;
    m_negPoints = neg;
    redraw();
}

void VideoCanvas::clearAll() {
    m_frame = QImage();
    m_composited = QImage();
    m_boxes.clear();
    m_masks.clear();
    m_posPoints.clear();
    m_negPoints.clear();
    clearBox();
    setText(tr("Open a video, pick a model and click Load.\n"
               "Then pause, click on an object / drag a box to add an "
               "instance, and press Play."));
    setPixmap(QPixmap());
}

void VideoCanvas::setInteractive(bool on) {
    m_interactive = on;
    setCursor(on ? Qt::CrossCursor : Qt::ArrowCursor);
}

bool VideoCanvas::isInsideImage(const QPointF& p) const {
    return !m_frame.isNull() && p.x() >= 0 && p.y() >= 0 &&
           p.x() < static_cast<double>(m_frame.width()) &&
           p.y() < static_cast<double>(m_frame.height());
}

QPointF VideoCanvas::screenToImage(const QPointF& screen) const {
    const QPixmap* pm = pixmap();
    if (!pm || pm->isNull() || m_frame.isNull()) return QPointF(-1, -1);
    const double offX = (width() - pm->width()) / 2.0;
    const double offY = (height() - pm->height()) / 2.0;
    const double ix = (screen.x() - offX) / pm->width() * m_frame.width();
    const double iy = (screen.y() - offY) / pm->height() * m_frame.height();
    return QPointF(ix, iy);
}

void VideoCanvas::clearBox() {
    m_hasBox = false;
    m_box = QRectF();
    redraw();
}

void VideoCanvas::addNegPoint(const QPointF& p) {
    m_negPoints.append(p);
    redraw();
}

QSize VideoCanvas::imageSize() const { return m_frame.size(); }

int VideoCanvas::hitTestInstance(const QPointF& p) const {
    const int px = static_cast<int>(p.x());
    const int py = static_cast<int>(p.y());
    for (int i = 0; i < m_masks.size(); ++i) {
        const QImage& mask = m_masks[i];
        if (mask.isNull()) continue;
        if (px >= 0 && px < mask.width() && py >= 0 && py < mask.height() &&
            mask.constScanLine(py)[px] > 127) {
            return m_boxes.value(i).id;
        }
    }
    return -1;
}

void VideoCanvas::mousePressEvent(QMouseEvent* e) {
    if (!m_interactive || m_frame.isNull()) {
        QLabel::mousePressEvent(e);
        return;
    }
    const QPointF ip = screenToImage(e->localPos());
    if (!isInsideImage(ip)) return;

    if (e->button() == Qt::LeftButton) {
        // Clicking an existing tracked mask refines that instance.
        const int hit = hitTestInstance(ip);
        if (hit >= 0) {
            emit instanceClicked(hit, ip);
            return;
        }
        m_dragging = true;
        m_dragStart = ip;
        m_dragRect = QRectF(ip, QSizeF(0, 0));
    } else if (e->button() == Qt::RightButton) {
        emit negPointAdded(ip);
    }
}

void VideoCanvas::mouseMoveEvent(QMouseEvent* e) {
    if (m_dragging) {
        const QPointF ip = screenToImage(e->localPos());
        m_dragRect = QRectF(m_dragStart, ip).normalized();
        redraw();
    }
}

void VideoCanvas::mouseReleaseEvent(QMouseEvent* e) {
    if (!m_dragging) {
        QLabel::mouseReleaseEvent(e);
        return;
    }
    m_dragging = false;
    QPointF ip = screenToImage(e->localPos());
    ip.setX(qBound(0.0, ip.x(), m_frame.width() - 1.0));
    ip.setY(qBound(0.0, ip.y(), m_frame.height() - 1.0));

    const double dx = ip.x() - m_dragStart.x();
    const double dy = ip.y() - m_dragStart.y();
    if (dx * dx + dy * dy > 25.0) {
        m_box = QRectF(m_dragStart, ip).normalized();
        m_hasBox = true;
        redraw();
        emit boxDrawn();
    } else {
        // Plain click without a mask hit: positive point (Points mode).
        emit posPointAdded(m_dragStart);
    }
}

void VideoCanvas::paintEvent(QPaintEvent* e) { QLabel::paintEvent(e); }

void VideoCanvas::resizeEvent(QResizeEvent* e) {
    QLabel::resizeEvent(e);
    if (!m_composited.isNull()) {
        setPixmap(QPixmap::fromImage(m_composited.scaled(
                size(), Qt::KeepAspectRatio, Qt::SmoothTransformation)));
    }
}

void VideoCanvas::redraw() {
    if (m_frame.isNull()) return;
    QImage display = m_frame.copy();
    QPainter p(&display);
    drawFrameAndMasks(p, display);
    p.end();
    m_composited = display;
    setPixmap(QPixmap::fromImage(display.scaled(size(), Qt::KeepAspectRatio,
                                                Qt::SmoothTransformation)));
    setText(QString());
}

void VideoCanvas::drawFrameAndMasks(QPainter& p, QImage& canvas) {
    // Mask tints (alpha 0.4, like upstream build_frame_overlay).
    for (int i = 0; i < m_masks.size(); ++i) {
        const QImage& mask = m_masks[i];
        if (mask.isNull()) continue;
        const QColor tint = m_boxes.value(i).color;
        for (int y = 0; y < mask.height() && y < canvas.height(); ++y) {
            const uchar* src = mask.constScanLine(y);
            QRgb* dst = reinterpret_cast<QRgb*>(canvas.scanLine(y));
            for (int x = 0; x < mask.width() && x < canvas.width(); ++x) {
                if (src[x] > 127) {
                    const QRgb base = dst[x];
                    dst[x] = qRgb(static_cast<int>(qRed(base) * 0.6 +
                                                   tint.red() * 0.4),
                                  static_cast<int>(qGreen(base) * 0.6 +
                                                   tint.green() * 0.4),
                                  static_cast<int>(qBlue(base) * 0.6 +
                                                   tint.blue() * 0.4));
                }
            }
        }
    }

    const double scaleX = static_cast<double>(canvas.width()) / m_frame.width();
    const double scaleY =
            static_cast<double>(canvas.height()) / m_frame.height();

    // Instance boxes + labels.
    for (const auto& inst : m_boxes) {
        p.setPen(QPen(inst.color, 2));
        p.setBrush(Qt::NoBrush);
        p.drawRect(QRectF(inst.box.left() * scaleX, inst.box.top() * scaleY,
                          inst.box.width() * scaleX,
                          inst.box.height() * scaleY));
        p.drawText(QPointF(inst.box.left() * scaleX + 2,
                           inst.box.top() * scaleY - 4),
                   QString("#%1 %2").arg(inst.id).arg(inst.score, 0, 'f', 2));
    }

    // Pending prompt points (green positive / red negative).
    p.setPen(QPen(Qt::white, 2));
    p.setBrush(QColor(0, 255, 0, 220));
    for (const auto& pt : m_posPoints) {
        p.drawEllipse(QPointF(pt.x() * scaleX, pt.y() * scaleY), 6, 6);
    }
    p.setBrush(QColor(255, 0, 0, 220));
    for (const auto& pt : m_negPoints) {
        p.drawEllipse(QPointF(pt.x() * scaleX, pt.y() * scaleY), 6, 6);
    }

    // Confirmed box (cyan) and drag-in-progress box (yellow).
    if (m_hasBox) {
        p.setPen(QPen(QColor(0, 255, 255, 220), 3));
        p.setBrush(Qt::NoBrush);
        p.drawRect(QRectF(m_box.left() * scaleX, m_box.top() * scaleY,
                          m_box.width() * scaleX, m_box.height() * scaleY));
    }
    if (m_dragging) {
        p.setPen(QPen(QColor(255, 255, 0, 180), 2));
        p.setBrush(Qt::NoBrush);
        p.drawRect(QRectF(m_dragRect.left() * scaleX, m_dragRect.top() * scaleY,
                          m_dragRect.width() * scaleX,
                          m_dragRect.height() * scaleY));
    }
}
