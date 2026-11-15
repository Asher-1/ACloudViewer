// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "YOLOVisualPromptLabel.h"

#include <QInputDialog>
#include <QLineEdit>
#include <QMouseEvent>
#include <QPainter>
#include <QResizeEvent>
#include <algorithm>

namespace {

// Overlay style (kept in one place so boxes and labels stay consistent).
const char* kBoxColor = "#ff9d2e";
const char* kLabelBgColor = "#b34700";

// Minimum drawn extent (widget pixels) below which a drag is treated as a
// click and discarded.
constexpr int kMinBoxExtent = 4;

}  // namespace

YOLOVisualPromptLabel::YOLOVisualPromptLabel(QWidget* parent) : QLabel(parent) {
    setAlignment(Qt::AlignCenter);
    setMouseTracking(false);
}

void YOLOVisualPromptLabel::setPromptImage(const QImage& image,
                                           const QSize& displaySize) {
    m_image = image;
    m_boxes.clear();
    m_boxNames.clear();
    m_rubberBandActive = false;
    m_displayTarget = displaySize;
    if (m_image.isNull()) {
        clear();
        m_pixmapRect = QRectF();
        m_imageSize = QSizeF();
        update();
        return;
    }
    m_imageSize = QSizeF(m_image.size());
    refreshPixmap();
}

void YOLOVisualPromptLabel::refreshPixmap() {
    if (m_image.isNull()) return;
    // Fixed target (legacy small-canvas use) or the live widget size (the
    // inline canvas that resizes with the dialog layout).
    const QSize target = m_displayTarget.isValid() && !m_displayTarget.isEmpty()
                                 ? m_displayTarget
                                 : size();
    if (target.isEmpty()) return;  // not laid out yet; resizeEvent re-fits
    // KeepAspectRatio + centered alignment (QLabel default): the displayed
    // rect is derived from the actual pixmap, so the coordinate mapping
    // stays exact for any DPI/size combination.
    const QPixmap pm = QPixmap::fromImage(m_image.scaled(
            target, Qt::KeepAspectRatio, Qt::SmoothTransformation));
    const QSizeF pmSize(pm.size());
    const QSizeF widgetSize(size());
    const QPointF topLeft(
            qMax<qreal>(0.0, (widgetSize.width() - pmSize.width()) / 2.0),
            qMax<qreal>(0.0, (widgetSize.height() - pmSize.height()) / 2.0));
    m_pixmapRect = QRectF(topLeft, pmSize);
    setPixmap(pm);
    update();
}

void YOLOVisualPromptLabel::clearPrompt() {
    m_image = QImage();
    m_imageSize = QSizeF();
    m_pixmapRect = QRectF();
    m_boxes.clear();
    m_boxNames.clear();
    m_rubberBandActive = false;
    clear();
    update();
}

void YOLOVisualPromptLabel::setDrawingEnabled(bool enabled) {
    m_drawingEnabled = enabled;
    if (!enabled) {
        m_rubberBandActive = false;
        setCursor(Qt::ArrowCursor);
        update();
    } else {
        setCursor(Qt::CrossCursor);
    }
}

void YOLOVisualPromptLabel::setBoxes(const QList<QRectF>& boxes) {
    m_boxes = boxes;
    m_boxNames.clear();  // programmatic box set: drop stale names
    emit boxesChanged();
    update();
}

QStringList YOLOVisualPromptLabel::boxNames() const {
    // Index-aligned with boxes(); unnamed prompts stay empty so the
    // backend's positional objectN label applies.
    QStringList names;
    names.reserve(m_boxes.size());
    for (int i = 0; i < m_boxes.size(); ++i) {
        names.append(i < m_boxNames.size() ? m_boxNames.at(i) : QString());
    }
    return names;
}

void YOLOVisualPromptLabel::removeLast() {
    if (m_boxes.isEmpty()) return;
    m_boxes.removeLast();
    if (!m_boxNames.isEmpty()) m_boxNames.removeLast();
    emit boxesChanged();
    update();
}

void YOLOVisualPromptLabel::clearBoxes() {
    if (m_boxes.isEmpty()) return;
    m_boxes.clear();
    m_boxNames.clear();
    emit boxesChanged();
    update();
}

QPointF YOLOVisualPromptLabel::toImageCoords(const QPointF& widgetPos) const {
    if (m_pixmapRect.isEmpty() || m_imageSize.isEmpty()) return QPointF();
    return QPointF(m_imageSize.width() * (widgetPos.x() - m_pixmapRect.left()) /
                           m_pixmapRect.width(),
                   m_imageSize.height() * (widgetPos.y() - m_pixmapRect.top()) /
                           m_pixmapRect.height());
}

QRectF YOLOVisualPromptLabel::toWidgetRect(const QRectF& imageRect) const {
    if (m_pixmapRect.isEmpty() || m_imageSize.isEmpty()) return QRectF();
    const qreal sx = m_pixmapRect.width() / m_imageSize.width();
    const qreal sy = m_pixmapRect.height() / m_imageSize.height();
    return QRectF(m_pixmapRect.left() + imageRect.left() * sx,
                  m_pixmapRect.top() + imageRect.top() * sy,
                  imageRect.width() * sx, imageRect.height() * sy);
}

void YOLOVisualPromptLabel::updateLabelFromBoxes() { update(); }

void YOLOVisualPromptLabel::paintEvent(QPaintEvent* event) {
    QLabel::paintEvent(event);
    if (m_imageSize.isEmpty()) return;

    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    const QFontMetrics fm(font());
    QPen pen{QColor(kBoxColor)};
    pen.setWidthF(2.0);

    auto drawBox = [&](const QRectF& imageRect, const QString& label) {
        const QRectF widgetRect = toWidgetRect(imageRect);
        if (widgetRect.isEmpty()) return;
        painter.setPen(pen);
        painter.setBrush(Qt::NoBrush);
        painter.drawRect(widgetRect);
        if (!label.isEmpty()) {
            const QRect textRect =
                    fm.boundingRect(label).adjusted(-3, -1, 3, 1);
            const QPointF anchor(
                    qMax<qreal>(widgetRect.left(), 0.0),
                    qMax<qreal>(widgetRect.top() - textRect.height() - 2, 0.0));
            painter.fillRect(QRectF(anchor, textRect.size()),
                             QColor(0, 0, 0, 150));
            painter.setPen(Qt::white);
            painter.drawText(QRectF(anchor, textRect.size()), Qt::AlignCenter,
                             label);
        }
    };

    for (int i = 0; i < m_boxes.size(); ++i) {
        // User-assigned name when set; official positional label otherwise.
        const QString name = m_boxNames.value(i);
        drawBox(m_boxes[i],
                name.isEmpty() ? QStringLiteral("object%1").arg(i) : name);
    }
    if (m_rubberBandActive) {
        pen.setStyle(Qt::DashLine);
        painter.setPen(pen);
        painter.drawRect(toWidgetRect(m_rubberBandCurrent));
    }
}

void YOLOVisualPromptLabel::mousePressEvent(QMouseEvent* event) {
    if (!m_drawingEnabled || event->button() != Qt::LeftButton ||
        m_imageSize.isEmpty() || !m_pixmapRect.contains(event->pos())) {
        QLabel::mousePressEvent(event);
        return;
    }
    m_rubberBandActive = true;
    m_rubberBandStart = toImageCoords(event->pos());
    m_rubberBandCurrent = QRectF(m_rubberBandStart, m_rubberBandStart);
    update();
}

void YOLOVisualPromptLabel::mouseMoveEvent(QMouseEvent* event) {
    if (!m_rubberBandActive) {
        QLabel::mouseMoveEvent(event);
        return;
    }
    const QPointF img = toImageCoords(event->pos());
    m_rubberBandCurrent =
            QRectF(QPointF(std::min(m_rubberBandStart.x(), img.x()),
                           std::min(m_rubberBandStart.y(), img.y())),
                   QPointF(std::max(m_rubberBandStart.x(), img.x()),
                           std::max(m_rubberBandStart.y(), img.y())));
    update();
}

void YOLOVisualPromptLabel::mouseReleaseEvent(QMouseEvent* event) {
    if (!m_rubberBandActive || event->button() != Qt::LeftButton) {
        QLabel::mouseReleaseEvent(event);
        return;
    }
    m_rubberBandActive = false;
    const QPointF img = toImageCoords(event->pos());
    QRectF box(QPointF(std::min(m_rubberBandStart.x(), img.x()),
                       std::min(m_rubberBandStart.y(), img.y())),
               QPointF(std::max(m_rubberBandStart.x(), img.x()),
                       std::max(m_rubberBandStart.y(), img.y())));
    // Discard tiny drags (clicks) and out-of-canvas slivers.
    const qreal sx = m_pixmapRect.width() / m_imageSize.width();
    const qreal sy = m_pixmapRect.height() / m_imageSize.height();
    const bool bigEnough = box.width() * sx >= kMinBoxExtent &&
                           box.height() * sy >= kMinBoxExtent;
    if (bigEnough && m_pixmapRect.intersects(toWidgetRect(box))) {
        // Clamp to the image bounds.
        box = box.intersected(QRectF(QPointF(0, 0), m_imageSize));
        if (!box.isEmpty()) {
            m_boxes.append(box);
            emit boxesChanged();
        }
    }
    m_rubberBandCurrent = QRectF();
    update();
}

void YOLOVisualPromptLabel::mouseDoubleClickEvent(QMouseEvent* event) {
    if (!m_drawingEnabled || event->button() != Qt::LeftButton ||
        m_imageSize.isEmpty()) {
        QLabel::mouseDoubleClickEvent(event);
        return;
    }
    // The double click also started a rubber band on press; cancel it.
    m_rubberBandActive = false;
    const QPointF widgetPos = event->pos();
    for (int i = m_boxes.size() - 1; i >= 0; --i) {
        if (!toWidgetRect(m_boxes[i]).contains(widgetPos)) continue;
        bool ok = false;
        const QString text = QInputDialog::getText(
                this, tr("Prompt name"),
                tr("Name for this example (empty = object%1)").arg(i),
                QLineEdit::Normal, m_boxNames.value(i), &ok);
        if (ok) {
            while (m_boxNames.size() < m_boxes.size()) {
                m_boxNames.append(QString());
            }
            m_boxNames[i] = text.trimmed();
            emit boxesChanged();
        }
        update();
        return;
    }
    // Not on any box: fall through to the default handling.
    QLabel::mouseDoubleClickEvent(event);
}

void YOLOVisualPromptLabel::resizeEvent(QResizeEvent* event) {
    QLabel::resizeEvent(event);
    // Re-fit the pixmap and the mapping rect when the widget resizes (the
    // inline canvas tracks the dialog layout; a fixed display target never
    // changes, so nothing to do there).
    if (m_image.isNull()) return;
    if (m_displayTarget.isValid() && !m_displayTarget.isEmpty()) return;
    refreshPixmap();
}