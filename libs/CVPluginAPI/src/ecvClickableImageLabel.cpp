// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvClickableImageLabel.h"

#include <QDialog>
#include <QHBoxLayout>
#include <QMouseEvent>
#include <QPainter>
#include <QScrollArea>
#include <QScrollBar>
#include <QVBoxLayout>
#include <QWheelEvent>
#include <algorithm>
#include <cmath>

namespace {

/** Strided opacity probe: true when the image carries an alpha channel with
 *  at least one non-opaque sample. Opaque images (regular photos, meshes
 *  rendered on opaque backdrops) must pass through byte-identical, so the
 *  checker backdrop below never costs them a repaint. */
bool hasMeaningfulAlpha(const QImage& image) {
    if (!image.hasAlphaChannel()) return false;
    const int w = image.width();
    const int h = image.height();
    const int64_t count = int64_t(w) * h;
    const int64_t samples = std::min<int64_t>(count, 4096);
    const int64_t stride = std::max<int64_t>(1, count / samples);
    for (int64_t idx = 0; idx < count; idx += stride) {
        if (image.pixelColor(int(idx % w), int(idx / w)).alpha() < 255) {
            return true;
        }
    }
    return false;
}

/** Transparency indicator composited under RGBA previews. Without it a matted
 *  image (e.g. an RMBG cut-out) paints its removed background in the host
 *  widget's color and reads as "the original, unchanged". The pattern is
 *  generated at display time: the thumbnail path composites in display
 *  pixels so the tiles stay readable at thumbnail scale, while the zoomable
 *  preview bakes the pattern into the image once so the tiles scale with the
 *  content under zoom and pan (the Photoshop-style document backdrop). */
QImage composeCheckerBackdrop(const QImage& image, int tile) {
    if (!hasMeaningfulAlpha(image) || tile <= 0) return image;
    QImage backdrop(image.size(), QImage::Format_ARGB32_Premultiplied);
    QPainter p(&backdrop);
    // Two neutral greys: light enough to keep white subjects readable, dark
    // enough to expose a white removed background.
    const QColor light(255, 255, 255);
    const QColor dark(204, 204, 204);
    p.fillRect(backdrop.rect(), light);
    for (int y = 0; y < image.height(); y += tile) {
        for (int x = 0; x < image.width(); x += tile) {
            if (((x / tile) + (y / tile)) & 1) {
                p.fillRect(x, y, std::min(tile, image.width() - x),
                           std::min(tile, image.height() - y), dark);
            }
        }
    }
    p.drawImage(0, 0, image);
    p.end();
    return backdrop;
}

/** Interaction controller behind the enlarged-preview dialog: fit-to-window
 *  on open, wheel zoom anchored at the cursor, drag panning, double-click
 *  reset, and an automatic re-fit while the user has not zoomed manually.
 *  Installed on the scroll viewport (wheel + drag) and the host window
 *  (show/resize re-fit). */
class PreviewZoomController : public QObject {
public:
    PreviewZoomController(QLabel* label, QScrollArea* scroll, QImage image)
        : QObject(scroll),
          m_label(label),
          m_scroll(scroll),
          m_image(std::move(image)) {
        // Working-copy cap: smooth-rescaling a 16K source on every wheel
        // tick would stall the GUI; ~4096 px keeps far more detail than the
        // dialog can display at any supported zoom.
        constexpr int kMaxWorkPx = 4096;
        if (m_image.width() > kMaxWorkPx || m_image.height() > kMaxWorkPx) {
            m_image =
                    m_image.scaled(kMaxWorkPx, kMaxWorkPx, Qt::KeepAspectRatio,
                                   Qt::SmoothTransformation);
        }
        m_scroll->viewport()->installEventFilter(this);
        if (QWidget* win = m_scroll->window()) {
            win->installEventFilter(this);
        }
        fitToWindow();
    }

protected:
    bool eventFilter(QObject* watched, QEvent* event) override {
        if (watched == m_scroll->viewport()) {
            switch (event->type()) {
                case QEvent::Wheel:
                    zoomAt(QCursor::pos(),
                           static_cast<QWheelEvent*>(event)->angleDelta().y() >=
                                           0
                                   ? 1.25
                                   : 0.8);
                    return true;
                case QEvent::MouseButtonPress:
                    if (static_cast<QMouseEvent*>(event)->button() ==
                        Qt::LeftButton) {
                        m_panOrigin = QCursor::pos();
                        m_panH = m_scroll->horizontalScrollBar()->value();
                        m_panV = m_scroll->verticalScrollBar()->value();
                        m_panning = true;
                        m_scroll->viewport()->setCursor(Qt::ClosedHandCursor);
                        return true;
                    }
                    break;
                case QEvent::MouseMove:
                    if (m_panning) {
                        const QPoint delta = QCursor::pos() - m_panOrigin;
                        m_scroll->horizontalScrollBar()->setValue(m_panH -
                                                                  delta.x());
                        m_scroll->verticalScrollBar()->setValue(m_panV -
                                                                delta.y());
                        return true;
                    }
                    break;
                case QEvent::MouseButtonRelease:
                    if (m_panning &&
                        static_cast<QMouseEvent*>(event)->button() ==
                                Qt::LeftButton) {
                        m_panning = false;
                        m_scroll->viewport()->unsetCursor();
                        return true;
                    }
                    break;
                case QEvent::MouseButtonDblClick:
                    fitToWindow();
                    return true;
                default:
                    break;
            }
            return QObject::eventFilter(watched, event);
        }
        // Host window shown / resized while the user has not zoomed: keep
        // the fit-to-window behavior of the plain preview dialog.
        if (event->type() == QEvent::Show ||
            (event->type() == QEvent::Resize && !m_userZoomed)) {
            fitToWindow();
        }
        return QObject::eventFilter(watched, event);
    }

private:
    void fitToWindow() {
        m_userZoomed = false;
        const QSize avail = m_scroll->viewport()->size() - QSize(16, 16);
        const double zw = double(avail.width()) / m_image.width();
        const double zh = double(avail.height()) / m_image.height();
        m_zoom = std::min(zw, zh);
        apply();
    }

    void zoomAt(const QPoint& cursorGlobalPos, double factor) {
        const double newZoom = std::min(8.0, std::max(0.05, m_zoom * factor));
        if (std::fabs(newZoom - m_zoom) < 1e-9 || m_image.isNull()) return;
        // Anchor: the image point currently under the cursor must stay
        // under the cursor after the rescale.
        const QPoint viewPos =
                m_scroll->viewport()->mapFromGlobal(cursorGlobalPos);
        const QSize oldSize = m_currentSize;
        const QPoint contentPos(
                m_scroll->horizontalScrollBar()->value() + viewPos.x(),
                m_scroll->verticalScrollBar()->value() + viewPos.y());
        m_zoom = newZoom;
        m_userZoomed = true;
        apply();
        if (oldSize.isEmpty()) return;
        const QPoint newContentPos(
                int(contentPos.x() * double(m_currentSize.width()) /
                    oldSize.width()),
                int(contentPos.y() * double(m_currentSize.height()) /
                    oldSize.height()));
        m_scroll->horizontalScrollBar()->setValue(newContentPos.x() -
                                                  viewPos.x());
        m_scroll->verticalScrollBar()->setValue(newContentPos.y() -
                                                viewPos.y());
    }

    void apply() {
        m_currentSize = QSize(std::max(1, qRound(m_image.width() * m_zoom)),
                              std::max(1, qRound(m_image.height() * m_zoom)));
        m_label->setPixmap(QPixmap::fromImage(m_image.scaled(
                m_currentSize, Qt::KeepAspectRatio, Qt::SmoothTransformation)));
        m_label->resize(m_currentSize);
    }

    QLabel* m_label;
    QScrollArea* m_scroll;
    QImage m_image;
    double m_zoom = 1.0;
    QSize m_currentSize;
    bool m_userZoomed = false;
    bool m_panning = false;
    QPoint m_panOrigin;
    int m_panH = 0;
    int m_panV = 0;
};

}  // namespace

ecvClickableImageLabel::ecvClickableImageLabel(QWidget* parent)
    : QLabel(parent) {
    setAlignment(Qt::AlignCenter);
    updateInteractiveState();
}

void ecvClickableImageLabel::setPreviewImage(const QImage& image,
                                             int thumbSize) {
    setPreviewImage(image, QSize(thumbSize, thumbSize));
}

void ecvClickableImageLabel::setPreviewImage(const QImage& image,
                                             const QSize& displaySize) {
    m_fullImage = image;
    if (image.isNull()) {
        clearPreview();
        return;
    }
    const QSize target = displaySize.isValid() && !displaySize.isEmpty()
                                 ? displaySize
                                 : QSize(96, 96);
    // Removed backgrounds must stay visible at thumbnail scale: composite
    // the checker in display pixels (post-scale), not in source pixels.
    QImage display =
            image.scaled(target, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    display = composeCheckerBackdrop(display, 8);
    setPixmap(QPixmap::fromImage(display));
    if (displaySize.isValid() && !displaySize.isEmpty()) {
        setFixedSize(target);
        setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
        setMinimumSize(target);
        setMaximumSize(target);
    }
    updateInteractiveState();
}

void ecvClickableImageLabel::setPreviewPixmap(const QPixmap& pixmap,
                                              int thumbSize) {
    if (pixmap.isNull()) {
        clearPreview();
        return;
    }
    setPreviewImage(pixmap.toImage(), thumbSize);
}

void ecvClickableImageLabel::clearPreview() {
    m_fullImage = QImage();
    clear();
    updateInteractiveState();
}

void ecvClickableImageLabel::showEnlargedImage(QWidget* parent,
                                               const QImage& image,
                                               const QString& title) {
    if (image.isNull()) {
        return;
    }
    // Bake the transparency backdrop into the zoom target once: the tiles
    // then scale with the content under zoom and pan (a document backdrop)
    // instead of staying a screen-fixed grid behind the image. Sizing keeps
    // the original dimensions — the composite does not change them.
    const int bakeTile =
            std::max(16, std::min(image.width(), image.height()) / 64);
    const QImage zoomable = composeCheckerBackdrop(image, bakeTile);

    QDialog dlg(parent);
    dlg.setWindowTitle(title.isEmpty() ? QObject::tr("Image Preview") : title);

    auto* label = new QLabel;
    label->setAlignment(Qt::AlignCenter);
    label->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);

    auto* scroll = new QScrollArea(&dlg);
    // The zoom controller owns the displayed size; the scroll area must not
    // rescale the widget behind its back.
    scroll->setWidgetResizable(false);
    scroll->setAlignment(Qt::AlignCenter);
    scroll->setWidget(label);

    auto* hint = new QLabel(
            QObject::tr("Wheel: zoom · Drag: pan · Double-click: fit"), &dlg);
    hint->setAlignment(Qt::AlignCenter);
    hint->setStyleSheet(
            QStringLiteral("color: palette(mid); font-size: 11px;"));

    dlg.resize(qMin(image.width() + 48, 1280), qMin(image.height() + 48, 900));

    auto* layout = new QVBoxLayout(&dlg);
    layout->setContentsMargins(8, 8, 8, 8);
    layout->setSpacing(4);
    layout->addWidget(scroll, 1);
    layout->addWidget(hint);

    // Owns wheel zoom, drag pan, and the fit-on-show/resize behavior.
    new PreviewZoomController(label, scroll, zoomable);

    dlg.exec();
}

void ecvClickableImageLabel::mousePressEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton && !m_fullImage.isNull()) {
        showEnlargedImage(window(), m_fullImage, m_dialogTitle);
        event->accept();
        return;
    }
    QLabel::mousePressEvent(event);
}

void ecvClickableImageLabel::enterEvent(QEvent* event) {
    if (!m_fullImage.isNull()) {
        setToolTip(tr("Click to enlarge"));
    }
    QLabel::enterEvent(event);
}

void ecvClickableImageLabel::leaveEvent(QEvent* event) {
    if (toolTip() == tr("Click to enlarge")) {
        setToolTip(QString());
    }
    QLabel::leaveEvent(event);
}

void ecvClickableImageLabel::updateInteractiveState() {
    setCursor(m_fullImage.isNull() ? Qt::ArrowCursor : Qt::PointingHandCursor);
}

QWidget* ecvClickableImageLabel::wrapWithTapToPreviewHint(
        ecvClickableImageLabel* label, QWidget* parent) {
    if (!label) {
        return nullptr;
    }
    auto* row = new QWidget(parent);
    auto* layout = new QHBoxLayout(row);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(4);

    auto* hint = new QLabel(QObject::tr("Tap to preview"), row);
    hint->setAlignment(Qt::AlignVCenter | Qt::AlignRight);
    hint->setStyleSheet(
            QStringLiteral("color: palette(mid); font-size: 11px;"));
    hint->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Preferred);

    auto* arrow = new QLabel(QStringLiteral("\u2192"), row);
    arrow->setAlignment(Qt::AlignVCenter);
    arrow->setStyleSheet(
            QStringLiteral("color: palette(mid); font-size: 13px; "
                           "font-weight: bold; padding-bottom: 1px;"));
    arrow->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Preferred);

    layout->addWidget(hint);
    layout->addWidget(arrow);
    layout->addWidget(label);
    return row;
}
