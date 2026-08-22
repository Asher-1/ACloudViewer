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
#include <QScrollArea>
#include <QVBoxLayout>

#include <algorithm>

namespace {

/** Event filter that rescales the image pixmap whenever the host dialog is
 *  resized, keeping the image centered and preserving aspect ratio. */
class RescaleFilter : public QObject {
public:
    RescaleFilter(QLabel* label, const QImage& image, QObject* parent)
        : QObject(parent), m_label(label), m_image(image) {}

protected:
    bool eventFilter(QObject* obj, QEvent* event) override {
        if (event->type() == QEvent::Resize && m_label && !m_image.isNull()) {
            auto* dlg = qobject_cast<QDialog*>(obj);
            if (dlg) {
                // Leave room for layout margins (8+8=16) and scrollbar (~18)
                const int margin = 48;
                const int availW = std::max(1, dlg->width() - margin);
                const int availH = std::max(1, dlg->height() - margin);
                // Always rescale to fit the window (even when the source
                // image is smaller), so the preview tracks resizes both ways
                m_label->setPixmap(QPixmap::fromImage(
                        m_image.scaled(availW, availH, Qt::KeepAspectRatio,
                                       Qt::SmoothTransformation)));
            }
        }
        return QObject::eventFilter(obj, event);
    }

private:
    QLabel* m_label;
    QImage m_image;
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
    setPixmap(QPixmap::fromImage(image.scaled(target, Qt::KeepAspectRatio,
                                              Qt::SmoothTransformation)));
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

    QDialog dlg(parent);
    dlg.setWindowTitle(title.isEmpty() ? QObject::tr("Image Preview") : title);

    // Initial scale-to-fit (capped to a reasonable max so the dialog
    // doesn't open full-screen for a 16K image).
    constexpr int kMaxPreviewW = 1200;
    constexpr int kMaxPreviewH = 820;
    QImage display = image;
    if (image.width() > kMaxPreviewW || image.height() > kMaxPreviewH) {
        display = image.scaled(kMaxPreviewW, kMaxPreviewH, Qt::KeepAspectRatio,
                               Qt::SmoothTransformation);
    }

    auto* label = new QLabel;
    label->setPixmap(QPixmap::fromImage(display));
    label->setAlignment(Qt::AlignCenter);

    auto* scroll = new QScrollArea(&dlg);
    scroll->setWidgetResizable(true);
    scroll->setAlignment(Qt::AlignCenter);
    scroll->setWidget(label);

    dlg.resize(qMin(display.width() + 48, 1280),
               qMin(display.height() + 48, 900));

    auto* layout = new QVBoxLayout(&dlg);
    layout->setContentsMargins(8, 8, 8, 8);
    layout->addWidget(scroll);

    // Install event filter so the image rescales when the window is resized
    dlg.installEventFilter(new RescaleFilter(label, image, &dlg));

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
