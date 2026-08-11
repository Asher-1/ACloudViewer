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
