// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "AnnotationPopup.h"

#include <QEvent>
#include <QHBoxLayout>
#include <QIcon>
#include <QLabel>
#include <QPainter>
#include <QPainterPath>
#include <QTimer>

AnnotationPopup::AnnotationPopup(QWidget* parent) : QWidget(parent) {
    // Match the source Popup flags: frameless, above everything, and never
    // focus-activating or input-accepting so the canvas keeps all events.
    setWindowFlags(Qt::WindowType::FramelessWindowHint |
                   Qt::WindowType::WindowStaysOnTopHint |
                   Qt::WindowType::WindowDoesNotAcceptFocus |
                   Qt::WindowType::WindowTransparentForInput);
    setAttribute(Qt::WidgetAttribute::WA_TranslucentBackground);
    setAttribute(Qt::WidgetAttribute::WA_ShowWithoutActivating);

    auto* layout = new QHBoxLayout(this);
    layout->setContentsMargins(12, 8, 12, 8);
    layout->setSpacing(6);
    m_icon = new QLabel(this);
    m_icon->setPixmap(QIcon(QStringLiteral(":/CC/plugin/qAnyLabeling/"
                                           "images/copy-green.svg"))
                              .pixmap(16, 16));
    m_icon->setFixedSize(16, 16);
    m_text = new QLabel(this);
    m_text->setTextFormat(Qt::PlainText);
    m_text->setAlignment(Qt::AlignVCenter | Qt::AlignLeft);
    layout->addWidget(m_icon);
    layout->addWidget(m_text);
    setLayout(layout);

    m_timer = new QTimer(this);
    m_timer->setSingleShot(true);
    connect(m_timer, &QTimer::timeout, this, &QWidget::hide);
}

void AnnotationPopup::showMessage(const QString& text, int msec) {
    m_msec = msec;
    m_text->setText(text);
    adjustSize();
    const QWidget* host = parentWidget() ? parentWidget() : window();
    const QRect hostRect = host->geometry();
    const int width = sizeHint().width();
    const int height = qMax(sizeHint().height(), 36);
    const QPoint position(hostRect.x() + (hostRect.width() - width) / 2,
                          hostRect.y() + 100);
    setGeometry(QRect(position, QSize(width, height)));
    show();
    raise();
    m_timer->start(m_msec);
}

void AnnotationPopup::paintEvent(QPaintEvent* event) {
    Q_UNUSED(event);
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    const QPalette& palette = this->palette();
    const QColor background =
            palette.color(QPalette::ColorGroup::Active, QPalette::ToolTipBase);
    QColor text =
            palette.color(QPalette::ColorGroup::Active, QPalette::ToolTipText);
    if (!background.isValid() || !text.isValid()) {
        // Offscreen palettes may omit tooltip roles; fall back to theme
        // colors that stay readable on both light and dark surfaces.
        painter.fillPath(QPainterPath(), Qt::gray);
        return;
    }
    m_text->setStyleSheet(QStringLiteral("background: transparent; color: %1")
                                  .arg(text.name()));
    QPainterPath path;
    path.addRoundedRect(QRectF(rect()), 10, 10);
    painter.fillPath(path, background);
}
