// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvExpanderButton.h"

#include <QIcon>
#include <QMouseEvent>

//-----------------------------------------------------------------------------
cvExpanderButton::cvExpanderButton(QWidget* parent)
    : QFrame(parent),
      m_layout(nullptr),
      m_iconLabel(nullptr),
      m_textLabel(nullptr),
      m_checked(false),
      m_pressed(false) {
    // Load icons from resources
    m_checkedPixmap =
            QIcon(":/Resources/images/svg/pqMinus.svg").pixmap(QSize(16, 16));
    m_uncheckedPixmap =
            QIcon(":/Resources/images/svg/pqPlus.svg").pixmap(QSize(16, 16));

    setupUi();
    updateIcon();

#if defined(Q_WS_WIN) || defined(Q_OS_WIN)
    setFrameShadow(QFrame::Sunken);
#endif
}

//-----------------------------------------------------------------------------
cvExpanderButton::~cvExpanderButton() = default;

//-----------------------------------------------------------------------------
void cvExpanderButton::setupUi() {
    // Setup frame properties - ParaView style
    setFrameShape(QFrame::StyledPanel);
    setFrameShadow(QFrame::Raised);
    setLineWidth(1);

    // Create layout
    m_layout = new QHBoxLayout(this);
    m_layout->setContentsMargins(6, 4, 6, 4);
    m_layout->setSpacing(6);

    // Icon label
    m_iconLabel = new QLabel(this);
    m_iconLabel->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
    m_layout->addWidget(m_iconLabel, 0);

    // Text label
    m_textLabel = new QLabel(this);
    m_textLabel->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    QFont font = m_textLabel->font();
    font.setBold(true);
    m_textLabel->setFont(font);
    m_layout->addWidget(m_textLabel, 1);

    setLayout(m_layout);

    // Set cursor to indicate clickable
    setCursor(Qt::PointingHandCursor);

    // Set minimum size
    setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
}

//-----------------------------------------------------------------------------
void cvExpanderButton::updateIcon() {
    if (m_iconLabel) {
        m_iconLabel->setPixmap(m_checked ? m_checkedPixmap : m_uncheckedPixmap);
    }
}

//-----------------------------------------------------------------------------
void cvExpanderButton::toggle() { setChecked(!m_checked); }

//-----------------------------------------------------------------------------
void cvExpanderButton::setChecked(bool checked) {
    if (m_checked == checked) {
        return;
    }

    m_checked = checked;
    updateIcon();
    emit toggled(m_checked);
}

//-----------------------------------------------------------------------------
void cvExpanderButton::setText(const QString& text) {
    if (m_textLabel) {
        m_textLabel->setText(text);
    }
}

//-----------------------------------------------------------------------------
QString cvExpanderButton::text() const {
    return m_textLabel ? m_textLabel->text() : QString();
}

//-----------------------------------------------------------------------------
void cvExpanderButton::mousePressEvent(QMouseEvent* evt) {
    if (evt->button() == Qt::LeftButton && evt->buttons() == Qt::LeftButton) {
        m_pressed = true;
    }
}

//-----------------------------------------------------------------------------
void cvExpanderButton::mouseReleaseEvent(QMouseEvent* evt) {
    if (m_pressed && evt->button() == Qt::LeftButton) {
        m_pressed = false;
        toggle();
    }
}
