// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvOverlayDialog.h"

// CV_CORE_LIB
#include <CVLog.h>

// CV_DB_LIB
#include <ecvDisplayTools.h>

// Qt
#include <QApplication>
#include <QEvent>
#include <QKeyEvent>
#include <QMessageBox>

// system
#include <cassert>

ccOverlayDialog::ccOverlayDialog(
        QWidget* parent /*=0*/,
        Qt::WindowFlags flags /*=Qt::FramelessWindowHint | Qt::Tool*/)
    : QDialog(parent, flags), m_associatedWin(nullptr), m_processing(false) {}

ccOverlayDialog::~ccOverlayDialog() { onLinkedWindowDeletion(); }

bool ccOverlayDialog::linkWith(QWidget* win) {
    if (m_processing) {
        CVLog::Warning(
                "[ccOverlayDialog] Can't change associated window while "
                "running/displayed!");
        return false;
    }

    // same dialog? nothing to do
    if (m_associatedWin == win) {
        return true;
    }

    if (m_associatedWin) {
        // we automatically detach the former dialog
        {
            QWidgetList topWidgets = QApplication::topLevelWidgets();
            foreach (QWidget* widget, topWidgets) {
                widget->removeEventFilter(this);
            }
            m_associatedWin->removeEventFilter(this);
            disconnect(m_associatedWin, &QObject::destroyed, this,
                       &ccOverlayDialog::onLinkedWindowDeletion);
            m_associatedWin->disconnect(this);
            m_associatedWin = nullptr;
        }
    }

    m_associatedWin = win;
    if (m_associatedWin) {
        QWidgetList topWidgets = QApplication::topLevelWidgets();
        foreach (QWidget* widget, topWidgets) {
            widget->installEventFilter(this);
        }
        m_associatedWin->installEventFilter(this);
        connect(m_associatedWin, &QObject::destroyed, this,
                &ccOverlayDialog::onLinkedWindowDeletion);
    }

    return true;
}

void ccOverlayDialog::onLinkedWindowDeletion(QObject* object /*=0*/) {
    if (m_processing) stop(false);

    linkWith(nullptr);
}

bool ccOverlayDialog::start() {
    if (m_processing) return false;

    m_processing = true;

    // auto-show
    show();

    return true;
}

void ccOverlayDialog::stop(bool accepted) {
    m_processing = false;

    // auto-hide
    hide();

    linkWith(nullptr);

    emit processFinished(accepted);
}

void ccOverlayDialog::reject() {
    if (QMessageBox::question(
                this, tr("Quit"), tr("Are you sure you want to quit dialog?"),
                QMessageBox::Ok, QMessageBox::Cancel) == QMessageBox::Cancel) {
        return;
    }

    QDialog::reject();

    stop(false);
}

void ccOverlayDialog::addOverridenShortcut(Qt::Key key) {
    m_overriddenKeys.push_back(key);
}

bool ccOverlayDialog::eventFilter(QObject* obj, QEvent* e) {
    if (e->type() == QEvent::KeyPress) {
        QKeyEvent* keyEvent = static_cast<QKeyEvent*>(e);

        if (m_overriddenKeys.contains(keyEvent->key())) {
            emit shortcutTriggered(keyEvent->key());
            return true;
        } else {
            return QDialog::eventFilter(obj, e);
        }
    } else {
        if (e->type() == QEvent::Show) {
            emit shown();
        }

        // standard event processing
        return QDialog::eventFilter(obj, e);
    }
}