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

// Qt
#include <QApplication>
#include <QEvent>
#include <QKeyEvent>
#include <QMessageBox>

// CV_DB_LIB
#include <ecvViewManager.h>

// system
#include <cassert>

ccOverlayDialog::ccOverlayDialog(
        QWidget* parent /*=0*/,
        Qt::WindowFlags flags /*=Qt::FramelessWindowHint | Qt::Tool*/)
    : QDialog(parent, flags), m_associatedWin(nullptr), m_processing(false) {}

ccOverlayDialog::~ccOverlayDialog() { onLinkedWindowDeletion(); }

bool ccOverlayDialog::linkWith(QWidget* win) {
    // same dialog? nothing to do
    if (m_associatedWin == win) {
        return true;
    }

    if (m_associatedWin) {
        // we automatically detach the former dialog
        {
            // Only remove event filter from the associated window and the main
            // window — not every top-level widget, which would intercept keys
            // across unrelated windows in a multi-view setup.
            if (auto* mainWin = qobject_cast<QWidget*>(parent())) {
                mainWin->removeEventFilter(this);
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
        // Scope event filter to the associated window and the main window only,
        // preventing shortcut interception from leaking into other views.
        if (auto* mainWin = qobject_cast<QWidget*>(parent())) {
            mainWin->installEventFilter(this);
        }
        m_associatedWin->installEventFilter(this);
        connect(m_associatedWin, &QObject::destroyed, this,
                &ccOverlayDialog::onLinkedWindowDeletion);
    }

    return true;
}

void ccOverlayDialog::bindToView(ecvGenericGLDisplay* view) {
    m_boundView = view;
}

void ccOverlayDialog::onLinkedWindowDeletion(QObject* object /*=0*/) {
    if (m_processing) stop(false);

    linkWith(nullptr);
}

bool ccOverlayDialog::start() {
    if (m_processing) return false;

    m_processing = true;

    // Auto-relink to the new active widget when the active view changes,
    // so overlay dialogs (qCompass, etc.) follow the active split view.
    // Also re-emit shown() to trigger repositionOverlayDialog so the
    // toolbar moves to the correct corner of the NEW active pane.
    connect(&ecvViewManager::instance(),
            &ecvViewManager::activeViewChanged, this,
            [this](ecvGenericGLDisplay* /*newActive*/,
                   ecvGenericGLDisplay* /*oldActive*/) {
                if (!m_processing) return;
                auto* newWidget = ecvViewManager::instance().activeWidget();
                if (newWidget && newWidget != m_associatedWin) {
                    linkWith(newWidget);
                    emit shown();
                }
            });

    // auto-show
    show();

    // Trigger repositioning now that the dialog is visible.
    // registerOverlayDialog hooks shown() → repositionOverlayDialog, but
    // the initial reposition runs before show() (when isVisible()==false)
    // and the event-filter-based shown() only fires from Show events on
    // the ASSOCIATED widget, not the dialog itself.
    emit shown();

    return true;
}

void ccOverlayDialog::stop(bool accepted) {
    m_processing = false;

    disconnect(&ecvViewManager::instance(),
               &ecvViewManager::activeViewChanged, this, nullptr);

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