// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QApplication>
#include <QElapsedTimer>
#include <QKeyEvent>
#include <QKeySequence>
#include <QObject>
#include <QThread>
#include <QWidget>

namespace cloudviewer::tests {

// Qt supplies event delivery; GoogleTest owns assertions and test reporting.
inline void pressSequence(QWidget* target, const QKeySequence& sequence) {
    for (int i = 0; i < sequence.count(); ++i) {
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
        const int combined = sequence[i].toCombined();
#else
        const int combined = sequence[i];
#endif
        const int key = combined & ~int(Qt::KeyboardModifierMask);
        const auto modifiers =
                Qt::KeyboardModifiers(combined & int(Qt::KeyboardModifierMask));
        QString text;
        if (!(modifiers &
              (Qt::ControlModifier | Qt::AltModifier | Qt::MetaModifier)) &&
            key >= Qt::Key_Space && key <= Qt::Key_AsciiTilde) {
            text = QChar(key);
            if (!(modifiers & Qt::ShiftModifier)) text = text.toLower();
        }
        QKeyEvent down(QEvent::KeyPress, key, modifiers, text);
        QApplication::sendEvent(target, &down);
        QKeyEvent up(QEvent::KeyRelease, key, modifiers, text);
        QApplication::sendEvent(target, &up);
        QApplication::processEvents();
    }
}

inline bool waitForWindowActive(QWidget* widget, int timeoutMs = 1000) {
    QElapsedTimer timer;
    timer.start();
    do {
        QApplication::processEvents();
        if (widget->isActiveWindow()) return true;
        QThread::msleep(1);
    } while (timer.elapsed() < timeoutMs);
    return false;
}

class SignalCounter : public QObject {
public:
    template <typename Sender, typename Signal>
    SignalCounter(Sender* sender, Signal signal) {
        QObject::connect(sender, signal, this, [this]() { ++m_count; });
    }
    int count() const { return m_count; }

private:
    int m_count = 0;
};

}  // namespace cloudviewer::tests
