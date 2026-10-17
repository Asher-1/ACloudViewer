// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QWidget>

class QLabel;
class QTimer;

/** Focus-free toast matching the source Popup widget: frameless,
 *  always-on-top, click-through, and auto-closing after a fixed interval.
 *  Used for review-loop progress so the toast never steals keyboard focus
 *  from the annotation canvas. */
class AnnotationPopup final : public QWidget {
    Q_OBJECT
public:
    explicit AnnotationPopup(QWidget* parent = nullptr);

    /** Show centered near the top of the parent window and auto-hide. */
    void showMessage(const QString& text, int msec = 1800);

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    QLabel* m_icon = nullptr;
    QLabel* m_text = nullptr;
    QTimer* m_timer = nullptr;
    int m_msec = 1800;
};
