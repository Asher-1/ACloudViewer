// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Qt
#include <QAbstractButton>
#include <QThread>

class ccQtHelpers {
public:
    //! Sets a button background color
    inline static void SetButtonColor(QAbstractButton* button,
                                      const QColor& col) {
        if (button) {
            // This is a data-color swatch, not application chrome. Keep its
            // text contrast explicit so a prior dark QSS palette cannot leak
            // into a white swatch after returning to a native window style.
            const int luminance =
                    col.red() * 299 + col.green() * 587 + col.blue() * 114;
            button->setStyleSheet(
                    QString("QAbstractButton { background-color: "
                            "rgb(%1,%2,%3); "
                            "color: %4; }")
                            .arg(col.red())
                            .arg(col.green())
                            .arg(col.blue())
                            .arg(luminance >= 128000 ? "black" : "white"));
        }
    }

    //! Returns the ideal number of threads/cores
    static int GetMaxThreadCount(int idealThreadCount) {
        if (idealThreadCount <= 4) {
            return idealThreadCount;
        } else if (idealThreadCount <= 8) {
            return idealThreadCount - 1;
        } else {
            return idealThreadCount - 2;
        }
    }

    //! Returns the ideal number of threads/cores with Qt Concurrent
    static int GetMaxThreadCount() {
        return GetMaxThreadCount(QThread::idealThreadCount());
    }
};
