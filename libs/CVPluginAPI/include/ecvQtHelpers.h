// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_QT_HELPERS_HEADER
#define ECV_QT_HELPERS_HEADER

// Qt
#include <QAbstractButton>

class ccQtHelpers {
public:
    //! Sets a button background color
    inline static void SetButtonColor(QAbstractButton* button,
                                      const QColor& col) {
        if (button)
            button->setStyleSheet(
                    QString("* { background-color: rgb(%1,%2,%3) }")
                            .arg(col.red())
                            .arg(col.green())
                            .arg(col.blue()));
    }
};

#endif  // ECV_QT_HELPERS_HEADER