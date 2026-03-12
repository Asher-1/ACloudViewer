// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/**
 * @file penstylebox.h
 * @brief Combo box widget for selecting Qt pen styles.
 */

#include <QComboBox>

#include "qVTK.h"

namespace Widgets {

/**
 * @class PenStyleBox
 * @brief QComboBox displaying Qt::PenStyle options with visual previews.
 */
class QVTK_ENGINE_LIB_API PenStyleBox : public QComboBox {
    Q_OBJECT

public:
    PenStyleBox(QWidget* parent = 0);
    void setStyle(const Qt::PenStyle& style);
    Qt::PenStyle style() const;

    static int styleIndex(const Qt::PenStyle& style);
    static Qt::PenStyle penStyle(int index);

private:
    static const Qt::PenStyle patterns[];
};

}  // namespace Widgets
