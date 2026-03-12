// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/**
 * @file colorseries.h
 * @brief Color series generator with predefined color schemes.
 */

#include <QColor>

#include "qVTK.h"

namespace Utils {

class ColorSeriesPrivate;

/**
 * @class ColorSeries
 * @brief Generates ordered color sequences from predefined schemes
 *        (Spectrum, Warm, Cool, Blues, WildFlower, Citrus).
 */
class QVTK_ENGINE_LIB_API ColorSeries {
public:
    enum Scheme { Spectrum, Warm, Cool, Blues, WildFlower, Citrus };

    ColorSeries();
    ~ColorSeries();

    void setScheme(Scheme scheme);
    Scheme scheme() const;

    QColor color(int index) const;
    QColor nextColor() const;

private:
    ColorSeriesPrivate* d_ptr;
    Q_DISABLE_COPY(ColorSeries)
};

}  // namespace Utils
