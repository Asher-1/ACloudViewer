// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QColor>
#include <QHash>
#include <QVector3D>
#include <algorithm>
#include <cmath>

enum class PointCloudColorMode { Semantic, Intensity, Rgb, Instance };

// X-AnyLabeling PointCloudAnnotationDialog._display_colors: labels stay packed
// as (instance << 16) | semantic, independently of the current annotation task.
inline QVector3D pointCloudDisplayColor(quint32 label,
                                        float intensity,
                                        const QColor* rgb,
                                        PointCloudColorMode mode,
                                        const QHash<quint16, QColor>& palette,
                                        double intensityLow = 0,
                                        double intensityHigh = 1,
                                        bool selected = false) {
    const quint16 semantic = quint16(label);
    const auto codeColor = [](quint64 code) {
        return QVector3D(float((code * 37 + 83) % 191 + 64) / 255.0f,
                         float((code * 73 + 41) % 191 + 64) / 255.0f,
                         float((code * 109 + 19) % 191 + 64) / 255.0f);
    };
    const auto semanticColor = [&] {
        const auto entry = palette.constFind(semantic);
        return entry == palette.cend()
                       ? codeColor(semantic)
                       : QVector3D(float(entry->redF()), float(entry->greenF()),
                                   float(entry->blueF()));
    };
    QVector3D color;
    if (mode == PointCloudColorMode::Intensity) {
        double normalized = (double(intensity) - intensityLow) /
                            std::max(intensityHigh - intensityLow, 1e-12);
        if (std::isnan(normalized))
            normalized = 0;
        else if (std::isinf(normalized))
            normalized = normalized > 0 ? 1 : 0;
        const float value = float(normalized);
        color = QVector3D(value, .25f + .65f * value, 1.f - .8f * value);
    } else if (mode == PointCloudColorMode::Rgb) {
        color = rgb ? QVector3D(float(rgb->red()) / 255.0f,
                                float(rgb->green()) / 255.0f,
                                float(rgb->blue()) / 255.0f)
                    : QVector3D(1, 1, 1);
    } else {
        color = semanticColor();
        if (mode == PointCloudColorMode::Instance && semantic)
            color = (label >> 16) ? codeColor(label) : color * .4f;
    }
    if (selected) color = color * .4f + QVector3D(1.f, .85f, .2f) * .6f;
    return color;
}
