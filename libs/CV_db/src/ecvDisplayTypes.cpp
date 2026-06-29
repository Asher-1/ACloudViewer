// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvDisplayTypes.h"

#include <CVLog.h>

#include <algorithm>

#include "ecvDisplayTools.h"

void ecvHotZone::updateInternalVariables(QWidget* win) {
    if (win) {
        font = win->font();
        pixelDeviceRatio = win->devicePixelRatioF();
        // Match CloudCompare: scale font/metrics by per-widget DPR so layout
        // aligns with ImageVis / vtkRenderWindow physical pixel space.
        static constexpr int kBaseFontPt = 11;
        static constexpr int kBaseMargin = 10;
        static constexpr int kBaseIconSize = 14;
        font.setPointSize(static_cast<int>(kBaseFontPt * pixelDeviceRatio));
        margin = static_cast<int>(kBaseMargin * pixelDeviceRatio);
        iconSize = static_cast<int>(kBaseIconSize * pixelDeviceRatio);
        font.setBold(true);
        CVLog::PrintDebug(QString("hotZone DPR: %1 fontPt: %2 margin: %3 icon: %4")
                                  .arg(pixelDeviceRatio)
                                  .arg(font.pointSize())
                                  .arg(margin)
                                  .arg(iconSize));
    }

    QFontMetrics metrics(font);
    bbv_labelRect = metrics.boundingRect(bbv_label);
    fs_labelRect = metrics.boundingRect(fs_label);
    psi_labelRect = metrics.boundingRect(psi_label);
    lsi_labelRect = metrics.boundingRect(lsi_label);

    auto textLayoutWidth = [&metrics](const QString& text) {
        return metrics.horizontalAdvance(text) + metrics.descent() / 2 + 4;
    };
    psi_textWidth = textLayoutWidth(psi_label);
    lsi_textWidth = textLayoutWidth(lsi_label);
    bbv_textWidth = textLayoutWidth(bbv_label);
    fs_textWidth = textLayoutWidth(fs_label);

    // CloudCompare row width: label + gap + minus + separator + plus
    psi_totalWidth =
            psi_textWidth + margin + iconSize + margin * 2 + iconSize;
    lsi_totalWidth =
            lsi_textWidth + margin + iconSize + margin * 2 + iconSize;
    bbv_totalWidth = bbv_textWidth + margin + iconSize;
    fs_totalWidth = fs_textWidth + margin + iconSize;

    textHeight = std::max(psi_labelRect.height(), bbv_labelRect.height());
    textHeight = std::max(lsi_labelRect.height(), textHeight);
    textHeight = std::max(fs_labelRect.height(), textHeight);
    textHeight = (3 * textHeight) / 4;
    yTextBottomLineShift = (iconSize / 2) + (textHeight / 2);
}

QRect ecvHotZone::rect(bool clickableItemsVisible,
                       bool bubbleViewModeEnabled,
                       bool fullScreenEnabled) const {
    int totalWidth = 0;
    if (clickableItemsVisible)
        totalWidth = std::max(psi_totalWidth, lsi_totalWidth);
    if (bubbleViewModeEnabled)
        totalWidth = std::max(totalWidth, bbv_totalWidth);
    if (fullScreenEnabled) totalWidth = std::max(totalWidth, fs_totalWidth);

    QPoint minAreaCorner(0, std::min(0, yTextBottomLineShift - textHeight));
    QPoint maxAreaCorner(totalWidth, std::max(iconSize, yTextBottomLineShift));
    int rowCount = clickableItemsVisible ? 2 : 0;
    rowCount += bubbleViewModeEnabled ? 1 : 0;
    rowCount += fullScreenEnabled ? 1 : 0;
    const int rowGap = iconSize + (margin * 3) / 4;
    maxAreaCorner.setY(maxAreaCorner.y() + rowGap * (rowCount - 1));

    const int pad = (margin * 3) / 4;
    QRect areaRect(minAreaCorner - QPoint(pad, pad),
                   maxAreaCorner + QPoint(pad, pad));

    return areaRect;
}
