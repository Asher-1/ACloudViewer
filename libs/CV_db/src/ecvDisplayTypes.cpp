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
        static constexpr int kBaseFontPt = 12;
        static constexpr int kBaseMargin = 16;
        static constexpr int kBaseIconSize = 16;
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

    psi_totalWidth =
            psi_labelRect.width() + margin + iconSize + margin * 2 + iconSize;
    lsi_totalWidth =
            lsi_labelRect.width() + margin + iconSize + margin * 2 + iconSize;
    bbv_totalWidth = bbv_labelRect.width() + margin + iconSize;
    fs_totalWidth = fs_labelRect.width() + margin + iconSize;

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
    maxAreaCorner.setY(maxAreaCorner.y() +
                       (iconSize + margin) * (rowCount - 1));

    QRect areaRect(minAreaCorner - QPoint(margin, margin) / 2,
                   maxAreaCorner + QPoint(margin, margin) / 2);

    return areaRect;
}
