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
        pixelDeviceRatio = ecvDisplayTools::GetPlatformAwareDPIScale();
        int fontSize = ecvDisplayTools::GetOptimizedFontSize(12);
        if (fontSize != pixelDeviceRatio) {
            font.setPointSize(fontSize);
        } else {
            font.setPointSize(12 * pixelDeviceRatio);
        }
        CVLog::PrintDebug(QString("pixelDeviceRatio: %1 and fontSize %2")
                                .arg(pixelDeviceRatio)
                                .arg(fontSize));
        margin *= pixelDeviceRatio;
        iconSize *= pixelDeviceRatio;
        font.setBold(true);
    }

    QFontMetrics metrics(font);
    bbv_labelRect = metrics.boundingRect(bbv_label);
    fs_labelRect = metrics.boundingRect(fs_label);
    psi_labelRect = metrics.boundingRect(psi_label);
    lsi_labelRect = metrics.boundingRect(lsi_label);

    psi_totalWidth =
            psi_labelRect.width() + margin + iconSize + margin + iconSize;
    lsi_totalWidth =
            lsi_labelRect.width() + margin + iconSize + margin + iconSize;
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

#ifdef Q_OS_MAC
    totalWidth = totalWidth + 3 * (margin + iconSize);
#endif
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
