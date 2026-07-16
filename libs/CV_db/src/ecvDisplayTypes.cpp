// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvDisplayTypes.h"

#include <CVLog.h>

#include <QGuiApplication>
#include <QScreen>
#include <algorithm>

#include "ecvDisplayTools.h"

void ecvHotZone::updateInternalVariables(QWidget* win) {
    if (win) {
        font = win->font();
        pixelDeviceRatio = win->devicePixelRatioF();

        // ====================================================================
        // DPI-Adaptive Font Sizing Strategy (First Principles)
        // ====================================================================
        // Goal: Ensure readable text across all platforms and resolutions
        // - Windows: DPI awareness via Qt
        // - macOS: Retina displays report 2.0 DPI
        // - Linux: DE-dependent, may need fallback detection
        // - 4K displays: Need significantly larger fonts

        static constexpr int kBaseFontPt =
                14;  // Increased base for better readability
        static constexpr int kBaseMargin = 16;  // Match CloudCompare spacing
        static constexpr int kBaseIconSize = 16;

        // ====================================================================
        // Fallback Detection: Physical Screen Size
        // ====================================================================
        // On Linux, 4K displays may report DPI=1.0 if scaling is not configured
        // Detect this case by checking physical screen dimensions
        bool isHighResolutionScreen = false;
        if (pixelDeviceRatio < 1.5) {
            QScreen* screen = nullptr;

            // Qt version compatibility: QWidget::screen() added in Qt 5.14
#if QT_VERSION >= QT_VERSION_CHECK(5, 14, 0)
            screen = win->screen();
#endif

            // Fallback to primary screen
            if (!screen) {
                screen = QGuiApplication::primaryScreen();
            }

            // Final safety check
            if (screen) {
                QSize pixelSize = screen->size();
                QSizeF physicalSize = screen->physicalSize();  // mm

                // Validate physical size (may be invalid in VMs/remote desktop)
                if (physicalSize.width() > 0 && physicalSize.height() > 0) {
                    qreal physicalDPI =
                            pixelSize.width() / (physicalSize.width() / 25.4);

                    // Detect high-resolution displays with low reported DPI
                    // Use 145 DPI threshold (with 5 DPI tolerance for
                    // measurement errors) Typical 4K 24": ~183 DPI, 1080p 24":
                    // ~92 DPI
                    if (physicalDPI > 145.0 && pixelSize.width() >= 3000) {
                        isHighResolutionScreen = true;
                        CVLog::PrintVerbose(
                                QString("[HotZone] High-res screen detected: "
                                        "%1x%2 @ %3 DPI (physical)")
                                        .arg(pixelSize.width())
                                        .arg(pixelSize.height())
                                        .arg(physicalDPI, 0, 'f', 1));
                    }
                } else {
                    CVLog::PrintDebug(QString(
                            "[HotZone] Physical size unavailable (VM/RDP?), "
                            "using DPI-based detection only"));
                }
            } else {
                CVLog::Warning(
                        "[HotZone] Cannot detect screen, using default DPI "
                        "scaling");
            }
        }

        // ====================================================================
        // DPI-based font scaling with tiered thresholds
        // ====================================================================
        int scaledFontSize;

        if (isHighResolutionScreen) {
            // Override for high-resolution screens with low reported DPI
            // Common on Linux 4K displays at 100% scaling
            scaledFontSize =
                    22;  // Larger than base but not as large as DPI=2.0
            CVLog::PrintVerbose(
                    "[HotZone] High-resolution screen override applied (22pt)");
        } else if (pixelDeviceRatio >= 2.5) {
            // Ultra-high DPI (4K+ at 150-200% scaling, or 5K/6K displays)
            scaledFontSize = 28;
            CVLog::PrintVerbose("[HotZone] Ultra-high DPI detected (>=2.5)");
        } else if (pixelDeviceRatio >= 2.0) {
            // High DPI (Retina, 4K at proper scaling)
            scaledFontSize = 24;
            CVLog::PrintVerbose("[HotZone] High DPI detected (>=2.0)");
        } else if (pixelDeviceRatio >= 1.5) {
            // Medium-high DPI (1440p at 125%, 1080p at 150%)
            scaledFontSize = 18;
            CVLog::PrintVerbose("[HotZone] Medium-high DPI detected (>=1.5)");
        } else if (pixelDeviceRatio >= 1.25) {
            // Slightly scaled (1080p at 125%)
            scaledFontSize = 16;
            CVLog::PrintVerbose(
                    "[HotZone] Slightly scaled DPI detected (>=1.25)");
        } else {
            // Standard DPI or low DPI
            scaledFontSize = static_cast<int>(kBaseFontPt * pixelDeviceRatio);
            scaledFontSize = std::max(scaledFontSize, 14);  // Minimum 14pt
            CVLog::PrintVerbose("[HotZone] Standard DPI detected");
        }

        font.setPointSize(scaledFontSize);

        // Scale margins and icons proportionally
        qreal effectiveScale =
                isHighResolutionScreen ? 1.5 : std::max(pixelDeviceRatio, 1.0);
        margin = static_cast<int>(kBaseMargin * effectiveScale);
        iconSize = static_cast<int>(kBaseIconSize * effectiveScale);

        // Font rendering quality settings
        font.setBold(true);
        font.setWeight(QFont::Bold);
        font.setStyleStrategy(QFont::PreferAntialias);
        font.setHintingPreference(QFont::PreferFullHinting);

        CVLog::PrintVerbose(
                QString("[HotZone] DPI: %1, Font: %2pt, Margin: %3, Icon: %4")
                        .arg(pixelDeviceRatio, 0, 'f', 2)
                        .arg(font.pointSize())
                        .arg(margin)
                        .arg(iconSize));
    }

#ifdef Q_OS_MACOS
    // VTK with HiDPI enabled renders at DPI = DPR * 72, which inflates
    // the HotZone relative to CloudCompare's native GL rendering.
    // Apply 2/3 reduction for visual parity with CloudCompare on macOS.
    if (pixelDeviceRatio >= 1.5) {
        constexpr qreal kMacDisplayScale = 2.0 / 3.0;
        font.setPointSize(std::max(
                static_cast<int>(font.pointSize() * kMacDisplayScale), 14));
        margin = std::max(static_cast<int>(margin * kMacDisplayScale), 16);
        iconSize = std::max(static_cast<int>(iconSize * kMacDisplayScale), 16);
    }
#endif

    QFontMetrics metrics(font);
    bbv_labelRect = metrics.boundingRect(bbv_label);
    fs_labelRect = metrics.boundingRect(fs_label);
    psi_labelRect = metrics.boundingRect(psi_label);
    lsi_labelRect = metrics.boundingRect(lsi_label);

    // QFontMetrics returns logical pixels. With VTK HiDPI enabled, the
    // render window DPI = UnscaledDPI * DPR, so VTK renders text at DPR
    // times the logical size. Scale metrics by DPR to match physical layout.
    const qreal metricsScale = std::max(pixelDeviceRatio, 1.0);

    auto textLayoutWidth = [&metrics, metricsScale](const QString& text,
                                                    const QRect& br) {
        int advance = metrics.horizontalAdvance(text);
        int brWidth = br.width();
        int baseWidth = std::max(advance, brWidth) + metrics.descent() / 2 + 4;
        return static_cast<int>(baseWidth * metricsScale);
    };
    psi_textWidth = textLayoutWidth(psi_label, psi_labelRect);
    lsi_textWidth = textLayoutWidth(lsi_label, lsi_labelRect);
    bbv_textWidth = textLayoutWidth(bbv_label, bbv_labelRect);
    fs_textWidth = textLayoutWidth(fs_label, fs_labelRect);

    psi_totalWidth = psi_textWidth + margin + iconSize + margin + iconSize;
    lsi_totalWidth = lsi_textWidth + margin + iconSize + margin + iconSize;
    bbv_totalWidth = bbv_textWidth + margin + iconSize;
    fs_totalWidth = fs_textWidth + margin + iconSize;

    int scaledMaxHeight =
            std::max(psi_labelRect.height(), bbv_labelRect.height());
    scaledMaxHeight = std::max(lsi_labelRect.height(), scaledMaxHeight);
    scaledMaxHeight = std::max(fs_labelRect.height(), scaledMaxHeight);
    textHeight = static_cast<int>((3 * scaledMaxHeight * metricsScale) / 4);
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
