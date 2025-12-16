// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ScaleBar.h"

#include <vtkActor2D.h>
#include <vtkAlgorithmOutput.h>
#include <vtkCamera.h>
#include <vtkCoordinate.h>
#include <vtkLineSource.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper2D.h>
#include <vtkProperty2D.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkSmartPointer.h>
#include <vtkTextActor.h>
#include <vtkTextProperty.h>

#include <QApplication>
#include <QCoreApplication>
#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
#include <QDesktopWidget>
#endif
#include <QProcessEnvironment>
#include <QScreen>
#include <QString>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>

ScaleBar::ScaleBar(vtkRenderer* renderer) {
    // Get cross-platform optimized DPI scaling
    dpiScale = getPlatformAwareDPIScale();

    // Create line segment
    auto lineSource = vtkSmartPointer<vtkLineSource>::New();
    lineSource->SetPoint1(0.0, 0.0, 0.0);
    lineSource->SetPoint2(100.0, 0.0, 0.0);  // Initial length

    auto mapper = vtkSmartPointer<vtkPolyDataMapper2D>::New();
    mapper->SetInputConnection(lineSource->GetOutputPort());

    lineActor = vtkSmartPointer<vtkActor2D>::New();
    lineActor->SetMapper(mapper);
    lineActor->GetProperty()->SetColor(1.0, 1.0, 1.0);
    lineActor->GetProperty()->SetLineWidth(3.0 * dpiScale);

    // Create text - using cross-platform optimized font size
    textActor = vtkSmartPointer<vtkTextActor>::New();
    textActor->SetInput("1 m");
    int optimizedFontSize = getOptimizedFontSize(18);
    textActor->GetTextProperty()->SetFontSize(optimizedFontSize);
    textActor->GetTextProperty()->SetColor(1.0, 1.0, 1.0);
    textActor->GetTextProperty()->SetJustificationToCentered();
    textActor->GetTextProperty()->SetVerticalJustificationToTop();
    // Initial position set to center, will be adjusted in update()
    textActor->SetPosition(100.0 * dpiScale, 25.0 * dpiScale);

    // Create tick marks
    leftTickActor = createTickActor(0.0, 0.0, 10.0 * dpiScale);
    rightTickActor = createTickActor(100.0, 0.0, 10.0 * dpiScale);

    if (renderer) {
        renderer->AddActor2D(lineActor);
        renderer->AddActor2D(textActor);
        renderer->AddActor2D(leftTickActor);
        renderer->AddActor2D(rightTickActor);
    }
}

ScaleBar::~ScaleBar() {}

void ScaleBar::setVisible(bool v) {
    visible = v;
    lineActor->SetVisibility(v);
    textActor->SetVisibility(v);
    leftTickActor->SetVisibility(v);
    rightTickActor->SetVisibility(v);
}

double ScaleBar::getDPIScale() {
    // DPI retrieval method compatible with different Qt versions
    if (!QApplication::instance()) {
        return 1.0;
    }

// Method 1: Qt 5.6+ uses QScreen::devicePixelRatio() (recommended)
#if QT_VERSION >= QT_VERSION_CHECK(5, 6, 0)
    QScreen* screen = QApplication::primaryScreen();
    if (screen) {
        return screen->devicePixelRatio();
    }
#endif

// Method 2: Qt 5.0+ uses QApplication::devicePixelRatio()
#if QT_VERSION >= QT_VERSION_CHECK(5, 0, 0)
    QCoreApplication* coreApp = QCoreApplication::instance();
    QApplication* app = qobject_cast<QApplication*>(coreApp);
    if (app) {
        return app->devicePixelRatio();
    }
#endif

// Method 3: Qt 4.x calculates using QDesktopWidget
#if QT_VERSION < QT_VERSION_CHECK(5, 0, 0)
    QDesktopWidget* desktop = QApplication::desktop();
    if (desktop) {
        // Calculate scaling from physical and logical DPI
        int physicalDPI = desktop->physicalDpiX();
        int logicalDPI = desktop->logicalDpiX();
        if (logicalDPI > 0) {
            return static_cast<double>(physicalDPI) / logicalDPI;
        }
    }
#endif

    // Method 4: Through environment variable or system detection
    const char* qt_scale_factor = qgetenv("QT_SCALE_FACTOR");
    if (qt_scale_factor) {
        bool ok;
        double scale = QString(qt_scale_factor).toDouble(&ok);
        if (ok && scale > 0) {
            return scale;
        }
    }

    return 1.0;  // Default scaling
}

int ScaleBar::getOptimizedFontSize(int baseFontSize) {
    // Get screen information
    QScreen* screen = QApplication::primaryScreen();
    if (!screen) {
        return baseFontSize;
    }

    // Get screen resolution information
    QSize screenSize = screen->size();
    int screenWidth = screenSize.width();
    int screenHeight = screenSize.height();
    int screenDPI = screen->physicalDotsPerInch();
    int dpiScale = static_cast<int>(getDPIScale());

    // Platform-specific base font size adjustment
    int platformBaseSize = baseFontSize;
#ifdef Q_OS_MAC
    // macOS: Default font is slightly larger, but need to account for Retina
    // display over-scaling
    platformBaseSize = baseFontSize;
    if (dpiScale > 1) {
        // Retina display: Use smaller font to avoid over-scaling
        platformBaseSize = std::max(12, baseFontSize - (dpiScale - 1) * 3);
    }
#elif defined(Q_OS_WIN)
    // Windows: Adjust font size based on DPI
    if (screenDPI > 120) {
        // High DPI display
        platformBaseSize = std::max(12, baseFontSize - 2);
    } else if (screenDPI < 96) {
        // Low DPI display
        platformBaseSize = baseFontSize + 2;
    }
#elif defined(Q_OS_LINUX)
    // Linux: Adjust based on screen resolution
    if (screenWidth >= 1920 && screenHeight >= 1080) {
        // High resolution display
        platformBaseSize = std::max(12, baseFontSize - 2);
    } else if (screenWidth < 1366) {
        // Low resolution display
        platformBaseSize = baseFontSize + 2;
    }
#endif

    // Resolution-specific adjustment
    int resolutionFactor = 0;
    if (screenWidth >= 2560 && screenHeight >= 1440) {
        // 2K and above resolution
        resolutionFactor = -1;
    } else if (screenWidth < 1366) {
        // Low resolution
        resolutionFactor = 1;
    }

    // Final font size calculation
    int finalSize = platformBaseSize + resolutionFactor;

    // Ensure font size is within reasonable range
    finalSize = std::max(10, std::min(32, finalSize));

    return finalSize;
}

double ScaleBar::getPlatformAwareDPIScale() {
    double dpiScale = getDPIScale();
    QScreen* screen = QApplication::primaryScreen();
    if (!screen) {
        return dpiScale;
    }

    // Get screen information
    QSize screenSize = screen->size();
    int screenWidth = screenSize.width();
    int screenHeight = screenSize.height();
    int screenDPI = screen->physicalDotsPerInch();

    // Platform-specific DPI scaling adjustment
    double adjustedScale = dpiScale;

#ifdef Q_OS_MAC
    // macOS: Retina display needs special handling
    if (dpiScale > 1) {
        // For UI elements, use smaller scaling to avoid over-scaling
        adjustedScale = 1.0 + (dpiScale - 1.0) * 0.6;
    }
#elif defined(Q_OS_WIN)
    // Windows: Adjust based on DPI settings
    if (screenDPI > 120) {
        // High DPI display, reduce scaling appropriately
        adjustedScale = std::min(adjustedScale, 1.4);
    } else if (screenDPI < 96) {
        // Low DPI display, increase scaling appropriately
        adjustedScale = std::max(adjustedScale, 1.0);
    }
#elif defined(Q_OS_LINUX)
    // Linux: Adjust based on resolution
    if (screenWidth >= 2560 && screenHeight >= 1440) {
        // Ultra high resolution, reduce scaling
        adjustedScale = std::min(adjustedScale, 1.2);
    } else if (screenWidth < 1366) {
        // Low resolution, increase scaling
        adjustedScale = std::max(adjustedScale, 1.0);
    }
#endif

    // Ensure scaling is within reasonable range
    adjustedScale = std::max(0.8, std::min(1.8, adjustedScale));

    return adjustedScale;
}

vtkSmartPointer<vtkActor2D> ScaleBar::createTickActor(double x,
                                                      double y,
                                                      double length) {
    auto lineSource = vtkSmartPointer<vtkLineSource>::New();
    lineSource->SetPoint1(x, y, 0.0);
    lineSource->SetPoint2(x, y + length, 0.0);  // Vertical tick mark

    auto mapper = vtkSmartPointer<vtkPolyDataMapper2D>::New();
    mapper->SetInputConnection(lineSource->GetOutputPort());

    auto actor = vtkSmartPointer<vtkActor2D>::New();
    actor->SetMapper(mapper);
    actor->GetProperty()->SetColor(1.0, 1.0, 1.0);
    actor->GetProperty()->SetLineWidth(2.0 * dpiScale);

    return actor;
}

void ScaleBar::update(vtkRenderer* renderer,
                      vtkRenderWindowInteractor* interactor) {
    if (!visible || !renderer || !renderer->GetRenderWindow()) return;

    // Dynamically update DPI scaling (in case window moves to different DPI
    // monitor)
    double currentDPIScale = getPlatformAwareDPIScale();
    if (std::abs(currentDPIScale - dpiScale) > 0.1) {
        dpiScale = currentDPIScale;
        // Update font size and line width - using cross-platform optimized font
        // size
        if (textActor) {
            int optimizedFontSize = getOptimizedFontSize(18);
            textActor->GetTextProperty()->SetFontSize(optimizedFontSize);
            textActor->GetTextProperty()->SetJustificationToCentered();
            textActor->GetTextProperty()->SetVerticalJustificationToTop();
        }
        if (lineActor) {
            lineActor->GetProperty()->SetLineWidth(3.0 * dpiScale);
        }
        if (leftTickActor) {
            leftTickActor->GetProperty()->SetLineWidth(2.0 * dpiScale);
        }
        if (rightTickActor) {
            rightTickActor->GetProperty()->SetLineWidth(2.0 * dpiScale);
        }
    }

    // Get window size
    int* size = renderer->GetRenderWindow()->GetSize();
    int winW = size[0];
    int winH = size[1];
    // Scale bar length (pixels), considering DPI scaling
    int barPixelLen = static_cast<int>(
            (winW / 6.0) * dpiScale);  // Approximately 1/6 of window width

    // Calculate ScaleBar position centered at window bottom
    double bottomMargin = 25.0 * dpiScale;  // Bottom margin
    double centerX =
            static_cast<double>(winW) / 2.0;  // Window center X coordinate
    double bottomY = bottomMargin;            // Bottom Y coordinate

    // Calculate ScaleBar start and end positions (centered)
    double p1[3] = {centerX - static_cast<double>(barPixelLen) / 2.0, bottomY,
                    0.0};  // Line left endpoint
    double p2[3] = {centerX + static_cast<double>(barPixelLen) / 2.0, bottomY,
                    0.0};  // Line right endpoint
    // Convert screen coordinates to world coordinates
    double world1[4], world2[4];
    renderer->SetDisplayPoint(static_cast<int>(p1[0]), static_cast<int>(p1[1]),
                              0);
    renderer->DisplayToWorld();
    memcpy(world1, renderer->GetWorldPoint(), sizeof(double) * 4);
    renderer->SetDisplayPoint(static_cast<int>(p2[0]), static_cast<int>(p2[1]),
                              0);
    renderer->DisplayToWorld();
    memcpy(world2, renderer->GetWorldPoint(), sizeof(double) * 4);
    // Calculate world distance
    double dx = (world2[0] / world2[3]) - (world1[0] / world1[3]);
    double dy = (world2[1] / world2[3]) - (world1[1] / world1[3]);
    double dz = (world2[2] / world2[3]) - (world1[2] / world1[3]);
    double dist = sqrt(dx * dx + dy * dy + dz * dz);
    // Select appropriate display unit
    double showLen = dist;
    std::string unit = "m";
    if (showLen < 0.01) {
        showLen *= 1000;
        unit = "mm";
    } else if (showLen < 1) {
        showLen *= 100;
        unit = "cm";
    } else if (showLen > 1000) {
        showLen /= 1000;
        unit = "km";
    }
    // Round to nice number
    double niceLen = showLen;
    if (showLen > 10)
        niceLen = round(showLen / 10) * 10;
    else if (showLen > 1)
        niceLen = round(showLen);
    else
        niceLen = round(showLen * 10) / 10.0;
    // Update line segment
    auto mapper = dynamic_cast<vtkPolyDataMapper2D*>(lineActor->GetMapper());
    if (mapper) {
        auto lineSource = dynamic_cast<vtkLineSource*>(
                mapper->GetInputConnection(0, 0)->GetProducer());
        if (lineSource) {
            lineSource->SetPoint1(p1[0], p1[1], 0.0);
            lineSource->SetPoint2(p2[0], p2[1], 0.0);
            lineSource->Update();
        }
    }
    // Update tick mark positions
    if (leftTickActor && rightTickActor) {
        auto leftMapper =
                dynamic_cast<vtkPolyDataMapper2D*>(leftTickActor->GetMapper());
        auto rightMapper =
                dynamic_cast<vtkPolyDataMapper2D*>(rightTickActor->GetMapper());

        if (leftMapper && rightMapper) {
            auto leftSource = dynamic_cast<vtkLineSource*>(
                    leftMapper->GetInputConnection(0, 0)->GetProducer());
            auto rightSource = dynamic_cast<vtkLineSource*>(
                    rightMapper->GetInputConnection(0, 0)->GetProducer());

            if (leftSource && rightSource) {
                double tickLength = 8.0 * dpiScale;
                leftSource->SetPoint1(p1[0], p1[1], 0.0);
                leftSource->SetPoint2(p1[0], p1[1] + tickLength, 0.0);
                leftSource->Update();

                rightSource->SetPoint1(p2[0], p2[1], 0.0);
                rightSource->SetPoint2(p2[0], p2[1] + tickLength, 0.0);
                rightSource->Update();
            }
        }
    }

    // Update text
    std::ostringstream oss;
    oss.precision(2);
    oss << std::fixed << niceLen << " " << unit;
    textActor->SetInput(oss.str().c_str());

    // Set text centered alignment
    textActor->GetTextProperty()->SetJustificationToCentered();
    textActor->GetTextProperty()->SetVerticalJustificationToTop();

    // Calculate text position: centered below the line
    double textX = centerX;  // Use window center X coordinate to ensure text is
                             // centered
    double textY = bottomY - 10.0 * dpiScale;  // Below the line
    textActor->SetPosition(textX, textY);
}