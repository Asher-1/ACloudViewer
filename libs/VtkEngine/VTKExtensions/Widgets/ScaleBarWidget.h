// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/**
 * @file ScaleBarWidget.h
 * @brief 2D scale bar overlay showing physical scale in the viewport.
 */

#include <vtkActor2D.h>
#include <vtkCamera.h>
#include <vtkLineSource.h>
#include <vtkPolyDataMapper2D.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkSmartPointer.h>
#include <vtkTextActor.h>

// Qt version compatibility handling
#include <QApplication>
#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
#include <QDesktopWidget>
#endif
#include <QScreen>

/**
 * @class ScaleBarWidget
 * @brief 2D scale bar showing physical distance in viewport coordinates.
 */
class ScaleBarWidget {
public:
    /// @param renderer VTK renderer for the scale bar
    ScaleBarWidget(vtkRenderer* renderer);
    ~ScaleBarWidget();
    /// @param renderer Renderer
    /// @param interactor Render window interactor
    void update(vtkRenderer* renderer, vtkRenderWindowInteractor* interactor);
    /// @param visible Whether to show the scale bar
    void setVisible(bool visible);

private:
    vtkSmartPointer<vtkActor2D> lineActor;
    vtkSmartPointer<vtkTextActor> textActor;
    vtkSmartPointer<vtkActor2D> leftTickActor;   // Left tick mark
    vtkSmartPointer<vtkActor2D> rightTickActor;  // Right tick mark
    double lastLength = 0.0;
    bool visible = true;
    double dpiScale = 1.0;

    // DPI retrieval method compatible with different Qt versions
    double getDPIScale();

    // Cross-platform font size optimization function
    int getOptimizedFontSize(int baseFontSize = 18);

    // Cross-platform DPI scaling function
    double getPlatformAwareDPIScale();

    // Create tick mark actor
    vtkSmartPointer<vtkActor2D> createTickActor(double x,
                                                double y,
                                                double length);
};