// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QPoint>
#include <QRect>
#include <QWidget>
#include <algorithm>
#include <cmath>

#include "CV_db.h"

/// Centralized coordinate-space conversion between logical pixels (Qt widget
/// coordinates, VTK render-window coordinates on macOS Retina) and physical
/// pixels (GL viewport, picking, depth buffer).
///
/// Coordinate spaces in ACloudViewer:
///
///   Logical (LP)   – QWidget::width()/height(), QMouseEvent x/y.
///
///   Physical (PP)  – glViewport, ccGLDrawContext::glW/glH,
///                     vtkRenderWindow::GetSize(), ImageVis overlay,
///                     depth-buffer reads, picking coordinates.
///
/// Rule of thumb:
///   Input events arrive in LP → multiply by DPR for GL/picking.
///   GL/picking results are in PP → divide by DPR for ImageVis/Qt overlay.
///
/// Usage:
///   using DC = ecvDisplayCoordinates;
///   int physX = DC::toPhysical(logicalX, dpr);
///   QRect lpRect = DC::toLogical(physRect, dpr);
class CV_DB_LIB_API ecvDisplayCoordinates {
public:
    /// Get device-pixel-ratio from a widget, defaulting to 1.0 if null.
    static double dprOf(const QWidget* w) {
        return w ? w->devicePixelRatioF() : 1.0;
    }

    // ---- Scalar conversions ----

    static int toPhysical(int logical, double dpr) {
        return static_cast<int>(std::round(logical * dpr));
    }

    static int toLogical(int physical, double dpr) {
        return (dpr > 1.0) ? static_cast<int>(std::round(physical / dpr))
                           : physical;
    }

    static double toPhysicalF(double logical, double dpr) {
        return logical * dpr;
    }

    static double toLogicalF(double physical, double dpr) {
        return (dpr > 1.0) ? physical / dpr : physical;
    }

    // ---- QPoint conversions ----

    static QPoint toPhysical(const QPoint& lp, double dpr) {
        return QPoint(toPhysical(lp.x(), dpr), toPhysical(lp.y(), dpr));
    }

    static QPoint toLogical(const QPoint& pp, double dpr) {
        return QPoint(toLogical(pp.x(), dpr), toLogical(pp.y(), dpr));
    }

    // ---- QRect conversions ----

    static QRect toPhysical(const QRect& lp, double dpr) {
        return QRect(toPhysical(lp.x(), dpr), toPhysical(lp.y(), dpr),
                     toPhysical(lp.width(), dpr), toPhysical(lp.height(), dpr));
    }

    static QRect toLogical(const QRect& pp, double dpr) {
        return QRect(toLogical(pp.x(), dpr), toLogical(pp.y(), dpr),
                     toLogical(pp.width(), dpr), toLogical(pp.height(), dpr));
    }

    // ---- Y-flip helpers ----
    // VTK display space: origin at bottom-left.
    // Qt widget space:   origin at top-left.

    /// Convert Qt Y (top-down) to VTK display Y (bottom-up).
    static int qtYToVtk(int qtY, int viewportHeight) {
        return viewportHeight - 1 - qtY;
    }

    /// Convert VTK display Y (bottom-up) to Qt Y (top-down).
    static int vtkYToQt(int vtkY, int viewportHeight) {
        return viewportHeight - 1 - vtkY;
    }

    // ---- Compound helpers ----

    /// Qt logical mouse position → VTK physical display coordinates.
    /// Combines toPhysical + Y-flip in a single call.
    /// This is the most common conversion in picking/input handlers.
    static QPoint qtToVtkPhysical(const QPoint& qtPos,
                                  int widgetHeight,
                                  double dpr) {
        int px = toPhysical(qtPos.x(), dpr);
        int py = toPhysical(widgetHeight, dpr) - 1 - toPhysical(qtPos.y(), dpr);
        return QPoint(px, py);
    }

    /// VTK physical display coordinates → Qt logical mouse position.
    static QPoint vtkPhysicalToQt(int vtkX,
                                  int vtkY,
                                  int physicalHeight,
                                  double dpr) {
        int qtX = toLogical(vtkX, dpr);
        int qtY = toLogical(physicalHeight - 1 - vtkY, dpr);
        return QPoint(qtX, qtY);
    }

private:
    ecvDisplayCoordinates() = delete;
};
