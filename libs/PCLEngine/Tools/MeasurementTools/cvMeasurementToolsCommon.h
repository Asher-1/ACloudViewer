// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef CV_MEASUREMENT_TOOLS_COMMON_H
#define CV_MEASUREMENT_TOOLS_COMMON_H

#include <vtkPointHandleRepresentation3D.h>
#include <vtkProperty.h>

namespace cvMeasurementTools {

// ParaView-style colors
constexpr double FOREGROUND_COLOR[3] = {1.0, 1.0,
                                        1.0};  // White for normal state
constexpr double INTERACTION_COLOR[3] = {
        0.0, 1.0, 0.0};  // Green for selected/interactive state
constexpr double RAY_COLOR[3] = {1.0, 0.0,
                                 0.0};  // Red for angle rays (ParaView default)
constexpr double ARC_COLOR[3] = {
        1.0, 0.1, 0.0};  // Orange-red for arc and text (ParaView default)

/**
 * @brief Configure 3D handle representation with ParaView-style properties
 * @param handle The 3D handle representation to configure
 */
inline void configureHandle3D(vtkPointHandleRepresentation3D* handle) {
    if (!handle) return;

    // Set normal (foreground) color - white
    if (auto* prop = handle->GetProperty()) {
        prop->SetColor(FOREGROUND_COLOR[0], FOREGROUND_COLOR[1],
                       FOREGROUND_COLOR[2]);
    }

    // Set selected (interaction) color - green
    if (auto* selectedProp = handle->GetSelectedProperty()) {
        selectedProp->SetColor(INTERACTION_COLOR[0], INTERACTION_COLOR[1],
                               INTERACTION_COLOR[2]);
    }

    // Configure cursor appearance - show only the crosshair axes, no
    // outline/shadows
    handle->AllOff();  // Turn off outline and all shadows

    // Enable smooth motion and translation mode for better handle movement
    handle->SmoothMotionOn();
    handle->TranslationModeOn();
}

}  // namespace cvMeasurementTools

#endif  // CV_MEASUREMENT_TOOLS_COMMON_H
