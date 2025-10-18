// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "ecvDrawableObject.h"

class QWidget;
class ccGBLSensor;
class ccScalarField;

//! Misc. tools for rendering of advanced structures
class ECV_DB_LIB_API ccRenderingTools {
public:
    //! Displays a depth buffer as an image
    static void ShowDepthBuffer(ccGBLSensor* lidar,
                                QWidget* parent = nullptr,
                                unsigned maxDim = 1024);

    //! Displays the colored scale corresponding to the currently activated
    //! context scalar field
    /** Its appearance depends on the scalar fields min and max displayed
            values, min and max saturation values, and also the selected
            color ramp.
            \param context OpenGL context description
    **/
    static void DrawColorRamp(const CC_DRAW_CONTEXT& context);

    //! See other version of DrawColorRamp
    static void DrawColorRamp(const CC_DRAW_CONTEXT& context,
                              const ccScalarField* sf,
                              QWidget* win,
                              int glW,
                              int glH,
                              float renderZoom = 1.0f);
};
