// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

/**
 * @file CustomVtkBoxWidget.h
 * @brief Custom box widget with restricted transformations
 * 
 * Extends vtkBoxWidget to provide selective control over transformation axes.
 * Allows enabling/disabling translation, rotation, and scaling on individual axes.
 */

#pragma once

#include "qPCL.h"

#include <vtkBoxWidget.h>

/**
 * @class CustomVtkBoxWidget
 * @brief VTK box widget with axis-specific transformation restrictions
 * 
 * This custom widget extends vtkBoxWidget to allow fine-grained control
 * over which transformations are permitted. Each transformation type
 * (translate, rotate, scale) can be enabled or disabled per axis.
 * 
 * Features:
 * - Per-axis translation control (X, Y, Z)
 * - Per-axis rotation control (X, Y, Z)
 * - Global scale control
 * - Maintains VTK's standard box widget interaction
 * 
 * Use cases:
 * - Constraining object manipulation to specific directions
 * - Creating 2D interaction in 3D space
 * - Restricting transformations for specific workflows
 * 
 * @see vtkBoxWidget
 */
class QPCL_ENGINE_LIB_API CustomVtkBoxWidget : public vtkBoxWidget {
public:
    /**
     * @brief Create new instance
     * @return Pointer to new CustomVtkBoxWidget
     */
    static CustomVtkBoxWidget *New();

    vtkTypeMacro(CustomVtkBoxWidget, vtkBoxWidget);

    /**
     * @brief Translate widget with axis restrictions
     * @param p1 Start point in world coordinates
     * @param p2 End point in world coordinates
     * 
     * Overrides base translation to respect axis-specific enable flags.
     */
    virtual void Translate(double *p1, double *p2) override;
    
    /**
     * @brief Scale widget with restrictions
     * @param p1 First point for scale calculation
     * @param p2 Second point for scale calculation
     * @param X Screen X coordinate
     * @param Y Screen Y coordinate
     * 
     * Overrides base scaling to respect scale enable flag.
     */
    virtual void Scale(double *p1, double *p2, int X, int Y) override;
    
    /**
     * @brief Rotate widget with axis restrictions
     * @param X Screen X coordinate
     * @param Y Screen Y coordinate
     * @param p1 First point for rotation calculation
     * @param p2 Second point for rotation calculation
     * @param vpn View plane normal
     * 
     * Overrides base rotation to respect axis-specific rotation flags.
     */
    virtual void Rotate(
            int X, int Y, double *p1, double *p2, double *vpn) override;

    /**
     * @brief Enable/disable translation along X axis
     * @param state true to enable, false to disable
     */
    void SetTranslateXEnabled(bool state) { m_translateX = state; }
    
    /**
     * @brief Enable/disable translation along Y axis
     * @param state true to enable, false to disable
     */
    void SetTranslateYEnabled(bool state) { m_translateY = state; }
    
    /**
     * @brief Enable/disable translation along Z axis
     * @param state true to enable, false to disable
     */
    void SetTranslateZEnabled(bool state) { m_translateZ = state; }
    
    /**
     * @brief Enable/disable rotation around X axis
     * @param state true to enable, false to disable
     */
    void SetRotateXEnabled(bool state) { m_rotateX = state; }
    
    /**
     * @brief Enable/disable rotation around Y axis
     * @param state true to enable, false to disable
     */
    void SetRotateYEnabled(bool state) { m_rotateY = state; }
    
    /**
     * @brief Enable/disable rotation around Z axis
     * @param state true to enable, false to disable
     */
    void SetRotateZEnabled(bool state) { m_rotateZ = state; }
    
    /**
     * @brief Enable/disable scaling
     * @param state true to enable, false to disable
     */
    void SetScaleEnabled(bool state) { m_scale = state; }

private:
    bool m_translateX = true;  ///< X-axis translation enabled
    bool m_translateY = true;  ///< Y-axis translation enabled
    bool m_translateZ = true;  ///< Z-axis translation enabled
    bool m_rotateX = true;     ///< X-axis rotation enabled
    bool m_rotateY = true;     ///< Y-axis rotation enabled
    bool m_rotateZ = true;     ///< Z-axis rotation enabled
    bool m_scale = true;       ///< Scaling enabled
};
