// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/**
 * @file cvSliceFilter.h
 * @brief Slice filter for extracting contours from planes.
 */

#include "cvCutFilter.h"

class vtkStripper;
class vtkCutter;

/**
 * @class cvSliceFilter
 * @brief Slices geometry with plane to produce contour polylines.
 */
class cvSliceFilter : public cvCutFilter {
    Q_OBJECT
public:
    /// @param parent Parent widget.
    explicit cvSliceFilter(QWidget* parent = nullptr);

    void apply();

    /// @return Slice contours as ccHObject.
    virtual ccHObject* getOutput() override;

    virtual void clearAllActor() override;

private:
    bool m_outputMode = false;

    vtkSmartPointer<vtkStripper> m_cutStrips;
    vtkSmartPointer<vtkCutter> m_cutter;
};
