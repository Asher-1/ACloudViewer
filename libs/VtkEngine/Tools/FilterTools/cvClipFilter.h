// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/**
 * @file cvClipFilter.h
 * @brief Clip filter using box, plane, or sphere to cut geometry.
 */

#include "cvCutFilter.h"

class vtkClipPolyData;
class vtkClipDataSet;

/**
 * @class cvClipFilter
 * @brief Clip geometry inside or outside box/plane/sphere.
 */
class cvClipFilter : public cvCutFilter {
    Q_OBJECT

public:
    /// @param parent Parent widget.
    explicit cvClipFilter(QWidget* parent = 0);
    ~cvClipFilter();

    virtual void apply() override;
    /// @return Clipped geometry as ccHObject.
    virtual ccHObject* getOutput() override;
    virtual void clearAllActor() override;

protected:
    vtkSmartPointer<vtkClipPolyData> m_PolyClip;
    vtkSmartPointer<vtkClipDataSet> m_DataSetClip;
};
