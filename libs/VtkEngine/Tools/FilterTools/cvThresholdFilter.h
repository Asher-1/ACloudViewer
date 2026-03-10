// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/**
 * @file cvThresholdFilter.h
 * @brief Threshold filter for scalar-based extraction.
 */

#include "cvIsoSurfaceFilter.h"

class vtkDataSetSurfaceFilter;

/**
 * @class cvThresholdFilter
 * @brief Extracts geometry within a scalar threshold range.
 */
class cvThresholdFilter : public cvIsoSurfaceFilter {
    Q_OBJECT
public:
    explicit cvThresholdFilter(QWidget* parent = nullptr);
    ~cvThresholdFilter();

    virtual void apply() override;

    virtual ccHObject* getOutput() override;

    virtual void clearAllActor() override;

    virtual void initFilter() override;

protected:
    vtkSmartPointer<vtkDataSetSurfaceFilter> m_dssFilter;
};
