// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "cvIsoSurfaceFilter.h"

class vtkDataSetSurfaceFilter;
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
