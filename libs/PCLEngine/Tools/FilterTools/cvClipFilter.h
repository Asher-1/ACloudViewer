// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "cvCutFilter.h"

class vtkClipPolyData;
class vtkClipDataSet;

class cvClipFilter : public cvCutFilter {
    Q_OBJECT

public:
    explicit cvClipFilter(QWidget* parent = 0);
    ~cvClipFilter();

    virtual void apply() override;
    virtual ccHObject* getOutput() override;
    virtual void clearAllActor() override;

protected:
    vtkSmartPointer<vtkClipPolyData> m_PolyClip;
    vtkSmartPointer<vtkClipDataSet> m_DataSetClip;
};
