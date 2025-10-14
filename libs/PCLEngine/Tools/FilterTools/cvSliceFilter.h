// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef TOOLS_SLICE_FILTER_H
#define TOOLS_SLICE_FILTER_H

#include "cvCutFilter.h"

class vtkStripper;
class vtkCutter;
class cvSliceFilter : public cvCutFilter {
    Q_OBJECT
public:
    explicit cvSliceFilter(QWidget* parent = nullptr);

    void apply();

    virtual ccHObject* getOutput() override;

    virtual void clearAllActor() override;

private:
    bool m_outputMode = false;

    vtkSmartPointer<vtkStripper> m_cutStrips;
    vtkSmartPointer<vtkCutter> m_cutter;
};

#endif  // TOOLS_SLICE_FILTER_H
