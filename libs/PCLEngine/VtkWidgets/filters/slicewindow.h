// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "cutwindow.h"

class vtkStripper;
class SliceWindow : public CutWindow {
    Q_OBJECT
public:
    explicit SliceWindow(QWidget* parent = nullptr);

    virtual void apply() override;

    inline void setOutputMode(bool state) { m_outputMode = state; }

    virtual ccHObject* getOutput() const override;

private:
    bool m_outputMode = false;

    vtkSmartPointer<vtkStripper> m_cutStrips;
};
