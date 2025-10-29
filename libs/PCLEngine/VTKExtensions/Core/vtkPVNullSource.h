// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "qPCL.h"  //needed for exports
#include "vtkPolyDataAlgorithm.h"

class QPCL_ENGINE_LIB_API vtkPVNullSource : public vtkPolyDataAlgorithm {
public:
    static vtkPVNullSource* New();
    vtkTypeMacro(vtkPVNullSource, vtkPolyDataAlgorithm);
    void PrintSelf(ostream& os, vtkIndent indent) override;

protected:
    vtkPVNullSource();
    ~vtkPVNullSource() override;

private:
    vtkPVNullSource(const vtkPVNullSource&) = delete;
    void operator=(const vtkPVNullSource&) = delete;
};
