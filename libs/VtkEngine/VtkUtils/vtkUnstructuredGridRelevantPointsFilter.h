// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/// @file vtkUnstructuredGridRelevantPointsFilter.h
/// @brief VTK filter that keeps only points referenced by cells.

#include "vtkUnstructuredGridAlgorithm.h"

/// @class vtkUnstructuredGridRelevantPointsFilter
/// @brief Removes unreferenced points from vtkUnstructuredGrid.
class vtkUnstructuredGridRelevantPointsFilter
    : public vtkUnstructuredGridAlgorithm {
public:
    static vtkUnstructuredGridRelevantPointsFilter *New();
    vtkTypeMacro(vtkUnstructuredGridRelevantPointsFilter,
                 vtkUnstructuredGridAlgorithm);
    void PrintSelf(ostream &os, vtkIndent indent);

protected:
    vtkUnstructuredGridRelevantPointsFilter() {};
    ~vtkUnstructuredGridRelevantPointsFilter() {};

    // Usual data generation method
    int RequestData(vtkInformation *,
                    vtkInformationVector **,
                    vtkInformationVector *);

private:
    vtkUnstructuredGridRelevantPointsFilter(
            const vtkUnstructuredGridRelevantPointsFilter &);
    void operator=(const vtkUnstructuredGridRelevantPointsFilter &);
};
