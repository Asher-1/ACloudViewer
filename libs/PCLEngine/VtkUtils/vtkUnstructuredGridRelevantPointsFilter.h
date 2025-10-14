// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef __vtkUnstructuredGridRelevantPointsFilter_h
#define __vtkUnstructuredGridRelevantPointsFilter_h

#include "vtkUnstructuredGridAlgorithm.h"

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

#endif
