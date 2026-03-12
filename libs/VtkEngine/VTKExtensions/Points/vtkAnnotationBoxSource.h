// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/**
 * @file vtkAnnotationBoxSource.h
 * @brief Poly data source for annotation box geometry
 */

#include "qVTK.h"
#include "vtkFiltersSourcesModule.h"  // For export macro
#include "vtkPolyDataAlgorithm.h"

/**
 * @class vtkAnnotationBoxSource
 * @brief Produces poly data for an annotation box
 */
class QVTK_ENGINE_LIB_API vtkAnnotationBoxSource : public vtkPolyDataAlgorithm {
public:
    static vtkAnnotationBoxSource *New();
    vtkTypeMacro(vtkAnnotationBoxSource, vtkPolyDataAlgorithm);

protected:
    vtkAnnotationBoxSource();
    ~vtkAnnotationBoxSource() override {}
    int RequestData(vtkInformation *,
                    vtkInformationVector **,
                    vtkInformationVector *) override;

private:
    vtkAnnotationBoxSource(const vtkAnnotationBoxSource &) = delete;
    void operator=(const vtkAnnotationBoxSource &) = delete;
};
