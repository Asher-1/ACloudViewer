// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef vtkAnnotationBoxSource_h
#define vtkAnnotationBoxSource_h

#include "qPCL.h"
#include "vtkFiltersSourcesModule.h"  // For export macro
#include "vtkPolyDataAlgorithm.h"

class QPCL_ENGINE_LIB_API vtkAnnotationBoxSource : public vtkPolyDataAlgorithm {
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

#endif
