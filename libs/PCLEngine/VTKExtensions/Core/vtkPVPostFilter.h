// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <string>  // for std::string

#include "qPCL.h"  // needed for export macro
#include "vtkDataObjectAlgorithm.h"

class QPCL_ENGINE_LIB_API vtkPVPostFilter : public vtkDataObjectAlgorithm {
public:
    static vtkPVPostFilter* New();
    vtkTypeMacro(vtkPVPostFilter, vtkDataObjectAlgorithm);
    void PrintSelf(ostream& os, vtkIndent indent) override;

    /**
     * We need to override this method because the composite data pipeline
     * is not what we want. Instead we need the PVCompositeDataPipeline
     * so that we can figure out what we conversion(s) we need to do
     */
    vtkExecutive* CreateDefaultExecutive() override;

    static std::string DefaultComponentName(int componentNumber,
                                            int componentCount);

protected:
    vtkPVPostFilter();
    ~vtkPVPostFilter() override;

    int FillInputPortInformation(int port, vtkInformation* info) override;
    int RequestDataObject(vtkInformation*,
                          vtkInformationVector**,
                          vtkInformationVector*) override;
    int RequestData(vtkInformation*,
                    vtkInformationVector**,
                    vtkInformationVector*) override;

    int DoAnyNeededConversions(vtkDataObject* output);
    int DoAnyNeededConversions(vtkDataSet* dataset);
    int DoAnyNeededConversions(vtkDataSet* output,
                               const char* requested_name,
                               int fieldAssociation,
                               const char* demangled_name,
                               const char* demagled_component_name);
    void CellDataToPointData(vtkDataSet* output, const char* name);
    void PointDataToCellData(vtkDataSet* output, const char* name);
    int ExtractComponent(vtkDataSetAttributes* dsa,
                         const char* requested_name,
                         const char* demangled_name,
                         const char* demagled_component_name);

private:
    vtkPVPostFilter(const vtkPVPostFilter&) = delete;
    void operator=(const vtkPVPostFilter&) = delete;
};
