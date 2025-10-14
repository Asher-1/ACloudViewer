// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "qPCL.h"  // For export macro
#include "vtkMultiBlockDataSetAlgorithm.h"

class vtkMultiProcessController;
class vtkUnstructuredGrid;
class vtkIdList;
class vtkFloatArray;
class vtkIdTypeArray;

class QPCL_ENGINE_LIB_API vtkPMergeConnected
    : public vtkMultiBlockDataSetAlgorithm {
public:
    static vtkPMergeConnected* New();
    vtkTypeMacro(vtkPMergeConnected, vtkMultiBlockDataSetAlgorithm);
    void PrintSelf(ostream& os, vtkIndent indent);

    struct FaceWithKey {
        int num_pts;
        vtkIdType *key, *orig;
    };
    struct cmp_ids;

protected:
    vtkPMergeConnected();
    ~vtkPMergeConnected();

    int RequestData(vtkInformation*,
                    vtkInformationVector**,
                    vtkInformationVector*);
    int FillOutputPortInformation(int port, vtkInformation* info);

private:
    vtkPMergeConnected(const vtkPMergeConnected&) = delete;
    void operator=(const vtkPMergeConnected&) = delete;

    // parallelism
    int NumProcesses;
    int MyId;
    vtkMultiProcessController* Controller;
    void SetController(vtkMultiProcessController* c);

    // filter
    void LocalToGlobalRegionId(vtkMultiProcessController* contr,
                               vtkMultiBlockDataSet* data);
    void MergeCellsOnRegionId(vtkUnstructuredGrid* ugrid,
                              int target,
                              vtkIdList* facestream);
    float MergeCellDataOnRegionId(vtkFloatArray* data_array,
                                  vtkIdTypeArray* rid_array,
                                  vtkIdType target);

    void delete_key(FaceWithKey* key);
    FaceWithKey* IdsToKey(vtkIdList* ids);
};
