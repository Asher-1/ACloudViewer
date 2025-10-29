// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <vector>  // needed for std::vector

#include "qPCL.h"  // needed for export macro
#include "vtkObject.h"
#include "vtkSmartPointer.h"  // needed for vtkSmartPointer.

class vtkDataObject;
class vtkMultiProcessController;
class vtkMultiProcessStream;

class QPCL_ENGINE_LIB_API vtkMultiProcessControllerHelper : public vtkObject {
public:
    static vtkMultiProcessControllerHelper* New();
    vtkTypeMacro(vtkMultiProcessControllerHelper, vtkObject);
    void PrintSelf(ostream& os, vtkIndent indent) override;

    /**
     * Reduce the stream to all processes calling the (*operation) for
     * reduction. The operation is assumed to be commutative.
     */
    static int ReduceToAll(vtkMultiProcessController* controller,
                           vtkMultiProcessStream& data,
                           void (*operation)(vtkMultiProcessStream& A,
                                             vtkMultiProcessStream& B),
                           int tag);

    /**
     * Utility method to merge pieces received from several processes. It does
     * not handle all data types, and hence not meant for non-paraview specific
     * use. Returns a new instance of data object containing the merged result
     * on success, else returns NULL. The caller is expected to release the
     * memory from the returned data-object.
     */
    static vtkDataObject* MergePieces(vtkDataObject** pieces,
                                      unsigned int num_pieces);

    /**
     * Overload where the merged pieces are combined into result.
     */
    static bool MergePieces(std::vector<vtkSmartPointer<vtkDataObject>>& pieces,
                            vtkDataObject* result);

protected:
    vtkMultiProcessControllerHelper();
    ~vtkMultiProcessControllerHelper() override;

private:
    vtkMultiProcessControllerHelper(const vtkMultiProcessControllerHelper&) =
            delete;
    void operator=(const vtkMultiProcessControllerHelper&) = delete;
};

// VTK-HeaderTest-Exclude: vtkMultiProcessControllerHelper.h
