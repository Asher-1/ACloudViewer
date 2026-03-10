// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/**
 * @file vtkMultiProcessControllerHelper.h
 * @brief Utilities for parallel data reduction and merging
 */

#include <vector>  // needed for std::vector

#include "qVTK.h"  // needed for export macro
#include "vtkObject.h"
#include "vtkSmartPointer.h"  // needed for vtkSmartPointer.

class vtkDataObject;
class vtkMultiProcessController;
class vtkMultiProcessStream;

/**
 * @class vtkMultiProcessControllerHelper
 * @brief Static helpers for MPI reduction and piece merging
 */
class QVTK_ENGINE_LIB_API vtkMultiProcessControllerHelper : public vtkObject {
public:
    static vtkMultiProcessControllerHelper* New();
    vtkTypeMacro(vtkMultiProcessControllerHelper, vtkObject);
    void PrintSelf(ostream& os, vtkIndent indent) override;

    /**
     * Reduce the stream to all processes using the given operation.
     * @param controller Multi-process controller
     * @param data Stream to reduce (in/out)
     * @param operation Commutative reduction function
     * @param tag MPI tag for communication
     * @return 1 on success, 0 on failure
     */
    static int ReduceToAll(vtkMultiProcessController* controller,
                           vtkMultiProcessStream& data,
                           void (*operation)(vtkMultiProcessStream& A,
                                             vtkMultiProcessStream& B),
                           int tag);

    /**
     * Merge data pieces from several processes into a single object.
     * @param pieces Array of data objects (one per process)
     * @param num_pieces Number of pieces
     * @return Merged data object or NULL on failure (caller owns)
     */
    static vtkDataObject* MergePieces(vtkDataObject** pieces,
                                      unsigned int num_pieces);

    /**
     * Merge pieces into an existing result object.
     * @param pieces Vector of data objects to merge
     * @param result Output object to store merged data
     * @return True on success
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
