// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/**
 * @file vtkDistributedTrivialProducer.h
 * @brief Trivial producer with global output registration for distributed use
 */

#include "qVTK.h"  // needed for export macro"
#include "vtkPVTrivialProducer.h"

struct vtkPVTrivialProducerStaticInternal;

/**
 * @class vtkDistributedTrivialProducer
 * @brief Registers/retrieves output by key for cross-process data sharing
 */
class QVTK_ENGINE_LIB_API vtkDistributedTrivialProducer
    : public vtkPVTrivialProducer {
public:
    static vtkDistributedTrivialProducer* New();
    vtkTypeMacro(vtkDistributedTrivialProducer, vtkPVTrivialProducer);
    void PrintSelf(ostream& os, vtkIndent indent) override;

    /**
     * Store a data object globally by key for use by TrivialProducer instances.
     * @param key Registration key
     * @param output Data object to store
     */
    static void SetGlobalOutput(const char* key, vtkDataObject* output);

    /**
     * Release global output by key, or clear all if key is NULL.
     * @param key Key to release, or NULL to clear all
     */
    static void ReleaseGlobalOutput(const char* key);

    /**
     * Set this instance's output from a previously registered global object.
     * @param key Key of registered data
     */
    virtual void UpdateFromGlobal(const char* key);

protected:
    vtkDistributedTrivialProducer();
    ~vtkDistributedTrivialProducer() override;

private:
    vtkDistributedTrivialProducer(const vtkDistributedTrivialProducer&) =
            delete;
    void operator=(const vtkDistributedTrivialProducer&) = delete;

    static vtkPVTrivialProducerStaticInternal* InternalStatic;
};
