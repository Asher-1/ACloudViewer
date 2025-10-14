// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "qPCL.h"  // needed for export macro"
#include "vtkPVTrivialProducer.h"

struct vtkPVTrivialProducerStaticInternal;

class QPCL_ENGINE_LIB_API vtkDistributedTrivialProducer
    : public vtkPVTrivialProducer {
public:
    static vtkDistributedTrivialProducer* New();
    vtkTypeMacro(vtkDistributedTrivialProducer, vtkPVTrivialProducer);
    void PrintSelf(ostream& os, vtkIndent indent) override;

    /**
     * Provide a global method to store a data object across processes and allow
     * a given instance of TrivialProducer to use it based on its registered
     * key.
     */
    static void SetGlobalOutput(const char* key, vtkDataObject* output);

    /**
     * Release a given Global output if a valid key (not NULL) is provided,
     * otherwise the global output map will be cleared.
     */
    static void ReleaseGlobalOutput(const char* key);

    /**
     * Update the current instance to use a previously registered global data
     * object as current output.
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
