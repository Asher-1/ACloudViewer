// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef vtkCommunicationErrorCatcher_h
#define vtkCommunicationErrorCatcher_h

#include <string>

#include "qPCL.h"            // needed for export macro
#include "vtkWeakPointer.h"  // needed for vtkWeakPointer.

class vtkCommunicator;
class vtkMultiProcessController;
class vtkObject;

class QPCL_ENGINE_LIB_API vtkCommunicationErrorCatcher {
public:
    vtkCommunicationErrorCatcher(vtkMultiProcessController*);
    vtkCommunicationErrorCatcher(vtkCommunicator*);
    virtual ~vtkCommunicationErrorCatcher();

    /**
     * Get the status of errors.
     */
    bool GetErrorsRaised() const { return this->ErrorsRaised; }

    /**
     * Get the combined error messages.
     */
    const std::string& GetErrorMessages() const { return this->ErrorMessages; }

private:
    vtkCommunicationErrorCatcher(const vtkCommunicationErrorCatcher&) = delete;
    void operator=(const vtkCommunicationErrorCatcher&) = delete;

    void Initialize();
    void OnErrorEvent(vtkObject* caller, unsigned long eventid, void* calldata);

    vtkWeakPointer<vtkMultiProcessController> Controller;
    vtkWeakPointer<vtkCommunicator> Communicator;

    bool ErrorsRaised;
    std::string ErrorMessages;
    unsigned long ControllerObserverId;
    unsigned long CommunicatorObserverId;
};

#endif
// VTK-HeaderTest-Exclude: vtkCommunicationErrorCatcher.h
