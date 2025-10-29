// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "vtkCommand.h"

template <class ClassT>
class vtkMemberFunctionCommand : public vtkCommand {
    typedef vtkMemberFunctionCommand<ClassT> ThisT;

public:
    typedef vtkCommand Superclass;

    const char* GetClassNameInternal() const override {
        return "vtkMemberFunctionCommand";
    }

    static ThisT* SafeDownCast(vtkObjectBase* o) {
        return dynamic_cast<ThisT*>(o);
    }

    static ThisT* New() { return new ThisT(); }

    void PrintSelf(ostream& os, vtkIndent indent) override {
        vtkCommand::PrintSelf(os, indent);
    }

    //@{
    /**
     * Set which class instance and member function will be called when a VTK
     * event is received.
     */
    void SetCallback(ClassT& object, void (ClassT::*method)()) {
        this->Object = &object;
        this->Method = method;
    }
    //@}

    void SetCallback(ClassT& object,
                     void (ClassT::*method2)(vtkObject*,
                                             unsigned long,
                                             void*)) {
        this->Object = &object;
        this->Method2 = method2;
    }

    void Execute(vtkObject* caller,
                 unsigned long event,
                 void* calldata) override {
        if (this->Object && this->Method) {
            (this->Object->*this->Method)();
        }
        if (this->Object && this->Method2) {
            (this->Object->*this->Method2)(caller, event, calldata);
        }
    }
    void Reset() {
        this->Object = 0;
        this->Method2 = 0;
        this->Method = 0;
    }

private:
    vtkMemberFunctionCommand() {
        this->Object = 0;
        this->Method = 0;
        this->Method2 = 0;
    }

    ~vtkMemberFunctionCommand() override {}

    ClassT* Object;
    void (ClassT::*Method)();
    void (ClassT::*Method2)(vtkObject* caller,
                            unsigned long event,
                            void* calldata);

    vtkMemberFunctionCommand(const vtkMemberFunctionCommand&) = delete;
    void operator=(const vtkMemberFunctionCommand&) = delete;
};

/**
 * Convenience function for creating vtkMemberFunctionCommand instances that
 * automatically deduces its arguments.

 * Usage:

 * vtkObject* subject = ...
 * foo* observer = ...
 * vtkCommand* adapter = vtkMakeMemberFunctionCommand(observer, &foo::bar);
 * subject->AddObserver(vtkCommand::AnyEvent, adapter);

 * See Also:
 * vtkMemberFunctionCommand, vtkCallbackCommand
 */

template <class ClassT>
vtkMemberFunctionCommand<ClassT>* vtkMakeMemberFunctionCommand(
        ClassT& object, void (ClassT::*method)()) {
    vtkMemberFunctionCommand<ClassT>* result =
            vtkMemberFunctionCommand<ClassT>::New();
    result->SetCallback(object, method);
    return result;
}

template <class ClassT>
vtkMemberFunctionCommand<ClassT>* vtkMakeMemberFunctionCommand(
        ClassT& object,
        void (ClassT::*method)(vtkObject*, unsigned long, void*)) {
    vtkMemberFunctionCommand<ClassT>* result =
            vtkMemberFunctionCommand<ClassT>::New();
    result->SetCallback(object, method);
    return result;
}
//-----------------------------------------------------------------------------
// VTK-HeaderTest-Exclude: vtkMemberFunctionCommand.h
