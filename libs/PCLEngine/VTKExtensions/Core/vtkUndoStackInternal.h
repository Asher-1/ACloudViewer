// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <string>
#include <vector>

#include "qPCL.h"
#include "vtkSmartPointer.h"
#include "vtkUndoSet.h"

class QPCL_ENGINE_LIB_API vtkUndoStackInternal {
public:
    struct Element {
        std::string Label;
        vtkSmartPointer<vtkUndoSet> UndoSet;
        Element(const char* label, vtkUndoSet* set) {
            this->Label = label;
            this->UndoSet = vtkSmartPointer<vtkUndoSet>::New();
            for (int i = 0, nb = set->GetNumberOfElements(); i < nb; i++) {
                this->UndoSet->AddElement(set->GetElement(i));
            }
        }
    };
    typedef std::vector<Element> VectorOfElements;
    VectorOfElements UndoStack;
    VectorOfElements RedoStack;
};
//****************************************************************************
// VTK-HeaderTest-Exclude: vtkUndoStackInternal.h
