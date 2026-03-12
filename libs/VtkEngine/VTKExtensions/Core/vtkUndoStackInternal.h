// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

/**
 * @file vtkUndoStackInternal.h
 * @brief Internal data structures for vtkUndoStack (undo/redo vectors)
 */

#include <string>
#include <vector>

#include "qVTK.h"
#include "vtkSmartPointer.h"
#include "vtkUndoSet.h"

/**
 * @class vtkUndoStackInternal
 * @brief Internal storage for vtkUndoStack undo and redo vectors
 */
class QVTK_ENGINE_LIB_API vtkUndoStackInternal {
public:
    /// Element in undo/redo stack (label + copy of undo set)
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
